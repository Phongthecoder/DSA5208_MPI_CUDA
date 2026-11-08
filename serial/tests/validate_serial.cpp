#include "particle_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

using namespace tracking;

constexpr std::array<std::uint64_t, 6> data_seeds{11, 29, 47, 65, 83, 101};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_invariants(const FilterResult& result, std::size_t n) {
    for (const auto& estimate : result.estimates) {
        require(std::abs(estimate.weight_sum - 1.0) < 1e-10, "normalized weights must sum to one");
        require(estimate.ess >= 1.0 && estimate.ess <= static_cast<double>(n), "ESS outside [1, N]");
        require(std::isfinite(estimate.log_predictive_density), "nonfinite predictive density");
        for (const double value : estimate.mean) {
            require(std::isfinite(value), "nonfinite posterior mean");
        }
    }
}

void validate_edge_cases() {
    std::vector<std::size_t> ancestors;
    systematic_ancestors({0, 0.5, 0, 0.5}, 0.0, ancestors);
    require(ancestors == std::vector<std::size_t>{1, 1, 3, 3}, "wrong ancestry at a CDF boundary");
    systematic_ancestors({0, 0, 1, 0}, std::nextafter(1.0, 0.0), ancestors);
    require(ancestors == std::vector<std::size_t>{2, 2, 2, 2}, "selected a zero-weight ancestor");
    systematic_ancestors({1, 1, 1, 1}, 0.0, ancestors);
    require(ancestors == std::vector<std::size_t>{0, 1, 2, 3}, "uniform weights should retain all particles");
    systematic_ancestors({0, 3, 2, 1, 0, 1, 0, 1}, 0.2, ancestors);
    require(ancestors == std::vector<std::size_t>{1, 1, 1, 2, 2, 3, 5, 7}, "wrong eight-particle ancestry");
    require(wrap_bearing(-pi) == pi, "wrong bearing endpoint");
    require(std::abs(wrap_bearing((-pi + 0.01) - (pi - 0.01)) - 0.02) < 1e-12,
            "wrong bearing residual across the branch cut");

    const auto data = simulate(Sensor::linear, 4, 13);
    const auto longer = simulate(Sensor::linear, 8, 13);
    const auto radar = simulate(Sensor::range_bearing, 4, 13);
    require(std::equal(data.observations.begin(), data.observations.end(), longer.observations.begin()),
            "changing T must preserve the observation prefix");
    require(data.truth == radar.truth, "changing sensors must preserve truth for a fixed seed");
    FilterOptions options;
    options.particles = 1;
    options.ess_ratio = 1.0;
    const auto adaptive = particle_filter(Sensor::linear, data.observations, options);
    options.always_resample = true;
    const auto always = particle_filter(Sensor::linear, data.observations, options);
    check_invariants(adaptive, 1);
    check_invariants(always, 1);
    require(adaptive.resampled_steps == 0, "ESS comparison should be strict");
    require(always.resampled_steps == data.observations.size(), "always must resample at every step");
    for (std::size_t i = 0; i < data.observations.size(); ++i) {
        require(adaptive.estimates[i].mean == always.estimates[i].mean, "one-particle trajectory changed");
        require(adaptive.estimates[i].ess == 1.0, "one particle must have ESS of one");
    }

    options.particles = 128;
    const auto extreme = particle_filter(Sensor::linear, {{1e6, -1e6}}, options);
    check_invariants(extreme, options.particles);
    require(extreme.log_likelihood < -1e8, "failed to exercise tiny likelihoods");
    std::cout << "PASS systematic ancestry, angle wrapping, observation reproducibility, ESS and tiny likelihoods\n";
}

struct GaussianEstimate {
    State mean{};
    Covariance covariance{};
};

struct KalmanResult {
    std::vector<GaussianEstimate> estimates;
    double log_likelihood = 0.0;
};

// Exact linear oracle. The two axes are independent (position, velocity) blocks.
KalmanResult kalman_filter(const std::vector<Observation>& observations) {
    KalmanResult result;
    GaussianEstimate posterior;
    posterior.mean = prior_mean;
    for (std::size_t i = 0; i < posterior.mean.size(); ++i) {
        posterior.covariance[i][i] = prior_sd[i] * prior_sd[i];
    }
    const double measurement_var = position_sd * position_sd;
    const double q_pp = acceleration_density * dt * dt * dt / 3.0;
    const double q_pv = acceleration_density * dt * dt / 2.0;
    const double q_vv = acceleration_density * dt;
    for (const Observation& observation : observations) {
        for (std::size_t p = 0; p < 2; ++p) {
            const std::size_t v = p + 2;
            auto& covariance = posterior.covariance;
            const double pp = covariance[p][p] + 2.0 * dt * covariance[p][v] +
                              dt * dt * covariance[v][v] + q_pp;
            const double pv = covariance[p][v] + dt * covariance[v][v] + q_pv;
            const double vv = covariance[v][v] + q_vv;
            const double innovation_var = pp + measurement_var;
            posterior.mean[p] += dt * posterior.mean[v];
            const double innovation = observation[p] - posterior.mean[p];
            posterior.mean[p] += pp / innovation_var * innovation;
            posterior.mean[v] += pv / innovation_var * innovation;
            covariance[p][p] = pp * measurement_var / innovation_var;
            covariance[p][v] = covariance[v][p] = pv * measurement_var / innovation_var;
            covariance[v][v] = vv - pv * pv / innovation_var;
            result.log_likelihood -= 0.5 * (std::log(2.0 * pi * innovation_var) +
                                             innovation * innovation / innovation_var);
        }
        result.estimates.push_back(posterior);
    }
    return result;
}

double squared_distance(const State& a, const State& b, std::size_t component) {
    const double x = a[component] - b[component];
    const double y = a[component + 1] - b[component + 1];
    return x * x + y * y;
}

struct LinearMetrics {
    double position_rmse;
    double velocity_rmse;
    double covariance_rmse;
    double log_likelihood_error;
};

void validate_linear() {
    constexpr std::size_t steps = 80;
    constexpr std::array<std::size_t, 3> particle_counts{512, 4096, 65536};
    std::array<Scenario, data_seeds.size()> scenarios;
    std::array<KalmanResult, data_seeds.size()> references;
    for (std::size_t i = 0; i < data_seeds.size(); ++i) {
        scenarios[i] = simulate(Sensor::linear, steps, data_seeds[i]);
        references[i] = kalman_filter(scenarios[i].observations);
    }

    std::cout << "\nLinear Gaussian (T=" << steps << ", " << data_seeds.size() << " seeds, ESS ratio 0.5)\n"
              << "particles,position_mean_rmse_m,velocity_mean_rmse_mps,standardized_covariance_rmse,"
                 "absolute_log_likelihood_error_per_step\n";
    const double observations = static_cast<double>(steps * data_seeds.size());
    std::array<LinearMetrics, particle_counts.size()> results{};
    for (std::size_t size = 0; size < particle_counts.size(); ++size) {
        FilterOptions options;
        options.particles = particle_counts[size];
        options.estimate_covariance = true;
        double position_sq = 0.0;
        double velocity_sq = 0.0;
        double covariance_sq = 0.0;
        double log_likelihood_error = 0.0;
        for (std::size_t i = 0; i < data_seeds.size(); ++i) {
            options.seed = 1729 + i;
            const auto fit = particle_filter(Sensor::linear, scenarios[i].observations, options);
            check_invariants(fit, options.particles);
            const auto& reference = references[i];
            log_likelihood_error += std::abs(fit.log_likelihood - reference.log_likelihood);
            for (std::size_t t = 0; t < steps; ++t) {
                const auto& exact = reference.estimates[t];
                position_sq += squared_distance(fit.estimates[t].mean, exact.mean, 0);
                velocity_sq += squared_distance(fit.estimates[t].mean, exact.mean, 2);
                for (std::size_t row = 0; row < 4; ++row) {
                    for (std::size_t col = 0; col < 4; ++col) {
                        const double scale = std::sqrt(exact.covariance[row][row] * exact.covariance[col][col]);
                        const double delta = (fit.covariances[t][row][col] - exact.covariance[row][col]) / scale;
                        covariance_sq += delta * delta;
                    }
                }
            }
        }
        const LinearMetrics metrics{std::sqrt(position_sq / observations), std::sqrt(velocity_sq / observations),
                                    std::sqrt(covariance_sq / (16 * observations)),
                                    log_likelihood_error / observations};
        results[size] = metrics;
        std::cout << options.particles << ',' << metrics.position_rmse << ',' << metrics.velocity_rmse << ','
                  << metrics.covariance_rmse << ',' << metrics.log_likelihood_error << '\n';
    }
    const auto& coarse = results.front();
    const auto& fine = results.back();
    require(fine.position_rmse < 0.6 && fine.position_rmse < coarse.position_rmse,
            "position means failed Kalman convergence");
    require(fine.velocity_rmse < 0.15 && fine.velocity_rmse < coarse.velocity_rmse,
            "velocity means failed Kalman convergence");
    require(fine.covariance_rmse < 0.06 && fine.covariance_rmse < coarse.covariance_rmse,
            "covariances failed Kalman convergence");
    require(fine.log_likelihood_error < 0.035 && fine.log_likelihood_error < coarse.log_likelihood_error,
            "log likelihood failed Kalman convergence");
    std::cout << "PASS Kalman comparison\n";
}

void validate_nonlinear() {
    constexpr std::size_t steps = 200;
    FilterOptions options;
    options.particles = 8192;
    const double observations = static_cast<double>(steps * data_seeds.size());
    double measurement_sq = 0.0;
    std::array<double, 2> position_sq{};
    std::array<double, 2> velocity_sq{};
    std::array<std::size_t, 2> resampled{};
    for (std::size_t i = 0; i < data_seeds.size(); ++i) {
        const auto data = simulate(Sensor::range_bearing, steps, data_seeds[i]);
        for (std::size_t t = 0; t < steps; ++t) {
            const auto [range, bearing] = data.observations[t];
            const State measured{radar_position[0] + range * std::cos(bearing),
                                 radar_position[1] + range * std::sin(bearing), 0.0, 0.0};
            measurement_sq += squared_distance(measured, data.truth[t], 0);
        }
        options.seed = 1729 + i;
        for (std::size_t policy = 0; policy < resampled.size(); ++policy) {
            options.always_resample = policy == 1;
            const auto fit = particle_filter(Sensor::range_bearing, data.observations, options);
            check_invariants(fit, options.particles);
            resampled[policy] += fit.resampled_steps;
            for (std::size_t t = 0; t < steps; ++t) {
                position_sq[policy] += squared_distance(fit.estimates[t].mean, data.truth[t], 0);
                velocity_sq[policy] += squared_distance(fit.estimates[t].mean, data.truth[t], 2);
            }
        }
    }
    std::cout << "\nRange-bearing (N=" << options.particles << ", T=" << steps << ", " << data_seeds.size()
              << " seeds)\npolicy,position_rmse_m,velocity_rmse_mps,mean_resampled_steps\n"
              << "raw_measurements," << std::sqrt(measurement_sq / observations) << ",,\n";
    for (std::size_t policy = 0; policy < resampled.size(); ++policy) {
        const double position_rmse = std::sqrt(position_sq[policy] / observations);
        const double velocity_rmse = std::sqrt(velocity_sq[policy] / observations);
        std::cout << (policy == 0 ? "ess_0.5" : "always") << ',' << position_rmse << ',' << velocity_rmse << ','
                  << static_cast<double>(resampled[policy]) / data_seeds.size() << '\n';
        require(position_sq[policy] < measurement_sq && velocity_rmse < 3.0,
                "particle estimates failed nonlinear tracking accuracy check");
    }
    require(resampled[0] > 0 && resampled[0] < resampled[1] && resampled[1] == steps * data_seeds.size(),
            "incorrect nonlinear resampling counts");
    std::cout << "PASS nonlinear tracking and resampling policies\n";
}

}  // namespace

int main() {
    try {
        std::cout << std::fixed << std::setprecision(6);
        validate_edge_cases();
        validate_linear();
        validate_nonlinear();
        std::cout << "\nAll serial validation checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Validation failed: " << error.what() << '\n';
        return 1;
    }
}

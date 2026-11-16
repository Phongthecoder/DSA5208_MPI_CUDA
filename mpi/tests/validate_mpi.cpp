#include "mpi_filter.hpp"
#include "philox.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

using namespace tracking;
using namespace tracking::parallel;
constexpr std::array<std::uint64_t, 6> seeds{11, 29, 47, 65, 83, 101};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool near(double a, double b) {
    return std::abs(a - b) <= 1e-9 * std::max({1.0, std::abs(a), std::abs(b)});
}

void validate_resampling_case(const std::vector<double>& weights, double u,
                              const std::vector<std::size_t>& expected) {
    const ParticlePartition partition(weights.size(), MPI_COMM_WORLD);
    std::vector<std::size_t> serial;
    systematic_ancestors(weights, u, serial);
    require(serial == expected, "serial oracle has unexpected ancestors");
    std::vector<State> states(partition.local_size()), output(states.size());
    std::vector<double> local_weights(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) {
        const auto id = partition.first() + i;
        states[i] = {static_cast<double>(id), -static_cast<double>(id), 7.0, 11.0};
        local_weights[i] = weights[id];
    }
    const double mass = std::accumulate(local_weights.begin(), local_weights.end(), 0.0);
    for (const auto method : {Resampling::centralized, Resampling::distributed}) {
        SystematicResampler resampler(partition, method, MPI_COMM_WORLD);
        const auto stats = resampler.run(states, local_weights, mass, u, output);
        require(output.size() == states.size(), "output partition size changed");
        for (std::size_t i = 0; i < output.size(); ++i) {
            const auto id = static_cast<double>(expected[partition.first() + i]);
            require(output[i] == State{id, -id, 7.0, 11.0}, "systematic ancestor or slot differs");
        }
        std::uint64_t actual_bytes = 0;
        MPI_Allreduce(&stats.off_rank_bytes, &actual_bytes, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        std::uint64_t expected_bytes = 0;
        if (method == Resampling::centralized) {
            expected_bytes = (weights.size() - partition.counts[0]) * (2 * sizeof(State) + sizeof(double));
        } else {
            for (std::size_t slot = 0; slot < expected.size(); ++slot) {
                const auto destination = std::upper_bound(partition.offsets.begin(), partition.offsets.end(),
                                                          static_cast<int>(slot));
                const auto source = std::upper_bound(partition.offsets.begin(), partition.offsets.end(),
                                                     static_cast<int>(expected[slot]));
                expected_bytes += source != destination ? sizeof(State) : 0;
            }
        }
        require(actual_bytes == expected_bytes, "resampling payload counter differs from the oracle");
    }
}

void validate_invariants(const FilterResult& fit, const FilterOptions& options) {
    for (const auto& estimate : fit.estimates) {
        require(std::abs(estimate.weight_sum - 1.0) < 1e-10, "global weights do not sum to one");
        require(estimate.ess >= 1.0 && estimate.ess <= options.particles, "global ESS is out of range");
        require(std::isfinite(estimate.log_predictive_density), "predictive log density is nonfinite");
        for (const auto value : estimate.mean) {
            require(std::isfinite(value), "state mean is nonfinite");
        }
    }
}

void compare_filters(const FilterResult& reference, const FilterResult& fit) {
    require(reference.resampled_steps == fit.resampled_steps, "number of resamples changed");
    require(near(reference.log_likelihood, fit.log_likelihood), "log likelihood changed");
    for (std::size_t t = 0; t < fit.estimates.size(); ++t) {
        require(near(reference.estimates[t].ess, fit.estimates[t].ess), "ESS changed");
        require(reference.estimates[t].resampled == fit.estimates[t].resampled, "resampling decision changed");
        for (std::size_t row = 0; row < 4; ++row) {
            require(near(reference.estimates[t].mean[row], fit.estimates[t].mean[row]), "posterior mean changed");
            for (std::size_t col = 0; col < 4; ++col) {
                require(near(reference.covariances[t][row][col], fit.covariances[t][row][col]),
                        "posterior covariance changed");
            }
        }
    }
}

void validate_parallel_filter() {
    require(philox(0, 0, 0) == std::array<std::uint32_t, 4>{0x6627e8d5, 0xe169c58d,
                                                              0xbc57ac4c, 0x9b00dbd8},
            "Philox known answer differs");
    validate_resampling_case({0, 3, 2, 1, 0, 1, 0, 1}, 0.2, {1, 1, 1, 2, 2, 3, 5, 7});
    validate_resampling_case({0, 0.5, 0, 0.5}, 0.0, {1, 1, 3, 3});
    validate_resampling_case({0, 0, 1, 0}, std::nextafter(1.0, 0.0), {2, 2, 2, 2});
    validate_resampling_case({0, 0, 0, 0, 0, 0, 1}, 0.0, {6, 6, 6, 6, 6, 6, 6});
    validate_resampling_case({1}, 0.0, {0});

    const auto data = simulate_on_root(Sensor::range_bearing, 48, 42, MPI_COMM_WORLD);
    FilterOptions options;
    options.particles = 2049;
    options.estimate_covariance = true;
    const auto one = particle_filter_mpi(Sensor::range_bearing, data.observations, options,
                                          Resampling::distributed, MPI_COMM_SELF);
    for (const auto method : {Resampling::centralized, Resampling::distributed}) {
        const auto result = particle_filter_mpi(Sensor::range_bearing, data.observations, options,
                                                 method, MPI_COMM_WORLD);
        validate_invariants(result.filter, options);
        compare_filters(one.filter, result.filter);
    }
    options.particles = 3;
    options.always_resample = true;
    const auto extreme = particle_filter_mpi(Sensor::linear, {{1e6, -1e6}}, options,
                                              Resampling::distributed, MPI_COMM_WORLD);
    validate_invariants(extreme.filter, options);
    require(extreme.filter.resampled_steps == 1, "every-step policy skipped resampling");
}

// Exact linear-Gaussian update for independent (position, velocity) axis pairs.
FilterResult kalman_oracle(const std::vector<Observation>& observations) {
    FilterResult reference;
    State mean = prior_mean;
    Covariance covariance{};
    for (std::size_t j = 0; j < 4; ++j) {
        covariance[j][j] = prior_sd[j] * prior_sd[j];
    }
    const double r = position_sd * position_sd;
    for (const auto& observation : observations) {
        for (std::size_t p = 0; p < 2; ++p) {
            const auto v = p + 2;
            const double pp = covariance[p][p] + 2.0 * dt * covariance[p][v] + dt * dt * covariance[v][v]
                              + acceleration_density * dt * dt * dt / 3.0;
            const double pv = covariance[p][v] + dt * covariance[v][v] + acceleration_density * dt * dt / 2.0;
            const double vv = covariance[v][v] + acceleration_density * dt;
            const double predicted = mean[p] + dt * mean[v];
            const double error = observation[p] - predicted;
            const double innovation = pp + r;
            mean[p] = predicted + pp * error / innovation;
            mean[v] += pv * error / innovation;
            covariance[p][p] = pp * r / innovation;
            covariance[v][p] = covariance[p][v] = pv * r / innovation;
            covariance[v][v] = vv - pv * pv / innovation;
            reference.log_likelihood -= 0.5 * (std::log(2.0 * pi * innovation) + error * error / innovation);
        }
        Estimate estimate;
        estimate.mean = mean;
        reference.estimates.push_back(estimate);
        reference.covariances.push_back(covariance);
    }
    return reference;
}

double squared_error(const State& left, const State& right, std::size_t first) {
    const double a = left[first] - right[first];
    const double b = left[first + 1] - right[first + 1];
    return a * a + b * b;
}

void validate_kalman(int rank) {
    constexpr int steps = 80;
    std::array<double, 4> coarse{};
    for (const std::size_t particles : {512, 65536}) {
        std::array<double, 4> error{};
        for (std::size_t i = 0; i < seeds.size(); ++i) {
            const auto data = simulate_on_root(Sensor::linear, steps, seeds[i], MPI_COMM_WORLD);
            const auto exact = kalman_oracle(data.observations);
            FilterOptions options;
            options.particles = particles;
            options.seed += i;
            options.estimate_covariance = true;
            const auto result = particle_filter_mpi(Sensor::linear, data.observations, options,
                                                     Resampling::distributed, MPI_COMM_WORLD);
            const auto& fit = result.filter;
            validate_invariants(fit, options);
            error[3] += std::abs(fit.log_likelihood - exact.log_likelihood);
            for (std::size_t t = 0; t < fit.estimates.size(); ++t) {
                error[0] += squared_error(fit.estimates[t].mean, exact.estimates[t].mean, 0);
                error[1] += squared_error(fit.estimates[t].mean, exact.estimates[t].mean, 2);
                for (std::size_t row = 0; row < 4; ++row) {
                    for (std::size_t col = 0; col < 4; ++col) {
                        const auto& cov = exact.covariances[t];
                        const double delta = fit.covariances[t][row][col] - cov[row][col];
                        error[2] += delta * delta / (cov[row][row] * cov[col][col]);
                    }
                }
            }
        }
        const double records = steps * seeds.size();
        error = {std::sqrt(error[0] / records), std::sqrt(error[1] / records),
                 std::sqrt(error[2] / (16 * records)), error[3] / records};
        if (rank == 0) {
            std::cout << "Kalman comparison N=" << particles << " position RMSE=" << error[0]
                      << " m, velocity RMSE=" << error[1] << " m/s, standardized covariance RMSE="
                      << error[2] << ", log-likelihood error/step=" << error[3] << '\n';
        }
        if (particles == 512) {
            coarse = error;
        } else {
            const std::array<double, 4> tolerances{0.6, 0.15, 0.06, 0.035};
            for (std::size_t i = 0; i < error.size(); ++i) {
                require(error[i] < coarse[i] && error[i] < tolerances[i], "MPI filter failed Kalman convergence");
            }
        }
    }
}

void validate_nonlinear(int rank) {
    constexpr int steps = 200;
    double mpi_error = 0.0;
    double serial_error = 0.0;
    double raw_error = 0.0;
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        const auto data = simulate_on_root(Sensor::range_bearing, steps, seeds[i], MPI_COMM_WORLD);
        FilterOptions options;
        options.particles = 8192;
        options.seed += i;
        const auto fit = particle_filter_mpi(Sensor::range_bearing, data.observations, options,
                                              Resampling::distributed, MPI_COMM_WORLD);
        if (rank == 0) {
            const auto reference = particle_filter(Sensor::range_bearing, data.observations, options);
            for (std::size_t t = 0; t < data.truth.size(); ++t) {
                const auto [range, angle] = data.observations[t];
                const State raw{radar_position[0] + range * std::cos(angle),
                                radar_position[1] + range * std::sin(angle), 0.0, 0.0};
                mpi_error += squared_error(fit.filter.estimates[t].mean, data.truth[t], 0);
                serial_error += squared_error(reference.estimates[t].mean, data.truth[t], 0);
                raw_error += squared_error(raw, data.truth[t], 0);
            }
        }
    }
    if (rank == 0) {
        const double records = steps * seeds.size();
        const double mpi_rmse = std::sqrt(mpi_error / records);
        const double serial_rmse = std::sqrt(serial_error / records);
        const double raw_rmse = std::sqrt(raw_error / records);
        std::cout << "Range-bearing position RMSE over six tracks: MPI=" << mpi_rmse
                  << " m, serial=" << serial_rmse << " m, observations=" << raw_rmse << " m\n";
        require(mpi_rmse < raw_rmse && std::abs(mpi_rmse - serial_rmse) < 0.25 * serial_rmse,
                "MPI tracking error is inconsistent with the serial reference");
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        validate_parallel_filter();
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << "PASS Philox, exact systematic ancestry and payload, rank-count consistency and invariants\n";
        }
        validate_kalman(rank);
        validate_nonlinear(rank);
        if (rank == 0) {
            std::cout << "All MPI checks passed on " << ranks << " ranks.\n";
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MPI validation rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}

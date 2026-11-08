#include "particle_filter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace tracking {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class StateSampler {
public:
    explicit StateSampler(std::uint64_t seed) : engine_(seed) {}

    State draw_normals() {
        return {normal_(engine_), normal_(engine_), normal_(engine_), normal_(engine_)};
    }

    State prior() { return prior(draw_normals()); }

    static State prior(const State& noise) {
        State state{};
        for (std::size_t i = 0; i < state.size(); ++i) {
            state[i] = prior_mean[i] + prior_sd[i] * noise[i];
        }
        return state;
    }

    void propagate(State& state) { propagate(state, draw_normals()); }

    void propagate(State& state, const State& noise) const {
        for (std::size_t axis = 0; axis < 2; ++axis) {
            const double z_position = noise[2 * axis];
            const double z_velocity = noise[2 * axis + 1];
            state[axis] += dt * state[axis + 2] + position_noise_sd_ * z_position;
            state[axis + 2] += shared_noise_scale_ * z_position + velocity_noise_sd_ * z_velocity;
        }
    }

private:
    std::mt19937_64 engine_;
    std::normal_distribution<double> normal_;
    // Cholesky factors of the two identical (position, velocity) blocks of Q.
    const double position_noise_sd_ = std::sqrt(acceleration_density * dt * dt * dt / 3.0);
    const double shared_noise_scale_ = acceleration_density * dt * dt / (2.0 * position_noise_sd_);
    const double velocity_noise_sd_ = std::sqrt(acceleration_density * dt -
                                               shared_noise_scale_ * shared_noise_scale_);
};

Observation expected_observation(Sensor sensor, const State& state) {
    if (sensor == Sensor::linear) {
        return {state[0], state[1]};
    }
    const double x = state[0] - radar_position[0];
    const double y = state[1] - radar_position[1];
    return {std::hypot(x, y), std::atan2(y, x)};
}

Observation observation_sd(Sensor sensor) {
    if (sensor == Sensor::linear) {
        return {position_sd, position_sd};
    }
    return {range_sd, bearing_sd};
}

}  // namespace

double wrap_bearing(double angle) {
    const double wrapped = std::remainder(angle, 2.0 * pi);
    return wrapped <= -pi ? pi : wrapped;
}

Scenario simulate(Sensor sensor, std::size_t steps, std::uint64_t seed) {
    StateSampler motion(seed);
    std::mt19937_64 measurement_engine(seed ^ 0x9e3779b97f4a7c15ULL);
    std::normal_distribution<double> normal;
    const Observation sd = observation_sd(sensor);
    Scenario data;
    data.truth.reserve(steps);
    data.observations.reserve(steps);

    State truth = motion.prior();
    for (std::size_t step = 0; step < steps; ++step) {
        motion.propagate(truth);
        Observation observation = expected_observation(sensor, truth);
        observation[0] += sd[0] * normal(measurement_engine);
        observation[1] += sd[1] * normal(measurement_engine);
        if (sensor == Sensor::range_bearing) {
            observation[1] = wrap_bearing(observation[1]);
        }
        data.truth.push_back(truth);
        data.observations.push_back(observation);
    }
    return data;
}

void systematic_ancestors(const std::vector<double>& weights, double u,
                          std::vector<std::size_t>& ancestors) {
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(total > 0.0 && std::isfinite(total) && u >= 0.0 && u < 1.0)) {
        throw std::invalid_argument("systematic resampling requires positive weight mass and u in [0, 1)");
    }
    ancestors.resize(weights.size());
    const double count = static_cast<double>(weights.size());
    // At large k, rounding can put the final point exactly at total mass.
    const double last_point = std::nextafter(total, 0.0);
    std::size_t parent = 0;
    double cumulative = weights[0];
    for (std::size_t k = 0; k < weights.size(); ++k) {
        const double point = std::min((static_cast<double>(k) + u) / count * total, last_point);
        while (cumulative <= point) {
            cumulative += weights[++parent];
        }
        ancestors[k] = parent;
    }
}

FilterResult particle_filter(Sensor sensor, const std::vector<Observation>& observations,
                             const FilterOptions& options) {
    const auto started = Clock::now();
    if (options.particles == 0 || observations.empty()) {
        throw std::invalid_argument("the particle and observation counts must be positive");
    }
    if (!(options.ess_ratio >= 0.0 && options.ess_ratio <= 1.0)) {
        throw std::invalid_argument("ESS ratio must be in [0, 1]");
    }

    const auto n = options.particles;
    const auto count = static_cast<double>(n);
    const double uniform_log_weight = -std::log(count);
    std::vector<State> particles(n);
    std::vector<State> resampled_particles(n);
    std::vector<double> log_weights(n, uniform_log_weight);
    std::vector<double> weights(n);
    std::vector<std::size_t> ancestors(n);
    StateSampler motion(options.seed ^ 0xd1b54a32d192ed03ULL);
    std::mt19937_64 resampling_engine(options.seed ^ 0x94d049bb133111ebULL);
    constexpr std::size_t batch_particles = 1024;
    std::vector<State> noise_batch(options.profile_gaussian ? std::min(n, batch_particles) : 0);
    const auto for_each_particle = [&](auto apply_noise) {
        if (noise_batch.empty()) {
            for (std::size_t i = 0; i < n; ++i) {
                apply_noise(i, motion.draw_normals());
            }
            return 0.0;
        }

        double gaussian_ms = 0.0;
        for (std::size_t offset = 0; offset < n;) {
            const auto batch_size = std::min(noise_batch.size(), n - offset);
            // Time the draws in batches to keep clock calls out of the per-particle loop.
            const auto batch_started = Clock::now();
            std::generate_n(noise_batch.begin(), batch_size, [&] { return motion.draw_normals(); });
            gaussian_ms += milliseconds(batch_started, Clock::now());
            for (std::size_t j = 0; j < batch_size; ++j) {
                apply_noise(offset + j, noise_batch[j]);
            }
            offset += batch_size;
        }
        return gaussian_ms;
    };

    FilterResult result;
    result.times.gaussian_prior_ms = for_each_particle([&](std::size_t i, const State& noise) {
        particles[i] = StateSampler::prior(noise);
    });
    result.estimates.resize(observations.size());
    if (options.estimate_covariance) {
        result.covariances.resize(observations.size());
    }
    result.particle_buffer_bytes = sizeof(State) * (particles.capacity() + resampled_particles.capacity()) +
                                   sizeof(double) * (weights.capacity() + log_weights.capacity()) +
                                   sizeof(std::size_t) * ancestors.capacity();
    const Observation sd = observation_sd(sensor);
    const double inverse_sd0 = 1.0 / sd[0];
    const double inverse_sd1 = 1.0 / sd[1];
    const double log_density_constant = -std::log(2.0 * pi * sd[0] * sd[1]);
    auto phase_start = Clock::now();
    result.times.initialize_ms = milliseconds(started, phase_start);

    for (std::size_t step = 0; step < observations.size(); ++step) {
        const auto& observation = observations[step];
        if (!std::isfinite(observation[0]) || !std::isfinite(observation[1])) {
            throw std::invalid_argument("observations must be finite");
        }
        double max_log_weight = -std::numeric_limits<double>::infinity();
        result.times.gaussian_motion_ms += for_each_particle([&](std::size_t i, const State& noise) {
            motion.propagate(particles[i], noise);
            const Observation predicted = expected_observation(sensor, particles[i]);
            const double error0 = (observation[0] - predicted[0]) * inverse_sd0;
            double residual1 = observation[1] - predicted[1];
            if (sensor == Sensor::range_bearing) {
                residual1 = wrap_bearing(residual1);
            }
            const double error1 = residual1 * inverse_sd1;
            log_weights[i] += log_density_constant - 0.5 * (error0 * error0 + error1 * error1);
            max_log_weight = std::max(max_log_weight, log_weights[i]);
        });
        const auto weighted = Clock::now();
        result.times.propagate_weight_ms += milliseconds(phase_start, weighted);

        if (!std::isfinite(max_log_weight)) {
            throw std::runtime_error("no particle has finite likelihood at an observation");
        }
        double shifted_sum = 0.0;
        double shifted_square_sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double shifted = std::exp(log_weights[i] - max_log_weight);
            weights[i] = shifted;
            shifted_sum += shifted;
            shifted_square_sum += shifted * shifted;
        }
        const double log_shifted_sum = std::log(shifted_sum);
        const double inverse_sum = 1.0 / shifted_sum;
        for (std::size_t i = 0; i < n; ++i) {
            weights[i] *= inverse_sum;
            log_weights[i] = (log_weights[i] - max_log_weight) - log_shifted_sum;
        }
        auto& estimate = result.estimates[step];
        // Clamp only floating-point roundoff at the theoretical ESS limits.
        estimate.ess = std::clamp(shifted_sum * shifted_sum / shifted_square_sum, 1.0, count);
        estimate.log_predictive_density = max_log_weight + log_shifted_sum;
        result.log_likelihood += estimate.log_predictive_density;
        const auto normalized = Clock::now();
        result.times.normalize_ess_ms += milliseconds(weighted, normalized);

        for (std::size_t i = 0; i < n; ++i) {
            estimate.weight_sum += weights[i];
            for (std::size_t component = 0; component < estimate.mean.size(); ++component) {
                estimate.mean[component] += weights[i] * particles[i][component];
            }
        }
        if (options.estimate_covariance) {
            auto& covariance = result.covariances[step];
            for (std::size_t i = 0; i < n; ++i) {
                State delta{};
                for (std::size_t component = 0; component < delta.size(); ++component) {
                    delta[component] = particles[i][component] - estimate.mean[component];
                }
                for (std::size_t row = 0; row < delta.size(); ++row) {
                    for (std::size_t col = row; col < delta.size(); ++col) {
                        covariance[row][col] += weights[i] * delta[row] * delta[col];
                    }
                }
            }
            for (std::size_t row = 0; row < covariance.size(); ++row) {
                for (std::size_t col = 0; col < row; ++col) {
                    covariance[row][col] = covariance[col][row];
                }
            }
        }
        const auto estimated = Clock::now();
        result.times.estimate_ms += milliseconds(normalized, estimated);

        if (options.always_resample || estimate.ess < options.ess_ratio * count) {
            // Use the high 53 random bits to obtain a double strictly below one.
            const double u = static_cast<double>(resampling_engine() >> 11) * 0x1.0p-53;
            systematic_ancestors(weights, u, ancestors);
            for (std::size_t i = 0; i < n; ++i) {
                resampled_particles[i] = particles[ancestors[i]];
            }
            particles.swap(resampled_particles);
            std::fill(log_weights.begin(), log_weights.end(), uniform_log_weight);
            estimate.resampled = true;
            ++result.resampled_steps;
            phase_start = Clock::now();
            result.times.resample_ms += milliseconds(estimated, phase_start);
        } else {
            phase_start = estimated;
        }
    }
    result.times.filter_ms = milliseconds(started, Clock::now());
    return result;
}

}  // namespace tracking

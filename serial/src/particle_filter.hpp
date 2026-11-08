#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tracking {

using State = std::array<double, 4>;  // px, py, vx, vy
using Observation = std::array<double, 2>;
using Covariance = std::array<State, 4>;

enum class Sensor { linear, range_bearing };

inline constexpr double pi = 3.14159265358979323846;
inline constexpr double dt = 1.0;
inline constexpr double acceleration_density = 0.04;
inline constexpr State prior_mean{-250.0, -60.0, 2.0, 3.0};
inline constexpr State prior_sd{20.0, 20.0, 1.5, 1.5};
inline constexpr Observation radar_position{0.0, 0.0};
inline constexpr double position_sd = 10.0;
inline constexpr double range_sd = 8.0;
inline constexpr double bearing_sd = 0.04;

struct Scenario {
    std::vector<State> truth;
    std::vector<Observation> observations;
};

struct FilterOptions {
    std::size_t particles = 10000;
    std::uint64_t seed = 1729;
    double ess_ratio = 0.5;
    bool always_resample = false;
    bool estimate_covariance = false;
};

struct Estimate {
    State mean{};
    double weight_sum = 0.0;
    double ess = 0.0;
    double log_predictive_density = 0.0;
    bool resampled = false;
};

struct PhaseTimes {
    double initialize_ms = 0.0;
    double propagate_weight_ms = 0.0;
    double normalize_ess_ms = 0.0;
    double estimate_ms = 0.0;
    double resample_ms = 0.0;
    double filter_ms = 0.0;
};

struct FilterResult {
    std::vector<Estimate> estimates;
    std::vector<Covariance> covariances;
    double log_likelihood = 0.0;
    std::size_t resampled_steps = 0;
    std::size_t particle_buffer_bytes = 0;
    PhaseTimes times;
};

double wrap_bearing(double angle);

// Entry 0 is time step 1. The shared Gaussian prior is at time step 0.
Scenario simulate(Sensor sensor, std::size_t steps, std::uint64_t seed);
FilterResult particle_filter(Sensor sensor, const std::vector<Observation>& observations,
                             const FilterOptions& options);

// Weights are finite and nonnegative with positive total mass. For u in [0, 1),
// point (k + u) / N belongs to the first ancestor whose normalized CDF exceeds it.
void systematic_ancestors(const std::vector<double>& weights, double u,
                          std::vector<std::size_t>& ancestors);

}  // namespace tracking

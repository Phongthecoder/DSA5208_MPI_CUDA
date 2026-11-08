#include "particle_filter.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    tracking::Sensor sensor = tracking::Sensor::range_bearing;
    std::size_t steps = 200;
    std::uint64_t data_seed = 42;
    tracking::FilterOptions filter;
    std::string trajectory_path;
};

template <typename UInt>
UInt parse_unsigned(std::string_view value, std::string_view flag) {
    UInt parsed{};
    const auto end = value.data() + value.size();
    const auto [ptr, error] = std::from_chars(value.data(), end, parsed);
    if (error != std::errc{} || ptr != end) {
        throw std::invalid_argument(std::string(flag) + " requires an unsigned integer");
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--profile-gaussian") {
            options.filter.profile_gaussian = true;
            continue;
        }
        if (i + 1 == argc) {
            throw std::invalid_argument("missing value for " + std::string(flag));
        }
        const std::string_view value = argv[++i];
        if (flag == "--model") {
            if (value == "linear") {
                options.sensor = tracking::Sensor::linear;
            } else if (value == "range-bearing") {
                options.sensor = tracking::Sensor::range_bearing;
            } else {
                throw std::invalid_argument("--model must be linear or range-bearing");
            }
        } else if (flag == "--particles") {
            options.filter.particles = parse_unsigned<std::size_t>(value, flag);
        } else if (flag == "--steps") {
            options.steps = parse_unsigned<std::size_t>(value, flag);
        } else if (flag == "--data-seed") {
            options.data_seed = parse_unsigned<std::uint64_t>(value, flag);
        } else if (flag == "--filter-seed") {
            options.filter.seed = parse_unsigned<std::uint64_t>(value, flag);
        } else if (flag == "--resample") {
            options.filter.always_resample = value == "always";
            if (!options.filter.always_resample) {
                std::size_t consumed = 0;
                options.filter.ess_ratio = std::stod(std::string(value), &consumed);
                if (consumed != value.size() || !(options.filter.ess_ratio >= 0.0 &&
                                                  options.filter.ess_ratio <= 1.0)) {
                    throw std::invalid_argument("--resample must be always or an ESS ratio in [0, 1]");
                }
            }
        } else if (flag == "--trajectory") {
            options.trajectory_path = value;
        } else {
            throw std::invalid_argument("unknown option " + std::string(flag));
        }
    }
    if (options.steps == 0 || options.filter.particles == 0) {
        throw std::invalid_argument("--steps and --particles must be positive");
    }
    return options;
}

void write_trajectory(const std::string& path, tracking::Sensor sensor, const tracking::Scenario& data,
                      const tracking::FilterResult& fit) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open trajectory file " + path);
    }
    const char* sensor_columns = sensor == tracking::Sensor::linear ? "observed_x_m,observed_y_m"
                                                                   : "observed_range_m,observed_bearing_rad";
    out << "step,time_s,true_x_m,true_y_m,true_vx_mps,true_vy_mps," << sensor_columns
        << ",estimated_x_m,estimated_y_m,estimated_vx_mps,estimated_vy_mps,ess,weight_sum,resampled,"
           "log_predictive_density\n"
        << std::setprecision(17);
    for (std::size_t i = 0; i < data.truth.size(); ++i) {
        out << i + 1 << ',' << static_cast<double>(i + 1) * tracking::dt;
        for (double value : data.truth[i]) {
            out << ',' << value;
        }
        for (double value : data.observations[i]) {
            out << ',' << value;
        }
        const auto& estimate = fit.estimates[i];
        for (double value : estimate.mean) {
            out << ',' << value;
        }
        out << ',' << estimate.ess << ',' << estimate.weight_sum << ',' << estimate.resampled
            << ',' << estimate.log_predictive_density << '\n';
    }
    out.close();
    if (!out) {
        throw std::runtime_error("failed to write trajectory file " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: serial_pf [--model linear|range-bearing] [--particles N] [--steps T]\n"
                     "                 [--data-seed SEED] [--filter-seed SEED]\n"
                     "                 [--resample RATIO|always] [--trajectory FILE] [--profile-gaussian]\n";
        return 0;
    }
    try {
        const Options options = parse_options(argc, argv);
        const auto started = Clock::now();
        const auto data = tracking::simulate(options.sensor, options.steps, options.data_seed);
        const auto simulated = Clock::now();
        const auto fit = tracking::particle_filter(options.sensor, data.observations, options.filter);

        double position_error_sq = 0.0;
        double velocity_error_sq = 0.0;
        for (std::size_t i = 0; i < data.truth.size(); ++i) {
            const auto& mean = fit.estimates[i].mean;
            const auto& truth = data.truth[i];
            for (std::size_t axis = 0; axis < 2; ++axis) {
                const double position_error = mean[axis] - truth[axis];
                const double velocity_error = mean[axis + 2] - truth[axis + 2];
                position_error_sq += position_error * position_error;
                velocity_error_sq += velocity_error * velocity_error;
            }
        }
        const double position_rmse = std::sqrt(position_error_sq / options.steps);
        const double velocity_rmse = std::sqrt(velocity_error_sq / options.steps);
        if (!options.trajectory_path.empty()) {
            write_trajectory(options.trajectory_path, options.sensor, data, fit);
        }

        const double workflow_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        const double simulate_ms = std::chrono::duration<double, std::milli>(simulated - started).count();
        const auto& times = fit.times;
        const char* profile_columns = options.filter.profile_gaussian
                                         ? ",gaussian_prior_ms,gaussian_motion_ms"
                                         : "";
        std::cout << "model,particles,steps,data_seed,filter_seed,resampling,position_rmse_m,velocity_rmse_mps,"
                     "log_likelihood,resampled_steps,particle_buffer_bytes,simulate_ms,initialize_ms,"
                     "propagate_weight_ms,normalize_ess_ms,estimate_ms,resample_ms,filter_ms,workflow_ms"
                  << profile_columns << '\n'
                  << std::setprecision(12)
                  << (options.sensor == tracking::Sensor::linear ? "linear" : "range-bearing") << ','
                  << options.filter.particles << ',' << options.steps << ',' << options.data_seed << ','
                  << options.filter.seed << ',';
        if (options.filter.always_resample) {
            std::cout << "always";
        } else {
            std::cout << options.filter.ess_ratio;
        }
        std::cout << ',' << position_rmse << ',' << velocity_rmse << ',' << fit.log_likelihood << ','
                  << fit.resampled_steps << ',' << fit.particle_buffer_bytes << ',' << simulate_ms << ','
                  << times.initialize_ms << ',' << times.propagate_weight_ms << ',' << times.normalize_ess_ms
                  << ',' << times.estimate_ms << ',' << times.resample_ms << ',' << times.filter_ms << ','
                  << workflow_ms;
        if (options.filter.profile_gaussian) {
            std::cout << ',' << times.gaussian_prior_ms << ',' << times.gaussian_motion_ms;
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "serial_pf: " << error.what() << '\n';
        return 1;
    }
}

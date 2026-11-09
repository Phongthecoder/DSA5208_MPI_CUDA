#include "mpi_filter.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace tracking;
using namespace tracking::parallel;

struct Options {
    Sensor sensor = Sensor::range_bearing;
    Resampling resampler = Resampling::distributed;
    FilterOptions filter;
    std::size_t steps = 256;
    std::uint64_t data_seed = 42;
    std::string trajectory;
};

template <typename UInt>
UInt unsigned_value(std::string_view value, std::string_view flag) {
    UInt parsed{};
    const auto* end = value.data() + value.size();
    const auto [ptr, error] = std::from_chars(value.data(), end, parsed);
    if (error != std::errc{} || ptr != end) {
        throw std::invalid_argument(std::string(flag) + " requires an unsigned integer");
    }
    return parsed;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i += 2) {
        const std::string_view flag = argv[i];
        if (i + 1 == argc) {
            throw std::invalid_argument("missing value for " + std::string(flag));
        }
        const std::string_view value = argv[i + 1];
        if (flag == "--model") {
            if (value == "range-bearing") {
                options.sensor = Sensor::range_bearing;
            } else if (value == "linear") {
                options.sensor = Sensor::linear;
            } else {
                throw std::invalid_argument("--model must be range-bearing or linear");
            }
        } else if (flag == "--resampler") {
            if (value == "centralized") {
                options.resampler = Resampling::centralized;
            } else if (value == "distributed") {
                options.resampler = Resampling::distributed;
            } else {
                throw std::invalid_argument("--resampler must be centralized or distributed");
            }
        } else if (flag == "--particles") {
            options.filter.particles = unsigned_value<std::size_t>(value, flag);
        } else if (flag == "--steps") {
            options.steps = unsigned_value<std::size_t>(value, flag);
        } else if (flag == "--data-seed") {
            options.data_seed = unsigned_value<std::uint64_t>(value, flag);
        } else if (flag == "--filter-seed") {
            options.filter.seed = unsigned_value<std::uint64_t>(value, flag);
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
            options.trajectory = value;
        } else {
            throw std::invalid_argument("unknown option " + std::string(flag));
        }
    }
    return options;
}

void write_track(const Options& options, const Scenario& data, const FilterResult& fit) {
    std::ofstream file(options.trajectory);
    if (!file) {
        throw std::runtime_error("cannot open trajectory file " + options.trajectory);
    }
    const char* observation_columns = options.sensor == Sensor::linear ? "observed_x_m,observed_y_m"
                                                                      : "observed_range_m,observed_bearing_rad";
    file << "step,time_s,true_x_m,true_y_m,true_vx_mps,true_vy_mps," << observation_columns
         << ",estimated_x_m,estimated_y_m,estimated_vx_mps,estimated_vy_mps,ess,weight_sum,resampled,"
            "log_predictive_density\n" << std::setprecision(17);
    for (std::size_t t = 0; t < data.truth.size(); ++t) {
        file << t + 1 << ',' << (t + 1) * dt;
        for (double value : data.truth[t]) {
            file << ',' << value;
        }
        for (double value : data.observations[t]) {
            file << ',' << value;
        }
        const auto& estimate = fit.estimates[t];
        for (double value : estimate.mean) {
            file << ',' << value;
        }
        file << ',' << estimate.ess << ',' << estimate.weight_sum << ',' << estimate.resampled
             << ',' << estimate.log_predictive_density << '\n';
    }
    file.close();
    if (!file) {
        throw std::runtime_error("failed to write trajectory file " + options.trajectory);
    }
}

void run(const Options& options, int rank, int ranks) {
    const auto nodes = count_nodes(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const auto scenario = simulate_on_root(options.sensor, options.steps, options.data_seed, MPI_COMM_WORLD);
    const auto result = particle_filter_mpi(options.sensor, scenario.observations, options.filter,
                                            options.resampler, MPI_COMM_WORLD);
    const auto& fit = result.filter;
    std::array<double, 4> squared_error{};
    if (rank == 0) {
        for (std::size_t t = 0; t < scenario.truth.size(); ++t) {
            for (std::size_t j = 0; j < squared_error.size(); ++j) {
                const double error = fit.estimates[t].mean[j] - scenario.truth[t][j];
                squared_error[j] += error * error;
            }
        }
        if (!options.trajectory.empty()) {
            write_track(options, scenario, fit);
        }
    }
    const double elapsed = 1000.0 * (MPI_Wtime() - start);
    double workflow_ms = 0.0;
    MPI_Reduce(&elapsed, &workflow_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        return;
    }

    const double position_rmse = std::sqrt((squared_error[0] + squared_error[1]) / options.steps);
    const double velocity_rmse = std::sqrt((squared_error[2] + squared_error[3]) / options.steps);
    const double pack_ratio = static_cast<double>(result.max_packed_particles) * ranks / options.filter.particles;
    const auto& times = fit.times;
    std::cout << "model,resampler,ranks,nodes,particles,steps,data_seed,filter_seed,resampling,position_rmse_m,"
                 "velocity_rmse_mps,log_likelihood,resampled_steps,initialize_ms,propagate_weight_ms,"
                 "normalize_estimate_ms,resample_ms,reduction_mpi_ms,resample_mpi_ms,max_rank_mpi_pct,"
                 "resample_off_rank_bytes,max_packed_particles,max_pack_ratio,filter_ms,workflow_ms\n"
              << std::setprecision(14)
              << (options.sensor == Sensor::linear ? "linear" : "range-bearing") << ','
              << (options.resampler == Resampling::distributed ? "distributed" : "centralized") << ','
              << ranks << ',' << nodes << ',' << options.filter.particles << ',' << options.steps << ','
              << options.data_seed << ',' << options.filter.seed << ',';
    if (options.filter.always_resample) {
        std::cout << "always";
    } else {
        std::cout << options.filter.ess_ratio;
    }
    std::cout << ',' << position_rmse << ',' << velocity_rmse << ',' << fit.log_likelihood << ','
              << fit.resampled_steps << ',' << times.initialize_ms << ',' << times.propagate_weight_ms
              << ',' << times.normalize_ess_ms + times.estimate_ms << ',' << times.resample_ms << ','
              << result.reduction_mpi_ms << ',' << result.resampling_mpi_ms << ',' << result.max_rank_mpi_pct
              << ',' << result.resampling_off_rank_bytes << ',' << result.max_packed_particles << ','
              << pack_ratio << ',' << times.filter_ms << ',' << workflow_ms << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            if (rank == 0) {
                std::cout << "Usage: mpiexec -n P mpi_pf [--model linear|range-bearing]\n"
                             "       [--particles N] [--steps T] [--data-seed SEED] [--filter-seed SEED]\n"
                             "       [--resampler centralized|distributed] [--resample RATIO|always]\n"
                             "       [--trajectory FILE]\n";
            }
        } else {
            run(parse(argc, argv), rank, ranks);
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mpi_pf rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}

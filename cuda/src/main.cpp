#include "cuda_filter.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace tracking;
using namespace tracking::gpu;

struct Options {
    enum class Mode { run, probe, validate } mode = Mode::run;
    Transport transport = Transport::staged;
    FilterOptions filter;
    std::size_t steps = 256;
    std::uint64_t data_seed = 42;
    std::string expected_gpu;
};

template <typename T>
T positive_integer(std::string_view value, std::string_view flag, bool allow_zero = false) {
    T number{};
    const auto* end = value.data() + value.size();
    const auto result = std::from_chars(value.data(), end, number);
    if (result.ec != std::errc{} || result.ptr != end || (!number && !allow_zero)) {
        throw std::invalid_argument("invalid integer for " + std::string(flag));
    }
    return number;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--probe" || flag == "--validate") {
            if (options.mode != Options::Mode::run) {
                throw std::invalid_argument("use one of --probe or --validate");
            }
            options.mode = flag == "--probe" ? Options::Mode::probe : Options::Mode::validate;
            continue;
        }
        if (i + 1 == argc) {
            throw std::invalid_argument("missing value for " + std::string(flag));
        }
        const std::string_view value = argv[++i];
        if (flag == "--transport" && (value == "staged" || value == "device")) {
            options.transport = value == "staged" ? Transport::staged : Transport::device;
        } else if (flag == "--particles") {
            options.filter.particles = positive_integer<std::size_t>(value, flag);
        } else if (flag == "--steps") {
            options.steps = positive_integer<std::size_t>(value, flag);
        } else if (flag == "--data-seed") {
            options.data_seed = positive_integer<std::uint64_t>(value, flag, true);
        } else if (flag == "--filter-seed") {
            options.filter.seed = positive_integer<std::uint64_t>(value, flag, true);
        } else if (flag == "--expect-gpu") {
            options.expected_gpu = value;
        } else {
            throw std::invalid_argument("unsupported option or value for " + std::string(flag));
        }
    }
    return options;
}

void print_probe(const GpuInfo& gpu, int rank, int ranks, MPI_Comm comm) {
    std::array<char, MPI_MAX_PROCESSOR_NAME> host{};
    int length = 0;
    MPI_Get_processor_name(host.data(), &length);
    std::array<char, MPI_MAX_PROCESSOR_NAME + 160> line{};
    std::snprintf(line.data(), line.size(), "%d\t%s\t%d\t%s\t%s\t%d",
                  rank, host.data(), gpu.local_rank, gpu.bus.c_str(), gpu.name.c_str(), gpu.cuda_aware);
    const int line_size = static_cast<int>(line.size());
    std::vector<char> lines(rank == 0 ? ranks * line.size() : 0);
    MPI_Gather(line.data(), line_size, MPI_CHAR, lines.data(), line_size, MPI_CHAR, 0, comm);
    if (rank == 0) {
        std::cout << "rank\thost\tlocal_rank\tpci_bus\tgpu\tcuda_aware_mpi\n";
        for (int r = 0; r < ranks; ++r) {
            std::cout << lines.data() + r * line.size() << '\n';
        }
    }
}

void run_filter(const Options& options, const GpuInfo& gpu, int rank, int ranks, MPI_Comm comm) {
    const auto nodes = parallel::count_nodes(comm);
    const double started = MPI_Wtime();
    const auto data = parallel::simulate_on_root(Sensor::range_bearing, options.steps, options.data_seed, comm);
    const auto result = particle_filter_cuda(Sensor::range_bearing, data.observations, options.filter,
                                             options.transport, comm);
    if (rank != 0) {
        return;
    }
    const auto& mpi = result.mpi;
    const auto& fit = mpi.filter;
    const auto& times = fit.times;
    double position_error = 0.0;
    double velocity_error = 0.0;
    for (std::size_t t = 0; t < data.truth.size(); ++t) {
        const auto& actual = fit.estimates[t].mean;
        const auto& truth = data.truth[t];
        const double dx = actual[0] - truth[0], dy = actual[1] - truth[1];
        const double dvx = actual[2] - truth[2], dvy = actual[3] - truth[3];
        position_error += dx * dx + dy * dy;
        velocity_error += dvx * dvx + dvy * dvy;
    }
    const double position_rmse = std::sqrt(position_error / data.truth.size());
    const double velocity_rmse = std::sqrt(velocity_error / data.truth.size());
    const double workflow_ms = 1000.0 * (MPI_Wtime() - started);
    const double pack_ratio = double(mpi.max_packed_particles) * ranks / options.filter.particles;
    std::cout << "model,resampler,transport,gpu,ranks,nodes,particles,steps,data_seed,filter_seed,resampling,"
                 "position_rmse_m,velocity_rmse_mps,log_likelihood,resampled_steps,initialize_ms,"
                 "propagate_weight_ms,normalize_estimate_ms,resample_ms,reduction_mpi_ms,resample_mpi_ms,"
                 "staging_ms,max_rank_mpi_pct,resample_off_rank_bytes,max_packed_particles,max_pack_ratio,"
                 "filter_ms,workflow_ms\n"
              << std::setprecision(13) << "range-bearing,distributed,"
              << (options.transport == Transport::staged ? "staged" : "device") << ',' << gpu.name << ','
              << ranks << ',' << nodes << ',' << options.filter.particles << ',' << options.steps << ','
              << options.data_seed << ',' << options.filter.seed << ',' << options.filter.ess_ratio << ','
              << position_rmse << ',' << velocity_rmse << ',' << fit.log_likelihood << ',' << fit.resampled_steps
              << ',' << times.initialize_ms << ',' << times.propagate_weight_ms << ',' << times.normalize_ess_ms
              << ',' << times.resample_ms << ',' << mpi.reduction_mpi_ms << ',' << mpi.resampling_mpi_ms
              << ',' << result.staging_ms << ',' << mpi.max_rank_mpi_pct << ','
              << mpi.resampling_off_rank_bytes << ',' << mpi.max_packed_particles << ',' << pack_ratio
              << ',' << times.filter_ms << ',' << workflow_ms << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, ranks = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    try {
        const auto options = parse(argc, argv);
        const auto gpu = select_gpu(comm);
        if (!options.expected_gpu.empty() && gpu.name != options.expected_gpu) {
            throw std::runtime_error("expected GPU " + options.expected_gpu + ", found " + gpu.name);
        }
        switch (options.mode) {
            case Options::Mode::probe:
                print_probe(gpu, rank, ranks, comm);
                break;
            case Options::Mode::validate:
                validate_cuda(options.transport, comm);
                break;
            case Options::Mode::run:
                run_filter(options, gpu, rank, ranks, comm);
                break;
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(comm, 1);
        return 1;
    }
}

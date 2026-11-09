#include "mpi_filter.hpp"
#include "philox.hpp"

#include <algorithm>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace {

using namespace tracking;
using namespace tracking::parallel;

int positive_int(std::string_view argument) {
    int value = 0;
    const auto end = argument.data() + argument.size();
    const auto [last, error] = std::from_chars(argument.data(), end, value);
    if (error != std::errc{} || last != end || value < 1) {
        throw std::invalid_argument("expected a positive integer");
    }
    return value;
}

void run(int argc, char** argv) {
    if (argc != 5) {
        throw std::invalid_argument("usage: resampling_bench N centralized|distributed uniform|skewed iterations");
    }
    const auto particles = positive_int(argv[1]);
    const std::string_view method = argv[2];
    const std::string_view shape = argv[3];
    const auto iterations = positive_int(argv[4]);
    if ((method != "distributed" && method != "centralized") || (shape != "uniform" && shape != "skewed")) {
        throw std::invalid_argument("invalid method or weight shape");
    }
    const ParticlePartition partition(particles, MPI_COMM_WORLD);
    const auto nodes = count_nodes(MPI_COMM_WORLD);
    SystematicResampler resampler(partition, method == "distributed" ? Resampling::distributed
                                                                      : Resampling::centralized, MPI_COMM_WORLD);
    std::vector<State> states(partition.local_size()), output(states.size());
    std::vector<double> weights(states.size());
    const int hot_particles = std::max(1, particles / 100);
    const double hot_weight = 0.9 / hot_particles;
    const double cold_weight = 0.1 / std::max(1, particles - hot_particles);
    for (std::size_t i = 0; i < states.size(); ++i) {
        const auto id = partition.first() + i;
        states[i] = {static_cast<double>(id), 0.0, 0.0, 0.0};
        weights[i] = shape == "uniform" ? 1.0 :
                     (id < static_cast<std::size_t>(hot_particles) ? hot_weight : cold_weight);
    }
    const double local_mass = std::accumulate(weights.begin(), weights.end(), 0.0);
    resampler.run(states, weights, local_mass, systematic_offset(1729, 0), output);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    double mpi_ms = 0.0;
    std::uint64_t bytes = 0;
    int max_packed = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto stats = resampler.run(states, weights, local_mass,
                                         systematic_offset(1729, iteration + 1), output);
        mpi_ms += stats.mpi_ms;
        bytes += stats.off_rank_bytes;
        max_packed = std::max(max_packed, stats.packed_particles);
    }
    std::array<double, 2> times{(MPI_Wtime() - start) * 1000.0, mpi_ms};
    MPI_Allreduce(MPI_IN_PLACE, times.data(), 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bytes, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_packed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (partition.rank == 0) {
        std::cout << "ranks,nodes,particles,resampler,weight_shape,iterations,ms_per_resample,"
                     "mpi_ms_per_resample,off_rank_bytes_per_resample,max_pack_ratio\n"
                  << std::setprecision(12) << partition.size << ',' << nodes << ',' << particles << ','
                  << method << ',' << shape << ',' << iterations << ',' << times[0] / iterations << ','
                  << times[1] / iterations << ',' << static_cast<double>(bytes) / iterations << ','
                  << static_cast<double>(max_packed) * partition.size / particles << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    try {
        run(argc, argv);
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "resampling_bench: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}

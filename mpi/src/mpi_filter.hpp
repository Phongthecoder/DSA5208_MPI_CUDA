#pragma once

#include "particle_filter.hpp"

#include <mpi.h>

namespace tracking::parallel {

enum class Resampling { centralized, distributed };

struct ParticlePartition {
    int rank;
    int size;
    std::vector<int> counts;
    std::vector<int> offsets;

    ParticlePartition(std::size_t particles, MPI_Comm comm);
    int local_size() const { return counts[rank]; }
    int first() const { return offsets[rank]; }
    int total() const { return offsets.back(); }
};

struct ResamplingStats {
    double mpi_ms = 0.0;
    std::uint64_t off_rank_bytes = 0;
    int packed_particles = 0;
};

// Owns scratch buffers and keeps the output in global slot order after each resample.
class SystematicResampler {
public:
    SystematicResampler(const ParticlePartition& partition, Resampling method, MPI_Comm comm);
    ~SystematicResampler();
    SystematicResampler(const SystematicResampler&) = delete;
    SystematicResampler& operator=(const SystematicResampler&) = delete;

    // All buffers have the local population size. local_mass is the sum of the nonnegative weights.
    ResamplingStats run(const std::vector<State>& states, const std::vector<double>& weights,
                        double local_mass, double u, std::vector<State>& output);

private:
    ResamplingStats centralized(const std::vector<State>& states, const std::vector<double>& weights,
                                double u, std::vector<State>& output);
    ResamplingStats distributed(const std::vector<State>& states, const std::vector<double>& weights,
                                double local_mass, double u, std::vector<State>& output);

    const ParticlePartition& partition_;
    Resampling method_;
    MPI_Comm comm_;
    MPI_Datatype state_type_;
    std::vector<State> root_states_;
    std::vector<State> packed_;
    std::vector<double> root_weights_;
    std::vector<std::size_t> ancestors_;
    std::vector<double> rank_prefix_;
    std::vector<int> send_counts_;
    std::vector<int> send_offsets_;
    std::vector<int> recv_counts_;
    std::vector<int> recv_offsets_;
};

struct MpiResult {
    FilterResult filter;
    double reduction_mpi_ms = 0.0;
    double resampling_mpi_ms = 0.0;
    double max_rank_mpi_pct = 0.0;
    std::uint64_t resampling_off_rank_bytes = 0;
    int max_packed_particles = 0;
};

// Returns the number of participating hosts on rank zero.
std::size_t count_nodes(MPI_Comm comm);

// Only rank zero retains the truth. All ranks receive the same observations.
Scenario simulate_on_root(Sensor sensor, std::size_t steps, std::uint64_t seed, MPI_Comm comm);

// Collective. All ranks supply the same observations and filter options.
MpiResult particle_filter_mpi(Sensor sensor, const std::vector<Observation>& observations,
                              const FilterOptions& options, Resampling method, MPI_Comm comm);

}  // namespace tracking::parallel

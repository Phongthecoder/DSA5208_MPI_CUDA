#pragma once

#include "mpi_filter.hpp"

#include <string>

namespace tracking::gpu {

enum class Transport { staged, device };

struct GpuInfo {
    std::string name;
    std::string bus;
    int local_rank;
    bool cuda_aware;
};

struct GpuResult {
    parallel::MpiResult mpi;
    double staging_ms = 0.0;
};

// One GPU per local MPI rank. Also checks that the device IDs are distinct.
GpuInfo select_gpu(MPI_Comm comm);

GpuResult particle_filter_cuda(Sensor sensor, const std::vector<Observation>& observations,
                               const FilterOptions& options, Transport transport, MPI_Comm comm);

void validate_cuda(Transport transport, MPI_Comm comm);

}  // namespace tracking::gpu

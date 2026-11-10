#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include "cuda_filter.hpp"
#include "philox.hpp"

#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tracking::gpu {
namespace {

using parallel::ParticlePartition;
using parallel::ResamplingStats;
constexpr int block_width = 256;

void check(cudaError_t error, const char* action) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(error));
    }
}

void finish_kernel() {
    check(cudaGetLastError(), "launch CUDA kernel");
    check(cudaDeviceSynchronize(), "finish CUDA kernel");
}

double elapsed_ms(double start) { return 1000.0 * (MPI_Wtime() - start); }

int blocks(int n) { return (n + block_width - 1) / block_width; }

bool supports_cuda_aware_mpi() {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() == 1;
#else
    return false;
#endif
}

template <typename T, bool pinned = false>
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(std::size_t n) { reserve(n); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { release(); }

    // Callers use this only for scratch storage. Growth need not preserve data.
    void reserve(std::size_t n) {
        n = std::max<std::size_t>(n, 1);
        if (capacity_ >= n) {
            return;
        }
        void* allocation = nullptr;
        if constexpr (pinned) {
            check(cudaMallocHost(&allocation, n * sizeof(T)), "allocate pinned memory");
        } else {
            check(cudaMalloc(&allocation, n * sizeof(T)), "allocate GPU memory");
        }
        release();
        data_ = static_cast<T*>(allocation);
        capacity_ = n;
    }

    T* data() const { return data_; }

private:
    void release() noexcept {
        if (data_) {
            if constexpr (pinned) {
                cudaFreeHost(data_);
            } else {
                cudaFree(data_);
            }
        }
    }

    T* data_ = nullptr;
    std::size_t capacity_ = 0;
};

struct GpuState {
    double value[4];
};

static_assert(sizeof(GpuState) == 4 * sizeof(double));

struct Model {
    double position_noise;
    double shared_noise;
    double velocity_noise;
    double inverse_sd0;
    double inverse_sd1;
    double log_constant;
    double radar_x;
    double radar_y;
    bool range_bearing;
};

Model make_model(Sensor sensor) {
    const double position_noise = std::sqrt(acceleration_density * dt * dt * dt / 3.0);
    const double shared_noise = acceleration_density * dt * dt / (2.0 * position_noise);
    const double velocity_noise = std::sqrt(acceleration_density * dt - shared_noise * shared_noise);
    const bool range_bearing = sensor == Sensor::range_bearing;
    const double sd0 = 1.0 / (range_bearing ? range_sd : position_sd);
    const double sd1 = 1.0 / (range_bearing ? bearing_sd : position_sd);
    return {position_noise, shared_noise, velocity_noise, sd0, sd1,
            std::log(sd0 * sd1 / (2.0 * pi)), radar_position[0], radar_position[1], range_bearing};
}

__device__ GpuState noise(std::uint64_t seed, std::uint64_t slot, std::uint64_t step) {
    const auto words = parallel::philox_words(seed, slot, step);
    GpuState output;
    for (int axis = 0; axis < 2; ++axis) {
        const double radius = sqrt(-2.0 * log(parallel::uniform_open(words.value[2 * axis])));
        const double angle = (2.0 * pi) * parallel::uniform_open(words.value[2 * axis + 1]);
        double sine, cosine;
        sincos(angle, &sine, &cosine);
        output.value[2 * axis] = radius * cosine;
        output.value[2 * axis + 1] = radius * sine;
    }
    return output;
}

__global__ void initialize(GpuState* states, double* log_weights, int n, int first,
                           std::uint64_t seed, GpuState mean, GpuState sd, double uniform) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const auto z = noise(seed, first + i, 0);
        for (int j = 0; j < 4; ++j) {
            states[i].value[j] = mean.value[j] + sd.value[j] * z.value[j];
        }
        log_weights[i] = uniform;
    }
}

struct Max {
    __host__ __device__ double operator()(double a, double b) const { return a > b ? a : b; }
};

__global__ void propagate(GpuState* states, double* log_weights, double* partial, int n,
                          int first, std::uint64_t seed, std::uint64_t step, Model model,
                          double obs0, double obs1) {
    using Reduction = cub::BlockReduce<double, block_width>;
    __shared__ typename Reduction::TempStorage scratch;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double log_weight = -CUDART_INF;
    if (i < n) {
        auto state = states[i];
        const auto z = noise(seed, first + i, step);
        for (int axis = 0; axis < 2; ++axis) {
            const double z0 = z.value[2 * axis];
            const double z1 = z.value[2 * axis + 1];
            state.value[axis] += dt * state.value[axis + 2] + model.position_noise * z0;
            state.value[axis + 2] += model.shared_noise * z0 + model.velocity_noise * z1;
        }
        double residual0 = obs0 - state.value[0];
        double residual1 = obs1 - state.value[1];
        if (model.range_bearing) {
            const double x = state.value[0] - model.radar_x;
            const double y = state.value[1] - model.radar_y;
            residual0 = obs0 - hypot(x, y);
            residual1 = remainder(obs1 - atan2(y, x), 2.0 * pi);
            if (residual1 <= -pi) {
                residual1 = pi;
            }
        }
        const double r0 = residual0 * model.inverse_sd0;
        const double r1 = residual1 * model.inverse_sd1;
        log_weight = log_weights[i] + (model.log_constant - 0.5 * (r0 * r0 + r1 * r1));
        states[i] = state;
        log_weights[i] = log_weight;
    }
    const auto maximum = Reduction(scratch).Reduce(log_weight, Max{});
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = maximum;
    }
}

struct Moments {
    double value[6];  // sum(w), sum(w^2), and sum(w*x) for each state component
};

struct AddMoments {
    __host__ __device__ Moments operator()(Moments a, const Moments& b) const {
        for (int j = 0; j < 6; ++j) {
            a.value[j] += b.value[j];
        }
        return a;
    }
};

__global__ void weigh(const GpuState* states, const double* log_weights, double* weights,
                      Moments* partial, int n, double global_max) {
    using Reduction = cub::BlockReduce<Moments, block_width>;
    __shared__ typename Reduction::TempStorage scratch;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    Moments value{};
    if (i < n) {
        const double weight = exp(log_weights[i] - global_max);
        weights[i] = weight;
        value.value[0] = weight;
        value.value[1] = weight * weight;
        for (int j = 0; j < 4; ++j) {
            value.value[j + 2] = weight * states[i].value[j];
        }
    }
    const auto sums = Reduction(scratch).Reduce(value, AddMoments{});
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sums;
    }
}

template <typename T, typename Op>
__global__ void finish_reduction(const T* partial, T* output, int n, T identity, Op op) {
    using Reduction = cub::BlockReduce<T, block_width>;
    __shared__ typename Reduction::TempStorage scratch;
    T local = identity;
    for (int i = threadIdx.x; i < n; i += block_width) {
        local = op(local, partial[i]);
    }
    const auto value = Reduction(scratch).Reduce(local, op);
    if (threadIdx.x == 0) {
        *output = value;
    }
}

__global__ void normalize(double* log_weights, int n, double maximum, double log_mass,
                          bool resampled) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        log_weights[i] = resampled ? -log_mass : (log_weights[i] - maximum) - log_mass;
    }
}

struct Points {
    int count;
    double offset;
    double mass;
    double cap;

    __host__ __device__ double at(int slot) const {
        const double point = (static_cast<double>(slot) + offset) / count * mass;
        return point < cap ? point : cap;
    }

    int first_at_or_after(double boundary) const {
        int lo = 0;
        int hi = count;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (at(mid) < boundary) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
};

__global__ void pack(const GpuState* states, const double* cdf, GpuState* destination,
                     int local_count, Points points, int first, int count, double rank_prefix) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const double target = points.at(first + i);
        int lo = 0;
        int hi = local_count - 1;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (rank_prefix + cdf[mid] <= target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        destination[i] = states[lo];
    }
}

class GpuResampler {
public:
    GpuResampler(const ParticlePartition& partition, Transport transport, MPI_Comm comm)
        : partition_(partition), transport_(transport), comm_(comm), cdf_(partition.local_size()),
          prefix_(partition.size + 1), send_counts_(partition.size), send_offsets_(partition.size),
          receive_counts_(partition.size), receive_offsets_(partition.size) {
        if (partition.local_size()) {
            check(cub::DeviceScan::InclusiveSum(nullptr, scratch_bytes_, cdf_.data(), cdf_.data(),
                                                partition.local_size()), "size CUB scan workspace");
            scratch_.reserve(scratch_bytes_);
        }
    }

    ResamplingStats run(const GpuState* input, const double* weights, GpuState* output,
                        double offset, double& staging_ms) {
        const int n = partition_.local_size();
        double mass = 0.0;
        if (n) {
            check(cub::DeviceScan::InclusiveSum(scratch_.data(), scratch_bytes_, weights, cdf_.data(), n),
                  "scan local weights");
            check(cudaMemcpy(&mass, cdf_.data() + n - 1, sizeof(mass), cudaMemcpyDeviceToHost),
                  "read local CDF mass");
        }
        double mpi_ms = 0.0;
        if (partition_.size == 1) {
            prefix_[1] = mass;
        } else {
            const double started = MPI_Wtime();
            MPI_Allgather(&mass, 1, MPI_DOUBLE, prefix_.data() + 1, 1, MPI_DOUBLE, comm_);
            mpi_ms = elapsed_ms(started);
        }
        for (int r = 1; r <= partition_.size; ++r) {
            prefix_[r] += prefix_[r - 1];
        }
        const double total = prefix_.back();
        if (!(total > 0 && std::isfinite(total))) {
            throw std::runtime_error("invalid CDF mass in GPU resampling");
        }
        const Points points{partition_.total(), offset, total, std::nextafter(total, 0.0)};
        const double start_mass = prefix_[partition_.rank];
        const int first = points.first_at_or_after(start_mass);
        const int end = points.first_at_or_after(prefix_[partition_.rank + 1]);
        const int count = end - first;

        GpuState* send = output;
        if (partition_.size > 1) {
            packed_.reserve(count);
            send = packed_.data();
        }
        if (count) {
            pack<<<blocks(count), block_width>>>(input, cdf_.data(), send, n, points, first,
                                                 count, start_mass);
            finish_kernel();
        }
        if (partition_.size == 1) {
            return {0.0, 0, count};
        }
        // Global output slots are ordered by sender CDF range. MPI displacements
        // express states as four doubles to keep the direct-buffer MPI path simple.
        for (int r = 0; r < partition_.size; ++r) {
            const int lo = std::max(first, partition_.offsets[r]);
            const int hi = std::min(end, partition_.offsets[r + 1]);
            send_counts_[r] = 4 * std::max(0, hi - lo);
            send_offsets_[r] = send_counts_[r] ? 4 * (lo - first) : 0;
        }
        double started = MPI_Wtime();
        MPI_Alltoall(send_counts_.data(), 1, MPI_INT, receive_counts_.data(), 1, MPI_INT, comm_);
        mpi_ms += elapsed_ms(started);
        int received = 0;
        for (int r = 0; r < partition_.size; ++r) {
            receive_offsets_[r] = received;
            received += receive_counts_[r];
        }
        if (received != 4 * n) {
            throw std::runtime_error("GPU resampling changed the local particle count");
        }

        GpuState* receive = output;
        if (transport_ == Transport::staged) {
            started = MPI_Wtime();
            pinned_send_.reserve(count);
            pinned_receive_.reserve(n);
            if (count) {
                check(cudaMemcpy(pinned_send_.data(), send, count * sizeof(GpuState),
                                 cudaMemcpyDeviceToHost), "stage packed particles");
            }
            send = pinned_send_.data();
            receive = pinned_receive_.data();
            staging_ms += elapsed_ms(started);
        }
        started = MPI_Wtime();
        MPI_Alltoallv(send, send_counts_.data(), send_offsets_.data(), MPI_DOUBLE,
                      receive, receive_counts_.data(), receive_offsets_.data(), MPI_DOUBLE, comm_);
        mpi_ms += elapsed_ms(started);
        if (transport_ == Transport::staged && n) {
            started = MPI_Wtime();
            check(cudaMemcpy(output, receive, n * sizeof(GpuState), cudaMemcpyHostToDevice),
                  "return resampled particles to GPU");
            staging_ms += elapsed_ms(started);
        }
        const auto off_rank = std::uint64_t(count - send_counts_[partition_.rank] / 4);
        return {mpi_ms, off_rank * sizeof(GpuState), count};
    }

private:
    const ParticlePartition& partition_;
    Transport transport_;
    MPI_Comm comm_;
    Buffer<double> cdf_;
    Buffer<unsigned char> scratch_;
    std::size_t scratch_bytes_ = 0;
    Buffer<GpuState> packed_;
    Buffer<GpuState, true> pinned_send_, pinned_receive_;
    std::vector<double> prefix_;
    std::vector<int> send_counts_, send_offsets_, receive_counts_, receive_offsets_;
};

void require_transport(Transport transport) {
    if (transport == Transport::device && !supports_cuda_aware_mpi()) {
        throw std::runtime_error("device transport requires CUDA-aware Open MPI");
    }
}

void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

}  // namespace

GpuInfo select_gpu(MPI_Comm comm) {
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int rank, size;
    MPI_Comm_rank(local, &rank);
    MPI_Comm_size(local, &size);
    int count = 0;
    check(cudaGetDeviceCount(&count), "count visible GPUs");
    if (!count || (count > 1 && rank >= count)) {
        throw std::runtime_error("MPI local rank has no visible GPU");
    }
    const int device = count == 1 ? 0 : rank;
    check(cudaSetDevice(device), "select CUDA device");
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, device), "read CUDA device properties");
    std::array<char, 32> bus{};
    check(cudaDeviceGetPCIBusId(bus.data(), static_cast<int>(bus.size()), device), "read GPU PCI ID");
    std::vector<char> buses(static_cast<std::size_t>(size) * bus.size());
    MPI_Allgather(bus.data(), static_cast<int>(bus.size()), MPI_CHAR, buses.data(),
                  static_cast<int>(bus.size()), MPI_CHAR, local);
    MPI_Comm_free(&local);
    for (int r = 0; r < size; ++r) {
        if (r != rank && std::strcmp(buses.data() + r * bus.size(), bus.data()) == 0) {
            throw std::runtime_error("multiple local MPI ranks selected the same physical GPU");
        }
    }
    return {props.name, bus.data(), rank, supports_cuda_aware_mpi()};
}

GpuResult particle_filter_cuda(Sensor sensor, const std::vector<Observation>& observations,
                               const FilterOptions& options, Transport transport, MPI_Comm comm) {
    require_transport(transport);
    if (!options.particles || options.particles > std::numeric_limits<int>::max() / 4 ||
        observations.empty() || !(options.ess_ratio >= 0.0 && options.ess_ratio <= 1.0) ||
        options.estimate_covariance || options.profile_gaussian) {
        throw std::invalid_argument("GPU filter needs 1 to INT_MAX/4 particles, observations, and valid ESS options");
    }
    MPI_Barrier(comm);
    const double started = MPI_Wtime();
    const ParticlePartition p(options.particles, comm);
    const int n = p.local_size();
    const int grid = blocks(n);
    const double count = static_cast<double>(p.total());
    const double log_count = std::log(count);
    const Model model = make_model(sensor);
    Buffer<GpuState> input(n), output(n);
    auto* states = input.data();
    auto* resampled = output.data();
    Buffer<double> log_weights(n), weights(n), max_partial(grid), max_value(1);
    Buffer<Moments> moment_partial(grid), moment_value(1);
    GpuResampler resampler(p, transport, comm);
    GpuResult result;
    auto& mpi = result.mpi;
    auto& fit = mpi.filter;
    auto& times = fit.times;
    fit.estimates.resize(observations.size());
    if (n) {
        const GpuState mean{{prior_mean[0], prior_mean[1], prior_mean[2], prior_mean[3]}};
        const GpuState sd{{prior_sd[0], prior_sd[1], prior_sd[2], prior_sd[3]}};
        initialize<<<grid, block_width>>>(states, log_weights.data(), n, p.first(),
                                         options.seed, mean, sd, -log_count);
        finish_kernel();
    }
    times.initialize_ms = elapsed_ms(started);

    for (std::size_t step = 0; step < observations.size(); ++step) {
        const auto& obs = observations[step];
        require(std::isfinite(obs[0]) && std::isfinite(obs[1]), "observations must be finite");
        double phase = MPI_Wtime();
        double local_max = -std::numeric_limits<double>::infinity();
        if (n) {
            propagate<<<grid, block_width>>>(states, log_weights.data(), max_partial.data(), n,
                                             p.first(), options.seed, step + 1, model, obs[0], obs[1]);
            check(cudaGetLastError(), "propagate and weight particles");
            finish_reduction<<<1, block_width>>>(max_partial.data(), max_value.data(), grid, local_max, Max{});
            check(cudaGetLastError(), "reduce local maximum");
            check(cudaMemcpy(&local_max, max_value.data(), sizeof(local_max), cudaMemcpyDeviceToHost),
                  "read local weight maximum");
        }
        times.propagate_weight_ms += elapsed_ms(phase);

        phase = MPI_Wtime();
        double global_max = 0.0;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        mpi.reduction_mpi_ms += elapsed_ms(phase);
        require(std::isfinite(global_max), "no particle has finite likelihood");
        Moments local{}, global{};
        if (n) {
            weigh<<<grid, block_width>>>(states, log_weights.data(), weights.data(), moment_partial.data(),
                                         n, global_max);
            check(cudaGetLastError(), "calculate shifted weights");
            finish_reduction<<<1, block_width>>>(moment_partial.data(), moment_value.data(), grid,
                                                 Moments{}, AddMoments{});
            check(cudaGetLastError(), "reduce local moments");
            check(cudaMemcpy(&local, moment_value.data(), sizeof(local), cudaMemcpyDeviceToHost),
                  "read local moments");
        }
        const double reduction = MPI_Wtime();
        MPI_Allreduce(local.value, global.value, 6, MPI_DOUBLE, MPI_SUM, comm);
        mpi.reduction_mpi_ms += elapsed_ms(reduction);
        const double inverse_mass = 1.0 / global.value[0];
        const double log_mass = std::log(global.value[0]);
        auto& estimate = fit.estimates[step];
        estimate.ess = std::clamp(global.value[0] * global.value[0] / global.value[1], 1.0, count);
        estimate.weight_sum = global.value[0] * inverse_mass;
        estimate.log_predictive_density = global_max + log_mass;
        fit.log_likelihood += estimate.log_predictive_density;
        for (int j = 0; j < 4; ++j) {
            estimate.mean[j] = global.value[j + 2] * inverse_mass;
        }
        estimate.resampled = options.always_resample || estimate.ess < options.ess_ratio * count;
        if (n && !estimate.resampled) {
            normalize<<<grid, block_width>>>(log_weights.data(), n, global_max, log_mass, false);
            finish_kernel();
        }
        times.normalize_ess_ms += elapsed_ms(phase);

        if (estimate.resampled) {
            phase = MPI_Wtime();
            const auto stats = resampler.run(states, weights.data(), resampled,
                                              parallel::systematic_offset(options.seed, step + 1),
                                              result.staging_ms);
            std::swap(states, resampled);
            if (n) {
                normalize<<<grid, block_width>>>(log_weights.data(), n, 0.0, log_count, true);
                finish_kernel();
            }
            ++fit.resampled_steps;
            mpi.resampling_mpi_ms += stats.mpi_ms;
            mpi.resampling_off_rank_bytes += stats.off_rank_bytes;
            mpi.max_packed_particles = std::max(mpi.max_packed_particles, stats.packed_particles);
            times.resample_ms += elapsed_ms(phase);
        }
    }
    times.filter_ms = elapsed_ms(started);
    mpi.max_rank_mpi_pct = 100.0 * (mpi.reduction_mpi_ms + mpi.resampling_mpi_ms) / times.filter_ms;
    std::array<double, 9> maxima{times.initialize_ms, times.propagate_weight_ms, times.normalize_ess_ms,
                                 times.resample_ms, times.filter_ms, mpi.reduction_mpi_ms,
                                 mpi.resampling_mpi_ms, mpi.max_rank_mpi_pct, result.staging_ms};
    MPI_Allreduce(MPI_IN_PLACE, maxima.data(), static_cast<int>(maxima.size()), MPI_DOUBLE, MPI_MAX, comm);
    times.initialize_ms = maxima[0];
    times.propagate_weight_ms = maxima[1];
    times.normalize_ess_ms = maxima[2];
    times.resample_ms = maxima[3];
    times.filter_ms = maxima[4];
    mpi.reduction_mpi_ms = maxima[5];
    mpi.resampling_mpi_ms = maxima[6];
    mpi.max_rank_mpi_pct = maxima[7];
    result.staging_ms = maxima[8];
    MPI_Allreduce(MPI_IN_PLACE, &mpi.resampling_off_rank_bytes, 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &mpi.max_packed_particles, 1, MPI_INT, MPI_MAX, comm);
    return result;
}

void validate_cuda(Transport transport, MPI_Comm comm) {
    require_transport(transport);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const std::array<std::pair<std::vector<double>, double>, 4> cases{{
        {{0, 3, 2, 1, 0, 1, 0, 1}, 0.2},
        {{0, 0.5, 0, 0.5, 0}, 0.0},
        {{0, 0, 0, 1, 0, 0, 0}, std::nextafter(1.0, 0.0)},
        {{1}, 0.0},
    }};
    for (const auto& test : cases) {
        const auto& weights = test.first;
        const ParticlePartition p(weights.size(), comm);
        const int n = p.local_size();
        std::vector<GpuState> state(n);
        for (int i = 0; i < n; ++i) {
            const double id = p.first() + i;
            state[i] = {{id, -id, 7.0, 11.0}};
        }
        Buffer<GpuState> input(n), output(n);
        Buffer<double> device_weights(n);
        if (n) {
            check(cudaMemcpy(input.data(), state.data(), n * sizeof(GpuState), cudaMemcpyHostToDevice),
                  "copy test particles");
            check(cudaMemcpy(device_weights.data(), weights.data() + p.first(), n * sizeof(double),
                             cudaMemcpyHostToDevice), "copy test weights");
        }
        GpuResampler resampler(p, transport, comm);
        double copies = 0.0;
        resampler.run(input.data(), device_weights.data(), output.data(), test.second, copies);
        if (n) {
            check(cudaMemcpy(state.data(), output.data(), n * sizeof(GpuState), cudaMemcpyDeviceToHost),
                  "read resampling test result");
        }
        std::vector<std::size_t> expected;
        systematic_ancestors(weights, test.second, expected);
        for (int i = 0; i < n; ++i) {
            const double id = static_cast<double>(expected[p.first() + i]);
            const auto* actual = state[i].value;
            require(actual[0] == id && actual[1] == -id && actual[2] == 7.0 && actual[3] == 11.0,
                    "GPU resampling ancestry differs from serial oracle");
        }
    }

    for (Sensor sensor : {Sensor::linear, Sensor::range_bearing}) {
        for (bool always : {false, true}) {
            FilterOptions options;
            options.particles = 4099;
            options.always_resample = always;
            const auto data = parallel::simulate_on_root(sensor, 48, 42, comm);
            const auto cpu = parallel::particle_filter_mpi(sensor, data.observations, options,
                                                          parallel::Resampling::distributed, comm).filter;
            const auto gpu = particle_filter_cuda(sensor, data.observations, options, transport, comm).mpi.filter;
            double max_mean_delta = 0.0, max_ess_delta = 0.0;
            for (std::size_t i = 0; i < cpu.estimates.size(); ++i) {
                const auto& a = cpu.estimates[i];
                const auto& b = gpu.estimates[i];
                require(a.resampled == b.resampled && std::abs(b.weight_sum - 1.0) < 1e-10 &&
                        b.ess >= 1 && b.ess <= options.particles,
                        "GPU ESS decision or normalized weights differ");
                for (int j = 0; j < 4; ++j) {
                    max_mean_delta = std::max(max_mean_delta, std::abs(a.mean[j] - b.mean[j]));
                    require(std::isfinite(b.mean[j]), "nonfinite GPU posterior mean");
                }
                max_ess_delta = std::max(max_ess_delta, std::abs(a.ess - b.ess));
            }
            require(max_mean_delta < 1e-6 && max_ess_delta < 1e-5 &&
                    std::abs(cpu.log_likelihood - gpu.log_likelihood) < 1e-6,
                    "GPU filter differs from fixed-seed CPU MPI reference");
            if (rank == 0) {
                std::cout << (sensor == Sensor::linear ? "linear" : "range-bearing")
                          << (always ? " ESS=always" : " ESS=0.5")
                          << " max_state_delta=" << max_mean_delta << " max_ESS_delta=" << max_ess_delta
                          << " log_likelihood_delta=" << std::abs(cpu.log_likelihood - gpu.log_likelihood)
                          << '\n';
            }
        }
    }
    if (rank == 0) {
        std::cout << "PASS exact systematic ancestry, zero-mass and empty ranks, and CPU MPI agreement on "
                  << ranks << " GPU ranks\n";
    }
}

}  // namespace tracking::gpu

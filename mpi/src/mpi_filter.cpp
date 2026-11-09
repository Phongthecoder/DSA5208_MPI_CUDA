#include "mpi_filter.hpp"
#include "philox.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

namespace tracking::parallel {
namespace {

constexpr std::size_t mpi_count_max = std::numeric_limits<int>::max();

double elapsed_ms(double start) {
    return 1000.0 * (MPI_Wtime() - start);
}

struct SystematicPoints {
    int count;
    double offset;
    double mass;
    double last;

    double operator()(int slot) const {
        return std::min((static_cast<double>(slot) + offset) / count * mass, last);
    }

    int below(double boundary) const {
        int lo = 0;
        int hi = count;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if ((*this)(mid) < boundary) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
};

struct Motion {
    const double position_noise_sd = std::sqrt(acceleration_density * dt * dt * dt / 3.0);
    const double shared_noise_scale = acceleration_density * dt * dt / (2.0 * position_noise_sd);
    const double velocity_noise_sd = std::sqrt(acceleration_density * dt -
                                               shared_noise_scale * shared_noise_scale);

    void propagate(State& state, const State& noise) const {
        for (std::size_t axis = 0; axis < 2; ++axis) {
            const double z_position = noise[2 * axis];
            const double z_velocity = noise[2 * axis + 1];
            state[axis] += dt * state[axis + 2] + position_noise_sd * z_position;
            state[axis + 2] += shared_noise_scale * z_position + velocity_noise_sd * z_velocity;
        }
    }
};

class Likelihood {
public:
    explicit Likelihood(Sensor sensor)
        : sensor_(sensor), inverse_sd0_(1.0 / (sensor == Sensor::linear ? position_sd : range_sd)),
          inverse_sd1_(1.0 / (sensor == Sensor::linear ? position_sd : bearing_sd)),
          constant_(std::log(inverse_sd0_ * inverse_sd1_ / (2.0 * pi))) {}

    double operator()(const State& state, const Observation& observation) const {
        double residual0 = observation[0] - state[0];
        double residual1 = observation[1] - state[1];
        if (sensor_ == Sensor::range_bearing) {
            const double x = state[0] - radar_position[0];
            const double y = state[1] - radar_position[1];
            residual0 = observation[0] - std::hypot(x, y);
            residual1 = wrap_bearing(observation[1] - std::atan2(y, x));
        }
        const double z0 = residual0 * inverse_sd0_;
        const double z1 = residual1 * inverse_sd1_;
        return constant_ - 0.5 * (z0 * z0 + z1 * z1);
    }

private:
    Sensor sensor_;
    double inverse_sd0_;
    double inverse_sd1_;
    double constant_;
};

}  // namespace

ParticlePartition::ParticlePartition(std::size_t particles, MPI_Comm comm) {
    if (particles == 0 || particles > mpi_count_max) {
        throw std::invalid_argument("particle count must be in [1, INT_MAX]");
    }
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    counts.resize(size);
    offsets.resize(size + 1);
    const int total = static_cast<int>(particles);
    for (int r = 0; r < size; ++r) {
        counts[r] = total / size + (r < total % size);
        offsets[r + 1] = offsets[r] + counts[r];
    }
}

SystematicResampler::SystematicResampler(const ParticlePartition& partition, Resampling method,
                                         MPI_Comm comm)
    : partition_(partition), method_(method), comm_(comm) {
    if (partition.size > 1 && method == Resampling::centralized && partition.rank == 0) {
        root_states_.resize(partition.total());
        root_weights_.resize(partition.total());
        packed_.resize(partition.total());
        ancestors_.resize(partition.total());
    }
    if (partition.size > 1 && method == Resampling::distributed) {
        rank_prefix_.resize(partition.size + 1);
        send_counts_.resize(partition.size);
        send_offsets_.resize(partition.size);
        recv_counts_.resize(partition.size);
        recv_offsets_.resize(partition.size);
        packed_.reserve(partition.local_size());
    }
    static_assert(sizeof(State) == 4 * sizeof(double));
    MPI_Type_contiguous(4, MPI_DOUBLE, &state_type_);
    MPI_Type_commit(&state_type_);
}

SystematicResampler::~SystematicResampler() {
    MPI_Type_free(&state_type_);
}

ResamplingStats SystematicResampler::run(const std::vector<State>& states,
                                          const std::vector<double>& weights, double local_mass,
                                          double u, std::vector<State>& output) {
    if (partition_.size == 1) {
        systematic_ancestors(weights, u, ancestors_);
        for (std::size_t i = 0; i < output.size(); ++i) {
            output[i] = states[ancestors_[i]];
        }
        return {0.0, 0, partition_.total()};
    }
    if (method_ == Resampling::centralized) {
        return centralized(states, weights, u, output);
    }
    return distributed(states, weights, local_mass, u, output);
}

ResamplingStats SystematicResampler::centralized(const std::vector<State>& states,
                                                   const std::vector<double>& weights, double u,
                                                   std::vector<State>& output) {
    const int local_size = partition_.local_size();
    double started = MPI_Wtime();
    MPI_Gatherv(states.data(), local_size, state_type_, root_states_.data(),
                partition_.counts.data(), partition_.offsets.data(), state_type_, 0, comm_);
    MPI_Gatherv(weights.data(), local_size, MPI_DOUBLE, root_weights_.data(),
                partition_.counts.data(), partition_.offsets.data(), MPI_DOUBLE, 0, comm_);
    double mpi_ms = elapsed_ms(started);
    if (partition_.rank == 0) {
        systematic_ancestors(root_weights_, u, ancestors_);
        for (std::size_t i = 0; i < packed_.size(); ++i) {
            packed_[i] = root_states_[ancestors_[i]];
        }
    }
    started = MPI_Wtime();
    MPI_Scatterv(packed_.data(), partition_.counts.data(), partition_.offsets.data(),
                 state_type_, output.data(), local_size, state_type_, 0, comm_);
    mpi_ms += elapsed_ms(started);

    // Charge each non-root contribution once, including the states sent back.
    const auto bytes = partition_.rank == 0 ? 0 : std::uint64_t(local_size) *
                       (2 * sizeof(State) + sizeof(double));
    return {mpi_ms, bytes, partition_.rank == 0 ? partition_.total() : 0};
}

ResamplingStats SystematicResampler::distributed(const std::vector<State>& states,
                                                   const std::vector<double>& weights,
                                                   double local_mass, double u,
                                                   std::vector<State>& output) {
    double started = MPI_Wtime();
    MPI_Allgather(&local_mass, 1, MPI_DOUBLE, rank_prefix_.data() + 1, 1, MPI_DOUBLE, comm_);
    double mpi_ms = elapsed_ms(started);
    // Every rank builds the same boundaries in rank order, including at zero-mass ranks.
    std::partial_sum(rank_prefix_.begin(), rank_prefix_.end(), rank_prefix_.begin());
    const double mass = rank_prefix_.back();
    if (!(mass > 0.0 && std::isfinite(mass))) {
        throw std::runtime_error("invalid global mass in systematic resampling");
    }
    const SystematicPoints point{partition_.total(), u, mass, std::nextafter(mass, 0.0)};
    const double rank_start = rank_prefix_[partition_.rank];
    const int first_slot = point.below(rank_start);
    const int last_slot = point.below(rank_prefix_[partition_.rank + 1]);
    const int packed = last_slot - first_slot;
    packed_.resize(packed);
    std::size_t parent = 0;
    double prefix = 0.0;
    for (int slot = first_slot; slot < last_slot; ++slot) {
        const double target = point(slot);
        while (rank_start + prefix <= target && parent < weights.size()) {
            prefix += weights[parent++];
        }
        if (parent == 0 || rank_start + prefix <= target) {
            throw std::runtime_error("systematic resampling could not locate a parent");
        }
        packed_[slot - first_slot] = states[parent - 1];
    }

    for (int rank = 0; rank < partition_.size; ++rank) {
        const int begin = std::max(first_slot, partition_.offsets[rank]);
        const int end = std::min(last_slot, partition_.offsets[rank + 1]);
        send_counts_[rank] = std::max(0, end - begin);
        send_offsets_[rank] = send_counts_[rank] ? begin - first_slot : 0;
    }
    started = MPI_Wtime();
    MPI_Alltoall(send_counts_.data(), 1, MPI_INT, recv_counts_.data(), 1, MPI_INT, comm_);
    mpi_ms += elapsed_ms(started);
    int received = 0;
    for (int rank = 0; rank < partition_.size; ++rank) {
        recv_offsets_[rank] = received;
        received += recv_counts_[rank];
    }
    if (received != partition_.local_size()) {
        throw std::runtime_error("systematic resampling changed the local particle count");
    }
    started = MPI_Wtime();
    MPI_Alltoallv(packed_.data(), send_counts_.data(), send_offsets_.data(), state_type_,
                  output.data(), recv_counts_.data(), recv_offsets_.data(), state_type_, comm_);
    mpi_ms += elapsed_ms(started);
    const auto off_rank = std::uint64_t(packed - send_counts_[partition_.rank]);
    return {mpi_ms, off_rank * sizeof(State), packed};
}

std::size_t count_nodes(MPI_Comm comm) {
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    std::array<char, MPI_MAX_PROCESSOR_NAME> name{};
    int length = 0;
    MPI_Get_processor_name(name.data(), &length);
    std::vector<char> names(rank == 0 ? static_cast<std::size_t>(ranks) * name.size() : 0);
    MPI_Gather(name.data(), static_cast<int>(name.size()), MPI_CHAR, names.data(),
               static_cast<int>(name.size()), MPI_CHAR, 0, comm);
    std::set<std::string> hosts;
    if (rank == 0) {
        for (int r = 0; r < ranks; ++r) {
            hosts.emplace(names.data() + r * name.size());
        }
    }
    return hosts.size();
}

Scenario simulate_on_root(Sensor sensor, std::size_t steps, std::uint64_t seed, MPI_Comm comm) {
    if (steps == 0 || steps > mpi_count_max / 2) {
        throw std::invalid_argument("time step count must be in [1, INT_MAX/2]");
    }
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    Scenario data;
    if (rank == 0) {
        data = simulate(sensor, steps, seed);
    } else {
        data.observations.resize(steps);
    }
    static_assert(sizeof(Observation) == 2 * sizeof(double));
    MPI_Bcast(data.observations.data(), static_cast<int>(2 * steps), MPI_DOUBLE, 0, comm);
    return data;
}

MpiResult particle_filter_mpi(Sensor sensor, const std::vector<Observation>& observations,
                              const FilterOptions& options, Resampling method, MPI_Comm comm) {
    if (observations.empty() || !(options.ess_ratio >= 0.0 && options.ess_ratio <= 1.0)) {
        throw std::invalid_argument("observations and an ESS ratio in [0, 1] are required");
    }
    MPI_Barrier(comm);
    const double started = MPI_Wtime();
    const ParticlePartition partition(options.particles, comm);
    const auto local_size = static_cast<std::size_t>(partition.local_size());
    const auto first = static_cast<std::uint64_t>(partition.first());
    const double count = static_cast<double>(partition.total());
    const double uniform_log_weight = -std::log(count);
    const Motion motion;
    const Likelihood log_likelihood(sensor);
    std::vector<State> states(local_size), resampled(local_size);
    std::vector<double> log_weights(local_size, uniform_log_weight), weights(local_size);
    SystematicResampler resampler(partition, method, comm);
    for (std::size_t i = 0; i < local_size; ++i) {
        const State noise = gaussian_noise(options.seed, first + i, 0);
        for (std::size_t component = 0; component < noise.size(); ++component) {
            states[i][component] = prior_mean[component] + prior_sd[component] * noise[component];
        }
    }

    MpiResult result;
    auto& fit = result.filter;
    auto& times = fit.times;
    fit.estimates.resize(observations.size());
    if (options.estimate_covariance) {
        fit.covariances.resize(observations.size());
    }
    times.initialize_ms = elapsed_ms(started);
    for (std::size_t step = 0; step < observations.size(); ++step) {
        const auto& observation = observations[step];
        if (!std::isfinite(observation[0]) || !std::isfinite(observation[1])) {
            throw std::invalid_argument("observations must be finite");
        }
        double phase = MPI_Wtime();
        double local_max = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < local_size; ++i) {
            motion.propagate(states[i], gaussian_noise(options.seed, first + i, step + 1));
            log_weights[i] += log_likelihood(states[i], observation);
            local_max = std::max(local_max, log_weights[i]);
        }
        times.propagate_weight_ms += elapsed_ms(phase);

        phase = MPI_Wtime();
        double global_max = 0.0;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        result.reduction_mpi_ms += elapsed_ms(phase);
        if (!std::isfinite(global_max)) {
            throw std::runtime_error("no particle has finite likelihood");
        }
        std::array<double, 6> local{}, global{};
        for (std::size_t i = 0; i < local_size; ++i) {
            const double weight = std::exp(log_weights[i] - global_max);
            weights[i] = weight;
            local[0] += weight;
            local[1] += weight * weight;
            for (std::size_t j = 0; j < states[i].size(); ++j) {
                local[j + 2] += weight * states[i][j];
            }
        }
        const double reduction_started = MPI_Wtime();
        MPI_Allreduce(local.data(), global.data(), 6, MPI_DOUBLE, MPI_SUM, comm);
        result.reduction_mpi_ms += elapsed_ms(reduction_started);
        const double inverse_sum = 1.0 / global[0];
        const double log_sum = std::log(global[0]);
        auto& estimate = fit.estimates[step];
        estimate.ess = std::clamp(global[0] * global[0] / global[1], 1.0, count);
        estimate.weight_sum = global[0] * inverse_sum;
        estimate.log_predictive_density = global_max + log_sum;
        fit.log_likelihood += estimate.log_predictive_density;
        for (std::size_t j = 0; j < estimate.mean.size(); ++j) {
            estimate.mean[j] = global[j + 2] * inverse_sum;
        }
        const bool do_resample = options.always_resample || estimate.ess < options.ess_ratio * count;
        if (!do_resample) {
            for (auto& log_weight : log_weights) {
                log_weight = (log_weight - global_max) - log_sum;
            }
        }
        times.normalize_ess_ms += elapsed_ms(phase);

        if (options.estimate_covariance) {
            phase = MPI_Wtime();
            // Compute covariance about the global mean. Include a weight-sum audit.
            std::array<double, 17> covariance{};
            for (std::size_t i = 0; i < local_size; ++i) {
                const double weight = weights[i] * inverse_sum;
                covariance.back() += weight;
                for (std::size_t row = 0; row < 4; ++row) {
                    const double delta = states[i][row] - estimate.mean[row];
                    for (std::size_t col = row; col < 4; ++col) {
                        covariance[4 * row + col] += weight * delta *
                                                     (states[i][col] - estimate.mean[col]);
                    }
                }
            }
            const double covariance_started = MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE, covariance.data(), 17, MPI_DOUBLE, MPI_SUM, comm);
            result.reduction_mpi_ms += elapsed_ms(covariance_started);
            estimate.weight_sum = covariance.back();
            for (std::size_t row = 0; row < 4; ++row) {
                for (std::size_t col = row; col < 4; ++col) {
                    fit.covariances[step][row][col] = fit.covariances[step][col][row] =
                        covariance[4 * row + col];
                }
            }
            times.estimate_ms += elapsed_ms(phase);
        }

        if (do_resample) {
            phase = MPI_Wtime();
            const auto stats = resampler.run(states, weights, local[0],
                                             systematic_offset(options.seed, step + 1), resampled);
            states.swap(resampled);
            std::fill(log_weights.begin(), log_weights.end(), uniform_log_weight);
            estimate.resampled = true;
            ++fit.resampled_steps;
            result.resampling_mpi_ms += stats.mpi_ms;
            result.resampling_off_rank_bytes += stats.off_rank_bytes;
            result.max_packed_particles = std::max(result.max_packed_particles, stats.packed_particles);
            times.resample_ms += elapsed_ms(phase);
        }
    }
    times.filter_ms = elapsed_ms(started);
    result.max_rank_mpi_pct = 100.0 * (result.reduction_mpi_ms + result.resampling_mpi_ms) / times.filter_ms;
    std::array<double, 9> maxima{times.initialize_ms, times.propagate_weight_ms,
                                 times.normalize_ess_ms, times.estimate_ms, times.resample_ms,
                                 times.filter_ms, result.reduction_mpi_ms, result.resampling_mpi_ms,
                                 result.max_rank_mpi_pct};
    MPI_Allreduce(MPI_IN_PLACE, maxima.data(), static_cast<int>(maxima.size()), MPI_DOUBLE, MPI_MAX, comm);
    times.initialize_ms = maxima[0];
    times.propagate_weight_ms = maxima[1];
    times.normalize_ess_ms = maxima[2];
    times.estimate_ms = maxima[3];
    times.resample_ms = maxima[4];
    times.filter_ms = maxima[5];
    result.reduction_mpi_ms = maxima[6];
    result.resampling_mpi_ms = maxima[7];
    result.max_rank_mpi_pct = maxima[8];
    MPI_Allreduce(MPI_IN_PLACE, &result.resampling_off_rank_bytes, 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &result.max_packed_particles, 1, MPI_INT, MPI_MAX, comm);
    return result;
}

}  // namespace tracking::parallel

#pragma once

#include "particle_filter.hpp"

#include <array>
#include <cmath>
#include <cstdint>

namespace tracking::parallel {

#ifdef __CUDACC__
#define TRACKING_HOST_DEVICE __host__ __device__
#else
#define TRACKING_HOST_DEVICE
#endif

struct PhiloxWords {
    std::uint32_t value[4];
};

// Philox4x32-10, as specified by Salmon et al. (SC11), Random123.
TRACKING_HOST_DEVICE inline PhiloxWords philox_words(std::uint64_t seed, std::uint64_t index,
                                                     std::uint64_t step) {
    PhiloxWords counter{{static_cast<std::uint32_t>(index), static_cast<std::uint32_t>(index >> 32),
                        static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(step >> 32)}};
    std::uint32_t key0 = static_cast<std::uint32_t>(seed);
    std::uint32_t key1 = static_cast<std::uint32_t>(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        const std::uint64_t product0 = std::uint64_t{0xd2511f53} * counter.value[0];
        const std::uint64_t product1 = std::uint64_t{0xcd9e8d57} * counter.value[2];
        counter = {{static_cast<std::uint32_t>(product1 >> 32) ^ counter.value[1] ^ key0,
                    static_cast<std::uint32_t>(product1),
                    static_cast<std::uint32_t>(product0 >> 32) ^ counter.value[3] ^ key1,
                    static_cast<std::uint32_t>(product0)}};
        key0 += 0x9e3779b9U;
        key1 += 0xbb67ae85U;
    }
    return counter;
}

inline std::array<std::uint32_t, 4> philox(std::uint64_t seed, std::uint64_t index,
                                          std::uint64_t step) {
    const auto words = philox_words(seed, index, step);
    return {words.value[0], words.value[1], words.value[2], words.value[3]};
}

TRACKING_HOST_DEVICE inline double uniform_open(std::uint32_t word) {
    return (static_cast<double>(word) + 0.5) * 0x1.0p-32;
}

#undef TRACKING_HOST_DEVICE

inline State gaussian_noise(std::uint64_t seed, std::uint64_t index, std::uint64_t step) {
    const auto words = philox(seed, index, step);
    State noise{};
    for (std::size_t axis = 0; axis < 2; ++axis) {
        const double radius = std::sqrt(-2.0 * std::log(uniform_open(words[2 * axis])));
        const double angle = (2.0 * pi) * uniform_open(words[2 * axis + 1]);
        noise[2 * axis] = radius * std::cos(angle);
        noise[2 * axis + 1] = radius * std::sin(angle);
    }
    return noise;
}

inline double systematic_offset(std::uint64_t seed, std::uint64_t step) {
    const auto words = philox(seed ^ 0x94d049bb133111ebULL, 0, step);
    const std::uint64_t bits = (std::uint64_t{words[0]} << 32) | words[1];
    return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

}  // namespace tracking::parallel

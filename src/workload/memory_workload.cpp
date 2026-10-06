#include "neuroswap/workload/memory_workload.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

namespace neuroswap::workload {
namespace {

constexpr std::size_t kBytesPerMiB = 1024U * 1024U;
constexpr std::size_t kWordsPerCacheLine = 8;  // 8 * sizeof(uint64_t) = 64 bytes

void validate_config(const WorkloadConfig& config) {
    if (config.memory_mb == 0 || config.iterations == 0) {
        throw std::invalid_argument("memory size and iterations must both be greater than zero");
    }
    if (config.memory_mb > std::numeric_limits<std::size_t>::max() / kBytesPerMiB) {
        throw std::invalid_argument("memory size is too large");
    }
}

void read_write_word(std::uint64_t& word, std::uint64_t salt, std::uint64_t& checksum) {
    word ^= salt + 0x9e3779b97f4a7c15ULL;
    checksum += word;
}

}  // namespace

AccessPattern parse_access_pattern(std::string_view value) {
    if (value == "sequential") {
        return AccessPattern::Sequential;
    }
    if (value == "strided") {
        return AccessPattern::Strided;
    }
    throw std::invalid_argument("pattern must be 'sequential' or 'strided'");
}

std::string_view access_pattern_name(AccessPattern pattern) {
    return pattern == AccessPattern::Sequential ? "sequential" : "strided";
}

MemoryWorkload::MemoryWorkload(const WorkloadConfig& config) : config_(config) {
    validate_config(config);
    const std::size_t allocated_bytes = config.memory_mb * kBytesPerMiB;
    const std::size_t word_count = allocated_bytes / sizeof(std::uint64_t);
    buffer_.resize(word_count);

    // Write every word once so every allocated page is deliberately touched.
    for (std::size_t index = 0; index < word_count; ++index) {
        buffer_[index] = static_cast<std::uint64_t>(index) ^ 0xa5a5a5a5a5a5a5a5ULL;
    }
}

void* MemoryWorkload::data() noexcept { return buffer_.data(); }

std::size_t MemoryWorkload::size_bytes() const noexcept {
    return buffer_.size() * sizeof(std::uint64_t);
}

WorkloadResult MemoryWorkload::run() {
    std::uint64_t checksum = 0;
    for (std::size_t iteration = 0; iteration < config_.iterations; ++iteration) {
        const std::uint64_t salt = static_cast<std::uint64_t>(iteration + 1);
        if (config_.pattern == AccessPattern::Sequential) {
            for (std::size_t index = 0; index < buffer_.size(); ++index) {
                read_write_word(buffer_[index], salt, checksum);
            }
        } else {
            // Visit the same words through cache-line-sized strides to provide
            // a deterministic alternative ordering without skipping any page.
            for (std::size_t offset = 0; offset < kWordsPerCacheLine; ++offset) {
                for (std::size_t index = offset; index < buffer_.size(); index += kWordsPerCacheLine) {
                    read_write_word(buffer_[index], salt, checksum);
                }
            }
        }
    }

    return WorkloadResult{size_bytes(), checksum};
}

WorkloadResult run_memory_workload(const WorkloadConfig& config) {
    MemoryWorkload workload(config);
    return workload.run();
}

}  // namespace neuroswap::workload

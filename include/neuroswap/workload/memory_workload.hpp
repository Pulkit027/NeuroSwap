#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace neuroswap::workload {

enum class AccessPattern { Sequential, Strided };

struct WorkloadConfig {
    std::size_t memory_mb{64};
    std::size_t iterations{5};
    AccessPattern pattern{AccessPattern::Sequential};
};

struct WorkloadResult {
    std::size_t allocated_bytes{};
    std::uint64_t checksum{};
};

class MemoryWorkload {
public:
    explicit MemoryWorkload(const WorkloadConfig& config);
    void* data() noexcept;
    std::size_t size_bytes() const noexcept;
    WorkloadResult run();

private:
    WorkloadConfig config_;
    std::vector<std::uint64_t> buffer_;
};

AccessPattern parse_access_pattern(std::string_view value);
std::string_view access_pattern_name(AccessPattern pattern);
WorkloadResult run_memory_workload(const WorkloadConfig& config);

}  // namespace neuroswap::workload

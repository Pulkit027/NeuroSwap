#pragma once

#include "neuroswap/workload/memory_workload.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace neuroswap::prefetcher {

struct OperationResult {
    bool success{false};
    int error_number{};
    std::string message;
};

class FileMapping {
public:
    FileMapping() = default;
    ~FileMapping();
    FileMapping(const FileMapping&) = delete;
    FileMapping& operator=(const FileMapping&) = delete;

    OperationResult open_and_map_readonly(const std::string& path);
    OperationResult advise_willneed() const;
    OperationResult close();
    const void* data() const noexcept;
    std::size_t size_bytes() const noexcept;

private:
    int file_descriptor_{-1};
    void* address_{nullptr};
    std::size_t size_bytes_{};
};

OperationResult create_deterministic_file(const std::string& path, std::size_t bytes);
OperationResult advise_drop_file_cache(const std::string& path);
std::uint64_t run_mapped_read_workload(const void* address, std::size_t bytes,
                                       std::size_t iterations,
                                       workload::AccessPattern pattern);

}  // namespace neuroswap::prefetcher

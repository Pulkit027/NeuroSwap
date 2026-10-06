#pragma once

#include <cstddef>
#include <string>

namespace neuroswap::pinner {
struct LockResult {
    bool success{false};
    std::size_t requested_bytes{};
    unsigned long long memlock_soft_limit_bytes{};
    int error_number{};
    std::string message;
};
struct UnlockResult { bool success{false}; int error_number{}; std::string message; };
class MemoryPinner {
public:
    LockResult lock(void* address, std::size_t bytes) const;
    UnlockResult unlock(void* address, std::size_t bytes) const;
};
}  // namespace neuroswap::pinner

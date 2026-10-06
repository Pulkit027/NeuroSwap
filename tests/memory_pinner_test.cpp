#include "neuroswap/pinner/memory_pinner.hpp"
#include <cassert>
#include <cerrno>
#include <cstdint>
int main() {
    const neuroswap::pinner::MemoryPinner pinner;
    const auto null_result = pinner.lock(nullptr, 64); assert(!null_result.success && null_result.error_number == EINVAL);
    std::uint64_t word = 0;
    const auto zero_result = pinner.lock(&word, 0); assert(!zero_result.success && zero_result.error_number == EINVAL);
    const auto over_limit = pinner.lock(&word, static_cast<std::size_t>(1) << 40U); assert(!over_limit.success && !over_limit.message.empty());
    const auto lock_result = pinner.lock(&word, sizeof(word));
    if (lock_result.success) { const auto unlock_result = pinner.unlock(&word, sizeof(word)); assert(unlock_result.success); }
    else { assert(lock_result.error_number != 0 && !lock_result.message.empty()); }
}

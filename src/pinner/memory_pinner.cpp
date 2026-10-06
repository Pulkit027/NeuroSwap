#include "neuroswap/pinner/memory_pinner.hpp"

#include <cerrno>
#include <cstring>
#include <limits>
#include <sstream>
#include <sys/mman.h>
#include <sys/resource.h>

namespace neuroswap::pinner {
namespace {
unsigned long long limit_to_bytes(rlim_t limit) {
    return limit == RLIM_INFINITY ? std::numeric_limits<unsigned long long>::max()
                                  : static_cast<unsigned long long>(limit);
}
std::string errno_message(int error_number) { return std::strerror(error_number); }
}  // namespace

LockResult MemoryPinner::lock(void* address, std::size_t bytes) const {
    if (address == nullptr) return {false, bytes, 0, EINVAL, "mlock() was not attempted: address is null"};
    if (bytes == 0) return {false, 0, 0, EINVAL, "mlock() was not attempted: size is zero"};
    rlimit limit{};
    if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0) {
        const int error_number = errno;
        return {false, bytes, 0, error_number, "getrlimit(RLIMIT_MEMLOCK) failed: " + errno_message(error_number)};
    }
    const auto soft_limit = limit_to_bytes(limit.rlim_cur);
    if (soft_limit != std::numeric_limits<unsigned long long>::max() && bytes > soft_limit) {
        std::ostringstream message;
        message << "mlock() was not attempted: requested " << bytes << " bytes exceeds RLIMIT_MEMLOCK soft limit of " << soft_limit << " bytes";
        return {false, bytes, soft_limit, ENOMEM, message.str()};
    }
    if (mlock(address, bytes) != 0) {
        const int error_number = errno;
        return {false, bytes, soft_limit, error_number, "mlock() failed: " + errno_message(error_number)};
    }
    return {true, bytes, soft_limit, 0, "mlock() succeeded"};
}

UnlockResult MemoryPinner::unlock(void* address, std::size_t bytes) const {
    if (address == nullptr) return {false, EINVAL, "munlock() was not attempted: address is null"};
    if (bytes == 0) return {false, EINVAL, "munlock() was not attempted: size is zero"};
    if (munlock(address, bytes) != 0) {
        const int error_number = errno;
        return {false, error_number, "munlock() failed: " + errno_message(error_number)};
    }
    return {true, 0, "munlock() succeeded"};
}
}  // namespace neuroswap::pinner

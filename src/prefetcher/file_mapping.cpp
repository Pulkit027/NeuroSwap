#include "neuroswap/prefetcher/file_mapping.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace neuroswap::prefetcher {
namespace {
constexpr std::size_t kWriteBlockBytes = 64U * 1024U;
constexpr std::size_t kWordsPerCacheLine = 8;

std::string error_text(int error_number) { return std::strerror(error_number); }

OperationResult errno_result(const std::string& operation) {
    const int error_number = errno;
    return {false, error_number, operation + " failed: " + error_text(error_number)};
}
}  // namespace

FileMapping::~FileMapping() { close(); }

OperationResult create_deterministic_file(const std::string& path, std::size_t bytes) {
    if (bytes == 0) return {false, EINVAL, "file creation was not attempted: size is zero"};
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return errno_result("open(" + path + ")");

    std::vector<unsigned char> block(kWriteBlockBytes);
    std::size_t written = 0;
    while (written < bytes) {
        const std::size_t count = std::min(block.size(), bytes - written);
        for (std::size_t index = 0; index < count; ++index) {
            block[index] = static_cast<unsigned char>((written + index) & 0xffU);
        }
        const ssize_t result = write(fd, block.data(), count);
        if (result < 0 || static_cast<std::size_t>(result) != count) {
            const auto failure = result < 0 ? errno_result("write(" + path + ")")
                                            : OperationResult{false, EIO, "write(" + path + ") wrote fewer bytes than requested"};
            close(fd);
            return failure;
        }
        written += count;
    }
    if (fsync(fd) != 0) { const auto failure = errno_result("fsync(" + path + ")"); close(fd); return failure; }
    if (close(fd) != 0) return errno_result("close(" + path + ")");
    return {true, 0, "deterministic test file created"};
}

OperationResult advise_drop_file_cache(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return errno_result("open(" + path + ")");
    const int result = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    const int close_result = close(fd);
    if (result != 0) return {false, result, "posix_fadvise(POSIX_FADV_DONTNEED) failed: " + error_text(result)};
    if (close_result != 0) return errno_result("close(" + path + ")");
    return {true, 0, "posix_fadvise(POSIX_FADV_DONTNEED) accepted"};
}

OperationResult FileMapping::open_and_map_readonly(const std::string& path, MapSharing sharing) {
    close();
    file_descriptor_ = open(path.c_str(), O_RDONLY);
    if (file_descriptor_ < 0) return errno_result("open(" + path + ")");
    struct stat status {};
    if (fstat(file_descriptor_, &status) != 0) { const auto failure = errno_result("fstat(" + path + ")"); close(); return failure; }
    if (status.st_size <= 0) { close(); return {false, EINVAL, "mmap() was not attempted: file size is zero"}; }
    size_bytes_ = static_cast<std::size_t>(status.st_size);
    const int flags = sharing == MapSharing::Shared ? MAP_SHARED : MAP_PRIVATE;
    address_ = mmap(nullptr, size_bytes_, PROT_READ, flags, file_descriptor_, 0);
    if (address_ == MAP_FAILED) { address_ = nullptr; const auto failure = errno_result("mmap(" + path + ")"); close(); return failure; }
    return {true, 0, "mmap() succeeded"};
}

OperationResult FileMapping::advise_willneed() const {
    if (address_ == nullptr || size_bytes_ == 0) return {false, EINVAL, "madvise() was not attempted: no valid mapping"};
    if (madvise(address_, size_bytes_, MADV_WILLNEED) != 0) return errno_result("madvise(MADV_WILLNEED)");
    return {true, 0, "madvise(MADV_WILLNEED) succeeded"};
}

OperationResult FileMapping::advise_willneed(std::size_t offset, std::size_t length) const {
    if (address_ == nullptr || size_bytes_ == 0) return {false, EINVAL, "madvise() was not attempted: no valid mapping"};
    if (length == 0 || offset > size_bytes_ || length > size_bytes_ - offset ||
        offset % static_cast<std::size_t>(sysconf(_SC_PAGESIZE)) != 0) {
        return {false, EINVAL, "madvise() was not attempted: range outside the mapping or not page-aligned"};
    }
    if (madvise(static_cast<char*>(address_) + offset, length, MADV_WILLNEED) != 0) {
        return errno_result("madvise(MADV_WILLNEED)");
    }
    return {true, 0, "madvise(MADV_WILLNEED) succeeded"};
}

OperationResult FileMapping::close() {
    if (address_ != nullptr) {
        if (munmap(address_, size_bytes_) != 0) return errno_result("munmap()");
        address_ = nullptr;
        size_bytes_ = 0;
    }
    if (file_descriptor_ >= 0) {
        if (::close(file_descriptor_) != 0) return errno_result("close()");
        file_descriptor_ = -1;
    }
    return {true, 0, "munmap() and close() succeeded"};
}

const void* FileMapping::data() const noexcept { return address_; }
std::size_t FileMapping::size_bytes() const noexcept { return size_bytes_; }

std::uint64_t run_mapped_read_workload(const void* address, std::size_t bytes,
                                       std::size_t iterations, workload::AccessPattern pattern) {
    if (address == nullptr || bytes == 0 || iterations == 0) return 0;
    const auto* words = static_cast<const volatile std::uint64_t*>(address);
    const std::size_t word_count = bytes / sizeof(std::uint64_t);
    std::uint64_t checksum = 0;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        if (pattern == workload::AccessPattern::Sequential) {
            for (std::size_t index = 0; index < word_count; ++index) checksum += words[index];
        } else {
            for (std::size_t offset = 0; offset < kWordsPerCacheLine; ++offset)
                for (std::size_t index = offset; index < word_count; index += kWordsPerCacheLine) checksum += words[index];
        }
    }
    return checksum;
}
}  // namespace neuroswap::prefetcher

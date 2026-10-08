#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace neuroswap::strategy {

struct ByteRange {
    std::uint64_t offset{};
    std::uint64_t length{};
};

struct TensorExtent {
    std::string name;
    std::uint64_t offset{};  // absolute file offset of the tensor data
    std::uint64_t size{};    // bytes up to the next tensor (includes alignment padding)
};

struct GgufLayout {
    bool ok{false};
    std::string error;
    std::uint32_t version{};
    std::uint64_t alignment{};
    std::uint64_t data_start{};
    std::vector<TensorExtent> tensors;  // sorted by offset
};

// Parses the GGUF (v2/v3, little-endian) header and tensor table of a mapped
// file. Tensor sizes are the distance between consecutive tensor offsets (the
// last tensor runs to end of file), so no per-quantization size table is needed.
// Reading the header faults in its pages (~5.7 MiB for the Qwen2.5 model).
GgufLayout parse_gguf_layout(const void* data, std::size_t size);

// Only the looked-up rows of the embedding are read per token, so it is not
// part of the per-token weight set (docs/STRATEGY_DESIGN.md section 3).
inline constexpr const char* kEmbeddingTensor = "token_embd.weight";

// File ranges of every tensor except kEmbeddingTensor, in file-offset order.
// Each tensor's start is rounded down and its end up to page_size (clamped to
// the last page of the file), and touching ranges are merged, so a boundary
// page shared with the embedding may be included.
std::vector<ByteRange> per_token_weight_ranges(const GgufLayout& layout, std::uint64_t file_size,
                                               std::uint64_t page_size);

std::uint64_t total_bytes(const std::vector<ByteRange>& ranges);

// The first `bytes` of `ranges`, in order; `bytes` is clamped to the total.
std::vector<ByteRange> select_prefix(const std::vector<ByteRange>& ranges, std::uint64_t bytes);

}  // namespace neuroswap::strategy

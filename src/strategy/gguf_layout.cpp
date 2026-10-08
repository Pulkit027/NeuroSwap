#include "neuroswap/strategy/gguf_layout.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace neuroswap::strategy {
namespace {
constexpr std::uint64_t kMaxEntries = 1'000'000;
constexpr int kMaxArrayDepth = 4;

// GGUF metadata value types.
enum : std::uint32_t {
    kU8 = 0, kI8, kU16, kI16, kU32, kI32, kF32, kBool, kString, kArray, kU64, kI64, kF64
};

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t scalar_size(std::uint32_t type) {
    switch (type) {
        case kU8: case kI8: case kBool: return 1;
        case kU16: case kI16: return 2;
        case kU32: case kI32: case kF32: return 4;
        case kU64: case kI64: case kF64: return 8;
        default: return 0;
    }
}

class Reader {
public:
    Reader(const unsigned char* data, std::size_t size) : data_(data), size_(size) {}

    template <typename T>
    T read() {
        need(sizeof(T));
        T value;
        std::memcpy(&value, data_ + position_, sizeof(T));
        position_ += sizeof(T);
        return value;
    }

    std::string read_string() {
        const auto length = read<std::uint64_t>();
        need(length);
        std::string value(reinterpret_cast<const char*>(data_ + position_), length);
        position_ += length;
        return value;
    }

    void skip(std::uint64_t bytes) { need(bytes); position_ += bytes; }
    std::size_t position() const noexcept { return position_; }

private:
    void need(std::uint64_t bytes) const {
        if (bytes > size_ - position_) {
            throw std::runtime_error("header truncated at byte " + std::to_string(position_));
        }
    }

    const unsigned char* data_;
    std::size_t size_;
    std::size_t position_{0};
};

void skip_value(Reader& reader, std::uint32_t type, int depth) {
    if (type == kString) {
        reader.skip(reader.read<std::uint64_t>());
        return;
    }
    if (type == kArray) {
        if (depth >= kMaxArrayDepth) throw std::runtime_error("metadata arrays nested too deeply");
        const auto element_type = reader.read<std::uint32_t>();
        const auto count = reader.read<std::uint64_t>();
        const auto element_size = scalar_size(element_type);
        if (element_size != 0) {
            if (count > UINT64_MAX / element_size) throw std::runtime_error("metadata array too large");
            reader.skip(count * element_size);
        } else {
            for (std::uint64_t index = 0; index < count; ++index) skip_value(reader, element_type, depth + 1);
        }
        return;
    }
    const auto size = scalar_size(type);
    if (size == 0) throw std::runtime_error("unknown metadata value type " + std::to_string(type));
    reader.skip(size);
}
}  // namespace

GgufLayout parse_gguf_layout(const void* data, std::size_t size) {
    GgufLayout layout;
    try {
        if (data == nullptr || size < 4 || std::memcmp(data, "GGUF", 4) != 0) {
            throw std::runtime_error("not a GGUF file (bad magic)");
        }
        Reader reader(static_cast<const unsigned char*>(data), size);
        reader.skip(4);
        layout.version = reader.read<std::uint32_t>();
        if (layout.version != 2 && layout.version != 3) {
            throw std::runtime_error("unsupported GGUF version " + std::to_string(layout.version));
        }
        const auto tensor_count = reader.read<std::uint64_t>();
        const auto kv_count = reader.read<std::uint64_t>();
        if (tensor_count > kMaxEntries || kv_count > kMaxEntries) {
            throw std::runtime_error("implausible tensor or metadata count");
        }

        layout.alignment = 32;  // GGUF default when general.alignment is absent
        for (std::uint64_t index = 0; index < kv_count; ++index) {
            const auto key = reader.read_string();
            const auto type = reader.read<std::uint32_t>();
            if (key == "general.alignment" && type == kU32) {
                layout.alignment = reader.read<std::uint32_t>();
            } else {
                skip_value(reader, type, 0);
            }
        }
        if (layout.alignment == 0 || (layout.alignment & (layout.alignment - 1)) != 0) {
            throw std::runtime_error("general.alignment is not a power of two");
        }

        std::vector<std::pair<std::string, std::uint64_t>> entries;
        entries.reserve(tensor_count);
        for (std::uint64_t index = 0; index < tensor_count; ++index) {
            auto name = reader.read_string();
            const auto dimensions = reader.read<std::uint32_t>();
            if (dimensions > 8) throw std::runtime_error("tensor " + name + " has too many dimensions");
            reader.skip(static_cast<std::uint64_t>(dimensions) * sizeof(std::uint64_t));
            reader.read<std::uint32_t>();  // ggml type; sizes come from offsets instead
            entries.emplace_back(std::move(name), reader.read<std::uint64_t>());
        }

        layout.data_start = align_up(reader.position(), layout.alignment);
        if (layout.data_start > size) throw std::runtime_error("tensor data starts beyond end of file");
        std::sort(entries.begin(), entries.end(),
                  [](const auto& left, const auto& right) { return left.second < right.second; });
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const auto begin = layout.data_start + entries[index].second;
            const auto end = index + 1 < entries.size() ? layout.data_start + entries[index + 1].second : size;
            if (begin >= size || end > size || end <= begin) {
                throw std::runtime_error("tensor " + entries[index].first + " has an invalid offset");
            }
            layout.tensors.push_back({entries[index].first, begin, end - begin});
        }
    } catch (const std::exception& error) {
        layout.ok = false;
        layout.error = error.what();
        layout.tensors.clear();
        return layout;
    }
    layout.ok = true;
    return layout;
}

std::vector<ByteRange> per_token_weight_ranges(const GgufLayout& layout, std::uint64_t file_size,
                                               std::uint64_t page_size) {
    std::vector<ByteRange> ranges;
    if (!layout.ok || page_size == 0) return ranges;
    const auto file_end = align_up(file_size, page_size);
    for (const auto& tensor : layout.tensors) {
        if (tensor.name == kEmbeddingTensor) continue;
        const auto begin = tensor.offset / page_size * page_size;
        const auto end = std::min(align_up(tensor.offset + tensor.size, page_size), file_end);
        if (!ranges.empty() && begin <= ranges.back().offset + ranges.back().length) {
            auto& last = ranges.back();
            last.length = std::max(last.offset + last.length, end) - last.offset;
        } else {
            ranges.push_back({begin, end - begin});
        }
    }
    return ranges;
}

std::uint64_t total_bytes(const std::vector<ByteRange>& ranges) {
    std::uint64_t total = 0;
    for (const auto& range : ranges) total += range.length;
    return total;
}

std::vector<ByteRange> select_prefix(const std::vector<ByteRange>& ranges, std::uint64_t bytes) {
    std::vector<ByteRange> selected;
    for (const auto& range : ranges) {
        if (bytes == 0) break;
        const auto take = std::min(bytes, range.length);
        selected.push_back({range.offset, take});
        bytes -= take;
    }
    return selected;
}

}  // namespace neuroswap::strategy

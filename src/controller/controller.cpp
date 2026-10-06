#include "neuroswap/controller/controller.hpp"

#include <stdexcept>

namespace neuroswap::controller {

Decision Controller::decide(const ControllerInput& input) const {
    if (input.workload_type == WorkloadType::FileBacked &&
        input.pattern == workload::AccessPattern::Sequential) {
        return {OperatingMode::Prefetched,
                "file-backed sequential access selected; request MADV_WILLNEED"};
    }
    if (input.workload_type == WorkloadType::MemorySensitive) {
        return {OperatingMode::Pinned,
                "memory-sensitive workload selected; attempt bounded mlock()"};
    }
    return {OperatingMode::Baseline,
            "normal memory workload selected; run without memory-management hint"};
}

WorkloadType parse_workload_type(std::string_view value) {
    if (value == "normal") return WorkloadType::NormalMemory;
    if (value == "sensitive") return WorkloadType::MemorySensitive;
    if (value == "file-backed") return WorkloadType::FileBacked;
    throw std::invalid_argument("type must be 'normal', 'sensitive', or 'file-backed'");
}

std::string_view workload_type_name(WorkloadType type) {
    switch (type) {
        case WorkloadType::NormalMemory: return "normal memory";
        case WorkloadType::MemorySensitive: return "memory-sensitive (configured)";
        case WorkloadType::FileBacked: return "file-backed";
    }
    return "unknown";
}

std::string_view operating_mode_name(OperatingMode mode) {
    switch (mode) {
        case OperatingMode::Baseline: return "BASELINE";
        case OperatingMode::Pinned: return "PINNED";
        case OperatingMode::Prefetched: return "PREFETCHED";
    }
    return "UNKNOWN";
}

}  // namespace neuroswap::controller

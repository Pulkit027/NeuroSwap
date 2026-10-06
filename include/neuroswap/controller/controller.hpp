#pragma once

#include "neuroswap/workload/memory_workload.hpp"

#include <cstddef>
#include <string_view>

namespace neuroswap::controller {

enum class WorkloadType { NormalMemory, MemorySensitive, FileBacked };
enum class OperatingMode { Baseline, Pinned, Prefetched };

struct ControllerInput {
    WorkloadType workload_type{WorkloadType::NormalMemory};
    std::size_t memory_mb{16};
    workload::AccessPattern pattern{workload::AccessPattern::Sequential};
};

struct Decision {
    OperatingMode mode{OperatingMode::Baseline};
    std::string_view reason;
};

class Controller {
public:
    // Deterministic configuration rules only; this controller does not predict
    // memory pressure or identify internal LLM memory regions.
    Decision decide(const ControllerInput& input) const;
};

WorkloadType parse_workload_type(std::string_view value);
std::string_view workload_type_name(WorkloadType type);
std::string_view operating_mode_name(OperatingMode mode);

}  // namespace neuroswap::controller

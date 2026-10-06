#include "neuroswap/controller/controller.hpp"

#include <cassert>

int main() {
    using namespace neuroswap::controller;
    const Controller controller;
    const auto baseline = controller.decide({WorkloadType::NormalMemory, 16, neuroswap::workload::AccessPattern::Sequential});
    const auto pinned = controller.decide({WorkloadType::MemorySensitive, 16, neuroswap::workload::AccessPattern::Strided});
    const auto prefetched = controller.decide({WorkloadType::FileBacked, 16, neuroswap::workload::AccessPattern::Sequential});
    const auto file_strided = controller.decide({WorkloadType::FileBacked, 16, neuroswap::workload::AccessPattern::Strided});
    assert(baseline.mode == OperatingMode::Baseline);
    assert(pinned.mode == OperatingMode::Pinned);
    assert(prefetched.mode == OperatingMode::Prefetched);
    assert(file_strided.mode == OperatingMode::Baseline);
    assert(controller.decide({WorkloadType::MemorySensitive, 16, neuroswap::workload::AccessPattern::Strided}).mode == pinned.mode);
}

#include "neuroswap/workload/memory_workload.hpp"

#include <cassert>

int main() {
    using namespace neuroswap::workload;

    const WorkloadConfig sequential{1, 1, AccessPattern::Sequential};
    const WorkloadConfig strided{1, 1, AccessPattern::Strided};
    const auto sequential_result = run_memory_workload(sequential);
    const auto strided_result = run_memory_workload(strided);

    assert(sequential_result.allocated_bytes == 1024U * 1024U);
    assert(strided_result.allocated_bytes == 1024U * 1024U);
    assert(sequential_result.checksum == strided_result.checksum);
}

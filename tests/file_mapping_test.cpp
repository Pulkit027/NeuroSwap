#include "neuroswap/prefetcher/file_mapping.hpp"

#include <cassert>
#include <string>
#include <unistd.h>

int main() {
    using namespace neuroswap::prefetcher;
    FileMapping mapping;
    const auto missing = mapping.open_and_map_readonly("/tmp/neuroswap_file_that_does_not_exist");
    assert(!missing.success && !missing.message.empty());
    const std::string path = "/tmp/neuroswap_file_mapping_test_" + std::to_string(getpid()) + ".bin";
    const auto zero = create_deterministic_file(path, 0);
    assert(!zero.success);
    const auto created = create_deterministic_file(path, 4096);
    assert(created.success);
    const auto mapped = mapping.open_and_map_readonly(path);
    assert(mapped.success && mapping.data() != nullptr && mapping.size_bytes() == 4096);
    const auto advice = mapping.advise_willneed();
    assert(advice.success || !advice.message.empty());
    const auto cleanup = mapping.close();
    assert(cleanup.success);
    unlink(path.c_str());
}

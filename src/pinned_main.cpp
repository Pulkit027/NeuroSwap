#include "neuroswap/pinner/memory_pinner.hpp"
#include "neuroswap/profiler/measurement.hpp"
#include "neuroswap/workload/memory_workload.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
std::size_t parse_positive_size(const std::string& value, const char* option) {
    try { std::size_t used = 0; const auto parsed = std::stoull(value, &used); if (used != value.size() || parsed == 0) throw std::invalid_argument("invalid"); return parsed; }
    catch (const std::exception&) { throw std::invalid_argument(std::string(option) + " requires a positive integer"); }
}
void usage(const char* program) { std::cout << "Usage: " << program << " [--memory-mb N] [--iterations N] [--pattern sequential|strided]\n"; }
}  // namespace

int main(int argc, char* argv[]) {
    try {
        neuroswap::workload::WorkloadConfig config;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help") { usage(argv[0]); return 0; }
            if (index + 1 >= argc) throw std::invalid_argument("missing value for " + argument);
            const std::string value = argv[++index];
            if (argument == "--memory-mb") config.memory_mb = parse_positive_size(value, "--memory-mb");
            else if (argument == "--iterations") config.iterations = parse_positive_size(value, "--iterations");
            else if (argument == "--pattern") config.pattern = neuroswap::workload::parse_access_pattern(value);
            else throw std::invalid_argument("unknown option: " + argument);
        }
        neuroswap::profiler::MeasurementScope measurement;
        neuroswap::workload::MemoryWorkload workload(config);
        const neuroswap::pinner::MemoryPinner pinner;
        const auto lock_result = pinner.lock(workload.data(), workload.size_bytes());
        if (!lock_result.success) { std::cerr << "Pinner error: " << lock_result.message << '\n'; return 1; }
        const auto result = workload.run();
        const auto metrics = measurement.finish();
        const auto unlock_result = pinner.unlock(workload.data(), workload.size_bytes());
        std::cout << "=== NeuroSwap Pinned ===\n"
                  << "Memory: " << config.memory_mb << " MiB\n"
                  << "Iterations: " << config.iterations << "\n"
                  << "Pattern: " << neuroswap::workload::access_pattern_name(config.pattern) << "\n"
                  << "Allocated bytes: " << result.allocated_bytes << "\n"
                  << std::fixed << std::setprecision(3) << "Execution time: " << metrics.wall_clock_ms << " ms\n"
                  << "Minor page faults: " << metrics.minor_page_faults << "\n"
                  << "Major page faults: " << metrics.major_page_faults << "\n"
                  << "Max RSS: " << metrics.max_rss_kb << " KiB\n"
                  << "mlock(): " << lock_result.message << "\n"
                  << "Pinned bytes: " << lock_result.requested_bytes << "\n"
                  << "RLIMIT_MEMLOCK soft limit: " << lock_result.memlock_soft_limit_bytes << " bytes\n"
                  << "munlock(): " << unlock_result.message << "\n"
                  << "Checksum: " << result.checksum << "\n";
        return unlock_result.success ? 0 : 1;
    } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; usage(argv[0]); return 1; }
}

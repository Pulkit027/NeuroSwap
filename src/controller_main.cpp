#include "neuroswap/controller/controller.hpp"
#include "neuroswap/pinner/memory_pinner.hpp"
#include "neuroswap/prefetcher/file_mapping.hpp"
#include "neuroswap/profiler/measurement.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr std::size_t kBytesPerMiB = 1024U * 1024U;
struct Config { std::size_t memory_mb{16}; std::size_t iterations{3}; neuroswap::workload::AccessPattern pattern{neuroswap::workload::AccessPattern::Sequential}; neuroswap::controller::WorkloadType type{neuroswap::controller::WorkloadType::NormalMemory}; };
std::size_t positive(const std::string& value, const char* option) {
    try { std::size_t used = 0; const auto parsed = std::stoull(value, &used); if (used != value.size() || parsed == 0) throw std::invalid_argument("invalid"); return parsed; }
    catch (const std::exception&) { throw std::invalid_argument(std::string(option) + " requires a positive integer"); }
}
void usage(const char* program) { std::cout << "Usage: " << program << " [--type normal|sensitive|file-backed] [--memory-mb N] [--iterations N] [--pattern sequential|strided]\n"; }
void print_measurements(const neuroswap::profiler::Measurement& metrics, std::uint64_t checksum) {
    std::cout << std::fixed << std::setprecision(3) << "Execution time: " << metrics.wall_clock_ms << " ms\n"
              << "Minor page faults: " << metrics.minor_page_faults << "\n"
              << "Major page faults: " << metrics.major_page_faults << "\n"
              << "Max RSS: " << metrics.max_rss_kb << " KiB\n"
              << "Checksum: " << checksum << "\n";
}
void run_baseline(const Config& config) {
    neuroswap::profiler::MeasurementScope timer;
    neuroswap::workload::MemoryWorkload workload({config.memory_mb, config.iterations, config.pattern});
    const auto result = workload.run();
    print_measurements(timer.finish(), result.checksum);
}
void run_pinned(const Config& config) {
    neuroswap::profiler::MeasurementScope timer;
    neuroswap::workload::MemoryWorkload workload({config.memory_mb, config.iterations, config.pattern});
    const neuroswap::pinner::MemoryPinner pinner;
    const auto lock = pinner.lock(workload.data(), workload.size_bytes());
    std::cout << "mlock(): " << lock.message << "\nPinned bytes: " << (lock.success ? lock.requested_bytes : 0) << "\n";
    if (!lock.success) throw std::runtime_error("Pinner could not run: " + lock.message);
    const auto result = workload.run();
    const auto metrics = timer.finish();
    const auto unlock = pinner.unlock(workload.data(), workload.size_bytes());
    std::cout << "munlock(): " << unlock.message << "\n";
    print_measurements(metrics, result.checksum);
    if (!unlock.success) throw std::runtime_error("Pinner cleanup failed: " + unlock.message);
}
void run_prefetched(const Config& config) {
    if (config.memory_mb > static_cast<std::size_t>(-1) / kBytesPerMiB) throw std::invalid_argument("file size is too large");
    const std::string path = "results/raw/controller_prefetch_data.bin";
    const auto created = neuroswap::prefetcher::create_deterministic_file(path, config.memory_mb * kBytesPerMiB);
    if (!created.success) throw std::runtime_error(created.message);
    const auto cache_advice = neuroswap::prefetcher::advise_drop_file_cache(path);
    neuroswap::profiler::MeasurementScope timer;
    neuroswap::prefetcher::FileMapping mapping;
    const auto mapped = mapping.open_and_map_readonly(path);
    if (!mapped.success) throw std::runtime_error(mapped.message);
    const auto advice = mapping.advise_willneed();
    const auto checksum = neuroswap::prefetcher::run_mapped_read_workload(mapping.data(), mapping.size_bytes(), config.iterations, config.pattern);
    const auto metrics = timer.finish();
    const auto cleanup = mapping.close();
    std::cout << "File preparation: " << created.message << "\nCache-reset advice: " << cache_advice.message << "\nmadvise(): " << advice.message << "\nCleanup: " << cleanup.message << "\n";
    print_measurements(metrics, checksum);
    if (!cleanup.success) throw std::runtime_error("Prefetcher cleanup failed: " + cleanup.message);
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        Config config;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help") { usage(argv[0]); return 0; }
            if (index + 1 >= argc) throw std::invalid_argument("missing value for " + argument);
            const std::string value = argv[++index];
            if (argument == "--type") config.type = neuroswap::controller::parse_workload_type(value);
            else if (argument == "--memory-mb") config.memory_mb = positive(value, "--memory-mb");
            else if (argument == "--iterations") config.iterations = positive(value, "--iterations");
            else if (argument == "--pattern") config.pattern = neuroswap::workload::parse_access_pattern(value);
            else throw std::invalid_argument("unknown option: " + argument);
        }
        const neuroswap::controller::Controller controller;
        const auto decision = controller.decide({config.type, config.memory_mb, config.pattern});
        std::cout << "=== NeuroSwap Controller ===\nRule-based controller: configuration-driven, not predictive.\n\nWorkload:\n"
                  << "Memory: " << config.memory_mb << " MiB\nIterations: " << config.iterations << "\nPattern: " << neuroswap::workload::access_pattern_name(config.pattern) << "\nType: " << neuroswap::controller::workload_type_name(config.type) << "\n\nController decision:\n"
                  << "Mode: " << neuroswap::controller::operating_mode_name(decision.mode) << "\nReason: " << decision.reason << "\n\nExperiment output:\n";
        if (decision.mode == neuroswap::controller::OperatingMode::Baseline) run_baseline(config);
        else if (decision.mode == neuroswap::controller::OperatingMode::Pinned) run_pinned(config);
        else run_prefetched(config);
        return 0;
    } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; usage(argv[0]); return 1; }
}

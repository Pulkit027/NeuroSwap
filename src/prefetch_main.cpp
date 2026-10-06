#include "neuroswap/prefetcher/file_mapping.hpp"
#include "neuroswap/profiler/measurement.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr std::size_t kBytesPerMiB = 1024U * 1024U;
struct Config { std::size_t file_mb{16}; std::size_t iterations{3}; neuroswap::workload::AccessPattern pattern{neuroswap::workload::AccessPattern::Sequential}; };
std::size_t positive(const std::string& value, const char* option) {
    try { std::size_t used = 0; const auto parsed = std::stoull(value, &used); if (used != value.size() || parsed == 0) throw std::invalid_argument("invalid"); return parsed; }
    catch (const std::exception&) { throw std::invalid_argument(std::string(option) + " requires a positive integer"); }
}
void usage(const char* program) { std::cout << "Usage: " << program << " [--file-mb N] [--iterations N] [--pattern sequential|strided]\n"; }
struct CaseResult { neuroswap::profiler::Measurement measurement; neuroswap::prefetcher::OperationResult cache_advice; neuroswap::prefetcher::OperationResult madvise; neuroswap::prefetcher::OperationResult cleanup; std::uint64_t checksum{}; };
CaseResult run_case(const std::string& path, const Config& config, bool use_prefetch) {
    CaseResult output{};
    output.cache_advice = neuroswap::prefetcher::advise_drop_file_cache(path);
    neuroswap::profiler::MeasurementScope timer;
    neuroswap::prefetcher::FileMapping mapping;
    const auto map_result = mapping.open_and_map_readonly(path);
    if (!map_result.success) throw std::runtime_error(map_result.message);
    output.madvise = use_prefetch ? mapping.advise_willneed() : neuroswap::prefetcher::OperationResult{true, 0, "not requested"};
    output.checksum = neuroswap::prefetcher::run_mapped_read_workload(mapping.data(), mapping.size_bytes(), config.iterations, config.pattern);
    output.measurement = timer.finish();
    output.cleanup = mapping.close();
    return output;
}
void print_case(const char* name, const CaseResult& result) {
    std::cout << "--- " << name << " ---\n" << std::fixed << std::setprecision(3)
              << "Execution time: " << result.measurement.wall_clock_ms << " ms\n"
              << "Minor page faults: " << result.measurement.minor_page_faults << "\n"
              << "Major page faults: " << result.measurement.major_page_faults << "\n"
              << "Max RSS: " << result.measurement.max_rss_kb << " KiB\n"
              << "Cache-reset advice: " << result.cache_advice.message << "\n"
              << "madvise(): " << result.madvise.message << "\n"
              << "Cleanup: " << result.cleanup.message << "\n"
              << "Checksum: " << result.checksum << "\n";
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
            if (argument == "--file-mb") config.file_mb = positive(value, "--file-mb");
            else if (argument == "--iterations") config.iterations = positive(value, "--iterations");
            else if (argument == "--pattern") config.pattern = neuroswap::workload::parse_access_pattern(value);
            else throw std::invalid_argument("unknown option: " + argument);
        }
        if (config.file_mb > static_cast<std::size_t>(-1) / kBytesPerMiB) throw std::invalid_argument("file size is too large");
        const std::string path = "results/raw/prefetch_test_data.bin";
        const auto create_result = neuroswap::prefetcher::create_deterministic_file(path, config.file_mb * kBytesPerMiB);
        if (!create_result.success) throw std::runtime_error(create_result.message);
        const auto baseline = run_case(path, config, false);
        const auto hinted = run_case(path, config, true);
        std::cout << "=== NeuroSwap Prefetch Experiment ===\n"
                  << "File: " << path << "\nFile size: " << config.file_mb << " MiB\nIterations: " << config.iterations << "\nPattern: " << neuroswap::workload::access_pattern_name(config.pattern) << "\n"
                  << "File preparation: " << create_result.message << "\n"
                  << "MADV_WILLNEED is an advisory hint; it does not guarantee prefetched pages.\n";
        print_case("Without prefetch hint", baseline);
        print_case("With MADV_WILLNEED", hinted);
        return baseline.cleanup.success && hinted.cleanup.success ? 0 : 1;
    } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; usage(argv[0]); return 1; }
}

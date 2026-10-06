#include "neuroswap/profiler/measurement.hpp"
#include "neuroswap/workload/memory_workload.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::size_t parse_positive_size(const std::string& value, const char* option) {
    std::size_t parsed = 0;
    try {
        std::size_t consumed = 0;
        parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || parsed == 0) {
            throw std::invalid_argument("invalid value");
        }
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer");
    }
    return parsed;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--memory-mb N] [--iterations N] [--pattern sequential|strided]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        neuroswap::workload::WorkloadConfig config;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help") {
                print_usage(argv[0]);
                return 0;
            }
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            const std::string value = argv[++index];
            if (argument == "--memory-mb") {
                config.memory_mb = parse_positive_size(value, "--memory-mb");
            } else if (argument == "--iterations") {
                config.iterations = parse_positive_size(value, "--iterations");
            } else if (argument == "--pattern") {
                config.pattern = neuroswap::workload::parse_access_pattern(value);
            } else {
                throw std::invalid_argument("unknown option: " + argument);
            }
        }

        neuroswap::profiler::MeasurementScope measurement;
        const auto result = neuroswap::workload::run_memory_workload(config);
        const auto metrics = measurement.finish();

        std::cout << "=== NeuroSwap Baseline ===\n"
                  << "Memory: " << config.memory_mb << " MiB\n"
                  << "Iterations: " << config.iterations << "\n"
                  << "Pattern: " << neuroswap::workload::access_pattern_name(config.pattern) << "\n"
                  << "Allocated bytes: " << result.allocated_bytes << "\n"
                  << std::fixed << std::setprecision(3)
                  << "Execution time: " << metrics.wall_clock_ms << " ms\n"
                  << "Minor page faults: " << metrics.minor_page_faults << "\n"
                  << "Major page faults: " << metrics.major_page_faults << "\n"
                  << "Max RSS: " << metrics.max_rss_kb << " KiB\n"
                  << "Checksum: " << result.checksum << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}

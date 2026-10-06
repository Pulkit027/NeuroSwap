#include "neuroswap/profiler/measurement.hpp"

#include <stdexcept>

namespace neuroswap::profiler {

MeasurementScope::MeasurementScope() : started_at_(std::chrono::steady_clock::now()) {
    if (getrusage(RUSAGE_SELF, &start_usage_) != 0) {
        throw std::runtime_error("getrusage(RUSAGE_SELF) failed");
    }
}

Measurement MeasurementScope::finish() const {
    rusage end_usage{};
    if (getrusage(RUSAGE_SELF, &end_usage) != 0) {
        throw std::runtime_error("getrusage(RUSAGE_SELF) failed");
    }

    const auto elapsed = std::chrono::steady_clock::now() - started_at_;
    return Measurement{
        std::chrono::duration<double, std::milli>(elapsed).count(),
        end_usage.ru_minflt - start_usage_.ru_minflt,
        end_usage.ru_majflt - start_usage_.ru_majflt,
        end_usage.ru_maxrss,
    };
}

}  // namespace neuroswap::profiler

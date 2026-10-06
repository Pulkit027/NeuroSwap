#pragma once

#include <chrono>
#include <sys/resource.h>

namespace neuroswap::profiler {

struct Measurement {
    double wall_clock_ms{};
    long minor_page_faults{};
    long major_page_faults{};
    long max_rss_kb{};
};

// Captures process resource usage before and after one experiment. ru_maxrss is
// reported as the process high-water mark (KiB) after the experiment.
class MeasurementScope {
public:
    MeasurementScope();
    Measurement finish() const;

private:
    std::chrono::steady_clock::time_point started_at_;
    rusage start_usage_{};
};

}  // namespace neuroswap::profiler

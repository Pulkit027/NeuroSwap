#pragma once

#include "neuroswap/prefetcher/file_mapping.hpp"
#include "neuroswap/strategy/gguf_layout.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace neuroswap::strategy {

enum class Strategy { Baseline, Prefetch, Pin };
Strategy parse_strategy(const std::string& value);
const char* strategy_name(Strategy strategy);

inline constexpr int kExitOk = 0;
inline constexpr int kExitError = 1;
inline constexpr int kExitUsage = 2;
inline constexpr int kExitCapRefused = 3;

inline constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;

// Inputs to pin_cap = min(room(L), memlock policy, this process's RLIMIT_MEMLOCK),
// room(L) = memory.max - peak_anon - other_rss - margin (docs/STRATEGY_DESIGN.md
// section 3). Defaults are the documented values.
struct PinBudget {
    std::optional<std::uint64_t> memory_max_bytes;           // nullopt: no cgroup limit
    std::uint64_t peak_anon_bytes{182156ULL * 1024ULL};       // measured, docs/CALIBRATION.md
    std::uint64_t other_rss_bytes{13ULL * kMiB};              // measured
    std::uint64_t margin_bytes{64ULL * kMiB};                 // assumption
    std::uint64_t memlock_policy_bytes{758604ULL * 1024ULL};  // measured root limit, kept fixed
    std::optional<std::uint64_t> memlock_rlimit_bytes;        // nullopt: unlimited
};

struct CapDecision {
    bool allowed{false};
    std::uint64_t requested_bytes{};
    std::uint64_t pin_cap_bytes{};
    std::optional<std::uint64_t> room_bytes;  // nullopt: no cgroup limit
    std::string binding_bound;                // "room(L)", "memlock_policy" or "rlimit_memlock"
    std::string message;
};

CapDecision check_pin_cap(const PinBudget& budget, std::uint64_t requested_bytes);

// mincore() residency over page-aligned ranges of a mapping.
struct Residency {
    bool measured{false};
    std::uint64_t pages_total{};
    std::uint64_t pages_resident{};
    int error_number{};
    double percent() const;
};
Residency measure_residency(const void* base, const std::vector<ByteRange>& ranges,
                            std::uint64_t page_size);

// Polls `measure` every `poll_interval` and stops at the first of:
//   "target"             percent() >= target_percent
//   "plateau"            resident pages have not grown for plateau_s
//   "timeout"            timeout_s has passed
//   "measurement_failed" mincore() failed (residency.measured is false)
// The last measurement is returned in `residency`; timed_out == (stopped_reason == "timeout").
struct WaitResult {
    double target_percent{};
    double timeout_s{};
    double plateau_s{};
    double wait_ms{};
    std::uint64_t polls{};
    std::string stopped_reason;
    bool timed_out{false};
    Residency residency;

    // Chunked prefetch only (prefetch_in_chunks); chunk_bytes == 0 otherwise.
    std::uint64_t chunk_bytes{};
    std::uint64_t chunks_requested{};
    std::uint64_t chunks_completed{};
    std::uint64_t madvise_calls{};
    int madvise_errno{};
    std::uint64_t bytes_resident_final{};
    std::optional<std::uint64_t> stopped_chunk_index;  // chunk the loop stopped in, if it stopped early
    ByteRange stopped_chunk;
    Residency stopped_chunk_residency;
    double stopped_chunk_wait_ms{};
    std::vector<double> chunk_wait_ms;  // per completed chunk: madvise() return to target reached
};
WaitResult wait_until_resident(const std::function<Residency()>& measure, double target_percent,
                               double timeout_s, double plateau_s,
                               std::chrono::milliseconds poll_interval);

// Prefetch [0, file_bytes) in chunks of chunk_bytes, using only madvise(MADV_WILLNEED):
// for each chunk, `advise` it (returns 0 or an errno), then poll `measure` on that chunk
// until it is at least target_percent resident. Stops at the first of:
//   "target"             every chunk reached target_percent
//   "plateau"            a chunk's resident pages did not grow for plateau_s
//   "timeout"            timeout_s passed (for the whole loop)
//   "madvise_failed"     `advise` returned an error
//   "measurement_failed" `measure` failed
// Then `residency` is `measure` over the whole file and bytes_resident_final its bytes.
WaitResult prefetch_in_chunks(std::uint64_t file_bytes, std::uint64_t chunk_bytes, std::uint64_t page_size,
                              const std::function<int(const ByteRange&)>& advise,
                              const std::function<Residency(const ByteRange&)>& measure,
                              double target_percent, double timeout_s, double plateau_s,
                              std::chrono::milliseconds poll_interval);

struct CallRecord {
    std::uint64_t offset{};
    std::uint64_t length{};
    bool attempted{false};  // false: refused by MemoryPinner before calling the syscall
    bool success{false};
    int error_number{};
    std::string message;
};

struct StrategyConfig {
    std::string model_path;
    Strategy strategy{Strategy::Baseline};
    std::optional<double> fraction;  // pin only, (0, 1]
    std::string cgroup_path;         // empty: stay in the current cgroup
    std::optional<double> wait_resident_percent;  // prefetch only, (0, 100]
    double wait_timeout_s{120.0};
    double wait_plateau_s{3.0};
    // 8 MiB: on this machine one madvise(WILLNEED) brought in at most 8 MiB (the device's
    // read_ahead_kb), so larger chunks plateaued at 8 MiB (docs/EXPERIMENT_PLAN.md).
    std::uint64_t prefetch_chunk_bytes{8ULL * 1024ULL * 1024ULL};
    PinBudget budget;                // memory_max and rlimit are filled in by run()
};

struct StrategyReport {
    std::string strategy;
    std::optional<double> fraction;
    std::string model_path;
    std::uint64_t file_size_bytes{};
    std::uint64_t page_size{};
    std::string cgroup_path;
    bool cgroup_joined{false};
    std::optional<std::uint64_t> rlimit_memlock_soft_bytes;

    std::optional<CallRecord> madvise;
    std::optional<std::uint64_t> prefetch_chunk_bytes;  // prefetch only
    std::optional<WaitResult> residency_wait;

    bool has_selection{false};
    std::uint32_t gguf_version{};
    std::uint64_t gguf_tensors{};
    std::uint64_t weight_set_bytes{};
    std::vector<ByteRange> selected_ranges;
    std::optional<PinBudget> budget;
    std::optional<CapDecision> cap;
    std::uint64_t bytes_requested{};
    std::uint64_t bytes_locked{};
    std::vector<CallRecord> mlock_calls;

    Residency before_file, after_file, before_selected, after_selected;
    double setup_wall_ms{};
    long setup_minor_faults{};
    long setup_major_faults{};
    std::vector<std::pair<std::string, long long>> helper_status_kb;

    bool held{false};
    std::string released_by;
    double held_ms{};
    Residency final_file, final_selected;
    std::vector<CallRecord> munlock_calls;

    std::vector<std::string> notes;
    int exit_code{kExitOk};
    std::string error;
};

std::string to_json(const StrategyReport& report);

// One helper run: run() applies the strategy and fills the report; release()
// takes final residency, munlocks and unmaps (it also happens on destruction).
class StrategySession {
public:
    explicit StrategySession(StrategyConfig config);
    int run();
    void release(const std::string& released_by, double held_ms);
    const StrategyReport& report() const noexcept { return report_; }

private:
    int finish(int exit_code, std::string error);

    StrategyConfig config_;
    prefetcher::FileMapping mapping_;
    std::vector<ByteRange> locked_;
    StrategyReport report_;
};

}  // namespace neuroswap::strategy

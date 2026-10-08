#include "neuroswap/strategy/strategy.hpp"

#include "neuroswap/pinner/memory_pinner.hpp"
#include "neuroswap/profiler/measurement.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>
#include <sys/resource.h>
#include <unistd.h>

namespace neuroswap::strategy {
namespace {
const char* const kNoKvNote =
    "This helper acts only on the model file's page-cache pages; it does not observe or "
    "control llama.cpp's KV cache or other anonymous memory.";
const char* const kResidencyNote =
    "Residency is the share of pages mincore() reported resident at that moment; it is "
    "not a performance measure.";
const char* const kMadviseNote =
    "madvise(MADV_WILLNEED) is advisory: the kernel may ignore it, read ahead "
    "asynchronously, or evict the pages again. after_file is measured immediately after "
    "madvise() returned.";
const char* const kMlockNote =
    "mlock() keeps the selected pages resident only while this helper holds them; it is "
    "bounded by RLIMIT_MEMLOCK and the cgroup's memory.max (locked pages are charged to it).";

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::string with_mib(std::uint64_t bytes) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%llu bytes (%.1f MiB)",
                  static_cast<unsigned long long>(bytes), static_cast<double>(bytes) / kMiB);
    return buffer;
}

std::string bound_description(const std::string& bound) {
    if (bound == "room(L)") return "room(L) = memory.max - peak_anon - other_rss - margin";
    if (bound == "memlock_policy") return "the RLIMIT_MEMLOCK policy cap (measured root limit, not raised)";
    return "this process's RLIMIT_MEMLOCK soft limit";
}

std::optional<std::uint64_t> read_memory_max(const std::string& cgroup) {
    const std::string path = cgroup + "/memory.max";
    std::ifstream file(path);
    std::string value;
    if (!(file >> value)) throw std::runtime_error("cannot read " + path);
    if (value == "max") return std::nullopt;
    std::size_t used = 0;
    const auto bytes = std::stoull(value, &used);
    if (used != value.size()) throw std::runtime_error("unexpected content in " + path + ": " + value);
    return bytes;
}

void join_cgroup(const std::string& cgroup) {
    const std::string path = cgroup + "/cgroup.procs";
    const int fd = open(path.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("open(" + path + ") failed: " + std::strerror(errno));
    const std::string pid = std::to_string(getpid());
    const ssize_t written = write(fd, pid.data(), pid.size());
    const int error_number = errno;
    close(fd);
    if (written != static_cast<ssize_t>(pid.size())) {
        throw std::runtime_error("write(" + path + ") failed: " + std::strerror(error_number));
    }
}

std::optional<std::uint64_t> memlock_soft_limit() {
    rlimit limit{};
    if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0) {
        throw std::runtime_error(std::string("getrlimit(RLIMIT_MEMLOCK) failed: ") + std::strerror(errno));
    }
    if (limit.rlim_cur == RLIM_INFINITY) return std::nullopt;
    return static_cast<std::uint64_t>(limit.rlim_cur);
}

std::vector<std::pair<std::string, long long>> read_self_status() {
    static const char* const keys[] = {"VmRSS", "VmHWM", "VmLck", "RssAnon", "RssFile"};
    std::vector<std::pair<std::string, long long>> values;
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const auto key = line.substr(0, colon);
        for (const char* wanted : keys) {
            if (key == wanted) values.emplace_back(key, std::stoll(line.substr(colon + 1)));
        }
    }
    return values;
}

// Reads one byte per page so every selected page is faulted in before mlock().
std::uint64_t touch_pages(const void* base, const std::vector<ByteRange>& ranges, std::uint64_t page_size) {
    const auto* bytes = static_cast<const volatile unsigned char*>(base);
    std::uint64_t checksum = 0;
    for (const auto& range : ranges) {
        for (auto offset = range.offset; offset < range.offset + range.length; offset += page_size) {
            checksum += bytes[offset];
        }
    }
    return checksum;
}

bool syscall_attempted(bool success, const std::string& message, const char* prefix) {
    return success || message.rfind(prefix, 0) == 0;
}

// ---- JSON output ----

std::string quote(const std::string& text) {
    std::string out = "\"";
    for (const unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof buffer, "\\u%04x", c);
                    out += buffer;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}

std::string number(double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.*f", precision, value);
    return buffer;
}

std::string optional_number(const std::optional<std::uint64_t>& value) {
    return value ? std::to_string(*value) : "null";
}

std::string boolean(bool value) { return value ? "true" : "false"; }

using Fields = std::vector<std::pair<std::string, std::string>>;

std::string object(const Fields& fields, int level) {
    if (fields.empty()) return "{}";
    const std::string indent(static_cast<std::size_t>(2 * (level + 1)), ' ');
    std::string out = "{\n";
    for (std::size_t index = 0; index < fields.size(); ++index) {
        out += indent + quote(fields[index].first) + ": " + fields[index].second;
        out += index + 1 < fields.size() ? ",\n" : "\n";
    }
    return out + std::string(static_cast<std::size_t>(2 * level), ' ') + "}";
}

std::string array(const std::vector<std::string>& items, int level) {
    if (items.empty()) return "[]";
    const std::string indent(static_cast<std::size_t>(2 * (level + 1)), ' ');
    std::string out = "[\n";
    for (std::size_t index = 0; index < items.size(); ++index) {
        out += indent + items[index] + (index + 1 < items.size() ? ",\n" : "\n");
    }
    return out + std::string(static_cast<std::size_t>(2 * level), ' ') + "]";
}

std::string residency_json(const Residency& residency, int level) {
    if (!residency.measured && residency.error_number == 0) return "null";
    return object({{"measured", boolean(residency.measured)},
                   {"pages_total", std::to_string(residency.pages_total)},
                   {"pages_resident", std::to_string(residency.pages_resident)},
                   {"percent_resident", number(residency.percent(), 3)},
                   {"errno", std::to_string(residency.error_number)}},
                  level);
}

std::string call_json(const CallRecord& call, int level) {
    return object({{"offset", std::to_string(call.offset)},
                   {"length", std::to_string(call.length)},
                   {"attempted", boolean(call.attempted)},
                   {"return_value", call.attempted ? (call.success ? "0" : "-1") : "null"},
                   {"errno", std::to_string(call.error_number)},
                   {"message", quote(call.message)}},
                  level);
}

std::string calls_json(const std::vector<CallRecord>& calls, int level) {
    std::vector<std::string> items;
    for (const auto& call : calls) items.push_back(call_json(call, level + 1));
    return array(items, level);
}
}  // namespace

Strategy parse_strategy(const std::string& value) {
    if (value == "baseline") return Strategy::Baseline;
    if (value == "prefetch") return Strategy::Prefetch;
    if (value == "pin") return Strategy::Pin;
    throw std::invalid_argument("--strategy must be baseline, prefetch or pin");
}

const char* strategy_name(Strategy strategy) {
    switch (strategy) {
        case Strategy::Baseline: return "baseline";
        case Strategy::Prefetch: return "prefetch";
        case Strategy::Pin: return "pin";
    }
    return "unknown";
}

CapDecision check_pin_cap(const PinBudget& budget, std::uint64_t requested_bytes) {
    CapDecision decision;
    decision.requested_bytes = requested_bytes;
    if (budget.memory_max_bytes) {
        const auto reserved = budget.peak_anon_bytes + budget.other_rss_bytes + budget.margin_bytes;
        decision.room_bytes = *budget.memory_max_bytes > reserved ? *budget.memory_max_bytes - reserved : 0;
    }
    decision.pin_cap_bytes = budget.memlock_policy_bytes;
    decision.binding_bound = "memlock_policy";
    if (decision.room_bytes && *decision.room_bytes < decision.pin_cap_bytes) {
        decision.pin_cap_bytes = *decision.room_bytes;
        decision.binding_bound = "room(L)";
    }
    if (budget.memlock_rlimit_bytes && *budget.memlock_rlimit_bytes < decision.pin_cap_bytes) {
        decision.pin_cap_bytes = *budget.memlock_rlimit_bytes;
        decision.binding_bound = "rlimit_memlock";
    }
    decision.allowed = requested_bytes <= decision.pin_cap_bytes;
    decision.message = decision.allowed
        ? "requested " + with_mib(requested_bytes) + " is within pin_cap " +
              with_mib(decision.pin_cap_bytes) + " (binding bound: " + decision.binding_bound + ")"
        : "pin refused before any mlock(): requested " + with_mib(requested_bytes) + " exceeds " +
              bound_description(decision.binding_bound) + " of " + with_mib(decision.pin_cap_bytes);
    return decision;
}

double Residency::percent() const {
    return pages_total == 0 ? 0.0 : 100.0 * static_cast<double>(pages_resident) / static_cast<double>(pages_total);
}

Residency measure_residency(const void* base, const std::vector<ByteRange>& ranges, std::uint64_t page_size) {
    Residency residency;
    std::vector<unsigned char> pages;
    auto* bytes = const_cast<unsigned char*>(static_cast<const unsigned char*>(base));
    for (const auto& range : ranges) {
        if (range.length == 0) continue;
        const auto begin = range.offset / page_size * page_size;
        const auto end = align_up(range.offset + range.length, page_size);
        pages.assign((end - begin) / page_size, 0);
        if (mincore(bytes + begin, end - begin, pages.data()) != 0) {
            residency.error_number = errno;
            residency.measured = false;
            return residency;
        }
        residency.pages_total += pages.size();
        for (const auto page : pages) residency.pages_resident += page & 1U;
    }
    residency.measured = true;
    return residency;
}

WaitResult wait_until_resident(const std::function<Residency()>& measure, double target_percent,
                               double timeout_s, double plateau_s,
                               std::chrono::milliseconds poll_interval) {
    using clock = std::chrono::steady_clock;
    WaitResult result;
    result.target_percent = target_percent;
    result.timeout_s = timeout_s;
    result.plateau_s = plateau_s;
    const auto started = clock::now();
    const auto deadline = started + std::chrono::duration<double>(timeout_s);
    const auto plateau = std::chrono::duration<double>(plateau_s);
    std::uint64_t most_resident = 0;
    auto last_growth = started;
    while (true) {
        result.residency = measure();
        ++result.polls;
        const auto now = clock::now();
        if (!result.residency.measured) { result.stopped_reason = "measurement_failed"; break; }
        if (result.residency.percent() >= target_percent) { result.stopped_reason = "target"; break; }
        if (result.polls == 1 || result.residency.pages_resident > most_resident) {
            most_resident = result.residency.pages_resident;
            last_growth = now;
        }
        if (now - last_growth >= plateau) { result.stopped_reason = "plateau"; break; }
        if (now >= deadline) { result.stopped_reason = "timeout"; break; }
        std::this_thread::sleep_for(poll_interval);
    }
    result.timed_out = result.stopped_reason == "timeout";
    result.wait_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}

StrategySession::StrategySession(StrategyConfig config) : config_(std::move(config)) {}

int StrategySession::finish(int exit_code, std::string error) {
    report_.exit_code = exit_code;
    report_.error = std::move(error);
    report_.helper_status_kb = read_self_status();
    return exit_code;
}

int StrategySession::run() {
    const profiler::MeasurementScope setup;
    auto done = [&](int exit_code, std::string error) {
        const auto measurement = setup.finish();
        report_.setup_wall_ms = measurement.wall_clock_ms;
        report_.setup_minor_faults = measurement.minor_page_faults;
        report_.setup_major_faults = measurement.major_page_faults;
        return finish(exit_code, std::move(error));
    };

    report_.strategy = strategy_name(config_.strategy);
    report_.fraction = config_.fraction;
    report_.model_path = config_.model_path;
    report_.cgroup_path = config_.cgroup_path;
    report_.page_size = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    report_.notes = {kNoKvNote, kResidencyNote};
    const auto page = report_.page_size;

    try {
        // Join the cgroup before any page of the model is touched, so every page
        // this helper faults in is charged to that cgroup.
        if (!config_.cgroup_path.empty()) {
            config_.budget.memory_max_bytes = read_memory_max(config_.cgroup_path);
            join_cgroup(config_.cgroup_path);
            report_.cgroup_joined = true;
        }
        config_.budget.memlock_rlimit_bytes = memlock_soft_limit();
        report_.rlimit_memlock_soft_bytes = config_.budget.memlock_rlimit_bytes;
    } catch (const std::exception& error) {
        return done(kExitError, error.what());
    }

    const auto mapped = mapping_.open_and_map_readonly(config_.model_path, prefetcher::MapSharing::Shared);
    if (!mapped.success) return done(kExitError, mapped.message);
    report_.file_size_bytes = mapping_.size_bytes();
    const std::vector<ByteRange> whole_file{{0, align_up(report_.file_size_bytes, page)}};
    report_.before_file = measure_residency(mapping_.data(), whole_file, page);

    if (config_.strategy == Strategy::Baseline) {
        report_.notes.push_back("baseline: no madvise() and no mlock(); residency is only observed.");
        report_.after_file = measure_residency(mapping_.data(), whole_file, page);
        return done(kExitOk, "");
    }

    if (config_.strategy == Strategy::Prefetch) {
        report_.notes.push_back(kMadviseNote);
        const auto advice = mapping_.advise_willneed();
        report_.madvise = CallRecord{0, report_.file_size_bytes,
                                     syscall_attempted(advice.success, advice.message, "madvise(MADV_WILLNEED) failed"),
                                     advice.success, advice.error_number, advice.message};
        report_.after_file = measure_residency(mapping_.data(), whole_file, page);
        if (!advice.success) return done(kExitError, advice.message);
        if (config_.wait_resident_percent) {
            report_.notes.push_back(
                "residency_wait polls mincore() every 10 ms after madvise() and stops when the target "
                "share of the file is resident, when residency has not grown for plateau_s (e.g. the "
                "file does not fit under memory.max), or at the timeout; the reason is recorded, not retried.");
            report_.residency_wait = wait_until_resident(
                [&] { return measure_residency(mapping_.data(), whole_file, page); },
                *config_.wait_resident_percent, config_.wait_timeout_s, config_.wait_plateau_s,
                std::chrono::milliseconds(10));
            if (!report_.residency_wait->residency.measured) {
                return done(kExitError, "mincore() failed while waiting for residency: " +
                                            std::string(std::strerror(report_.residency_wait->residency.error_number)));
            }
        }
        return done(kExitOk, "");
    }

    // Pin.
    report_.notes.push_back(kMlockNote);
    const auto layout = parse_gguf_layout(mapping_.data(), mapping_.size_bytes());
    if (!layout.ok) {
        return done(kExitError, "GGUF parse failed: " + layout.error + "; no fallback range is used");
    }
    report_.has_selection = true;
    report_.gguf_version = layout.version;
    report_.gguf_tensors = layout.tensors.size();
    const auto weight_set = per_token_weight_ranges(layout, report_.file_size_bytes, page);
    report_.weight_set_bytes = total_bytes(weight_set);
    const auto fraction = config_.fraction.value_or(0.0);
    const auto wanted = static_cast<std::uint64_t>(fraction * static_cast<double>(report_.weight_set_bytes));
    report_.bytes_requested = std::min(wanted / page * page, report_.weight_set_bytes);
    report_.budget = config_.budget;
    if (report_.bytes_requested == 0) {
        return done(kExitError, "fraction selects zero pages of the weight set");
    }

    report_.cap = check_pin_cap(config_.budget, report_.bytes_requested);
    if (!report_.cap->allowed) return done(kExitCapRefused, report_.cap->message);

    report_.selected_ranges = select_prefix(weight_set, report_.bytes_requested);
    report_.before_selected = measure_residency(mapping_.data(), report_.selected_ranges, page);
    touch_pages(mapping_.data(), report_.selected_ranges, page);

    const pinner::MemoryPinner pinner;
    auto* base = const_cast<unsigned char*>(static_cast<const unsigned char*>(mapping_.data()));
    std::string failure;
    for (const auto& range : report_.selected_ranges) {
        const auto lock = pinner.lock(base + range.offset, range.length);
        report_.mlock_calls.push_back({range.offset, range.length,
                                       syscall_attempted(lock.success, lock.message, "mlock() failed"),
                                       lock.success, lock.error_number, lock.message});
        if (!lock.success) {
            failure = lock.message;
            break;
        }
        locked_.push_back(range);
    }
    if (!failure.empty()) {
        // No partial pin and no fallback to baseline: undo what was locked.
        for (const auto& range : locked_) {
            const auto unlock = pinner.unlock(base + range.offset, range.length);
            report_.munlock_calls.push_back({range.offset, range.length,
                                             syscall_attempted(unlock.success, unlock.message, "munlock() failed"),
                                             unlock.success, unlock.error_number, unlock.message});
        }
        locked_.clear();
        report_.after_selected = measure_residency(mapping_.data(), report_.selected_ranges, page);
        return done(kExitError, "mlock failed: " + failure + "; nothing left locked, no fallback to baseline");
    }
    report_.bytes_locked = total_bytes(locked_);
    report_.after_selected = measure_residency(mapping_.data(), report_.selected_ranges, page);
    report_.after_file = measure_residency(mapping_.data(), whole_file, page);
    return done(kExitOk, "");
}

void StrategySession::release(const std::string& released_by, double held_ms) {
    report_.held = true;
    report_.released_by = released_by;
    report_.held_ms = held_ms;
    if (mapping_.data() == nullptr) return;
    const auto page = report_.page_size;
    report_.final_file = measure_residency(mapping_.data(), {{0, align_up(report_.file_size_bytes, page)}}, page);
    if (report_.has_selection && !report_.selected_ranges.empty()) {
        report_.final_selected = measure_residency(mapping_.data(), report_.selected_ranges, page);
    }
    const pinner::MemoryPinner pinner;
    auto* base = const_cast<unsigned char*>(static_cast<const unsigned char*>(mapping_.data()));
    for (const auto& range : locked_) {
        const auto unlock = pinner.unlock(base + range.offset, range.length);
        report_.munlock_calls.push_back({range.offset, range.length,
                                         syscall_attempted(unlock.success, unlock.message, "munlock() failed"),
                                         unlock.success, unlock.error_number, unlock.message});
    }
    locked_.clear();
    mapping_.close();
}

std::string to_json(const StrategyReport& report) {
    Fields budget_fields;
    if (report.budget) {
        const auto& budget = *report.budget;
        budget_fields = {{"memory_max_bytes", optional_number(budget.memory_max_bytes)},
                         {"peak_anon_bytes", std::to_string(budget.peak_anon_bytes)},
                         {"other_rss_bytes", std::to_string(budget.other_rss_bytes)},
                         {"margin_bytes", std::to_string(budget.margin_bytes)},
                         {"memlock_policy_bytes", std::to_string(budget.memlock_policy_bytes)},
                         {"rlimit_memlock_bytes", optional_number(budget.memlock_rlimit_bytes)}};
        if (report.cap) {
            budget_fields.push_back({"room_bytes", optional_number(report.cap->room_bytes)});
            budget_fields.push_back({"pin_cap_bytes", std::to_string(report.cap->pin_cap_bytes)});
            budget_fields.push_back({"binding_bound", quote(report.cap->binding_bound)});
            budget_fields.push_back({"allowed", boolean(report.cap->allowed)});
            budget_fields.push_back({"message", quote(report.cap->message)});
        }
    }

    std::string selection = "null";
    if (report.has_selection) {
        std::vector<std::string> ranges;
        for (const auto& range : report.selected_ranges) {
            ranges.push_back(object({{"offset", std::to_string(range.offset)},
                                     {"length", std::to_string(range.length)}}, 3));
        }
        selection = object({{"method", quote("GGUF tensor table: all tensors except token_embd.weight, "
                                             "page-aligned, merged, first bytes_requested in file-offset order")},
                            {"gguf_version", std::to_string(report.gguf_version)},
                            {"gguf_tensors", std::to_string(report.gguf_tensors)},
                            {"excluded_tensor", quote(kEmbeddingTensor)},
                            {"weight_set_bytes", std::to_string(report.weight_set_bytes)},
                            {"ranges", array(ranges, 2)}},
                           1);
    }

    Fields status;
    for (const auto& [key, value] : report.helper_status_kb) status.push_back({key, std::to_string(value)});

    std::string hold = "null";
    if (report.held) {
        hold = object({{"released_by", quote(report.released_by)},
                       {"held_ms", number(report.held_ms, 3)},
                       {"final_file", residency_json(report.final_file, 2)},
                       {"final_selected", residency_json(report.final_selected, 2)},
                       {"munlock_calls", calls_json(report.munlock_calls, 2)}},
                      1);
    } else if (!report.munlock_calls.empty()) {
        hold = object({{"munlock_calls", calls_json(report.munlock_calls, 2)}}, 1);
    }

    std::vector<std::string> notes;
    for (const auto& note : report.notes) notes.push_back(quote(note));

    return object({{"source", quote("neuroswap_measured")},
                   {"strategy", quote(report.strategy)},
                   {"fraction", report.fraction ? number(*report.fraction, 6) : "null"},
                   {"model_path", quote(report.model_path)},
                   {"file_size_bytes", std::to_string(report.file_size_bytes)},
                   {"page_size", std::to_string(report.page_size)},
                   {"cgroup", report.cgroup_path.empty() ? "null"
                                  : object({{"path", quote(report.cgroup_path)},
                                            {"joined", boolean(report.cgroup_joined)}}, 1)},
                   {"rlimit_memlock_soft_bytes", optional_number(report.rlimit_memlock_soft_bytes)},
                   {"exit_code", std::to_string(report.exit_code)},
                   {"error", report.error.empty() ? "null" : quote(report.error)},
                   {"setup_wall_ms", number(report.setup_wall_ms, 3)},
                   {"setup_minor_faults", std::to_string(report.setup_minor_faults)},
                   {"setup_major_faults", std::to_string(report.setup_major_faults)},
                   {"madvise", report.madvise ? call_json(*report.madvise, 1) : "null"},
                   {"residency_wait", report.residency_wait
                        ? object({{"target_percent", number(report.residency_wait->target_percent, 3)},
                                  {"timeout_s", number(report.residency_wait->timeout_s, 3)},
                                  {"plateau_s", number(report.residency_wait->plateau_s, 3)},
                                  {"stopped_reason", quote(report.residency_wait->stopped_reason)},
                                  {"wait_ms", number(report.residency_wait->wait_ms, 3)},
                                  {"polls", std::to_string(report.residency_wait->polls)},
                                  {"timed_out", boolean(report.residency_wait->timed_out)},
                                  {"residency", residency_json(report.residency_wait->residency, 2)}},
                                 1)
                        : "null"},
                   {"selection", selection},
                   {"pin_budget", report.budget ? object(budget_fields, 1) : "null"},
                   {"bytes_requested", std::to_string(report.bytes_requested)},
                   {"bytes_locked", std::to_string(report.bytes_locked)},
                   {"mlock_calls", calls_json(report.mlock_calls, 1)},
                   {"residency", object({{"before_file", residency_json(report.before_file, 2)},
                                         {"after_file", residency_json(report.after_file, 2)},
                                         {"before_selected", residency_json(report.before_selected, 2)},
                                         {"after_selected", residency_json(report.after_selected, 2)}},
                                        1)},
                   {"helper_status_kb", object(status, 1)},
                   {"hold", hold},
                   {"notes", array(notes, 1)}},
                  0) + "\n";
}

}  // namespace neuroswap::strategy

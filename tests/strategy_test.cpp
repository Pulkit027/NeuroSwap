#include "neuroswap/strategy/strategy.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

namespace {
using namespace neuroswap::strategy;

void put_u32(std::string& out, std::uint32_t value) { out.append(reinterpret_cast<const char*>(&value), 4); }
void put_u64(std::string& out, std::uint64_t value) { out.append(reinterpret_cast<const char*>(&value), 8); }
void put_str(std::string& out, const std::string& text) { put_u64(out, text.size()); out += text; }

// Tiny GGUF v3 file: alignment = page size, three F32 tensors laid out as
// output.weight (2 pages), token_embd.weight (1 page), blk.0.ffn.weight (3 pages).
std::string write_tiny_gguf(const std::string& path, std::uint64_t page) {
    std::string bytes = "GGUF";
    put_u32(bytes, 3);
    put_u64(bytes, 3);  // tensors
    put_u64(bytes, 1);  // metadata entries
    put_str(bytes, "general.alignment");
    put_u32(bytes, 4);  // uint32
    put_u32(bytes, static_cast<std::uint32_t>(page));
    const struct { const char* name; std::uint64_t pages; std::uint64_t offset_pages; } tensors[] = {
        {"output.weight", 2, 0}, {"token_embd.weight", 1, 2}, {"blk.0.ffn.weight", 3, 3}};
    for (const auto& tensor : tensors) {
        put_str(bytes, tensor.name);
        put_u32(bytes, 1);
        put_u64(bytes, tensor.pages * page / 4);
        put_u32(bytes, 0);  // F32
        put_u64(bytes, tensor.offset_pages * page);
    }
    bytes.resize(page, '\0');  // header padded to the first page boundary
    for (std::uint64_t index = 0; index < 6 * page; ++index) bytes += static_cast<char>(index & 0x7f);
    std::ofstream(path, std::ios::binary) << bytes;
    return path;
}

StrategyConfig config_for(const std::string& path, Strategy strategy) {
    StrategyConfig config;
    config.model_path = path;
    config.strategy = strategy;
    return config;
}
}  // namespace

int main() {
    const auto page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    const std::string path = "/tmp/neuroswap_strategy_test_" + std::to_string(getpid()) + ".gguf";
    write_tiny_gguf(path, page);

    // GGUF layout and per-token weight set.
    {
        std::ifstream file(path, std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        const auto layout = parse_gguf_layout(bytes.data(), bytes.size());
        assert(layout.ok && layout.version == 3 && layout.alignment == page);
        assert(layout.data_start == page && layout.tensors.size() == 3);
        assert(layout.tensors[1].name == "token_embd.weight" && layout.tensors[1].size == page);
        const auto set = per_token_weight_ranges(layout, bytes.size(), page);
        assert(set.size() == 2 && total_bytes(set) == 5 * page);
        assert(set[0].offset == page && set[0].length == 2 * page);
        assert(set[1].offset == 4 * page && set[1].length == 3 * page);
        const auto prefix = select_prefix(set, 3 * page);
        assert(prefix.size() == 2 && prefix[1].offset == 4 * page && prefix[1].length == page);

        const auto bad = parse_gguf_layout("NOPE", 4);
        assert(!bad.ok && !bad.error.empty());
    }

    // pin_cap: each bound can bind, and refusals name the bound and both values.
    {
        PinBudget budget;
        budget.memory_max_bytes = 300 * kMiB;  // room = 300 - 177.9 - 13 - 64 MiB
        const auto room = check_pin_cap(budget, 100 * kMiB);
        assert(!room.allowed && room.binding_bound == "room(L)" && room.room_bytes);
        assert(room.message.find(std::to_string(100 * kMiB)) != std::string::npos);
        assert(room.message.find(std::to_string(room.pin_cap_bytes)) != std::string::npos);

        PinBudget unlimited;  // no cgroup: the 740.8 MiB policy cap binds
        const auto policy = check_pin_cap(unlimited, 800 * kMiB);
        assert(!policy.allowed && policy.binding_bound == "memlock_policy" && !policy.room_bytes);
        assert(check_pin_cap(unlimited, 700 * kMiB).allowed);

        unlimited.memlock_rlimit_bytes = 64 * kMiB;
        const auto rlimit = check_pin_cap(unlimited, 100 * kMiB);
        assert(!rlimit.allowed && rlimit.binding_bound == "rlimit_memlock");

        PinBudget tiny;
        tiny.memory_max_bytes = 10 * kMiB;  // less than the reserved memory: room is 0
        assert(*check_pin_cap(tiny, page).room_bytes == 0 && !check_pin_cap(tiny, page).allowed);
    }

    // baseline: nothing applied, residency observed.
    {
        StrategySession session(config_for(path, Strategy::Baseline));
        assert(session.run() == kExitOk);
        const auto& report = session.report();
        assert(report.bytes_locked == 0 && report.mlock_calls.empty() && !report.madvise);
        assert(report.before_file.measured && report.before_file.pages_total == 7);
        const auto json = to_json(report);
        assert(json.find("\"source\": \"neuroswap_measured\"") != std::string::npos);
        assert(json.find("\"strategy\": \"baseline\"") != std::string::npos);
    }

    // prefetch: madvise over the whole file, residency before and after.
    {
        StrategySession session(config_for(path, Strategy::Prefetch));
        const int code = session.run();
        const auto& report = session.report();
        assert(report.madvise && report.madvise->attempted);
        assert(code == (report.madvise->success ? kExitOk : kExitError));
        assert(report.after_file.measured && report.after_file.pages_total == 7);
        assert(report.after_file.pages_resident <= report.after_file.pages_total);
        assert(to_json(report).find("advisory") != std::string::npos);
    }

    // residency wait, deterministic fake measurements for each stop reason.
    {
        auto fake = [](std::uint64_t resident) {
            Residency r;
            r.measured = true;
            r.pages_total = 1000;
            r.pages_resident = resident;
            return r;
        };

        // target: reached on the 3rd poll.
        const std::uint64_t sequence[] = {10, 500, 1000};
        std::size_t call = 0;
        const auto target = wait_until_resident([&] { return fake(sequence[call++]); },
                                                99.0, 5.0, 3.0, std::chrono::milliseconds(1));
        assert(target.stopped_reason == "target" && !target.timed_out && target.polls == 3);
        assert(target.residency.pages_resident == 1000);

        // plateau: stuck at 50% (e.g. file does not fit), stops after plateau_s, long before the timeout.
        const auto plateau = wait_until_resident([&] { return fake(500); },
                                                 99.0, 5.0, 0.05, std::chrono::milliseconds(5));
        assert(plateau.stopped_reason == "plateau" && !plateau.timed_out);
        assert(plateau.wait_ms >= 50.0 && plateau.wait_ms < 5000.0 && plateau.polls >= 2);

        // plateau clock restarts on growth: grows for 5 polls, then stalls.
        std::uint64_t grow = 0;
        const auto late = wait_until_resident([&] { grow = std::min<std::uint64_t>(grow + 100, 500); return fake(grow); },
                                              99.0, 5.0, 0.05, std::chrono::milliseconds(5));
        assert(late.stopped_reason == "plateau" && late.residency.pages_resident == 500);

        // timeout: keeps growing by one page per poll, never reaches the target or a plateau.
        std::uint64_t pages = 0;
        const auto timeout = wait_until_resident([&] { return fake(++pages); },
                                                 99.0, 0.05, 3.0, std::chrono::milliseconds(5));
        assert(timeout.stopped_reason == "timeout" && timeout.timed_out);
        assert(timeout.wait_ms >= 50.0 && timeout.residency.percent() < 99.0);

        // measurement failure: stops at once.
        const auto failed = wait_until_resident([] { Residency r; r.error_number = 12; return r; },
                                                99.0, 5.0, 3.0, std::chrono::milliseconds(1));
        assert(failed.stopped_reason == "measurement_failed" && failed.polls == 1 && !failed.residency.measured);
    }

    // residency wait on the real temp file through the prefetch session.
    {
        auto config = config_for(path, Strategy::Prefetch);
        config.wait_resident_percent = 100.0;
        config.wait_timeout_s = 5.0;
        StrategySession session(config);
        assert(session.run() == kExitOk);
        const auto& wait = session.report().residency_wait;
        assert(wait && wait->residency.measured && wait->residency.pages_total == 7 && wait->polls >= 1);
        assert(wait->stopped_reason == "target" || wait->stopped_reason == "plateau" ||
               wait->stopped_reason == "timeout");
        if (wait->stopped_reason == "target") assert(wait->residency.pages_resident == 7);
        else assert(wait->residency.percent() < 100.0);
        const auto json = to_json(session.report());
        assert(json.find("\"residency_wait\": {") != std::string::npos);
        assert(json.find("\"stopped_reason\": ") != std::string::npos);
    }

    // pin within the cap: 60% of 5 pages = 3 pages.
    {
        auto config = config_for(path, Strategy::Pin);
        config.fraction = 0.6;
        StrategySession session(config);
        const int code = session.run();
        const auto& report = session.report();
        assert(report.weight_set_bytes == 5 * page && report.bytes_requested == 3 * page);
        if (code == kExitOk) {
            assert(report.bytes_locked == 3 * page && report.mlock_calls.size() == 2);
            assert(report.after_selected.measured && report.after_selected.pages_resident == 3);
            session.release("test", 0.0);
            assert(session.report().munlock_calls.size() == 2);
            for (const auto& call : session.report().munlock_calls) assert(call.success);
        } else if (code == kExitCapRefused) {
            // Only possible if this process's RLIMIT_MEMLOCK is below 3 pages.
            assert(report.cap && report.cap->binding_bound == "rlimit_memlock" && report.mlock_calls.empty());
        } else {
            assert(code == kExitError && !report.mlock_calls.empty() && report.bytes_locked == 0);
        }
    }

    // pin above a small cap: refused before any mlock, exit code 3, no fallback.
    {
        auto config = config_for(path, Strategy::Pin);
        config.fraction = 1.0;
        config.budget.memlock_policy_bytes = page;
        StrategySession session(config);
        assert(session.run() == kExitCapRefused);
        const auto& report = session.report();
        assert(report.cap && !report.cap->allowed && report.cap->binding_bound != "room(L)");
        assert(report.mlock_calls.empty() && report.bytes_locked == 0);
        assert(!report.before_selected.measured && report.selected_ranges.empty());
        assert(report.error.find("refused before any mlock()") != std::string::npos);
        assert(report.error.find(std::to_string(5 * page)) != std::string::npos);
        assert(report.error.find(std::to_string(report.cap->pin_cap_bytes)) != std::string::npos);
    }

    // pin on a file that is not GGUF: error, no fallback range.
    {
        const std::string other = path + ".bin";
        std::ofstream(other, std::ios::binary) << std::string(2 * page, 'x');
        auto config = config_for(other, Strategy::Pin);
        config.fraction = 0.5;
        StrategySession session(config);
        assert(session.run() == kExitError);
        assert(session.report().error.find("GGUF parse failed") != std::string::npos);
        assert(session.report().mlock_calls.empty());
        std::remove(other.c_str());
    }

    std::remove(path.c_str());
}

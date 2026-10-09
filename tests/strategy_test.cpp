#include "neuroswap/strategy/strategy.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

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

    // chunked prefetch, deterministic: a fake madvise records the advised ranges, a fake
    // mincore reports per-chunk residency that rises over successive polls.
    {
        constexpr std::uint64_t kPage = 4096, kChunk = 4 * kPage, kFile = 14 * kPage;  // chunks of 4,4,4,2 pages
        struct Fake {
            std::vector<ByteRange> advised;
            std::map<std::uint64_t, std::uint64_t> resident;  // chunk offset -> resident pages
            std::uint64_t stuck_offset = UINT64_MAX;          // this chunk never passes 1 page
            std::uint64_t step = 2;                           // pages gained per poll
            int fail_at_call = -1;
            int advise(const ByteRange& r) {
                if (static_cast<int>(advised.size()) == fail_at_call) return EINVAL;
                advised.push_back(r);
                resident[r.offset] = 0;
                return 0;
            }
            Residency measure(const ByteRange& r) {
                Residency out;
                out.measured = true;
                out.pages_total = (r.length + kPage - 1) / kPage;
                if (r.offset == 0 && r.length == kFile) {  // final whole-file measurement
                    for (const auto& [offset, pages] : resident) out.pages_resident += pages;
                    return out;
                }
                auto& pages = resident[r.offset];
                const auto cap = r.offset == stuck_offset ? 1 : out.pages_total;
                pages = std::min<std::uint64_t>(pages + step, cap);
                out.pages_resident = pages;
                return out;
            }
        };
        auto run = [&](Fake& fake, double timeout_s, double plateau_s) {
            return prefetch_in_chunks(
                kFile, kChunk, kPage, [&](const ByteRange& r) { return fake.advise(r); },
                [&](const ByteRange& r) { return fake.measure(r); }, 99.0, timeout_s, plateau_s,
                std::chrono::milliseconds(1));
        };

        // progression: every chunk reaches the target; ranges cover the file in order.
        Fake all;
        const auto done = run(all, 5.0, 3.0);
        assert(done.stopped_reason == "target" && !done.timed_out && !done.stopped_chunk_index);
        assert(done.chunks_requested == 4 && done.chunks_completed == 4 && done.madvise_calls == 4);
        assert(all.advised.size() == 4 && all.advised[3].offset == 12 * kPage && all.advised[3].length == 2 * kPage);
        assert(done.chunk_wait_ms.size() == 4);
        assert(done.residency.pages_total == 14 && done.residency.pages_resident == 14);
        assert(done.bytes_resident_final == kFile);

        // plateau mid-file: chunk 2 sticks at 1 of 4 pages; chunk 3 is never advised.
        Fake mid;
        mid.stuck_offset = 2 * kChunk;
        const auto plateau = run(mid, 5.0, 0.05);
        assert(plateau.stopped_reason == "plateau" && plateau.stopped_chunk_index == 2u);
        assert(plateau.chunks_completed == 2 && plateau.madvise_calls == 3 && mid.advised.size() == 3);
        assert(plateau.stopped_chunk.offset == 2 * kChunk && plateau.stopped_chunk_residency.pages_resident == 1);
        assert(plateau.stopped_chunk_wait_ms >= 50.0 && plateau.chunk_wait_ms.size() == 2);
        assert(plateau.residency.pages_resident == 9 && plateau.bytes_resident_final == 9 * kPage);

        // timeout: residency grows one page per poll, too slowly for a 50 ms overall limit.
        Fake slow;
        slow.step = 1;
        const auto timeout = prefetch_in_chunks(
            kFile, kChunk, kPage, [&](const ByteRange& r) { return slow.advise(r); },
            [&](const ByteRange& r) { auto out = slow.measure(r); return out; }, 99.0, 0.05, 3.0,
            std::chrono::milliseconds(20));
        assert(timeout.stopped_reason == "timeout" && timeout.timed_out && timeout.stopped_chunk_index);
        assert(timeout.chunks_completed < timeout.chunks_requested && timeout.wait_ms >= 50.0);

        // madvise failure on the 2nd chunk.
        Fake failing;
        failing.fail_at_call = 1;
        const auto failed = run(failing, 5.0, 3.0);
        assert(failed.stopped_reason == "madvise_failed" && failed.madvise_errno == EINVAL);
        assert(failed.stopped_chunk_index == 1u && failed.chunks_completed == 1);
    }

    // default chunk size is 8 MiB and is recorded for prefetch only.
    {
        assert(StrategyConfig{}.prefetch_chunk_bytes == 8 * kMiB);
        StrategySession prefetch(config_for(path, Strategy::Prefetch));
        assert(prefetch.run() == kExitOk);
        assert(prefetch.report().prefetch_chunk_bytes == 8 * kMiB);
        assert(to_json(prefetch.report()).find("\"prefetch_chunk_bytes\": 8388608") != std::string::npos);
        StrategySession baseline(config_for(path, Strategy::Baseline));
        assert(baseline.run() == kExitOk && !baseline.report().prefetch_chunk_bytes);
        assert(to_json(baseline.report()).find("\"prefetch_chunk_bytes\": null") != std::string::npos);
    }

    // chunked prefetch through the session on the temp file: 1-page chunks, 7 chunks.
    {
        auto config = config_for(path, Strategy::Prefetch);
        config.prefetch_chunk_bytes = page;
        config.wait_resident_percent = 99.0;
        config.wait_timeout_s = 5.0;
        StrategySession session(config);
        assert(session.run() == kExitOk);
        const auto& report = session.report();
        assert(report.madvise && report.madvise->success);
        const auto& wait = report.residency_wait;
        assert(wait && wait->chunk_bytes == page && wait->chunks_requested == 7);
        assert(wait->madvise_calls >= 1 && wait->madvise_calls <= 7);
        if (wait->stopped_reason == "target") assert(wait->chunks_completed == 7);
        else assert(wait->stopped_chunk_index && wait->chunks_completed < 7);
        assert(wait->residency.pages_total == 7);
        const auto json = to_json(report);
        assert(json.find("\"chunks_completed\": ") != std::string::npos);
        assert(json.find("\"final_percent_file\": ") != std::string::npos);
        assert(json.find("\"chunk_wait_ms\": ") != std::string::npos);
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

# Phase 3 strategy design: pin and prefetch for the real model

Status: design only, nothing implemented. Numbers marked *measured* come from
`results/raw/llm/smoke.jsonl`, `results/raw/llm/cgroup_test.jsonl` or the
diagnostic snapshots described in section 1. Numbers marked *inferred* are
derived from those and from the GGUF tensor table; they still need confirmation.
None of the numbers here are benchmarks.

**Decision: every Phase 3 experiment, baseline included, runs llama.cpp with
`--no-repack`**, so all weights stay file-backed and every strategy is compared
against the same configuration. `scripts/run_one.py --repack off` (the default)
passes `--no-repack` and records `"repack": "off"` in each record.
Phase 1 results and the existing `smoke.jsonl` / `cgroup_test.jsonl` were taken
with repack on and are not directly comparable to Phase 3 runs.

## 1. Where the model lives in memory

### Peaks from the existing samples (100 ms sampling, MiB)

| run | limit | cold | peak RssFile | peak RssAnon | VmHWM |
|---|---|---|---|---|---|
| smoke wc0000 (25 tok) | none | no | 1076.4 | 710.3 | 1782.4 |
| smoke wc0075 (102 tok) | none | no | 1076.5 | 719.6 | 1791.8 |
| smoke wc0150 (266 tok) | none | no | 1076.5 | 741.7 | 1813.9 |
| smoke wc0225 (1171 tok) | none | no | 1076.5 | 774.4 | 1846.6 |
| cgroup_test wc0225 (1171 tok) | 1200M | yes | 1068.1 | 774.4 | 1192.2 |

In the 1200M run the RssFile peak happens during load; by the last sample
RssFile is 208 MiB while RssAnon stays 774 MiB, and the process read
38,938 MiB from disk (`io_read_bytes`) for a 1,065.6 MiB file (*measured*).

### Diagnostic runs (warm cache, no limit, 9-token prompt, `-lv 4`)

Run once with default settings and once with `--no-repack`; smaps taken 5 s in.

| | default (repack on) | `--no-repack` |
|---|---|---|
| llama.cpp `CPU_Mapped model buffer` | 1059.89 MiB | 1059.89 MiB |
| llama.cpp `CPU_REPACK model buffer` | 596.53 MiB | (none) |
| KV buffer | 56.00 MiB | 56.00 MiB |
| compute buffer (reserved) | 313.77 MiB | 313.77 MiB |
| .gguf mapping Rss (smaps) | 1,085,328 kB | 1,085,328 kB |
| Anonymous (smaps_rollup) | 723,808 kB | 114,732 kB |
| Locked | 0 | 0 |

### User-measured snapshots (one run each, ~15 s into generation)

Single-snapshot observations, one run per setting; not benchmarks.

| | RssFile | RssAnon |
|---|---|---|
| default (repack on) | 1,097,892 kB (1072.2 MiB) | 723,912 kB (706.9 MiB) |
| `--no-repack` | 1,097,904 kB (1072.2 MiB) | 113,052 kB (110.4 MiB) |

These agree with the diagnostic runs above: file-backed RSS is the same in
both settings; repacking adds ~596 MiB of anonymous memory.

Caveat: with repack on, the whole file is still resident (RssFile is the same
in both rows), so some tensors may be computed from their anonymous repacked
copies while their file pages sit unused. An RSS snapshot cannot tell which
copy is actually read during compute. This is one more reason to use
`--no-repack`: then the file mapping is the only copy of the weights.

### Answer

- **The weights are file-backed (an mmap of the .gguf, `load_mode = mmap`),
  but with the default settings llama.cpp also copies 596.53 MiB of them into
  anonymous memory (`CPU_REPACK`).** `REPACK = 1` in `system_info` only says the
  backend supports repacking; the `CPU_REPACK model buffer size` line and the
  ~609 MiB drop in anonymous memory with `--no-repack` show it actually happens.
- Which tensors are repacked (*inferred*): the GGUF has 721.7 MiB of Q4_K tensors,
  of which `token_embd.weight` is 125.19 MiB; 721.7 - 125.2 = 596.5 MiB, equal to
  the REPACK buffer. So all Q4_K matrices except the embedding are repacked; Q6_K
  (337.6 MiB, includes `output.weight` 182.6 MiB), F32 (0.6 MiB) and
  `token_embd` stay mmap-served.
- Consequence with repack on: after load, the 596.5 MiB of Q4_K file pages are
  no longer read (*inferred*); the per-token file-backed working set is about
  338 MiB plus the embedding rows that are looked up. Anonymous memory (repacked
  weights + KV + compute) is about 710-774 MiB and cannot be reclaimed (swap = 0).
- Llama.cpp's own mmap path already prefetches: `posix_fadvise(SEQUENTIAL)`, then
  `MAP_POPULATE` over the whole file (no lazy ranges were used in our runs), plus
  `POSIX_MADV_WILLNEED` (source: `src/llama-mmap.cpp` lines 482-520, commit 43fe9c6).
  That is why warm runs show `majflt = 0` and RssFile jumps to ~1076 MiB at once.

### Still to measure

1. ~~Anonymous peak with `--no-repack` for a long prompt.~~ Measured in
   `docs/CALIBRATION.md`: 182,156 kB (177.9 MiB) for wc0225 (1171 tokens).
2. ~~Root's memlock limits under sudo.~~ Measured: 758,604 KiB soft and hard
   (section 3).
3. Why only ~208 MiB of file pages stayed resident at the end of the 1200M run
   when anon (774) + hot file set (~338) would be ~1112 MiB: record the cgroup's
   `memory.stat` and `memory.peak` at the end of each limited run.

## 2. Review-2 modules and what has to change

| module | what it does | API |
|---|---|---|
| `pinner::MemoryPinner` | `mlock`/`munlock` on any address range; refuses before calling `mlock` if `bytes` > RLIMIT_MEMLOCK soft limit | `LockResult lock(void*, size_t) const`, `UnlockResult unlock(void*, size_t) const` |
| `prefetcher::FileMapping` | opens a file, `mmap(PROT_READ, MAP_PRIVATE)` of the **whole** file, `madvise(MADV_WILLNEED)` over the **whole** mapping | `open_and_map_readonly(path)`, `advise_willneed()`, `close()`, `data()`, `size_bytes()` |
| `prefetcher::advise_drop_file_cache` | `posix_fadvise(DONTNEED)` on a path | free function |
| `controller::Controller` | fixed rules mapping a configured workload type to Baseline/Pinned/Prefetched | `decide(ControllerInput)` |

Today they only act on the synthetic in-process workload (`neuroswap_pinned`,
`neuroswap_prefetch`). To act on the real model from a separate helper:

- **Pin:** `FileMapping` can map the .gguf (MAP_PRIVATE read-only pages are
  the shared page-cache pages, nothing is copied). `MemoryPinner::lock` accepts
  any sub-range (`data() + offset`, needs a `const_cast`). Missing pieces: a
  page-aligned range API and a touch pass. `lock()` refuses above the
  RLIMIT_MEMLOCK soft limit; the helper does not raise the limit (section 3)
  and refuses larger pins itself, before calling `lock()`.
- **Prefetch:** needs `advise_willneed(offset, length)` for ranges; the
  whole-file version stays as is.
- **Verification:** a new residency function wrapping `mincore()` over a range.
- The Review-2 modules and tests are left unchanged; new functions and a new
  executable go next to them, with their own tests.

## 3. Pin memory budget (all runs with `--no-repack`)

### Inputs

| quantity | value | status |
|---|---|---|
| model file / tensor data | 1,065.6 / 1,059.9 MiB | measured (file size, GGUF table) |
| `token_embd.weight` (only looked-up rows are read) | 125.2 MiB | GGUF table |
| mmap-served set read every token (all tensors except `token_embd`) | ~934.7 MiB | inferred |
| RssAnon, `--no-repack`, short prompt | 110.4-112 MiB | measured, single snapshots |
| RssAnon growth from 25 to 1171 prompt tokens | +64.1 MiB | measured with repack on (710.3 -> 774.4) |
| RssAnon, `--no-repack`, 1171-token prompt | 177.9 MiB (182,156 kB) | measured, single runs (`docs/CALIBRATION.md`) |
| other file-backed RSS (binary, libraries) | ~13 MiB | measured |
| safety margin for kernel memory charged to the cgroup | 64 MiB | assumption, not measured |
| RLIMIT_MEMLOCK, root under sudo | 758,604 KiB (740.8 MiB), soft = hard | measured (`ulimit -l` / `ulimit -Hl` under sudo); user's own soft limit is 64 MiB |

### Rules

Pinned pages are unevictable and charged to the cgroup. With swap = 0, if
pinned + anonymous memory exceeds memory.max the cgroup OOM killer runs. The
unpinned weights also need room to cycle through the page cache, otherwise
pinning makes thrashing worse than baseline.

    room(L)   = L - peak_anon - other_rss - margin
    pin_cap   = min(room(L), 740.8 MiB)   (RLIMIT_MEMLOCK under sudo)
    left(L,p) = room(L) - pinned(p)       (page cache left for unpinned weights)

A pin request above `pin_cap` is refused before any `mlock` call, with a clear
message naming which bound was exceeded (`room(L)` or RLIMIT_MEMLOCK) and both
values, and the run is recorded as a strategy setup failure (no fallback).

Policy note: the 740.8 MiB memlock limit is treated as a fixed cap and is not
raised. A root process normally has CAP_SYS_RESOURCE (may raise the hard limit)
and CAP_IPC_LOCK (the kernel then does not apply RLIMIT_MEMLOCK to `mlock`),
so this is a project rule, not a kernel guarantee. It keeps the
pin bound explicit and the same as `MemoryPinner::lock`'s own check.

Fractions are of the ~934.7 MiB mmap-served set, chosen in file-offset order:
25% ≈ 234 MiB, 50% ≈ 467 MiB, 75% ≈ 701 MiB.

### Budget table (calibrated levels)

Inputs: peak anon 177.9 MiB (measured, 1171-token prompt), other_rss 13 MiB
(measured), margin 64 MiB (assumption), memlock 740.8 MiB (measured).
Fractions of 934.7 MiB: 25% = 233.7, 50% = 467.4, 75% = 701.0 MiB.
Levels from `docs/CALIBRATION.md`.

| memory.max | room(L) | pin_cap | binding bound | 25% left | 50% left | 75% left |
|---|---|---|---|---|---|---|
| none | unbounded | 740.8 | memlock | n/a | n/a | n/a |
| 1100M | 845.1 | 740.8 | memlock | 611.4 | 377.7 | 144.1 |
| 1000M | 745.1 | 740.8 | memlock (by 4.3 MiB) | 511.4 | 277.7 | 44.1 (too tight) |

- **1100M:** 25%, 50% and 75% are all below `pin_cap` and feasible.
- **1000M:** all three are below `pin_cap`, but 75% leaves only 44.1 MiB of page
  cache for the remaining ~234 MiB of mmap-served weights plus the embedding
  rows, and the 64 MiB margin it relies on is an assumption. 75% at 1000M is
  not recommended; 25% and 50% are feasible.
- **none:** no cgroup bound; only the memlock cap applies (all three fractions
  are below 740.8 MiB).
- 100% of the mmap-served set (934.7 MiB) exceeds the memlock cap at every level
  and is refused.
- These are computed bounds, not measured outcomes; whether pinning helps or
  hurts at these levels is what the experiments will measure.

### For reference: repack on (not used in Phase 3)

Peak anon 774.4 MiB (measured) leaves ~349 MiB of room at 1200M, and only the
~338 MiB Q6_K + F32 part of the file is read after load (inferred).

**The helper refuses** to pin more than `pin_cap` and records the requested
fraction, the computed cap, the inputs used, and the bytes actually locked.

## 4. What we can and cannot claim

- We control **model weights through the page cache** (pin with mlock, hint with
  madvise) and the cgroup budget. We do **not** control the KV cache: it lives
  in llama.cpp's anonymous memory (56 MiB here) and is only observed.
- With repack on, 596.5 MiB of the weights are anonymous copies inside
  llama.cpp that the helper cannot pin, prefetch or evict. Phase 3 therefore
  runs with `--no-repack`; claims apply only to that configuration.
- `madvise(MADV_WILLNEED)` is advisory: the kernel may ignore it, read ahead
  asynchronously, or evict the pages again. Effects must be measured (mincore),
  never assumed.
- `mlock` is bounded: by RLIMIT_MEMLOCK (740.8 MiB under sudo, kept as a fixed cap), by the cgroup's
  memory.max (locked pages are charged) and by available RAM. It does not
  reserve memory; it only keeps faulted-in pages resident.
- NeuroSwap does not replace or bypass the kernel's memory management.
- llama.cpp already does whole-file `MAP_POPULATE` + `WILLNEED`; a "prefetch"
  strategy can only add timing (before start) or re-warming (during decode),
  and any gain has to be measured against that baseline.
- No performance claims until the runs exist.

## 5. Design

### Components

- `neuroswap_helper` (new C++ executable, reuses `FileMapping`, `MemoryPinner`):
  `--model PATH --mode pin|prefetch --ranges FILE --max-lock-bytes N`.
- `scripts/model_ranges.py` (new): parses the GGUF tensor table (stdlib only),
  writes `data/model_ranges.json` with every tensor's byte range and type, the
  model sha256, and the selected ranges per fraction (all tensors except
  `token_embd`, since `--no-repack` keeps every weight mmap-served).
  Deterministic order: file offset.
- `run_one.py`: new strategies `pin25`, `pin50`, `pin75`, `prefetch`; starts and
  stops the helper; stores the helper report in the run record.

### Helper lifecycle (one run)

1. `run_one` (root): if cold, sync + drop_caches; create `neuroswap_run`.
2. `run_one` starts the helper **inside the same cgroup** (same pre-exec join
   as llama). Reason: a page-cache page is charged to the cgroup of the process
   that first faults it. A helper outside the cgroup would put the pinned or
   prefetched pages on another cgroup's bill and effectively give llama memory
   beyond memory.max, which would make the comparison invalid.
3. Helper: open + mmap the .gguf read-only; `mincore` snapshot; for each
   selected range: `madvise(WILLNEED)`, read one byte per page, then
   (pin) `mlock`; `mincore` again; read `VmLck`. Prints one JSON line `ready`
   with requested/locked bytes, `pin_cap` and its inputs, RLIMIT_MEMLOCK,
   residency before/after, timings.
4. `run_one` waits for `ready` (timeout, e.g. 120 s). If the helper fails or
   times out: write a record with `error = "strategy setup failed: ..."`, kill
   the helper, do **not** run llama, no fallback to baseline.
5. `run_one` starts llama in the cgroup, samples as today; wall clock for the
   run starts at helper start (setup cost is part of the strategy).
6. After llama exits: read `memory.events`, `memory.stat`, `memory.peak`;
   close the helper's stdin; helper takes a final `mincore` of the pinned
   ranges and whole file, `munlock`, `munmap`, prints a `final` JSON line, exits.
7. `run_one` kills the helper if still alive (in `finally`), then removes the
   cgroup.

Prefetch mode: same lifecycle without `mlock`. Variant A: WILLNEED once before
llama starts. Variant B (later): re-issue WILLNEED on the mmap-served set every
N ms during decode. Both must be compared to llama.cpp's built-in populate.

### Verifying a strategy really took effect

| check | source | pass condition (pin) |
|---|---|---|
| bytes locked | helper `VmLck`, smaps `Locked` | == requested range bytes |
| residency of pinned ranges | helper `mincore`, at ready and at final | 100% at both |
| cgroup accounting | `memory.stat` `unevictable` | ≈ locked bytes |
| effect on llama | `ru_majflt`, `io_read_bytes`, RssFile samples | reported, not assumed |

A pin run that fails the first two checks is recorded with
`strategy_effective = false` and excluded from pin-vs-baseline comparisons.
For prefetch: residency of the target ranges at `ready` and at llama start,
since readahead is asynchronous.

### Rules for the experiment matrix

- Limited runs must be cold. In a warm run the model's page-cache pages are
  already charged to another cgroup, so memory.max does not bind them.
  `run_one` should refuse `--memory-limit` without `--cold`.
- `--no-repack` for every strategy, baseline included (decided); recorded in
  every row as `repack`.
- Record llama.cpp's buffer-size lines (`-lv 4`) in each record so the repack
  state is verified per run, not assumed.

### Risks and fallbacks

| risk | mitigation |
|---|---|
| OOM from pinning too much | cap from measured anon peak + margin; start at 25% |
| OOM killer picks the helper instead of llama | record both exit statuses; mark run invalid |
| helper's own memory charged to the cgroup | C++ helper (small RSS); report its peak RSS |
| pin request above RLIMIT_MEMLOCK (740.8 MiB) | refused before `mlock` with a clear message; limit not raised |
| stale cgroup or leftover helper | `finally` kill + rmdir; refuse to start if `neuroswap_run` exists (already done) |
| prefetch shows no effect because llama already populates | report as a finding; try the during-decode variant |
| repack silently on (flag ignored or changed upstream) | record `"repack"` and check the `-lv 4` log has no `CPU_REPACK` buffer; mark run invalid otherwise |
| WSL2: drop_caches and disk timing are VM-wide and noisy | repeat runs; keep other load idle; report variance |

### Decisions

1. Repack setting: **decided, `--no-repack` for all Phase 3 runs.**
2. Helper language: **decided, C++, reusing the Review-2 `MemoryPinner` and
   `FileMapping` (Review-2 modules and tests unchanged).**
3. Cold start: **decided, `--cold` is required for every `--memory-limit` run**
   (enforcement in `run_one.py` not implemented yet).
4. Budget levels: **decided from the `--no-repack` calibration**: none, 1100M,
   1000M (`docs/CALIBRATION.md`).
5. RLIMIT_MEMLOCK: **decided, kept at the measured 740.8 MiB and not raised**;
   pins above `min(room(L), 740.8 MiB)` are refused.

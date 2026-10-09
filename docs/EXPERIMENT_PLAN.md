# Experiment plan: pilot and final batch

**The pilot is not a result.** Its data are used to plan the final batch and
may be used as ML training data. Nothing measured in the pilot is reported as a
finding about whether a strategy helps. The final batch (section "Final batch")
has not been run; this document records its design only.

## Pilot purpose

1. **Noise:** how much the same condition (prompt, limit, strategy) varies
   across 3 repetitions, to choose the repetition count for the main run.
2. **Run time:** measured time per run for each limit and strategy, to size the
   main plan and the prompt count per bucket.
3. **Whether any condition differs** at all from baseline, beyond the
   repetition noise, to decide which strategies and fractions are worth the
   main run's time. This is a go/no-go screen, not an effect estimate.
4. **Harness check under sudo:** cgroup join, cold starts, helper refusals,
   resume after interruption.

## Pilot plans

Both plans use `--no-repack` (`"repack": "off"`), 3 repetitions, order seed 7,
and the strategies baseline, prefetch, pin 0.25 and pin 0.5. Prompts are drawn
from `data/manifest.csv` per bucket with seed 42 (`scripts/run_experiment.py`
`select_prompts`).

| plan | file | limits | prompts | runs | output (not committed) |
|---|---|---|---|---|---|
| pilot | `experiments/pilot.json` | none, 1100M | 3 per bucket (12) | 12 x 2 x 4 x 3 = 288 | `results/raw/llm/pilot.jsonl` |
| pilot_tight | `experiments/pilot_tight.json` | 1000M | 1 per bucket (4) | 4 x 1 x 4 x 3 = 48 | `results/raw/llm/pilot_tight.jsonl` |

Selected prompts (from `--dry-run`):

| bucket | pilot | pilot_tight |
|---|---|---|
| <64 | wc0003 wc0014 wc0035 | wc0014 |
| 64-256 | wc0092 wc0103 wc0106 | wc0078 |
| 256-1024 | wc0161 wc0163 wc0219 | wc0185 |
| 1024-1500 | wc0228 wc0229 wc0279 | wc0256 |

Levels come from `docs/CALIBRATION.md`: 1100M is the moderate,
prompt-dependent level; 1000M is the severe level, run on a subset because
each run took 96-127 s in calibration. Pin fractions 0.25 and 0.5 are both
below `pin_cap` at 1100M and 1000M (`docs/STRATEGY_DESIGN.md` section 3);
0.75 is left out because it leaves only 44 MiB of page cache at 1000M.

## How runs are executed

- `scripts/run_experiment.py` expands a plan into runs with a stable `run_id`
  (sha256 of `prompt_id|memory_limit|strategy|pin_fraction|repetition`, first
  16 hex characters), shuffles them with the order seed, and runs each through
  `scripts/run_one.py`.
- Each record gets two extra keys, `run_id` and `plan_name`; everything else
  is the `run_one.py` record unchanged.
- Resumable: run_ids already in the output are skipped, errored ones too unless
  `--retry-errors`. A retried run adds a new line; analysis uses the last
  record per run_id.
- Ctrl+C: the current run is stopped, its record discarded, the cgroup removed;
  the output stays valid line by line. Rerun the same command to resume.
- A status file `<output>.status.json` shows progress and the ETA (mean of the
  runs measured in the current session).
- The runner needs root for plans with a memory limit or `cold_unlimited`, and
  refuses to start when the 1-minute load average is above 1.0 unless `--force`.
- `prompts.exclude_plans` removes every prompt selected by the listed plans
  before sampling; the dry run prints the overlap, and the runner refuses to
  run if it is not 0. `cold_unlimited: true` makes `none` runs cold as well.

Commands (run by the user, in a separate terminal):

```
.venv/bin/python scripts/run_experiment.py experiments/pilot.json --dry-run
sudo .venv/bin/python scripts/run_experiment.py experiments/pilot.json
sudo .venv/bin/python scripts/run_experiment.py experiments/pilot_tight.json
sudo .venv/bin/python scripts/run_experiment.py experiments/final.json
```

## Known confounds in this design

- **Pilot: unlimited runs are warm, limited runs are cold.** By default
  `run_one.py` drops caches only for limited runs (the final batch sets
  `cold_unlimited`, so this applies to the pilot only). A `none` run therefore starts with whatever the
  previous run (in shuffled order) left in the page cache, so its cache state
  is not controlled. Comparisons are made within a limit, not across limits.
- **In the pilot, strategy comparisons at `none` are confounded with cache
  state.** Whether a baseline, prefetch or pin run at `none` finds the model
  already cached depends on which run came before it, not on the strategy.
  Pilot results at `none` are therefore only used for run time and the
  harness check, not to compare strategies. **The final experiments must run
  the `none` level cold too** (sync + drop_caches before every run), so that
  every level starts from the same cache state.
- **Order:** the shuffle spreads conditions over time, but drift (thermal,
  background load on the Windows host of WSL2) is not otherwise controlled.
- **Single machine, WSL2:** disk timing and `drop_caches` are VM-wide.

## Pilot observations (not results)

**Pilot observations, not results.** Single machine, 3 repetitions per
condition, 3 prompts per bucket (pilot) or 1 (pilot_tight). Numbers are from
`results/raw/llm/pilot.jsonl`, `pilot_tight.jsonl`, their `.status.json` files
and `scripts/analyze_pilot.py`. None of them says whether a strategy helps.

### Completion and run time

| plan | records | errors | run time |
|---|---|---|---|
| pilot | 288 | 0 | 2026-10-08 15:04:18-19:49:36 UTC (4 h 45 min), mean 23.1 s per run |
| pilot_tight | 48 | 0 | last session: 45 runs, 14:06:41-14:58:45 UTC, mean 67.9 s per run |

### Prefetch residency (fields under `helper_report`)

**The pilot's prefetch rows are not a test of prefetch.** In the pilot the
helper issued a single `madvise(MADV_WILLNEED)` over the whole file. Under cold,
limited runs that left about 0.75% of the file (2,048 pages, 8 MiB) resident
before llama.cpp started, so those rows measure "one madvise plus a 3 s wait",
not a prefetched model. `madvise` returned 0 in every prefetch run.

| level | runs | wait ended on | pages resident at end of wait (of 272,784) |
|---|---|---|---|
| 1000M (cold) | 12 | plateau in 12 | 2,048 (8.0 MiB) in every run |
| 1100M (cold) | 36 | plateau in 36 | 2,048 (8.0 MiB) in every run |
| none (warm) | 36 | target in 19, plateau in 17 | 272,784 (target) or 232,270-250,580 (plateau); 231,521-272,784 were already resident before `madvise` |

- At 1000M, `residency_wait.wait_ms` was 3007.3-3021.6 ms and
  `strategy_setup_ms` 3054.4-3074.8 ms, so the prefetch setup cost was almost
  entirely the 3 s plateau wait (1100M: setup medians 3061.6-3070.3 ms).

### Follow-up: chunked madvise (observation, after the pilot)

Single runs on 2026-10-09, without root: the model file was evicted with
`posix_fadvise(POSIX_FADV_DONTNEED)` (cold for that file), no memory limit, no
cgroup, `neuroswap_strategy --strategy prefetch --wait-resident 99
--wait-plateau-s 3 --wait-timeout 120`.

| madvise pattern | stopped | chunks completed | resident at end (of 272,784 pages) | wait |
|---|---|---|---|---|
| one call over the whole file | plateau | - | 2,048 (0.751%) | 3,010 ms |
| 32 MiB chunks | plateau in chunk 0 (2,048 of its 8,192 pages) | 0 / 34 | 2,048 (0.751%) | 3,024 ms |
| 8 MiB chunks | target | 134 / 134 | 272,784 (100.0%) | 2,398 ms (per chunk: median 10.4 ms, max 21.1 ms) |

- The model's block device (`/dev/sdd`) has `read_ahead_kb = 8192`, i.e.
  8 MiB = 2,048 pages, the amount one `madvise` call brought in every time.
  This is consistent with a per-call cap at the device readahead size; it has
  not been checked against kernel source.
- The same 2,048-page stop appeared without a memory limit, so it is not
  caused by memory pressure.
- Since then the helper prefetches in 8 MiB chunks by default
  (`--prefetch-chunk-mib 8`, also passed by `run_one.py`), and each record
  carries `prefetch_chunk_mib`.
- Not yet observed: chunked prefetch as root after `drop_caches`, and inside
  the 1100M cgroup, where the whole file cannot stay resident.

### Spread across repetitions (baseline, 1100M, `total_wall_s`)

Per prompt: median and range (max - min) over its 3 repetitions.

| bucket | per-prompt medians (s) | per-prompt ranges (s) |
|---|---|---|
| <64 | 4.15, 6.77, 7.88 | 0.52, 0.13, 5.91 |
| 64-256 | 9.58, 7.47, 8.98 | 1.82, 1.32, 1.92 |
| 256-1024 | 41.10, 28.97, 54.30 | 16.46, 23.49, 24.44 |
| 1024-1500 | 58.12, 55.75, 67.81 | 27.44, 4.64, 4.81 |

### Per-condition medians of `total_wall_s` (s; n = 9 at 1100M, 3 at 1000M)

At 1100M the 9 runs pool 3 prompts, so the IQR includes differences between
prompts. At 1000M each bucket is one prompt.

| bucket | limit | baseline | prefetch | pin 0.25 | pin 0.5 |
|---|---|---|---|---|---|
| <64 | 1100M | 6.769 | 9.673 | 8.341 | 7.337 |
| 64-256 | 1100M | 8.978 | 12.967 | 11.205 | 10.291 |
| 256-1024 | 1100M | 41.098 | 49.856 | 38.362 | 43.046 |
| 1024-1500 | 1100M | 58.443 | 64.805 | 61.221 | 58.138 |
| <64 | 1000M | 16.534 | 14.507 | 18.302 | 13.051 |
| 64-256 | 1000M | 96.452 | 58.001 | 77.771 | 61.986 |
| 256-1024 | 1000M | 104.036 | 107.508 | 87.147 | 54.746 |
| 1024-1500 | 1000M | 115.386 | 121.015 | 101.768 | 71.485 |

IQRs, min/max, major faults and per-prompt paired differences are printed by
`.venv/bin/python scripts/analyze_pilot.py results/raw/llm/<file>.jsonl`.

## Final batch

**Design only; not run yet.** Plan: `experiments/final.json`.

Purpose: the measured comparison of baseline, prefetch and pin 0.25 at `none`
and 1100M on prompts that were never part of the pilot, and the held-out
**ML test set** (see "Data use").

| setting | value |
|---|---|
| prompts | 3 per bucket (12), seed 2026, excluding all 15 pilot prompts (`prompts.exclude_plans`) |
| limits | none, 1100M |
| cache | every run cold, `none` included (`cold_unlimited: true`) |
| strategies | baseline, prefetch (8 MiB madvise chunks), pin 0.25 |
| repetitions | 2, order seed 11, `--no-repack` |
| runs | 12 x 2 x 3 x 2 = 144 |
| output | `results/raw/llm/final.jsonl` (not committed) |

Selected prompts (from `--dry-run`; overlap with the pilot prompts: 0, also
checked independently of the runner):

| bucket | final prompts |
|---|---|
| <64 | wc0017 wc0043 wc0067 |
| 64-256 | wc0089 wc0107 wc0144 |
| 256-1024 | wc0206 wc0215 wc0224 |
| 1024-1500 | wc0225 wc0258 wc0285 |

Prefetch in the final batch uses 8 MiB chunks (see "Follow-up: chunked
madvise"); unlike the pilot, these rows are intended as a test of prefetch.
How far residency gets inside the 1100M cgroup is recorded per run in
`helper_report.residency_wait` and is not known in advance.

## Data use

- **Held-out rule:** the 12 final-batch prompts listed above are the **ML test
  set**. Their records (`results/raw/llm/final.jsonl`) are never used for
  training, feature selection, threshold or hyperparameter choices.
- **Pilot data may be used for ML training** (15 prompts, 336 records), never as
  test data. Pilot and final records are kept in separate files and are never
  pooled into one test set.
- Any later plan that adds prompts must exclude the final-batch prompts from
  training (add `experiments/final.json` to its `prompts.exclude_plans`).
- All records here are measured telemetry (`source = "neuroswap_measured"`) and
  stay separate from the external dataset (`source = "external"`).
- Raw outputs stay under `results/raw/llm/`, which is not committed.

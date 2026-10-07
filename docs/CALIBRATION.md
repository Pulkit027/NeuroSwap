# Phase 3 calibration (baseline, `--no-repack`)

**Single-run calibration observations, not benchmark results.** Each row is
one run; there are no repeats, so no variance, and nothing here supports a
performance claim. The purpose is only to choose memory budget levels for
the strategy experiments.

Source: `results/raw/llm/calibration.jsonl` (10 records, 2026-10-07 06:41-06:53 UTC,
not committed). All 10 records have `source = "neuroswap_measured"`,
`strategy = "baseline"`, `repack = "off"` (`--no-repack` in `command`),
`cold = true`, `run_index = 0`, `exit_code = 0`, `error = null`.
Two prompts were used: `wc0000` (bucket `<64`) and `wc0225` (bucket `1024-1500`).

## Runs

Columns are record fields: `prompt_id`, `prompt_tokens`, `memory_limit`,
`wall_s`, `llama_perf.eval_tps` (decode tok/s), `rusage.ru_majflt`,
`rusage.ru_maxrss_kb`, `cgroup_memory_events.max`,
`cgroup_memory_events.oom_kill`. `null` = no cgroup (`memory_limit = none`).

| prompt_id | prompt_tokens | memory_limit | wall_s | eval_tps | ru_majflt | ru_maxrss_kb | events.max | events.oom_kill |
|---|---|---|---|---|---|---|---|---|
| wc0000 | 25 | none | 10.256 | 10.62 | 16 | 1213968 | null | null |
| wc0225 | 1171 | none | 39.704 | 11.19 | 14 | 1279376 | null | null |
| wc0000 | 25 | 1100M | 8.824 | 10.41 | 49 | 1113648 | 390 | 0 |
| wc0225 | 1171 | 1100M | 74.521 | 1.5 | 7183 | 1125156 | 109116 | 0 |
| wc0000 | 25 | 1000M | 96.399 | 0.68 | 12447 | 1023356 | 243845 | 0 |
| wc0225 | 1171 | 1000M | 126.707 | 0.65 | 13129 | 1022756 | 264076 | 0 |
| wc0000 | 25 | 900M | 93.958 | 0.73 | 13136 | 920532 | 228650 | 0 |
| wc0225 | 1171 | 900M | 104.134 | 0.91 | 12909 | 920496 | 220757 | 0 |
| wc0000 | 25 | 800M | 86.018 | 0.78 | 13702 | 818580 | 378420 | 0 |
| wc0225 | 1171 | 800M | 108.066 | 0.84 | 13197 | 818144 | 225255 | 0 |

No run hit the OOM killer (`oom` and `oom_kill` are 0 in every cgroup run).

## Supporting observations from the samples

Peaks over the 100 ms `samples`; `io_read_bytes` is the last sample's
cumulative value (bytes the process caused to be read from storage).

| prompt_id | memory_limit | peak RssAnon_kb | peak RssFile_kb | last io_read_bytes |
|---|---|---|---|---|
| wc0000 | none | 116460 | 1097872 | 1134911488 |
| wc0225 | none | 182156 | 1097820 | 1135042560 |
| wc0000 | 1100M | 116460 | 1040760 | 1352712192 |
| wc0225 | 1100M | 182156 | 1065340 | 22997381120 |
| wc0000 | 1000M | 116460 | 938824 | 55517163520 |
| wc0225 | 1000M | 182156 | 952208 | 57562791936 |
| wc0000 | 900M | 116460 | 848268 | 50510331904 |
| wc0225 | 900M | 182156 | 851016 | 48835624960 |
| wc0000 | 800M | 116460 | 735732 | 60962775040 |
| wc0225 | 800M | 182160 | 735284 | 57319948288 |

- Peak anonymous memory with `--no-repack` was 116,460 kB for the 25-token
  prompt and 182,156 kB (177.9 MiB) for the 1171-token prompt, at every limit.
  `docs/STRATEGY_DESIGN.md` used an inferred ~175 MiB for the long prompt.
- Without a limit, each run read about the model file size once
  (1,134,911,488 and 1,135,042,560 bytes; file is 1,117,320,736 bytes).
- At 1000M and below, each run read between 48.8 and 61.0 GB for that
  1.1 GB file.

## Proposed budget levels

| level | memory.max | role |
|---|---|---|
| none | unlimited | reference: no pressure |
| moderate | 1100M | prompt-dependent pressure; full prompt set |
| severe | 1000M | thrashing for both prompts; subset of prompts only |

Reasoning, from the single runs above:

- **1100M is prompt-dependent.** The 25-token prompt ran with 49 major faults
  and 10.41 decode tok/s, close to the unlimited run (16, 10.62). The
  1171-token prompt took 7183 major faults and 1.5 tok/s, and read
  22,997,381,120 bytes. Its anonymous peak is ~64 MiB larger, which leaves less
  of the 1100M for model pages. This is the level where a strategy could plausibly matter
  for some prompts and not others, so it gets the full prompt set.
- **1000M thrashes regardless of prompt.** Both prompts ran at 0.65-0.68 tok/s
  with 12,447-13,129 major faults and wall times of 96-127 s per run. Because
  each run is that expensive, 1000M is run on a subset of prompts only.
- **900M and 800M are dropped.** They show the same thrash plateau as 1000M:
  decode 0.73-0.91 tok/s, 12,909-13,702 major faults, 86-108 s per run, so
  they add cost without adding a distinct regime.
- 1200M was not part of this calibration. `docs/STRATEGY_DESIGN.md` estimated
  that with `--no-repack` 1200M may not create pressure at all. The 1100M
  result for the short prompt (49 major faults) is consistent with that, but
  1200M itself was not measured.

Caveats: two prompts, one run each, cold start, on WSL2 where disk timing is
noisy. The levels should be re-checked once repeats exist. The pin budget at 1100M and
1000M follows the `room(L)` rule in `docs/STRATEGY_DESIGN.md` section 3,
using a measured long-prompt anon peak of 177.9 MiB in place of the
inferred 175 MiB.

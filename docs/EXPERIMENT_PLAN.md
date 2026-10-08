# Pilot experiment plan

**The pilot is not a result.** Its data are used only to plan the main
experiment. Nothing measured in the pilot is reported as a finding about
whether a strategy helps.

## Purpose

1. **Noise:** how much the same condition (prompt, limit, strategy) varies
   across 3 repetitions, to choose the repetition count for the main run.
2. **Run time:** measured time per run for each limit and strategy, to size the
   main plan and the prompt count per bucket.
3. **Whether any condition differs** at all from baseline, beyond the
   repetition noise, to decide which strategies and fractions are worth the
   main run's time. This is a go/no-go screen, not an effect estimate.
4. **Harness check under sudo:** cgroup join, cold starts, helper refusals,
   resume after interruption.

## Plans

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
- The runner needs root for limited plans and refuses to start when the
  1-minute load average is above 1.0 unless `--force`.

Commands (run by the user, in a separate terminal):

```
.venv/bin/python scripts/run_experiment.py experiments/pilot.json --dry-run
sudo .venv/bin/python scripts/run_experiment.py experiments/pilot.json
sudo .venv/bin/python scripts/run_experiment.py experiments/pilot_tight.json
```

## Known confounds in this design

- **Unlimited runs are warm, limited runs are cold.** `run_one.py` drops caches
  only for limited runs. A `none` run therefore starts with whatever the
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

## Data use

- Pilot data are **never reused as final held-out test data**. The 15 pilot
  prompt IDs above (wc0014 is in both plans) must be excluded from any
  held-out set in the main experiment, and pilot records are not pooled with
  main-experiment records.
- Pilot records are measured telemetry (`source = "neuroswap_measured"`) and
  stay separate from the external dataset (`source = "external"`).
- Raw outputs stay under `results/raw/llm/`, which is not committed.

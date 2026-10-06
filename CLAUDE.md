# NeuroSwap rules (read first)

## Data
- No synthetic datasets. Primary dataset: WildChat-1M user prompts (ODC-BY); fallback: databricks-dolly-15k.
- Keep external dataset and NeuroSwap-measured telemetry separate and tagged (source = external | measured).

## Claims
- No performance or ML accuracy claims before measurement.
- No direct KV-cache control claims. Only model weights and page cache are controlled; KV is observed.
- NeuroSwap does not replace or bypass the Linux kernel. mlock is bounded by limits; madvise is advisory.

## Environment
- Repo lives in ~/NeuroSwap (WSL2 Ubuntu 24.04). Never work from /mnt/c.
- Free tools only. No paid APIs or cloud.
- Privileged steps (sudo, cgroup, drop_caches, ulimit) are run by the user, not by Claude Code.

## Workflow
- Preserve Review-2 modules and tests; all tests must pass before moving on.
- One phase at a time, one git branch per phase, commit before and after.
- Long experiment runs are started by the user in a separate terminal.

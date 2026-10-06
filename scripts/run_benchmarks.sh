#!/usr/bin/env bash
set -euo pipefail

repetitions="${1:-5}"
if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [positive-repetition-count]" >&2
    exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p results/raw

for run in $(seq 1 "$repetitions"); do
    suffix=$(printf '%02d' "$run")
    ./build/neuroswap_baseline --memory-mb 16 --iterations 3 --pattern sequential \
        > "results/raw/baseline_run_${suffix}.txt"
    ./build/neuroswap_pinned --memory-mb 16 --iterations 3 --pattern sequential \
        > "results/raw/pinned_run_${suffix}.txt"
    # Each invocation intentionally records matched no-hint and WILLNEED cases
    # for the same deterministic file configuration.
    ./build/neuroswap_prefetch --file-mb 16 --iterations 3 --pattern sequential \
        > "results/raw/prefetch_run_${suffix}.txt"
done

echo "Completed ${repetitions} runs each for baseline, pinned, and paired prefetch experiments."

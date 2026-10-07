#!/usr/bin/env bash
# Smoke test for run_one.py: first prompt of each bucket (manifest order),
# no memory limit, warm cache, baseline. Needs no root.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
out=results/raw/llm/smoke.jsonl
mkdir -p results/raw/llm

failures=0
for bucket in "<64" "64-256" "256-1024" "1024-1500"; do
    prompt_id=$(awk -F, -v b="$bucket" 'NR > 1 && $5 == b { print $1; exit }' data/manifest.csv)
    if [[ -z "$prompt_id" ]]; then
        echo "no prompt for bucket $bucket" >&2
        exit 1
    fi
    .venv/bin/python scripts/run_one.py --prompt-id "$prompt_id" --memory-limit none \
        --strategy baseline --run-index 0 --out "$out" || failures=$((failures + 1))
done
echo "wrote $out ($failures failed)"
exit $((failures > 0))

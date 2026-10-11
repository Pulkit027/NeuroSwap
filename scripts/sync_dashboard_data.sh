#!/usr/bin/env bash
# Copy the exported dashboard data (results/dashboard/*.json, from
# scripts/export_dashboard_json.py) into dashboard/public/data/, where the
# dashboard loads it. The copies are committed so the dashboard works after a clone.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$project_root/results/dashboard"
dst="$project_root/dashboard/public/data"
files=(overview experiments strategy_comparison ml_metrics feature_importance predictions policy_results)

for name in "${files[@]}"; do
    if [[ ! -f "$src/$name.json" ]]; then
        echo "missing $src/$name.json; run scripts/export_dashboard_json.py first" >&2
        exit 1
    fi
done
mkdir -p "$dst"
for name in "${files[@]}"; do
    cp "$src/$name.json" "$dst/$name.json"
done
echo "copied ${#files[@]} files to $dst"

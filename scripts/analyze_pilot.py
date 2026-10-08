#!/usr/bin/env python3
"""Summarize a run_experiment.py output file (results/raw/llm/*.jsonl).

Keeps the last record per run_id (a retried run appends a new line), then prints:

1. Per (bucket, memory_limit, strategy, pin_fraction): record count, error count,
   and over error-free records: median / IQR / min / max of total_wall_s,
   median inference_wall_s, median strategy_setup_ms, median ru_majflt.
2. Paired differences vs baseline: per prompt and memory limit, the median over
   repetitions of each condition minus the median over repetitions of baseline,
   so differences between prompts cancel.

Only numbers are printed; nothing here is a performance claim. Uses pandas.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BUCKET_ORDER = ["<64", "64-256", "256-1024", "1024-1500"]
GROUP = ["bucket", "memory_limit", "strategy", "pin_fraction"]


def load(path):
    """Records of a jsonl file; a trailing incomplete line is skipped with a warning."""
    lines = Path(path).read_text().split("\n")
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            if number == len(lines):
                print(f"warning: skipped incomplete last line {number}", file=sys.stderr)
                continue
            raise SystemExit(f"{path}: line {number} is not valid JSON")
    return records


def to_frame(records):
    rows = {}
    no_id = 0
    for r in records:
        key = r.get("run_id")
        if key is None:
            no_id += 1
            key = f"no_run_id_{no_id}"
        helper = r.get("helper_report") or {}
        rows[key] = {  # later lines replace earlier ones: last record per run_id wins
            "run_id": key,
            "prompt_id": r.get("prompt_id"),
            "bucket": r.get("bucket"),
            "memory_limit": r.get("memory_limit"),
            "strategy": r.get("strategy"),
            "pin_fraction": helper.get("fraction") if r.get("strategy") == "pin" else None,
            "run_index": r.get("run_index"),
            "error": r.get("error"),
            "total_wall_s": r.get("total_wall_s"),
            "inference_wall_s": r.get("inference_wall_s"),
            "strategy_setup_ms": r.get("strategy_setup_ms"),
            "ru_majflt": (r.get("rusage") or {}).get("ru_majflt"),
        }
    if no_id:
        print(f"warning: {no_id} record(s) without run_id kept as separate runs", file=sys.stderr)
    frame = pd.DataFrame(list(rows.values()))
    frame["pin_fraction"] = frame["pin_fraction"].map(lambda f: "-" if f is None or pd.isna(f) else f"{f:g}")
    frame["has_error"] = frame["error"].notna()
    for column in ["total_wall_s", "inference_wall_s", "strategy_setup_ms", "ru_majflt"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, len(records)


def limit_order(limit):
    if limit == "none":
        return (0, 0)
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    value = int(limit[:-1]) * units[limit[-1]] if limit[-1] in units else int(limit)
    return (1, -value)


def sort_frame(frame):
    order = {
        "bucket": lambda s: s.map(lambda b: BUCKET_ORDER.index(b) if b in BUCKET_ORDER else len(BUCKET_ORDER)),
        "memory_limit": lambda s: s.map(limit_order),
        "strategy": lambda s: s.map({"baseline": 0, "prefetch": 1, "pin": 2}).fillna(3),
    }
    keys = [c for c in ["bucket", "memory_limit", "prompt_id", "strategy", "pin_fraction"] if c in frame]
    return frame.sort_values(keys, key=lambda s: order[s.name](s) if s.name in order else s).reset_index(drop=True)


def summary_table(frame):
    ok = frame[~frame["has_error"]]
    counts = frame.groupby(GROUP).agg(n=("run_id", "size"), errors=("has_error", "sum"))
    stats = ok.groupby(GROUP).agg(
        n_ok=("run_id", "size"),
        total_median_s=("total_wall_s", "median"),
        total_iqr_s=("total_wall_s", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        total_min_s=("total_wall_s", "min"),
        total_max_s=("total_wall_s", "max"),
        inference_median_s=("inference_wall_s", "median"),
        setup_median_ms=("strategy_setup_ms", "median"),
        majflt_median=("ru_majflt", "median"),
    )
    table = counts.join(stats, how="left").reset_index()
    table["n_ok"] = table["n_ok"].fillna(0).astype(int)
    return sort_frame(table)


def paired_table(frame):
    """Per prompt and limit: median over repetitions per condition, minus baseline."""
    ok = frame[~frame["has_error"]]
    per_prompt = ok.groupby(["bucket", "memory_limit", "prompt_id", "strategy", "pin_fraction"]).agg(
        reps=("run_id", "size"),
        total_median_s=("total_wall_s", "median"),
        inference_median_s=("inference_wall_s", "median"),
        majflt_median=("ru_majflt", "median"),
    ).reset_index()
    base = per_prompt[per_prompt["strategy"] == "baseline"].drop(columns=["strategy", "pin_fraction"])
    base = base.rename(columns={c: f"base_{c}" for c in ["reps", "total_median_s", "inference_median_s", "majflt_median"]})
    other = per_prompt[per_prompt["strategy"] != "baseline"]
    paired = other.merge(base, on=["bucket", "memory_limit", "prompt_id"], how="left")
    paired["diff_total_s"] = paired["total_median_s"] - paired["base_total_median_s"]
    paired["diff_inference_s"] = paired["inference_median_s"] - paired["base_inference_median_s"]
    paired["diff_majflt"] = paired["majflt_median"] - paired["base_majflt_median"]
    columns = ["bucket", "memory_limit", "prompt_id", "strategy", "pin_fraction", "reps", "base_reps",
               "base_total_median_s", "total_median_s", "diff_total_s", "diff_inference_s",
               "base_majflt_median", "majflt_median", "diff_majflt"]
    return sort_frame(paired[columns])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path, help="e.g. results/raw/llm/pilot_tight.jsonl")
    ap.add_argument("--csv", type=Path, help="also write the per-condition table to this CSV file")
    args = ap.parse_args()

    frame, n_lines = to_frame(load(args.jsonl))
    print(f"file {args.jsonl}: {n_lines} lines, {len(frame)} runs after keeping the last record per run_id, "
          f"{int(frame['has_error'].sum())} with an error")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    table = summary_table(frame)
    print("\n== Per condition (error-free records; IQR/min/max are of total_wall_s) ==")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    paired = paired_table(frame)
    print("\n== Paired vs baseline: per prompt, median over repetitions, condition minus baseline ==")
    for (bucket, limit), group in paired.groupby(["bucket", "memory_limit"], sort=False):
        print(f"\n-- bucket {bucket}, memory_limit {limit}")
        print(group.drop(columns=["bucket", "memory_limit"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

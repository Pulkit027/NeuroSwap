#!/usr/bin/env python3
"""Build the ML train/test tables from measured runs.

Train = results/raw/llm/pilot.jsonl only, without its prefetch rows (the pilot
prefetch used one whole-file madvise and left ~0.75% of the model resident, so it
is not a test of prefetch) and without pilot_tight (1000M, a level not in the test).
Test  = results/raw/llm/final.jsonl only (the held-out prompts, docs/EXPERIMENT_PLAN.md).

The last record per run_id is used; records with an error are dropped and counted.
Both tables carry source = "neuroswap_measured". No prompt text is read or written.
"""

import argparse
import csv
import sys
from pathlib import Path

import nsdata

COLUMNS = ["run_id", "prompt_id", "bucket", "prompt_tokens", "memory_limit", "strategy",
           "pin_fraction", "cold", "total_wall_s", "inference_wall_s", "strategy_setup_ms",
           "ru_majflt", "source"]


def to_row(record):
    return {
        "run_id": record["run_id"],
        "prompt_id": record["prompt_id"],
        "bucket": record["bucket"],
        "prompt_tokens": record["prompt_tokens"],
        "memory_limit": record["memory_limit"],
        "strategy": record["strategy"],
        "pin_fraction": nsdata.pin_fraction(record),
        "cold": record["cold"],
        "total_wall_s": record["total_wall_s"],
        "inference_wall_s": record["inference_wall_s"],
        "strategy_setup_ms": record["strategy_setup_ms"],
        "ru_majflt": (record.get("rusage") or {}).get("ru_majflt"),
        "source": "neuroswap_measured",
    }


def build(pilot_path, final_path):
    """Returns (train_rows, test_rows, counts)."""
    counts = {}
    pilot = nsdata.load_last_records(pilot_path)
    final = nsdata.load_last_records(final_path)
    counts["pilot_records"] = len(pilot)
    counts["final_records"] = len(final)
    counts["pilot_errors_dropped"] = sum(r.get("error") is not None for r in pilot)
    counts["final_errors_dropped"] = sum(r.get("error") is not None for r in final)
    counts["pilot_prefetch_dropped"] = sum(r.get("error") is None and r["strategy"] == "prefetch" for r in pilot)

    train = [to_row(r) for r in pilot if r.get("error") is None and r["strategy"] != "prefetch"]
    test = [to_row(r) for r in final if r.get("error") is None]
    for name, rows, plan in (("train", train, "pilot"), ("test", test, "final")):
        records = pilot if plan == "pilot" else final
        plans = {r.get("plan_name") for r in records}
        if plans != {plan}:
            raise SystemExit(f"{name}: expected plan_name {plan!r} only, found {sorted(map(str, plans))}")

    leaked = {r["prompt_id"] for r in train} & {r["prompt_id"] for r in test}
    if leaked:
        raise SystemExit(f"final (test) prompts found in train: {sorted(leaked)}")
    if any(r["strategy"] == "prefetch" for r in train):
        raise SystemExit("pilot prefetch rows found in train")
    return train, test, counts


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def describe(name, rows):
    from collections import Counter
    cells = Counter((r["memory_limit"], nsdata.strategy_label(r["strategy"], r["pin_fraction"]), r["cold"])
                    for r in rows)
    print(f"{name}: {len(rows)} rows, {len({r['prompt_id'] for r in rows})} prompts")
    for (limit, strategy, cold), n in sorted(cells.items(), key=lambda kv: (nsdata.limit_mb(kv[0][0]), kv[0][1])):
        print(f"  memory_limit {limit:>5}  strategy {strategy:<9} cold {str(cold):<5}  rows {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", type=Path, default=nsdata.RAW / "pilot.jsonl")
    ap.add_argument("--final", type=Path, default=nsdata.RAW / "final.jsonl")
    ap.add_argument("--out-dir", type=Path, default=nsdata.REPO / "results/processed")
    args = ap.parse_args()

    train, test, counts = build(args.pilot, args.final)
    write_csv(args.out_dir / "train.csv", train)
    write_csv(args.out_dir / "test.csv", test)

    print(f"pilot: {counts['pilot_records']} runs (last record per run_id), "
          f"{counts['pilot_errors_dropped']} with an error dropped, "
          f"{counts['pilot_prefetch_dropped']} prefetch rows dropped (single whole-file madvise)")
    print(f"final: {counts['final_records']} runs, {counts['final_errors_dropped']} with an error dropped")
    print("pilot_tight (1000M) is not used.")
    describe("train (pilot)", train)
    describe("test (final)", test)
    print("note: pilot 'none' runs were warm (no cache drop); all final runs, 'none' included, were cold. "
          "The 'none' rows of train and test therefore differ in starting cache state.")
    print("prompt overlap train/test: 0")
    print(f"wrote {args.out_dir / 'train.csv'} and {args.out_dir / 'test.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

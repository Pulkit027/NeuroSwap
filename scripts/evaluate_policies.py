#!/usr/bin/env python3
"""Score strategy-selection policies on the held-out test set (final batch).

Policies: always-baseline, always-pin0.25, always-prefetch, rule-based
(pin0.25 if memory-limited and prompt_tokens > threshold, threshold chosen on
TRAIN only) and ML (lowest random-forest prediction among baseline and pin0.25,
the conditions seen in training).

Every policy is scored with MEASURED median total_wall_s per (prompt, limit,
condition) from the test set, setup cost included. Only numbers are printed.
Output: results/policies/policy_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

import nsdata
import policies


def summarize(name, scored, conditions):
    row = {"policy": name, "cells": len(scored), "mean_s": scored["latency"].mean()}
    for limit in sorted(scored["memory_limit"].unique(), key=nsdata.limit_mb):
        row[f"mean_s_{limit}"] = scored[scored["memory_limit"] == limit]["latency"].mean()
    for label in conditions:
        row[f"picked_{label}"] = int((scored["chosen"] == label).sum())
    diff = scored["diff_vs_baseline"]
    row.update({"mean_diff_vs_baseline_s": diff.mean(), "median_diff_vs_baseline_s": diff.median(),
                "cells_lower": int((diff < 0).sum()), "cells_higher": int((diff > 0).sum()),
                "cells_equal": int((diff == 0).sum())})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=nsdata.REPO / "results/processed")
    ap.add_argument("--models", type=Path, default=nsdata.REPO / "results/ml/models.joblib")
    ap.add_argument("--out", type=Path, default=nsdata.REPO / "results/policies/policy_results.json")
    args = ap.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    saved = joblib.load(args.models)
    forest = saved["models"]["random_forest"]

    threshold, threshold_table = policies.choose_threshold(train)
    print("== Rule threshold candidates on TRAIN (mean measured total_wall_s over train cells) ==")
    print(threshold_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"chosen threshold (train only): prompt_tokens > {threshold} -> pin0.25 when memory-limited")
    pinned_on_train = int(threshold_table.loc[threshold_table["threshold"] == threshold, "pinned_cells"].item())
    if threshold >= train["prompt_tokens"].max():
        print(f"note: {threshold} is the largest train prompt_tokens, so on train the rule pins "
              f"{pinned_on_train} cells (it is 'never pin' on train); on test it pins only cells with "
              f"more tokens than any train prompt.")

    measured = policies.measured_medians(test)
    test_cells = policies.cells(test)
    conditions = ["baseline", "pin0.25", "prefetch"]
    policy_set = {
        "always-baseline": policies.always("baseline"),
        "always-pin0.25": policies.always("pin0.25"),
        "always-prefetch": policies.always("prefetch"),
        f"rule (tokens > {threshold})": policies.rule_policy(threshold),
        "ml (random forest)": policies.ml_policy(forest),
    }
    scored = {name: policies.score(test_cells, policy, measured) for name, policy in policy_set.items()}
    summary = pd.DataFrame([summarize(name, s, conditions) for name, s in scored.items()])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(f"\n== Policies on the held-out TEST set: {len(test_cells)} cells (prompt x limit), "
          f"latency = measured median total_wall_s over repetitions, setup included ==")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    per_cell = test_cells.copy()
    per_cell["baseline_s"] = scored["always-baseline"]["latency"].to_numpy()
    for name, s in scored.items():
        if name != "always-baseline":
            per_cell[f"{name}: chosen"] = s["chosen"].to_numpy()
            per_cell[f"{name}: diff_s"] = s["diff_vs_baseline"].to_numpy()
    print("\n== Paired per prompt and limit: policy latency minus always-baseline latency (s) ==")
    print(per_cell.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    base_mean = summary.loc[summary["policy"] == "always-baseline", "mean_s"].item()
    lower = summary[(summary["policy"] != "always-baseline") & (summary["mean_s"] < base_mean)]
    print()
    if lower.empty:
        statement = ("No policy has a lower mean measured total_wall_s than always-baseline on the test set "
                     f"(always-baseline mean {base_mean:.3f} s).")
    else:
        statement = ("Policies with a lower mean measured total_wall_s than always-baseline "
                     f"({base_mean:.3f} s) on the test set: " +
                     ", ".join(f"{r.policy} ({r.mean_s:.3f} s)" for r in lower.itertuples()) + ".")
    print(statement)
    for limit in sorted(test_cells["memory_limit"].unique(), key=nsdata.limit_mb):
        column = f"mean_s_{limit}"
        base = summary.loc[summary["policy"] == "always-baseline", column].item()
        lower_limit = summary[(summary["policy"] != "always-baseline") & (summary[column] < base)]["policy"].tolist()
        print(f"  limit {limit}: always-baseline mean {base:.3f} s; lower mean: "
              f"{', '.join(lower_limit) if lower_limit else 'none'}")
    labels = [nsdata.strategy_label(s, f) for s, f in zip(test["strategy"], test["pin_fraction"])]
    spread = (test.assign(label=labels).groupby(["prompt_id", "memory_limit", "label"])["total_wall_s"]
              .agg(lambda v: v.max() - v.min()))
    reps = int(test.groupby(["prompt_id", "memory_limit", "strategy"]).size().min())
    print(f"repetition spread on test (max - min of total_wall_s over the {reps} repetitions of one cell and "
          f"condition, {len(spread)} cell-conditions): median {spread.median():.3f} s, "
          f"max {spread.max():.3f} s")
    for limit in sorted(test_cells["memory_limit"].unique(), key=nsdata.limit_mb):
        part = spread[spread.index.get_level_values("memory_limit") == limit]
        print(f"  limit {limit}: median {part.median():.3f} s, max {part.max():.3f} s")
    print("notes: no significance test was run; each cell has only the repetitions above. ML chooses only "
          "between baseline and pin0.25: prefetch is not in train and pin0.5 is not in test. The "
          "CV-selected model (median per bucket+limit) ignores strategy, so the strategy-aware random "
          "forest is used for the ML policy. Train 'none' runs were warm, test runs cold.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": "neuroswap_measured",
        "scored_on": "results/processed/test.csv (final batch, held out)",
        "latency": "median total_wall_s over repetitions per (prompt, limit, condition), setup included",
        "rule_threshold_prompt_tokens": int(threshold),
        "rule_threshold_candidates_train": threshold_table.to_dict("records"),
        "summary": summary.to_dict("records"),
        "per_cell": per_cell.to_dict("records"),
        "statement": statement,
        "repetition_spread_s": {"median": float(spread.median()), "max": float(spread.max()),
                                "repetitions_per_cell": reps},
    }, indent=2, default=float) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

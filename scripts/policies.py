"""Strategy-selection policies and their scorer (used by scripts/evaluate_policies.py).

A cell is one (prompt_id, memory_limit) with its bucket and prompt_tokens. A
policy maps a cell to a condition label: "baseline", "prefetch" or "pin0.25".
Policies are scored with MEASURED latencies only: the median total_wall_s over
repetitions of the chosen (prompt, limit, condition), setup cost included.
"""

import pandas as pd

import nsdata

PIN = "pin0.25"


def measured_medians(frame):
    """{(prompt_id, memory_limit, label): median total_wall_s} of a train/test table."""
    labels = [nsdata.strategy_label(s, f) for s, f in zip(frame["strategy"], frame["pin_fraction"])]
    grouped = frame.assign(label=labels).groupby(["prompt_id", "memory_limit", "label"])["total_wall_s"].median()
    return grouped.to_dict()


def cells(frame):
    """One row per (prompt_id, memory_limit) with bucket and prompt_tokens, in a stable order."""
    out = frame[["prompt_id", "memory_limit", "bucket", "prompt_tokens"]].drop_duplicates()
    return out.sort_values(["memory_limit", "prompt_id"]).reset_index(drop=True)


def always(label):
    return lambda cell: label


def rule_policy(threshold):
    """Review-2 controller idea: pin a memory-limited, memory-sensitive run; here
    'sensitive' means prompt_tokens above a threshold chosen on train."""
    def choose(cell):
        return PIN if cell["memory_limit"] != "none" and cell["prompt_tokens"] > threshold else "baseline"
    return choose


def choose_threshold(train):
    """Threshold for rule_policy with the lowest mean measured train latency.
    Candidates: -1 (pin every limited cell) and every distinct train prompt_tokens
    value (the largest one means never pin). Ties go to the higher threshold,
    i.e. fewer pinned cells. Returns (threshold, table of candidates)."""
    measured = measured_medians(train)
    train_cells = cells(train)
    rows = []
    for threshold in [-1] + sorted(train["prompt_tokens"].unique().tolist()):
        scored = score(train_cells, rule_policy(threshold), measured)
        rows.append({"threshold": threshold, "mean_total_wall_s": scored["latency"].mean(),
                     "pinned_cells": int((scored["chosen"] == PIN).sum())})
    table = pd.DataFrame(rows)
    best = table["mean_total_wall_s"].min()
    threshold = table[table["mean_total_wall_s"] == best]["threshold"].max()
    return threshold, table


def ml_policy(model, candidates=("baseline", PIN)):
    """Pick the candidate condition with the lowest predicted total_wall_s (ties: first
    candidate). Only conditions seen in training are offered."""
    def choose(cell):
        rows = []
        for label in candidates:
            strategy, fraction = ("pin", float(label[3:])) if label.startswith("pin") else (label, 0.0)
            rows.append({"prompt_id": cell["prompt_id"], "bucket": cell["bucket"],
                         "prompt_tokens": cell["prompt_tokens"], "memory_limit": cell["memory_limit"],
                         "strategy": strategy, "pin_fraction": fraction})
        predicted = model.predict(pd.DataFrame(rows))
        return candidates[int(min(range(len(candidates)), key=lambda i: predicted[i]))]
    return choose


def score(cell_table, policy, measured):
    """Per cell: the chosen condition, its measured median latency, the measured
    always-baseline latency and the difference (chosen - baseline)."""
    rows = []
    for cell in cell_table.to_dict("records"):
        chosen = policy(cell)
        key = (cell["prompt_id"], cell["memory_limit"], chosen)
        base_key = (cell["prompt_id"], cell["memory_limit"], "baseline")
        if key not in measured or base_key not in measured:
            raise KeyError(f"no measured runs for {key if key not in measured else base_key}")
        rows.append({**cell, "chosen": chosen, "latency": measured[key],
                     "baseline_latency": measured[base_key], "diff_vs_baseline": measured[key] - measured[base_key]})
    return pd.DataFrame(rows)

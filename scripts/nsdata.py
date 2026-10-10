"""Shared loading of run_experiment.py output (results/raw/llm/*.jsonl).

Standard library only, so every script and test can import it.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results/raw/llm"
BUCKET_ORDER = ["<64", "64-256", "256-1024", "1024-1500"]
NONE_LIMIT_MB = 1_000_000  # memory_limit "none" as a number: far above any real limit


def load_last_records(path):
    """Records of a jsonl file, keeping the last record per run_id (a retried run
    appends a new line). A trailing incomplete line is skipped; any other bad line
    raises ValueError. Records without run_id are kept as they are."""
    lines = Path(path).read_text().split("\n")
    by_id, no_id = {}, []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            if number == len(lines):
                continue
            raise ValueError(f"{path}: line {number} is not valid JSON")
        if "run_id" in record:
            by_id.pop(record["run_id"], None)  # keep file order of the last occurrence
            by_id[record["run_id"]] = record
        else:
            no_id.append(record)
    return list(by_id.values()) + no_id


def pin_fraction(record):
    """Pin fraction of a run (helper_report.fraction); 0.0 when nothing is pinned."""
    if record.get("strategy") != "pin":
        return 0.0
    return float((record.get("helper_report") or {}).get("fraction") or 0.0)


def limit_mb(memory_limit):
    """cgroup memory.max string ("1100M", "1G", "none") in MiB."""
    if memory_limit == "none":
        return NONE_LIMIT_MB
    units = {"K": 1 / 1024, "M": 1, "G": 1024}
    if memory_limit[-1] in units:
        return float(memory_limit[:-1]) * units[memory_limit[-1]]
    return float(memory_limit) / (1024 * 1024)


def strategy_label(strategy, fraction):
    """'baseline', 'prefetch' or 'pin0.25' style label of a condition."""
    return f"pin{fraction:g}" if strategy == "pin" else strategy

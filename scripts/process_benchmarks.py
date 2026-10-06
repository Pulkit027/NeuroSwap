#!/usr/bin/env python3
"""Parse real NeuroSwap benchmark outputs into run-level and aggregate CSVs."""

import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"

METRICS = {
    "execution_time_ms": r"Execution time: ([0-9]+(?:\.[0-9]+)?) ms",
    "minor_page_faults": r"Minor page faults: ([0-9]+)",
    "major_page_faults": r"Major page faults: ([0-9]+)",
    "max_rss_kib": r"Max RSS: ([0-9]+) KiB",
}


def extract_metrics(text: str, source: Path) -> dict:
    values = {}
    for name, pattern in METRICS.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"{source.name}: missing {name}")
        values[name] = float(match.group(1)) if name == "execution_time_ms" else int(match.group(1))
    return values


def read_single(experiment: str, run_number: int, path: Path) -> dict:
    record = {"experiment": experiment, "run_number": run_number}
    record.update(extract_metrics(path.read_text(), path))
    return record


def read_prefetch(run_number: int, path: Path) -> list[dict]:
    text = path.read_text()
    without = re.search(r"--- Without prefetch hint ---\n(.*?)(?=--- With MADV_WILLNEED ---)", text, re.S)
    with_hint = re.search(r"--- With MADV_WILLNEED ---\n(.*)", text, re.S)
    if not without or not with_hint:
        raise ValueError(f"{path.name}: missing paired prefetch sections")
    output = []
    for experiment, section in (("file_backed_no_hint", without.group(1)),
                                ("file_backed_willneed", with_hint.group(1))):
        record = {"experiment": experiment, "run_number": run_number}
        record.update(extract_metrics(section, path))
        output.append(record)
    return output


def main() -> None:
    repetitions = int(sys.argv[1]) if len(sys.argv) == 2 else 5
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    records = []
    for run in range(1, repetitions + 1):
        suffix = f"{run:02d}"
        records.append(read_single("baseline", run, RAW / f"baseline_run_{suffix}.txt"))
        records.append(read_single("pinned", run, RAW / f"pinned_run_{suffix}.txt"))
        records.extend(read_prefetch(run, RAW / f"prefetch_run_{suffix}.txt"))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    run_fields = ["experiment", "run_number", "execution_time_ms", "minor_page_faults", "major_page_faults", "max_rss_kib"]
    with (PROCESSED / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(records)

    statistics_rows = []
    experiment_order = ["baseline", "pinned", "file_backed_no_hint", "file_backed_willneed"]
    for experiment in experiment_order:
        rows = [row for row in records if row["experiment"] == experiment]
        if len(rows) != repetitions:
            raise ValueError(f"{experiment}: expected {repetitions} records, found {len(rows)}")
        timings = [row["execution_time_ms"] for row in rows]
        statistics_rows.append({
            "experiment": experiment,
            "mean_execution_time_ms": f"{statistics.mean(timings):.3f}",
            "min_execution_time_ms": f"{min(timings):.3f}",
            "max_execution_time_ms": f"{max(timings):.3f}",
            "mean_minor_faults": f"{statistics.mean(row['minor_page_faults'] for row in rows):.3f}",
            "mean_major_faults": f"{statistics.mean(row['major_page_faults'] for row in rows):.3f}",
            "mean_max_rss_kib": f"{statistics.mean(row['max_rss_kib'] for row in rows):.3f}",
        })
    stat_fields = list(statistics_rows[0])
    with (PROCESSED / "benchmark_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stat_fields)
        writer.writeheader()
        writer.writerows(statistics_rows)

    print(f"Verified {len(records)} CSV records against {repetitions * 3} raw benchmark files.")


if __name__ == "__main__":
    main()

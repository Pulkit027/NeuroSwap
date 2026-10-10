#!/usr/bin/env python3
"""Export dashboard data to results/dashboard/*.json.

Every file has: source ("neuroswap_measured"), kind ("measured",
"derived_from_measured" or "model_output"), generated_at, inputs, and data.
Only quantities that were measured, or computed from measured runs, are
exported; model predictions are marked as model_output. No prompt text.

Inputs: results/raw/llm/{pilot,pilot_tight,final}.jsonl (and .status.json),
results/ml/{metrics,feature_importance}.json, results/ml/predictions.csv,
results/processed/test.csv, results/policies/policy_results.json.
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import nsdata

PLANS = ["pilot", "pilot_tight", "final"]
PILOT_PREFETCH_NOTE = "pilot prefetch: one whole-file madvise, ~0.75% resident; not a test of prefetch"


def clean(value):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(out_dir, name, kind, inputs, data):
    payload = {"source": "neuroswap_measured", "kind": kind,
               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "inputs": inputs, "data": clean(data)}
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return path


def run_rows(plan, records):
    rows = []
    for r in records:
        helper = r.get("helper_report") or {}
        wait = helper.get("residency_wait") or {}
        prefetch_note = None
        if r.get("strategy") == "prefetch" and plan in ("pilot", "pilot_tight"):
            prefetch_note = PILOT_PREFETCH_NOTE
        rows.append({
            "run_id": r.get("run_id"), "plan_name": plan, "prompt_id": r.get("prompt_id"),
            "bucket": r.get("bucket"), "prompt_tokens": r.get("prompt_tokens"),
            "memory_limit": r.get("memory_limit"), "strategy": r.get("strategy"),
            "pin_fraction": nsdata.pin_fraction(r), "repetition": r.get("run_index"), "cold": r.get("cold"),
            "error": r.get("error"),
            "total_wall_s": r.get("total_wall_s"), "inference_wall_s": r.get("inference_wall_s"),
            "strategy_setup_ms": r.get("strategy_setup_ms"),
            "ru_majflt": (r.get("rusage") or {}).get("ru_majflt"),
            "ru_maxrss_kb": (r.get("rusage") or {}).get("ru_maxrss_kb"),
            "eval_tps": (r.get("llama_perf") or {}).get("eval_tps"),
            "prefetch_final_percent": wait.get("final_percent_file", (wait.get("residency") or {}).get("percent_resident")),
            "note": prefetch_note,
        })
    return rows


def comparison(frame):
    ok = frame[frame["error"].isna()].copy()
    ok["condition"] = [nsdata.strategy_label(s, f) for s, f in zip(ok["strategy"], ok["pin_fraction"])]
    rows = []
    for (plan, bucket, limit, condition), g in ok.groupby(["plan_name", "bucket", "memory_limit", "condition"]):
        total = g["total_wall_s"]
        rows.append({"plan_name": plan, "bucket": bucket, "memory_limit": limit, "condition": condition,
                     "n": int(len(g)), "prompts": int(g["prompt_id"].nunique()), "cold": sorted(set(g["cold"])),
                     "total_wall_s_median": total.median(),
                     "total_wall_s_iqr": total.quantile(0.75) - total.quantile(0.25),
                     "total_wall_s_min": total.min(), "total_wall_s_max": total.max(),
                     "inference_wall_s_median": g["inference_wall_s"].median(),
                     "strategy_setup_ms_median": g["strategy_setup_ms"].median(),
                     "ru_majflt_median": g["ru_majflt"].median(),
                     "note": PILOT_PREFETCH_NOTE if condition == "prefetch" and plan != "final" else None})
    order = {b: i for i, b in enumerate(nsdata.BUCKET_ORDER)}
    return sorted(rows, key=lambda r: (PLANS.index(r["plan_name"]), order.get(r["bucket"], 9),
                                       -nsdata.limit_mb(r["memory_limit"]), r["condition"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=nsdata.REPO / "results/dashboard")
    args = ap.parse_args()
    repo = nsdata.REPO
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Runs from every plan that exists.
    all_rows, plan_info = [], {}
    for plan in PLANS:
        path = nsdata.RAW / f"{plan}.jsonl"
        if not path.exists():
            continue
        records = nsdata.load_last_records(path)
        all_rows += run_rows(plan, records)
        status_path = path.with_name(path.name + ".status.json")
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        plan_info[plan] = {"runs": len(records), "errors": sum(r.get("error") is not None for r in records),
                           "prompts": len({r.get("prompt_id") for r in records}),
                           "memory_limits": sorted({r.get("memory_limit") for r in records}, key=nsdata.limit_mb),
                           "conditions": sorted({nsdata.strategy_label(r.get("strategy"), nsdata.pin_fraction(r))
                                                 for r in records}),
                           "cold": sorted({r.get("cold") for r in records}),
                           "session_started_at": status.get("started_at"), "session_updated_at": status.get("updated_at"),
                           "llama_cpp_commit": sorted({r.get("llama_cpp_commit") for r in records} - {None}),
                           "model_sha256": sorted({r.get("model_sha256") for r in records} - {None}),
                           "repack": sorted({r.get("repack") for r in records} - {None})}
    frame = pd.DataFrame(all_rows)
    raw_inputs = [f"results/raw/llm/{p}.jsonl" for p in plan_info]

    ml_metrics = json.loads((repo / "results/ml/metrics.json").read_text())
    importance = json.loads((repo / "results/ml/feature_importance.json").read_text())
    policy = json.loads((repo / "results/policies/policy_results.json").read_text())

    written.append(write(args.out_dir, "overview", "derived_from_measured", raw_inputs + [
        "results/ml/metrics.json", "results/policies/policy_results.json"], {
        "plans": plan_info,
        "ml": {"train_rows": ml_metrics["train_rows"], "test_rows": ml_metrics["test_rows"],
               "selected_model": ml_metrics["selected_model"], "selection": ml_metrics["selection"]},
        "policy_statement": policy["statement"],
        "notes": ["pilot 'none' runs were warm; final runs, 'none' included, were cold",
                  PILOT_PREFETCH_NOTE,
                  "final prefetch used 8 MiB madvise chunks",
                  "pilot data may be used for training; final prompts are the held-out test set"]}))
    written.append(write(args.out_dir, "experiments", "measured", raw_inputs, all_rows))
    written.append(write(args.out_dir, "strategy_comparison", "derived_from_measured", raw_inputs, comparison(frame)))
    written.append(write(args.out_dir, "ml_metrics", "derived_from_measured", ["results/ml/metrics.json"], ml_metrics))
    written.append(write(args.out_dir, "feature_importance", "model_output", ["results/ml/feature_importance.json"],
                         importance))

    predictions = pd.read_csv(repo / "results/ml/predictions.csv")
    test = pd.read_csv(repo / "results/processed/test.csv")
    joined = predictions.merge(test[["run_id", "prompt_id", "bucket", "prompt_tokens", "memory_limit",
                                     "strategy", "pin_fraction"]], on="run_id", how="left")
    written.append(write(args.out_dir, "predictions", "model_output",
                         ["results/ml/predictions.csv", "results/processed/test.csv"],
                         {"model": ml_metrics["selected_model"], "actual": "measured total_wall_s (s)",
                          "predicted": "model output (s)", "rows": joined.to_dict("records")}))
    written.append(write(args.out_dir, "policy_results", "derived_from_measured",
                         ["results/policies/policy_results.json"], policy))

    for path in written:
        payload = json.loads(path.read_text())
        size = len(payload["data"]) if isinstance(payload["data"], list) else len(payload["data"].get("rows", payload["data"]))
        shown = path.relative_to(repo) if path.is_relative_to(repo) else path
        print(f"wrote {shown}  kind={payload['kind']}  entries={size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

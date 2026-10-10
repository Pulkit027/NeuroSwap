#!/usr/bin/env python3
"""Train and compare latency models (target total_wall_s).

Models (scripts/nsmodels.py): (a) train median per bucket + limit, (b) linear
regression on prompt_tokens per limit, (c) random forest (fixed seed).
The model is chosen by leave-one-prompt-out cross-validation on TRAIN only
(grouped by prompt_id); the test set (final batch) is used only to report.

Test caveats printed with the results:
- prefetch never occurs in train (pilot prefetch rows were excluded), so test
  prefetch rows are an unseen strategy and are reported separately;
- pilot "none" runs were warm, all final runs are cold.

Outputs in results/ml/: models.joblib, metrics.json, feature_importance.json,
predictions.csv (run_id, actual, predicted) for the selected model.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

import nsdata
import nsmodels

TARGET = nsmodels.TARGET


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if len(actual) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    return {"n": int(len(actual)),
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None}


def cross_validate(make_model, train):
    """Out-of-fold predictions with one fold per training prompt."""
    predicted = np.empty(len(train))
    for fit_idx, held_idx in LeaveOneGroupOut().split(train, groups=train["prompt_id"]):
        model = make_model().fit(train.iloc[fit_idx])
        predicted[held_idx] = model.predict(train.iloc[held_idx])
    return predicted


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def print_table(title, rows, columns):
    print(f"\n== {title} ==")
    frame = pd.DataFrame(rows, columns=columns)
    print(frame.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=nsdata.REPO / "results/processed")
    ap.add_argument("--out-dir", type=Path, default=nsdata.REPO / "results/ml")
    args = ap.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    if set(train["prompt_id"]) & set(test["prompt_id"]):
        raise SystemExit("train and test share prompts")
    print(f"train: {len(train)} rows, {train['prompt_id'].nunique()} prompts; "
          f"test: {len(test)} rows, {test['prompt_id'].nunique()} prompts")
    print("train strategies:", ", ".join(sorted(nsdata.strategy_label(s, f) for s, f in
                                                 set(zip(train["strategy"], train["pin_fraction"])))),
          "| test strategies:", ", ".join(sorted(nsdata.strategy_label(s, f) for s, f in
                                                 set(zip(test["strategy"], test["pin_fraction"])))))

    factories = {m.name: type(m) for m in nsmodels.all_models()}

    # 1. Cross-validation on train only.
    cv = {}
    for name, factory in factories.items():
        cv[name] = metrics(train[TARGET], cross_validate(factory, train))
    selected = min(cv, key=lambda name: cv[name]["mae"])
    print_table(f"Leave-one-prompt-out CV on TRAIN ({train['prompt_id'].nunique()} folds), target total_wall_s (s)",
                [[n, m["n"], fmt(m["mae"]), fmt(m["rmse"]), fmt(m["r2"])] for n, m in cv.items()],
                ["model", "n", "MAE", "RMSE", "R2"])
    print(f"selected by lowest CV MAE: {selected}")

    # 2. Fit on all of train, evaluate on test (reporting only).
    fitted = {name: factory().fit(train) for name, factory in factories.items()}
    known = test["strategy"].isin(["baseline", "pin"])
    subsets = {"all test rows": np.ones(len(test), bool),
               "baseline + pin rows": known.to_numpy(),
               "prefetch rows (unseen strategy)": (~known).to_numpy()}
    for limit in sorted(test["memory_limit"].unique(), key=nsdata.limit_mb):
        subsets[f"baseline + pin rows, limit {limit}"] = (known & (test["memory_limit"] == limit)).to_numpy()
    test_metrics, test_rows = {}, []
    for name, model in fitted.items():
        predicted = model.predict(test)
        test_metrics[name] = {}
        for subset, mask in subsets.items():
            m = metrics(test[TARGET][mask], predicted[mask])
            test_metrics[name][subset] = m
            test_rows.append([name, subset, m["n"], fmt(m["mae"]), fmt(m["rmse"]), fmt(m["r2"])])
    print_table("Held-out TEST (final batch), target total_wall_s (s)", test_rows,
                ["model", "rows", "n", "MAE", "RMSE", "R2"])
    print("note: prefetch does not occur in train, so the prefetch rows are an unseen strategy for every model;")
    print("      models (a) and (b) ignore strategy, the forest sees prefetch only as an all-zero-trained one-hot column.")
    print("note: train 'none' rows were warm (pilot), test 'none' rows are cold (final).")

    # 3. Forest feature importances (impurity-based, fitted on train).
    importances = fitted["random_forest"].importances()
    print_table("Random forest feature importances (impurity-based, train fit)",
                [[k, f"{v:.4f}"] for k, v in sorted(importances.items(), key=lambda kv: -kv[1])],
                ["feature", "importance"])
    print("note: strategy_prefetch is all zero in train, so its importance is 0 by construction.")

    # 4. Save.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "selected": selected, "features": nsmodels.FEATURES,
                 "target": TARGET, "trained_on": "results/processed/train.csv"},
                args.out_dir / "models.joblib")
    predictions = pd.DataFrame({"run_id": test["run_id"], "actual": test[TARGET],
                                "predicted": fitted[selected].predict(test)})
    predictions.to_csv(args.out_dir / "predictions.csv", index=False, float_format="%.4f")
    (args.out_dir / "metrics.json").write_text(json.dumps({
        "source": "neuroswap_measured", "target": TARGET, "selected_model": selected,
        "selection": "lowest leave-one-prompt-out CV MAE on train",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "cv_train": cv, "test": test_metrics,
        "notes": ["prefetch rows in test are an unseen strategy (no prefetch in train)",
                  "train 'none' rows were warm, test rows are all cold"]}, indent=2) + "\n")
    (args.out_dir / "feature_importance.json").write_text(json.dumps({
        "source": "neuroswap_measured", "model": "random_forest", "method": "impurity-based, train fit",
        "importances": importances}, indent=2) + "\n")
    print(f"\nwrote {args.out_dir}/models.joblib, metrics.json, feature_importance.json, "
          f"predictions.csv (selected model: {selected})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

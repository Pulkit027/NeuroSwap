"""Latency models for scripts/train_ml.py and the ML policy (picklable, module-level).

Every model has fit(frame) and predict(frame) on DataFrames with the columns of
results/processed/train.csv; the target is total_wall_s.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import nsdata

STRATEGIES = ["baseline", "pin", "prefetch"]
FEATURES = ["prompt_tokens", "memory_limit_mb"] + [f"strategy_{s}" for s in STRATEGIES] + ["pin_fraction"]
TARGET = "total_wall_s"


def features(frame):
    """Feature matrix: prompt_tokens, memory_limit in MiB (none = 1,000,000),
    strategy one-hot over a fixed category list, pin_fraction (0 when not pinning)."""
    out = pd.DataFrame(index=frame.index)
    out["prompt_tokens"] = frame["prompt_tokens"].astype(float)
    out["memory_limit_mb"] = frame["memory_limit"].map(nsdata.limit_mb).astype(float)
    for strategy in STRATEGIES:
        out[f"strategy_{strategy}"] = (frame["strategy"] == strategy).astype(float)
    out["pin_fraction"] = frame["pin_fraction"].astype(float)
    return out[FEATURES]


class BucketLimitMedian:
    """(a) Trivial baseline: train median of total_wall_s per (bucket, memory_limit);
    falls back to the limit median, then the overall median. Ignores strategy."""
    name = "median_bucket_limit"

    def fit(self, frame):
        self.cell = frame.groupby(["bucket", "memory_limit"])[TARGET].median().to_dict()
        self.limit = frame.groupby("memory_limit")[TARGET].median().to_dict()
        self.overall = float(frame[TARGET].median())
        return self

    def predict(self, frame):
        return np.array([self.cell.get((b, l), self.limit.get(l, self.overall))
                         for b, l in zip(frame["bucket"], frame["memory_limit"])], dtype=float)


class PerLimitLinear:
    """(b) Linear regression of total_wall_s on prompt_tokens, one fit per memory_limit
    (a limit unseen in training uses a fit on all rows). Ignores strategy."""
    name = "linear_tokens_per_limit"

    def fit(self, frame):
        self.models = {limit: LinearRegression().fit(group[["prompt_tokens"]].astype(float), group[TARGET])
                       for limit, group in frame.groupby("memory_limit")}
        self.fallback = LinearRegression().fit(frame[["prompt_tokens"]].astype(float), frame[TARGET])
        return self

    def predict(self, frame):
        out = np.empty(len(frame))
        for i, (tokens, limit) in enumerate(zip(frame["prompt_tokens"], frame["memory_limit"])):
            model = self.models.get(limit, self.fallback)
            out[i] = model.predict(pd.DataFrame({"prompt_tokens": [float(tokens)]}))[0]
        return out


class Forest:
    """(c) RandomForestRegressor on FEATURES, fixed seed, default hyperparameters."""
    name = "random_forest"

    def __init__(self, seed=42, n_estimators=300):
        self.seed = seed
        self.n_estimators = n_estimators

    def fit(self, frame):
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=1)
        self.model.fit(features(frame), frame[TARGET])
        return self

    def predict(self, frame):
        return self.model.predict(features(frame))

    def importances(self):
        return dict(zip(FEATURES, self.model.feature_importances_))


def all_models():
    return [BucketLimitMedian(), PerLimitLinear(), Forest()]

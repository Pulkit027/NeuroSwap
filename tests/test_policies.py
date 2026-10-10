"""Tests for scripts/policies.py on a tiny hand-made fixture.

Run: .venv/bin/python -m unittest tests/test_policies.py -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import policies  # noqa: E402


def table(rows):
    """rows: (prompt, tokens, limit, strategy, fraction, [latencies per repetition])"""
    out = []
    for prompt, tokens, limit, strategy, fraction, latencies in rows:
        for latency in latencies:
            out.append({"prompt_id": prompt, "bucket": "<64", "prompt_tokens": tokens, "memory_limit": limit,
                        "strategy": strategy, "pin_fraction": fraction, "total_wall_s": latency})
    return pd.DataFrame(out)


# short: 10 tokens, long: 500 tokens. At 1100M pinning helps only the long prompt.
FIXTURE = table([
    ("short", 10, "none", "baseline", 0.0, [5.0, 7.0]),     # median 6
    ("short", 10, "none", "pin", 0.25, [6.0, 6.0]),         # 6
    ("short", 10, "none", "prefetch", 0.0, [9.0, 9.0]),     # 9
    ("short", 10, "1100M", "baseline", 0.0, [8.0, 8.0]),    # 8
    ("short", 10, "1100M", "pin", 0.25, [9.0, 11.0]),       # 10
    ("short", 10, "1100M", "prefetch", 0.0, [8.0, 8.0]),    # 8
    ("long", 500, "none", "baseline", 0.0, [20.0, 20.0]),   # 20
    ("long", 500, "none", "pin", 0.25, [21.0, 21.0]),       # 21
    ("long", 500, "none", "prefetch", 0.0, [20.0, 22.0]),   # 21
    ("long", 500, "1100M", "baseline", 0.0, [50.0, 60.0]),  # 55
    ("long", 500, "1100M", "pin", 0.25, [40.0, 40.0]),      # 40
    ("long", 500, "1100M", "prefetch", 0.0, [55.0, 55.0]),  # 55
])


class FixedModel:
    """Predicts from a lookup, to test the ML policy without training."""
    def __init__(self, lookup):
        self.lookup = lookup

    def predict(self, frame):
        return [self.lookup[(r.prompt_id, r.memory_limit, r.strategy)] for r in frame.itertuples()]


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.measured = policies.measured_medians(FIXTURE)
        self.cells = policies.cells(FIXTURE)

    def test_measured_medians(self):
        self.assertEqual(self.measured[("short", "none", "baseline")], 6.0)
        self.assertEqual(self.measured[("long", "1100M", "baseline")], 55.0)
        self.assertEqual(self.measured[("short", "1100M", "pin0.25")], 10.0)
        self.assertEqual(len(self.cells), 4)

    def test_score_always_policies(self):
        base = policies.score(self.cells, policies.always("baseline"), self.measured)
        self.assertAlmostEqual(base["latency"].mean(), (6 + 8 + 20 + 55) / 4)
        self.assertTrue((base["diff_vs_baseline"] == 0).all())
        pin = policies.score(self.cells, policies.always("pin0.25"), self.measured)
        self.assertAlmostEqual(pin["latency"].mean(), (6 + 10 + 21 + 40) / 4)
        diffs = dict(zip(zip(pin["prompt_id"], pin["memory_limit"]), pin["diff_vs_baseline"]))
        self.assertEqual(diffs, {("short", "none"): 0.0, ("short", "1100M"): 2.0,
                                 ("long", "none"): 1.0, ("long", "1100M"): -15.0})

    def test_rule_policy_pins_only_limited_long_prompts(self):
        rule = policies.score(self.cells, policies.rule_policy(100), self.measured)
        chosen = dict(zip(zip(rule["prompt_id"], rule["memory_limit"]), rule["chosen"]))
        self.assertEqual(chosen, {("short", "none"): "baseline", ("short", "1100M"): "baseline",
                                  ("long", "none"): "baseline", ("long", "1100M"): "pin0.25"})
        self.assertAlmostEqual(rule["latency"].mean(), (6 + 8 + 20 + 40) / 4)

    def test_threshold_chosen_on_given_data_with_ties_to_higher(self):
        threshold, candidates = policies.choose_threshold(FIXTURE)
        # Any threshold in [10, 500) pins only the long 1100M cell; the higher one wins the tie.
        self.assertEqual(threshold, 10)
        self.assertEqual(int(candidates.loc[candidates["threshold"] == 10, "pinned_cells"].item()), 1)
        self.assertGreater(candidates.loc[candidates["threshold"] == 500, "mean_total_wall_s"].item(),
                           candidates.loc[candidates["threshold"] == 10, "mean_total_wall_s"].item())

    def test_ml_policy_picks_lowest_prediction_among_candidates(self):
        lookup = {(p, l, s): 1.0 for p in ("short", "long") for l in ("none", "1100M") for s in ("baseline", "pin")}
        lookup[("long", "1100M", "pin")] = 0.5
        scored = policies.score(self.cells, policies.ml_policy(FixedModel(lookup)), self.measured)
        chosen = dict(zip(zip(scored["prompt_id"], scored["memory_limit"]), scored["chosen"]))
        self.assertEqual(chosen[("long", "1100M")], "pin0.25")
        self.assertEqual(chosen[("short", "none")], "baseline")  # tie -> first candidate
        self.assertNotIn("prefetch", set(scored["chosen"]))

    def test_missing_measurement_raises(self):
        partial = FIXTURE[FIXTURE["strategy"] != "prefetch"]
        with self.assertRaises(KeyError):
            policies.score(policies.cells(partial), policies.always("prefetch"), policies.measured_medians(partial))


if __name__ == "__main__":
    unittest.main()

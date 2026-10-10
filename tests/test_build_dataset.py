"""Tests for scripts/build_dataset.py: leakage and filtering rules.

Run: .venv/bin/python -m unittest tests/test_build_dataset.py -v
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import build_dataset  # noqa: E402
import run_experiment  # noqa: E402


def record(run_id, plan, prompt, strategy, limit="1100M", fraction=None, error=None, total=10.0):
    return {"run_id": run_id, "plan_name": plan, "prompt_id": prompt, "bucket": "<64", "prompt_tokens": 20,
            "memory_limit": limit, "strategy": strategy, "cold": limit != "none",
            "helper_report": {"fraction": fraction} if strategy != "baseline" else None,
            "total_wall_s": total, "inference_wall_s": total - 1, "strategy_setup_ms": 5.0,
            "rusage": {"ru_majflt": 42}, "error": error}


class BuildDatasetFixtureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, records):
        path = self.dir / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    def test_filters_and_fields(self):
        pilot = self.write("pilot.jsonl", [
            record("a", "pilot", "p1", "baseline"),
            record("b", "pilot", "p1", "prefetch"),
            record("c", "pilot", "p1", "pin", fraction=0.5, error="first attempt failed"),
            record("c", "pilot", "p1", "pin", fraction=0.5, total=7.5),  # retried: last record wins
            record("d", "pilot", "p2", "baseline", error="refused"),
        ])
        final = self.write("final.jsonl", [
            record("x", "final", "f1", "baseline"),
            record("y", "final", "f1", "prefetch"),
        ])
        train, test, counts = build_dataset.build(pilot, final)
        self.assertEqual([r["run_id"] for r in train], ["a", "c"])
        self.assertNotIn("prefetch", {r["strategy"] for r in train})
        self.assertEqual(counts["pilot_prefetch_dropped"], 1)
        self.assertEqual(counts["pilot_errors_dropped"], 1)
        pin = next(r for r in train if r["run_id"] == "c")
        self.assertEqual((pin["pin_fraction"], pin["total_wall_s"], pin["ru_majflt"]), (0.5, 7.5, 42))
        self.assertEqual(next(r for r in train if r["run_id"] == "a")["pin_fraction"], 0.0)
        self.assertEqual({r["source"] for r in train + test}, {"neuroswap_measured"})
        self.assertEqual({r["strategy"] for r in test}, {"baseline", "prefetch"})  # test keeps prefetch

    def test_final_prompt_in_pilot_is_refused(self):
        pilot = self.write("pilot.jsonl", [record("a", "pilot", "shared", "baseline")])
        final = self.write("final.jsonl", [record("x", "final", "shared", "baseline")])
        with self.assertRaises(SystemExit) as raised:
            build_dataset.build(pilot, final)
        self.assertIn("final (test) prompts found in train", str(raised.exception))

    def test_wrong_plan_refused(self):
        pilot = self.write("pilot.jsonl", [record("a", "pilot_tight", "p1", "baseline")])
        final = self.write("final.jsonl", [record("x", "final", "f1", "baseline")])
        with self.assertRaises(SystemExit):
            build_dataset.build(pilot, final)


@unittest.skipUnless((REPO / "results/processed/train.csv").exists(), "run scripts/build_dataset.py first")
class BuiltDatasetTest(unittest.TestCase):
    def rows(self, name):
        with open(REPO / "results/processed" / name, newline="") as f:
            return list(csv.DictReader(f))

    def test_no_final_prompt_in_train(self):
        final_plan = run_experiment.load_plan(REPO / "experiments/final.json")
        final_prompts = {pid for pid, _ in run_experiment.select_prompts(final_plan)}
        train_prompts = {r["prompt_id"] for r in self.rows("train.csv")}
        test_prompts = {r["prompt_id"] for r in self.rows("test.csv")}
        self.assertEqual(train_prompts & final_prompts, set())
        self.assertEqual(train_prompts & test_prompts, set())
        self.assertEqual(test_prompts, final_prompts)

    def test_no_pilot_prefetch_rows_in_train(self):
        self.assertNotIn("prefetch", {r["strategy"] for r in self.rows("train.csv")})


if __name__ == "__main__":
    unittest.main()

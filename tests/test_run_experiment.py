"""Tests for scripts/run_experiment.py with a fake run_one stub (no root needed).

Run: .venv/bin/python -m unittest tests/test_run_experiment.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts/run_experiment.py"
sys.path.insert(0, str(REPO / "scripts"))
import run_experiment  # noqa: E402

STUB = textwrap.dedent('''
    import argparse, json, os, signal, sys, time
    ap = argparse.ArgumentParser()
    for flag in ("--prompt-id", "--memory-limit", "--strategy", "--repack", "--out"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--run-index", type=int, required=True)
    ap.add_argument("--pin-fraction", type=float)
    a, _ = ap.parse_known_args()
    state = os.environ["STUB_STATE"]
    count_file = os.path.join(state, "count")
    n = int(open(count_file).read()) + 1 if os.path.exists(count_file) else 1
    open(count_file, "w").write(str(n))
    with open(os.path.join(state, "calls.jsonl"), "a") as f:
        f.write(json.dumps(vars(a)) + "\\n")
    rec = {"source": "neuroswap_measured", "prompt_id": a.prompt_id, "memory_limit": a.memory_limit,
           "strategy": a.strategy, "run_index": a.run_index, "error": None, "stub_call": n}
    if os.environ.get("STUB_NO_RECORD_AT") == str(n):
        sys.exit(2)
    if os.environ.get("STUB_INTERRUPT_AT") == str(n):
        open(a.out, "a").write(json.dumps(rec) + "\\n")  # like run_one's finally, then Ctrl+C
        os.kill(os.getppid(), signal.SIGINT)
        time.sleep(30)
    if os.environ.get("STUB_ERROR_STRATEGY") == a.strategy:
        rec["error"] = "stub error"
        open(a.out, "a").write(json.dumps(rec) + "\\n")
        sys.exit(1)
    open(a.out, "a").write(json.dumps(rec) + "\\n")
''')


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.stub = self.dir / "stub_run_one.py"
        self.stub.write_text(STUB)
        self.state = self.dir / "state"
        self.state.mkdir()
        self.out = self.dir / "out.jsonl"
        # 4 prompts x 1 limit x 2 strategies x 2 repetitions = 16 runs, no root needed.
        self.plan_path = self.dir / "plan.json"
        self.plan = {"plan_name": "stubtest", "prompts": {"per_bucket": 1, "seed": 42, "manifest": "data/manifest.csv"},
                     "memory_limits": ["none"],
                     "strategies": [{"strategy": "baseline"}, {"strategy": "pin", "fraction": 0.25}],
                     "repetitions": 2, "order_seed": 7, "repack": "off", "output": str(self.out)}
        self.plan_path.write_text(json.dumps(self.plan))

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, *extra, env=None, plan=None):
        full_env = {**os.environ, "STUB_STATE": str(self.state), **(env or {})}
        return subprocess.run([sys.executable, str(RUNNER), str(plan or self.plan_path), "--force",
                               "--run-one", str(self.stub), *extra],
                              capture_output=True, text=True, env=full_env, timeout=120)

    def records(self):
        return [json.loads(line) for line in self.out.read_text().splitlines()]

    def calls(self):
        path = self.state / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def expected_ids(self):
        return {r["run_id"] for r in run_experiment.expand(run_experiment.load_plan(self.plan_path))}

    def test_run_ids_stable_across_invocations(self):
        def dry_ids(plan):
            out = subprocess.run([sys.executable, str(RUNNER), str(plan), "--dry-run"],
                                 capture_output=True, text=True, check=True).stdout
            return [line.split()[1] for line in out.splitlines() if line.strip()[:1].isdigit()]
        for name in ("pilot", "pilot_tight"):
            plan = REPO / "experiments" / f"{name}.json"
            first, second = dry_ids(plan), dry_ids(plan)
            self.assertEqual(first, second)
            self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(dry_ids(REPO / "experiments/pilot.json")), 288)
        self.assertEqual(len(dry_ids(REPO / "experiments/pilot_tight.json")), 48)
        # The id depends only on the run's content, not on the order seed.
        plan = run_experiment.load_plan(self.plan_path)
        reordered = dict(plan, order_seed=8)
        a, b = run_experiment.expand(plan), run_experiment.expand(reordered)
        self.assertEqual({r["run_id"] for r in a}, {r["run_id"] for r in b})
        self.assertNotEqual([r["run_id"] for r in a], [r["run_id"] for r in b])
        self.assertEqual(run_experiment.run_id("wc0001", "1100M", "pin", 0.5, 2),
                         run_experiment.run_id("wc0001", "1100M", "pin", 0.50, 2))

    def test_dry_run_executes_nothing(self):
        result = self.runner("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("16 total, 0 already recorded, 16 to run", result.stdout)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.calls(), [])

    def test_full_run_adds_keys_and_status(self):
        result = self.runner()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        self.assertEqual({r["run_id"] for r in records}, self.expected_ids())
        self.assertEqual(len(records), 16)
        for r in records:
            self.assertEqual(r["plan_name"], "stubtest")
            self.assertEqual(r["source"], "neuroswap_measured")  # stub keys kept
        self.assertIn("[16/16]", result.stdout)
        self.assertIn("ETA", result.stdout)
        status = json.loads(self.out.with_name("out.jsonl.status.json").read_text())
        self.assertEqual((status["state"], status["done_this_session"]), ("finished", 16))
        pin_calls = [c for c in self.calls() if c["strategy"] == "pin"]
        self.assertTrue(pin_calls and all(c["pin_fraction"] == 0.25 for c in pin_calls))
        self.assertTrue(all(c["repack"] == "off" for c in self.calls()))
        # Second invocation: everything recorded, nothing executed.
        again = self.runner()
        self.assertEqual(again.returncode, 0)
        self.assertIn("16 already recorded, 0 to run", again.stdout)
        self.assertEqual(len(self.calls()), 16)

    def test_interrupt_then_resume(self):
        result = self.runner(env={"STUB_INTERRUPT_AT": "5"})
        self.assertEqual(result.returncode, 130, result.stderr)
        self.assertIn("partial record was discarded", result.stderr)
        records = self.records()  # every line parses
        self.assertEqual(len(records), 4)
        self.assertNotIn(5, [r["stub_call"] for r in records])
        status = json.loads(self.out.with_name("out.jsonl.status.json").read_text())
        self.assertEqual(status["state"], "interrupted")

        resumed = self.runner()
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("4 already recorded, 12 to run", resumed.stdout)
        records = self.records()
        self.assertEqual(len(records), 16)
        self.assertEqual({r["run_id"] for r in records}, self.expected_ids())

    def test_errors_skipped_unless_retry(self):
        result = self.runner(env={"STUB_ERROR_STRATEGY": "pin"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(r["error"] is not None for r in self.records()), 8)

        skip = self.runner()
        self.assertIn("16 already recorded, 0 to run (8 errored run_ids, rerun with --retry-errors)", skip.stdout)
        self.assertEqual(len(self.calls()), 16)

        retry = self.runner("--retry-errors")
        self.assertEqual(retry.returncode, 0, retry.stderr)
        self.assertIn("8 already recorded, 8 to run", retry.stdout)
        self.assertEqual(len(self.calls()), 24)
        last = {}
        for r in self.records():
            last[r["run_id"]] = r
        self.assertEqual(len(last), 16)
        self.assertTrue(all(r["error"] is None for r in last.values()))

    def test_run_without_record_stops(self):
        result = self.runner(env={"STUB_NO_RECORD_AT": "3"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("stopping (nothing written for this run)", result.stderr)
        self.assertEqual(len(self.records()), 2)

    def test_trailing_partial_line_repaired(self):
        ids = sorted(self.expected_ids())
        self.out.write_text(json.dumps({"run_id": ids[0], "error": None}) + "\n" + '{"run_id": "trunc')
        result = self.runner()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.out.with_name("out.jsonl.partial").read_text(), '{"run_id": "trunc\n')
        self.assertIn("1 already recorded, 15 to run", result.stdout)
        self.assertEqual(len(self.records()), 16)

    def test_invalid_middle_line_refused(self):
        self.out.write_text('{"run_id": "a", "error": null}\nnot json\n{"run_id": "b", "error": null}\n')
        result = self.runner()
        self.assertEqual(result.returncode, 2)
        self.assertIn("line 2 is not valid JSON", result.stderr)
        self.assertEqual(self.calls(), [])

    @unittest.skipIf(os.geteuid() == 0, "root may run limited plans")
    def test_limited_plan_needs_root(self):
        limited = self.dir / "limited.json"
        limited.write_text(json.dumps(dict(self.plan, memory_limits=["1100M"])))
        result = self.runner(plan=limited)
        self.assertEqual(result.returncode, 2)
        self.assertIn("need root", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_bad_plan_refused(self):
        bad = self.dir / "bad.json"
        bad.write_text(json.dumps(dict(self.plan, strategies=[{"strategy": "pin"}])))
        result = self.runner(plan=bad)
        self.assertEqual(result.returncode, 2)
        self.assertIn("pin needs 0 < fraction <= 1", result.stderr)


if __name__ == "__main__":
    unittest.main()

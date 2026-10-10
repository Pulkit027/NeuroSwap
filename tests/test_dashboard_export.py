"""Checks on scripts/export_dashboard_json.py output (needs the Phase 5 inputs).

Run: .venv/bin/python -m unittest tests/test_dashboard_export.py -v
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INPUTS = [REPO / "results/ml/metrics.json", REPO / "results/policies/policy_results.json",
          REPO / "results/raw/llm/final.jsonl"]


@unittest.skipUnless(all(p.exists() for p in INPUTS), "run build_dataset, train_ml and evaluate_policies first")
class DashboardExportTest(unittest.TestCase):
    def test_every_file_has_source_kind_and_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(REPO / "scripts/export_dashboard_json.py"), "--out-dir", tmp],
                           check=True, capture_output=True, text=True)
            files = {p.stem: json.loads(p.read_text(), parse_constant=self.fail) for p in Path(tmp).glob("*.json")}
        self.assertEqual(set(files), {"overview", "experiments", "strategy_comparison", "ml_metrics",
                                      "feature_importance", "predictions", "policy_results"})
        for name, payload in files.items():
            self.assertEqual(payload["source"], "neuroswap_measured", name)
            self.assertIn(payload["kind"], {"measured", "derived_from_measured", "model_output"}, name)
        self.assertEqual(files["predictions"]["kind"], "model_output")
        runs = files["experiments"]["data"]
        self.assertTrue(all("prompt" not in r for r in runs))  # ids only, never prompt text
        pilot_prefetch = [r for r in runs if r["plan_name"] == "pilot" and r["strategy"] == "prefetch"]
        self.assertTrue(pilot_prefetch and all(r["note"] for r in pilot_prefetch))


if __name__ == "__main__":
    unittest.main()

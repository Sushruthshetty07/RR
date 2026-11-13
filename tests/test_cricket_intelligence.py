"""
Unit tests for Strategic Cricket Intelligence & Composure Proxies.
"""

import sys
import unittest
import hashlib
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cricket_intelligence import generate_cricket_intelligence_summary, wilson_confidence_interval

DATA_PROCESSED = ROOT / "data" / "processed"


class TestCricketIntelligence(unittest.TestCase):
    def setUp(self):
        self.ci_path = DATA_PROCESSED / "cricket_intelligence_summary.json"
        if not self.ci_path.exists():
            generate_cricket_intelligence_summary()
        with open(self.ci_path, "r", encoding="utf-8") as f:
            self.ci = json.load(f)

    def test_top_level_keys(self):
        self.assertIn("metadata", self.ci)
        self.assertIn("batting_intelligence", self.ci)
        self.assertIn("bowling_intelligence", self.ci)
        self.assertIn("strategic_profiles", self.ci)

    def test_league_baselines(self):
        base = self.ci["metadata"]["league_baselines"]
        self.assertIn("post_dot_high_pressure", base)
        self.assertIn("post_boundary_bowling", base)

        # Post-dot high pressure baseline sanity bounds
        pd_base = base["post_dot_high_pressure"]
        self.assertGreaterEqual(pd_base["strike_rate"], 100.0)
        self.assertLessEqual(pd_base["strike_rate"], 150.0)
        self.assertGreaterEqual(pd_base["dismissal_pct"], 4.0)
        self.assertLessEqual(pd_base["dismissal_pct"], 15.0)

        # Post-boundary bowling baseline sanity bounds
        pb_base = base["post_boundary_bowling"]
        self.assertGreaterEqual(pb_base["economy_rpo"], 8.0)
        self.assertLessEqual(pb_base["economy_rpo"], 11.0)
        self.assertGreaterEqual(pb_base["repeat_boundary_pct"], 15.0)
        self.assertLessEqual(pb_base["repeat_boundary_pct"], 35.0)

    def test_wilson_interval_logic(self):
        """Tests the Wilson confidence interval bounds and edge cases."""
        # Edge case: n = 0
        self.assertEqual(wilson_confidence_interval(0, 0), (0.0, 0.0))
        # 0 observed out of 20
        low, high = wilson_confidence_interval(0, 20)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 10.0)
        self.assertLess(high, 20.0)
        # Symmetrical case: 50 out of 100
        low, high = wilson_confidence_interval(50, 100)
        self.assertAlmostEqual(low, 40.4, places=1)
        self.assertAlmostEqual(high, 59.6, places=1)

    def test_eligible_player_coverage(self):
        bat_intel = self.ci["batting_intelligence"]
        bowl_intel = self.ci["bowling_intelligence"]
        self.assertGreaterEqual(len(bat_intel), 180)
        self.assertGreaterEqual(len(bowl_intel), 190)

    def test_small_sample_insufficient_evidence_guard(self):
        """Verifies that players with N < 15 post-event opportunities are assigned Insufficient Evidence."""
        bat_intel = self.ci["batting_intelligence"]
        # Sanju Samson (2258) has 12 post-dot opps in HP (< 15)
        samson = bat_intel.get("2258")
        self.assertIsNotNone(samson)
        self.assertEqual(samson["post_dot_opps"], 12)
        self.assertIn("Insufficient Evidence", samson["composure_tier"])

    def test_strategic_profiles_4_questions(self):
        profiles = self.ci["strategic_profiles"]
        self.assertGreaterEqual(len(profiles), 8)
        for cid, p in profiles.items():
            self.assertIn("q1_discovery", p)
            self.assertIn("q2_significance", p)
            self.assertIn("q3_rr_implication", p)
            self.assertIn("q4_scouting_decision", p)
            # Ensure no speculative auction price tags remain
            self.assertNotIn("₹", p["q4_scouting_decision"])

    def test_official_csv_immutability(self):
        """Verifies that the four official CSV deliverables remain unchanged."""
        expected_hashes = {
            "official_top10_high_pressure_batters.csv": "a15f8064ba7235907e91ce1a50f4b7c7",
            "official_top10_high_pressure_bowlers.csv": "ffc8b156d8f01d77157b397c0371f397",
            "official_top10_overall_reliability_batters.csv": "22a073f5a42cb21c3530436824f5e96f",
            "official_top10_overall_reliability_bowlers.csv": "5eab5d69240d0ea2981463557e9a3440",
        }
        for fname, exp_md5 in expected_hashes.items():
            fpath = DATA_PROCESSED / fname
            self.assertTrue(fpath.exists(), f"Missing file: {fname}")
            with open(fpath, "rb") as f:
                actual_md5 = hashlib.md5(f.read()).hexdigest()
            self.assertEqual(
                actual_md5.lower(), exp_md5.lower(),
                f"Official CSV modified! Hash mismatch for {fname}"
            )


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Weight Sensitivity and Ranking Robustness module.
"""

import unittest
import numpy as np
import pandas as pd

from src.weight_sensitivity import (
    SCENARIOS,
    compute_scenario_rankings,
    run_weight_sensitivity_analysis,
)


class TestWeightSensitivity(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 30
        self.bat_synth = pd.DataFrame({
            "canonical_player_id": np.arange(1, n + 1),
            "player_name": [f"Player {i}" for i in range(1, n + 1)],
            "elig_primary": True,
            "neff": np.random.uniform(35.0, 300.0, size=n),
            "neff_hp": np.random.uniform(0.0, 150.0, size=n),
            "raa_post": np.random.normal(0, 5, size=n),
            "raa_sd": np.random.uniform(0.5, 2.0, size=n),
            "surv_post": np.random.normal(0, 0.5, size=n),
            "surv_sd": np.random.uniform(0.1, 0.4, size=n),
            "cons_post": np.random.uniform(0.4, 0.8, size=n),
            "cons_sd": np.random.uniform(0.05, 0.15, size=n),
            "hp_raa_post": np.random.normal(0, 8, size=n),
            "hp_raa_sd": np.random.uniform(1.0, 3.0, size=n),
        })

        self.bowl_synth = pd.DataFrame({
            "canonical_player_id": np.arange(1, n + 1),
            "player_name": [f"Bowler {i}" for i in range(1, n + 1)],
            "elig_primary": True,
            "neff": np.random.uniform(35.0, 300.0, size=n),
            "neff_hp": np.random.uniform(0.0, 150.0, size=n),
            "rsaa_post": np.random.normal(0, 0.5, size=n),
            "rsaa_sd": np.random.uniform(0.05, 0.2, size=n),
            "waa_post": np.random.normal(0, 0.3, size=n),
            "waa_sd": np.random.uniform(0.05, 0.15, size=n),
            "cons_post": np.random.uniform(0.4, 0.8, size=n),
            "cons_sd": np.random.uniform(0.05, 0.15, size=n),
            "hp_impact_post": np.random.normal(0, 0.6, size=n),
            "hp_impact_sd": np.random.uniform(0.1, 0.3, size=n),
        })

    def test_scenarios_count(self):
        self.assertEqual(len(SCENARIOS), 4)
        names = [sc.name for sc in SCENARIOS]
        self.assertIn("Baseline", names)
        self.assertIn("Balanced", names)
        self.assertIn("Pressure-Focused", names)
        self.assertIn("Reliability-Focused", names)

    def test_compute_scenario_rankings(self):
        res = compute_scenario_rankings(self.bat_synth, "batting", SCENARIOS[0])
        self.assertIn("pri_Baseline", res.columns)
        self.assertIn("rank_Baseline", res.columns)
        self.assertEqual(int(res["rank_Baseline"].min()), 1)

    def test_full_sensitivity_run(self):
        res = run_weight_sensitivity_analysis(self.bat_synth, self.bowl_synth)
        self.assertIn("batting", res)
        self.assertIn("bowling", res)
        corr_mat = res["batting"]["correlation_matrix"]
        self.assertEqual(corr_mat["Baseline"]["Baseline"], 1.0)
        self.assertGreater(corr_mat["Baseline"]["Balanced"], 0.70)


if __name__ == "__main__":
    unittest.main()

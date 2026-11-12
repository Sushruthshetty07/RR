"""
Unit tests for Player Evaluation Dimensions module.
"""

import unittest
import numpy as np
import pandas as pd

from src.player_evaluation import (
    classify_hp_evidence_status,
    evaluate_separated_dimensions,
)


class TestPlayerEvaluation(unittest.TestCase):
    def test_classify_hp_evidence_status(self):
        self.assertIn("Zero HP Evidence", classify_hp_evidence_status(0.0))
        self.assertIn("Insufficient HP Evidence", classify_hp_evidence_status(8.5))
        self.assertIn("Moderate HP Evidence", classify_hp_evidence_status(22.0))
        self.assertIn("Robust HP Evidence", classify_hp_evidence_status(45.0))

    def test_evaluate_separated_dimensions(self):
        bat = pd.DataFrame({
            "canonical_player_id": [1, 2, 3, 4],
            "player_name": ["Opener A", "Clutch B", "Middle C", "Tail D"],
            "elig_primary": [True, True, True, True],
            "pri_score_lcb": [98.5, 95.0, 90.0, 50.0],
            "neff": [200.0, 150.0, 120.0, 60.0],
            "neff_hp": [0.0, 80.0, 10.0, 5.0],
            "hp_raa_post": [-1.0, 25.0, 5.0, -2.0],
            "hp_raa_lcb": [-2.0, 21.0, 2.0, -4.0],
        })

        bowl = pd.DataFrame({
            "canonical_player_id": [10, 11, 12, 13],
            "player_name": ["Bowler A", "Bowler B", "Bowler C", "Bowler D"],
            "elig_primary": [True, True, True, True],
            "pri_score_lcb": [99.0, 94.0, 88.0, 60.0],
            "neff": [250.0, 180.0, 140.0, 50.0],
            "neff_hp": [40.0, 50.0, 12.0, 0.0],
            "hp_impact_post": [0.6, 0.4, 0.1, -0.2],
            "hp_impact_lcb": [0.5, 0.3, 0.0, -0.3],
        })

        res = evaluate_separated_dimensions(bat, bowl)
        bat_res = res["batting"]
        self.assertEqual(len(bat_res["top10_overall_reliability"]), 4)
        # Clutch B should lead robust high pressure (Opener A has Neff_HP=0 so not in robust)
        self.assertEqual(bat_res["top10_high_pressure_robust"][0]["player_name"], "Clutch B")
        # Opener A should appear in zero HP exposure list
        self.assertEqual(bat_res["pp_specialists_zero_hp"][0]["player_name"], "Opener A")


if __name__ == "__main__":
    unittest.main()

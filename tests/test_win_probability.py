"""
Unit tests for Win Probability, Leverage, and WPA module.
"""

import unittest
import numpy as np
import pandas as pd

from src.win_probability import (
    prepare_wp_features,
    build_pipeline,
    evaluate_model_predictions,
    compute_counterfactual_state,
)


class TestWinProbability(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df_chase = pd.DataFrame({
            "match_id": np.repeat(np.arange(1, 11), 10),
            "innings": 2,
            "legal_balls_remaining_eff": np.random.randint(6, 120, size=n),
            "wickets_in_hand_before": np.random.randint(1, 11, size=n),
            "runs_needed_eff": np.random.randint(10, 180, size=n),
            "rrr_eff": np.random.uniform(6.0, 15.0, size=n),
            "crr_before": np.random.uniform(6.0, 10.0, size=n),
            "match_phase": np.random.choice(["Powerplay", "Middle", "Death"], size=n),
            "venue_id": 6,
            "is_ipl": 1,
            "season_year": 2024.0,
            "bat_win": np.random.randint(0, 2, size=n),
            "score_before": np.random.randint(20, 150, size=n),
        })

    def test_prepare_wp_features_chase(self):
        X, y, groups = prepare_wp_features(self.df_chase, is_chase=True)
        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(len(groups), 100)
        self.assertIn("rrr_minus_crr", X.columns)
        self.assertIn("log_runs_needed", X.columns)

    def test_pipeline_build_and_fit(self):
        for m_type in ["logistic_regression", "random_forest", "gradient_boosting"]:
            pipe = build_pipeline(m_type, is_chase=True)
            X, y, _ = prepare_wp_features(self.df_chase, is_chase=True)
            pipe.fit(X, y)
            preds = pipe.predict_proba(X)[:, 1]
            self.assertEqual(len(preds), 100)
            self.assertTrue(np.all((preds >= 0.0) & (preds <= 1.0)))

    def test_evaluation_metrics_ranges(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.4, 0.85])
        metrics = evaluate_model_predictions(y_true, y_prob)
        self.assertGreaterEqual(metrics["roc_auc"], 0.8)
        self.assertLess(metrics["brier_score"], 0.15)
        self.assertLess(metrics["mean_calibration_error"], 0.25)

    def test_counterfactual_state_monotonicity(self):
        cf_six = compute_counterfactual_state(self.df_chase, "six", is_chase=True)
        cf_wkt = compute_counterfactual_state(self.df_chase, "wicket", is_chase=True)
        self.assertTrue(np.all(cf_six["runs_needed"] <= self.df_chase["runs_needed_eff"]))
        self.assertTrue(np.all(cf_wkt["wickets_in_hand_before"] <= self.df_chase["wickets_in_hand_before"]))


if __name__ == "__main__":
    unittest.main()

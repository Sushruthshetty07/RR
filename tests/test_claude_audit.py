"""Unit tests for the independent Claude review audit module and generated audit artifacts."""

import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.claude_audit import audit_delivery_and_p11_accounting, audit_feature_leakage
from src.match_state import reconstruct_match_situations
from src.pressure import compute_pressure_scores

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"


class TestClaudeAudit(unittest.TestCase):
    def test_feature_leakage_audit_passes_on_synthetic_pool(self) -> None:
        df = pd.DataFrame({"match_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "runs": [0, 1] * 5})
        res = audit_feature_leakage(df)
        self.assertEqual(res["bat_outcome_overlap"], [])
        self.assertEqual(res["bowl_outcome_overlap"], [])
        self.assertEqual(res["max_gkf_match_overlap"], 0)
        self.assertTrue(res["preprocessing_fit_inside_fold"])
        self.assertTrue(res["extras_multiplier_fit_inside_fold"])
        self.assertTrue(res["dls_pre_stoppage_leakage_free"])
        self.assertTrue(res["loo_inn1_self_excluded"])

    def test_delivery_and_p11_accounting_exact_on_synthetic_data(self) -> None:
        comps = pd.DataFrame([
            {"comp_id": 1, "comp_name": "Indian Premier League", "match_format": "T20", "season": "2024"},
            {"comp_id": 2, "comp_name": "Emerging Teams Asia Cup", "match_format": "T20", "season": "2024"},
        ])
        matches = pd.DataFrame([
            {
                "match_id": 1001, "comp_id": 1, "match_number": 1, "match_date": "2024-04-01",
                "stage": "League", "venue_id": 1, "venue_country": "India",
                "home_team_id": 1, "home_team": "RR", "away_team_id": 2, "away_team": "MI",
                "first_inn_revised_overs": np.nan, "second_inn_revised_overs": np.nan,
                "revised_target": np.nan, "knock_out": 0, "win_team_id": 1, "result": "Won",
                "man_of_the_match": 10,
            },
            {
                "match_id": 1002, "comp_id": 2, "match_number": 2, "match_date": "2024-10-01",
                "stage": "League", "venue_id": 2, "venue_country": "Oman",
                "home_team_id": 3, "home_team": "IND A", "away_team_id": 4, "away_team": "SL A",
                "first_inn_revised_overs": np.nan, "second_inn_revised_overs": np.nan,
                "revised_target": np.nan, "knock_out": 0, "win_team_id": 3, "result": "Won",
                "man_of_the_match": 20,
            },
        ])
        bbb = pd.DataFrame([
            {"match_id": 1001, "innings": 1, "overid": 1, "over_ball_num": 1, "innings_ball_num": 1,
             "batting_team_id": 1, "bowling_team_id": 2, "striker_id": 10, "non_striker_id": 11, "bowler_id": 30,
             "runs": 4, "extras_type": np.nan, "extras1_runs": 0, "extras_2_runs": 0, "is_wicket": 0,
             "dismissal_type_id": np.nan, "dismissal_type": np.nan, "dismissed_player_id": np.nan},
            {"match_id": 1001, "innings": 1, "overid": 1, "over_ball_num": 2, "innings_ball_num": 2,
             "batting_team_id": 1, "bowling_team_id": 2, "striker_id": 10, "non_striker_id": 11, "bowler_id": 30,
             "runs": 0, "extras_type": "wide", "extras1_runs": 1, "extras_2_runs": 0, "is_wicket": 0,
             "dismissal_type_id": np.nan, "dismissal_type": np.nan, "dismissed_player_id": np.nan},
            {"match_id": 1001, "innings": 1, "overid": 1, "over_ball_num": 2, "innings_ball_num": 3,
             "batting_team_id": 1, "bowling_team_id": 2, "striker_id": 10, "non_striker_id": 11, "bowler_id": 30,
             "runs": 1, "extras_type": "Overstep", "extras1_runs": 1, "extras_2_runs": 0, "is_wicket": 0,
             "dismissal_type_id": np.nan, "dismissal_type": np.nan, "dismissed_player_id": np.nan},
            {"match_id": 1002, "innings": 1, "overid": 1, "over_ball_num": 1, "innings_ball_num": 1,
             "batting_team_id": 3, "bowling_team_id": 4, "striker_id": 20, "non_striker_id": 21, "bowler_id": 40,
             "runs": 1, "extras_type": np.nan, "extras1_runs": 0, "extras_2_runs": 0, "is_wicket": 0,
             "dismissal_type_id": np.nan, "dismissal_type": np.nan, "dismissed_player_id": np.nan},
        ])
        p11 = pd.DataFrame([
            {"match_id": 1001, "team_id": 1, "player_id": 10, "batting_order": 1, "is_played": 1, "is_captain": 0, "is_wk": 0, "is_impact": 0, "is_impact_off": 0},
            {"match_id": 1001, "team_id": 1, "player_id": 10, "batting_order": 2, "is_played": 1, "is_captain": 0, "is_wk": 0, "is_impact": 0, "is_impact_off": 0},
            {"match_id": 1001, "team_id": 1, "player_id": 11, "batting_order": 2, "is_played": 1, "is_captain": 0, "is_wk": 0, "is_impact": 0, "is_impact_off": 0},
        ])
        state = compute_pressure_scores(reconstruct_match_situations(bbb, matches), matches, comps)
        acct = audit_delivery_and_p11_accounting(bbb, state, matches, comps, p11)
        self.assertEqual(acct["n_raw_deliveries"], 4)
        self.assertEqual(acct["n_emerging_total"], 1)
        self.assertEqual(acct["n_primary_reg_no_emg_rows"], 3)
        self.assertEqual(acct["n_primary_bat_balls_faced"], 2)  # excludes wide
        self.assertEqual(acct["n_primary_bowl_legal_balls"], 1)  # excludes wide + Overstep NB
        self.assertEqual(acct["playing_eleven_raw_rows"], 3)
        self.assertEqual(acct["playing_eleven_dup_keys"], 1)
        self.assertEqual(acct["playing_eleven_extra_rows_dropped"], 1)
        self.assertEqual(acct["playing_eleven_unique_rows"], 2)

    def test_persisted_claude_audit_outputs_invariants(self) -> None:
        summary_path = DATA_PROCESSED / "claude_audit_summary.json"
        self.assertTrue(summary_path.exists(), "claude_audit_summary.json must exist")
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        acct = summary["accounting"]
        self.assertEqual(acct["n_raw_deliveries"], 200999)
        self.assertEqual(acct["n_primary_reg_no_emg_rows"], 193144)
        self.assertEqual(acct["n_primary_bat_balls_faced"], 185702)
        self.assertEqual(acct["n_primary_bowl_legal_balls"], 184884)
        self.assertEqual(acct["playing_eleven_raw_rows"], 28381)
        self.assertEqual(acct["playing_eleven_dup_keys"], 35)
        self.assertEqual(acct["playing_eleven_extra_rows_dropped"], 40)
        self.assertEqual(acct["playing_eleven_unique_rows"], 28341)

        leak = summary["leakage"]
        self.assertEqual(leak["bat_outcome_overlap"], [])
        self.assertEqual(leak["bowl_outcome_overlap"], [])
        self.assertEqual(leak["max_gkf_match_overlap"], 0)

        pp_alt = summary["pp_alt_summary"]
        self.assertTrue(np.isfinite(pp_alt["spearman_rho_vs_approved"]))
        self.assertGreater(pp_alt["spearman_rho_vs_approved"], 0.9)
        self.assertEqual(pp_alt["top10_overlap_vs_approved"], 9)

        hp_grid = pd.read_csv(DATA_PROCESSED / "claude_hp_weight_gate_grid.csv")
        self.assertEqual(len(hp_grid), 12)
        self.assertTrue(hp_grid["bat_rho_full_pool"].notna().all())
        self.assertTrue((hp_grid["bat_top10_overlap_vs_approved"] <= 10).all())
        self.assertTrue((hp_grid["bowl_top10_overlap_vs_approved"] <= 10).all())

        for fn in ["claude_bat_bootstrap_ranks.csv", "claude_bowl_bootstrap_ranks.csv"]:
            df = pd.read_csv(DATA_PROCESSED / fn)
            self.assertTrue((df["rank_p05"] <= df["median_rank"] + 1e-9).all())
            self.assertTrue((df["median_rank"] <= df["rank_p95"] + 1e-9).all())
            self.assertTrue(((df["top10_inc_pct"] >= 0.0) & (df["top10_inc_pct"] <= 100.0)).all())

        t20i_df = pd.read_csv(DATA_PROCESSED / "claude_t20i_weight_summary.csv")
        self.assertEqual(len(t20i_df), 10)
        self.assertTrue(t20i_df["spearman_rho_vs_050"].notna().all())


if __name__ == "__main__":
    unittest.main()

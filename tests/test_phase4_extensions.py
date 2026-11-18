"""
Unit and integration tests for Phase 4 exploratory extensions (12 proposal items).

Tests zero-leakage guarantees, exact bowling extras/wicket attribution, match-cluster
bootstrap DEFF behavior under intra-match correlation, split-half repeatability,
non-bowler wicket exclusion in Wicket Quality, left-censored tenure proxy, and
preservation of approved Phase 3-4 outputs.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DATA_PROCESSED
from src.performance import PerformanceConfig
from src.phase4_extensions import (
    _cluster_bootstrap_player_stats,
    _evaluate_tenure_proxy_and_progression,
    _split_half_repeatability,
    audit_match_state_invariants,
    build_proposal_classification_table,
    evaluate_phase_dominance_and_wicket_quality,
    evaluate_win_probability_and_leverage,
)


def _make_synthetic_deliveries(n_matches: int = 12, balls_per_match: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Builds a minimal zero-leakage synthetic primary-deliveries + matches + comps dataset."""
    rng = np.random.default_rng(123)
    m_rows = []
    d_rows = []
    for m in range(1, n_matches + 1):
        win_t = 100 if (m % 2 == 1) else 200
        m_rows.append(
            {
                "match_id": m,
                "comp_id": 1,
                "match_date": f"2024-04-{10 + (m % 15):02d}",
                "venue_id": 10 if m <= 8 else 20,
                "venue_country": "India",
                "home_team_id": 100,
                "home_team": "RR",
                "away_team_id": 200,
                "away_team": "CSK",
                "win_team_id": np.nan if m == n_matches else win_t,
                "result": "No Result" if m == n_matches else "Won",
            }
        )
        for inn in (1, 2):
            score = 0
            wkts = 0
            for b in range(1, balls_per_match + 1):
                striker = 1001 if (b % 2 == 1) else 1002
                bowler = 2001 if ((m + b) % 2 == 0) else 2002
                ex_type = ""
                ex1 = 0
                runs = int(rng.choice([0, 1, 4, 6], p=[0.4, 0.35, 0.15, 0.10]))
                is_legal = True
                if b == 3 and m % 3 == 0:
                    ex_type = "bye"
                    ex1 = 4
                    runs = 0
                elif b == 5 and m % 4 == 0:
                    ex_type = "wide"
                    ex1 = 1
                    runs = 0
                    is_legal = False

                is_wkt = False
                d_type = ""
                is_bw = False
                d_pid = np.nan
                if b == 12 and m % 2 == 0:
                    # Non-bowler run out on top-order batter (PI < 60)
                    is_wkt = True
                    d_type = "run out"
                    is_bw = False
                    d_pid = striker
                elif b == 10 and m % 2 == 1:
                    # Bowled on top-order batter (PI < 60)
                    is_wkt = True
                    d_type = "Bowled"
                    is_bw = True
                    d_pid = striker
                elif b == 14 and m % 3 == 1:
                    # Caught on tailender (PI < 60)
                    is_wkt = True
                    d_type = "Caught"
                    is_bw = True
                    d_pid = striker

                tot_r = runs + ex1
                conc = runs + (ex1 if ex_type in ("wide", "Overstep", "other") else 0)
                leg_rem = max(0, 120 - b)
                d_rows.append(
                    {
                        "match_id": m,
                        "innings": inn,
                        "overid": (b - 1) // 6 + 1,
                        "over_ball_num": (b - 1) % 6 + 1,
                        "innings_ball_num": b,
                        "batting_team_id": 100 if inn == 1 else 200,
                        "bowling_team_id": 200 if inn == 1 else 100,
                        "canonical_striker_id": striker,
                        "canonical_bowler_id": bowler,
                        "canonical_striker_name": f"Batter_{striker}",
                        "canonical_bowler_name": f"Bowler_{bowler}",
                        "canonical_dismissed_player_id": d_pid,
                        "batting_order": 9 if b == 14 else 2,
                        "batting_type": "Right Hand",
                        "bowling_type": "Right Arm Fast",
                        "pace_or_spin": "Pace",
                        "delivery_style": "Yorker" if b % 5 == 0 else "Seamed In",
                        "runs": runs,
                        "extras_type": ex_type,
                        "extras1_runs": ex1,
                        "extras_2_runs": 0,
                        "ball_total_runs": tot_r,
                        "bowler_runs_conceded": conc,
                        "is_legal_ball": is_legal,
                        "is_batter_ball_faced": ex_type != "wide",
                        "is_wicket": is_wkt,
                        "dismissal_type": d_type,
                        "is_bowler_wicket": is_bw,
                        "striker_out": int(is_wkt and d_pid == striker),
                        "bowler_wkt_int": int(is_bw),
                        "score_before": score,
                        "wickets_before": wkts,
                        "wickets_in_hand_before": max(0, 10 - wkts),
                        "max_legal_balls_eff": 120,
                        "legal_balls_remaining_eff": leg_rem,
                        "crr_before": 7.5,
                        "target_eff": 160.0 if inn == 2 else np.nan,
                        "runs_needed_eff": max(0.0, 160.0 - score) if inn == 2 else np.nan,
                        "match_phase": "Powerplay" if b <= 12 else "Death",
                        "pressure_bat": 70.0 if b > 14 else 35.0,
                        "pressure_bowl": 70.0 if b > 14 else 35.0,
                        "is_super_over": False,
                        "dls_status": "standard",
                        "dls_ambiguous": False,
                        "comp_tier": "IPL",
                        "is_ipl_int": 1,
                        "season_year": 2024,
                        "venue_id": 10 if m <= 8 else 20,
                        "ipl_phase": "1_powerplay" if b <= 12 else "1_death",
                        "inn_phase": f"{inn}_{'powerplay' if b <= 12 else 'death'}",
                        "bat_exp_runs": 1.35,
                        "bat_exp_wkt": 0.05,
                        "bowl_exp_runs_legal_raw": 1.35,
                        "bowl_exp_runs": 1.40,
                        "bowl_exp_wkt": 0.05,
                    }
                )
                score += tot_r
                wkts += int(is_wkt)
    comps = pd.DataFrame([{"comp_id": 1, "comp_name": "Indian Premier League 2024", "season": "2024"}])
    return pd.DataFrame(d_rows), pd.DataFrame(m_rows), comps


class TestPhase4Extensions(unittest.TestCase):
    def test_classification_table_covers_12_items(self):
        cls_df = build_proposal_classification_table()
        self.assertEqual(len(cls_df), 12)
        self.assertListEqual(cls_df["proposal_id"].tolist(), list(range(1, 13)))
        allowed = {
            "1. Already implemented",
            "2. Partially implemented",
            "3. Missing and worth implementing",
            "4. Technically unsuitable or unsupported by the available data",
        }
        self.assertTrue(set(cls_df["classification"]).issubset(allowed))

    def test_match_state_invariants_audit(self):
        df, matches, _ = _make_synthetic_deliveries()
        res = audit_match_state_invariants(df, matches)
        self.assertTrue(res["inv_ball_total_runs_exact"])
        self.assertTrue(res["inv_bowler_runs_conceded_exact"])
        self.assertTrue(res["inv_byes_legbyes_excluded_from_bowler"])
        self.assertTrue(res["inv_legal_ball_definition_exact"])
        self.assertTrue(res["inv_non_bowler_wkts_excluded"])
        self.assertTrue(res["inv_super_over_flag_exact"])
        self.assertTrue(res["inv_super_overs_excluded_in_modeling_pool"])
        self.assertTrue(res["inv_no_match_outcome_columns_in_deliveries"])
        self.assertTrue(res["inv_first_ball_pre_state_zero"])

    def test_cluster_bootstrap_detects_intramatch_correlation(self):
        # Construct deliveries with strong positive match-level cluster shift -> DEFF must exceed 1.0
        rows = []
        for m in range(1, 41):
            cluster_shock = 30.0 if (m % 2 == 0) else -30.0
            for b in range(10):
                rows.append({"player_id": 1, "match_id": m, "w": 1.0, "d": cluster_shock + (b % 3 - 1)})
        synth = pd.DataFrame(rows)
        bs = _cluster_bootstrap_player_stats(synth, "player_id", "w", "d", n_boot=300, seed=7)
        self.assertGreater(float(bs.loc[0, "deff"]), 4.0)
        self.assertGreater(float(bs.loc[0, "boot_raw_se"]), float(bs.loc[0, "iid_se"]) * 2.0)

    def test_split_half_repeatability_recovers_positive_signal(self):
        rows = []
        for pid in range(1, 25):
            skill = float(pid - 12) * 3.0
            for m in range(1, 11):
                rows.append(
                    {
                        "canonical_player_id": pid,
                        "match_id": m,
                        "match_date_dt": pd.Timestamp("2024-01-01") + pd.Timedelta(days=m),
                        "inn_impact": skill + ((-1.0) ** m),
                        "sw": 4.0,
                    }
                )
        tab, stats = _split_half_repeatability(pd.DataFrame(rows), min_matches_half=4)
        self.assertEqual(len(tab), 24)
        self.assertGreater(stats["odd_even_repeatability"]["pearson_r"], 0.90)
        self.assertGreater(stats["chrono_split_half_repeatability"]["pearson_r"], 0.90)

    def test_wp_excludes_abandoned_and_computes_leverage(self):
        df, matches, _ = _make_synthetic_deliveries(n_matches=14)
        bat_stub = pd.DataFrame(
            {
                "canonical_player_id": [1001, 1002],
                "player_name": ["B1", "B2"],
                "elig_primary": [True, True],
                "hp_raa_post": [5.0, -2.0],
                "pri_score_lcb": [70.0, 45.0],
            }
        )
        bowl_stub = pd.DataFrame(
            {
                "canonical_player_id": [2001, 2002],
                "player_name": ["L1", "L2"],
                "elig_primary": [True, True],
                "hp_impact_post": [1.0, -0.5],
                "pri_score_lcb": [68.0, 42.0],
            }
        )
        wp_res = evaluate_win_probability_and_leverage(df, matches, bat_stub, bowl_stub)
        wp_d = wp_res["wp_deliveries"]
        # Match 14 had win_team_id = NaN and must be excluded
        self.assertNotIn(14, set(wp_d["match_id"].unique()))
        self.assertTrue((wp_d["li_combined_pp"] >= 0.0).all())

    def test_wicket_quality_excludes_non_bowler_runouts(self):
        df, matches, comps = _make_synthetic_deliveries(n_matches=10)
        bat_df = pd.DataFrame(
            {
                "canonical_player_id": [1001, 1002],
                "player_name": ["B1", "B2"],
                "elig_primary": [True, True],
                "elig_broad": [True, True],
                "pri_score_lcb": [75.0, 45.0],
            }
        )
        bowl_df = pd.DataFrame(
            {
                "canonical_player_id": [2001, 2002],
                "player_name": ["L1", "L2"],
                "elig_primary": [True, True],
                "elig_broad": [True, True],
                "pri_score_lcb": [65.0, 50.0],
                "rsaa_lcb": [1.0, 0.0],
                "waa_post": [0.2, 0.0],
                "cons_lcb": [0.6, 0.5],
                "hp_impact_lcb": [0.5, -0.2],
            }
        )
        res = evaluate_phase_dominance_and_wicket_quality(df, matches, comps, bat_df, bowl_df)
        self.assertEqual(res["summary"]["n_non_bowler_wkts_excluded_in_wq"], 10)
        self.assertEqual(res["summary"]["n_bowler_wkts_in_wq"], 18)
        self.assertGreater(float(res["bat_pdi"]["pdi_powerplay_neff"].sum()), 0.0)
        self.assertGreater(float(res["bowl_pdi_wq"]["pdi_death_neff"].sum()), 0.0)
        self.assertGreater(float(res["bowl_pdi_wq"]["death_pace_balls"].sum()), 0.0)

    def test_tenure_left_censoring_flag(self):
        # One player debuts in April 2022 (left-censored), another in May 2025
        pri = pd.DataFrame(
            {
                "canonical_striker_id": [1, 1, 2, 2],
                "canonical_bowler_id": [10, 10, 10, 10],
                "match_id": [1, 2, 3, 4],
                "match_date_dt": pd.to_datetime(["2022-04-02", "2026-04-02", "2025-05-01", "2026-05-01"]),
                "is_batter_ball_faced": [True, True, True, True],
                "is_legal_ball": [True, True, True, True],
                "runs": [1, 4, 0, 6],
                "bat_exp_runs": [1.2, 1.2, 1.2, 1.2],
                "bowl_exp_runs": [1.2, 1.2, 1.2, 1.2],
                "bowler_runs_conceded": [1, 4, 0, 6],
            }
        )
        bat_stub = pd.DataFrame({"canonical_player_id": [1, 2], "player_name": ["Old", "New"], "elig_primary": [True, True]})
        bowl_stub = pd.DataFrame({"canonical_player_id": [10], "player_name": ["Bw"], "elig_primary": [True]})
        res = _evaluate_tenure_proxy_and_progression(pri, bat_stub, bowl_stub)
        bt = res["bat_tenure"].set_index("canonical_player_id")
        self.assertTrue(bool(bt.loc[1, "left_censored_pre2022"]))
        self.assertFalse(bool(bt.loc[2, "left_censored_pre2022"]))
        self.assertAlmostEqual(float(bt.loc[1, "tenure_months"]), 48.0, delta=0.5)

    def test_real_extension_outputs_preserve_approved_shortlist(self):
        ext_bat_path = DATA_PROCESSED / "phase4_ext_bat_comparison.csv"
        ext_bowl_path = DATA_PROCESSED / "phase4_ext_bowl_comparison.csv"
        if not ext_bat_path.exists() or not ext_bowl_path.exists():
            self.skipTest("Extension CSVs not yet generated.")
        bat_short = pd.read_csv(DATA_PROCESSED / "phase4_bat_shortlist_cards.csv")
        bowl_short = pd.read_csv(DATA_PROCESSED / "phase4_bowl_shortlist_cards.csv")
        ext_bat = pd.read_csv(ext_bat_path)
        ext_bowl = pd.read_csv(ext_bowl_path)
        self.assertEqual(len(bat_short), 10)
        self.assertEqual(len(bowl_short), 10)
        self.assertEqual(int(ext_bat["is_phase4_shortlist"].sum()), 10)
        self.assertEqual(int(ext_bowl["is_phase4_shortlist"].sum()), 10)


if __name__ == "__main__":
    unittest.main()

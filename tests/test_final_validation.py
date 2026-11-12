"""
Unit tests for final independent validation, multiseed cluster bootstrap,
enriched RoE partitions, same-season PDI & spin-hitting, and temporal holdout.
"""

import unittest
import numpy as np
import pandas as pd

from src.final_validation import (
    analyze_pressure_intelligence,
    audit_enriched_roe_and_pressure_delta,
    evaluate_same_season_pdi_and_archetypes,
    reconstruct_and_audit_sample_matches,
    run_multiseed_cluster_bootstrap,
    run_temporal_holdout_validation,
)
from src.match_state import reconstruct_match_situations
from src.performance import (
    PerformanceConfig,
    evaluate_batting_pressure,
    evaluate_bowling_pressure,
    fit_predict_oof_baselines,
)
from src.player_identity import apply_canonical_identity
from src.pressure import compute_loo_inn1_baselines, compute_pressure_scores


def _build_synthetic_t20_dataset(n_matches: int = 24) -> dict:
    """Builds a realistic synthetic dataset across 2023-2025 with diverse extras, wickets & Super Over."""
    rows_bbb = []
    rows_m = []
    rows_p11 = []
    rng = np.random.default_rng(123)

    comps = pd.DataFrame(
        [
            {"comp_id": 1, "comp_name": "Indian Premier League", "match_format": "T20", "season": "2023"},
            {"comp_id": 2, "comp_name": "Indian Premier League", "match_format": "T20", "season": "2024"},
            {"comp_id": 3, "comp_name": "Indian Premier League", "match_format": "T20", "season": "2025"},
        ]
    )

    batters = list(range(101, 117))
    bowlers = list(range(201, 213))

    for m_idx in range(n_matches):
        mid = 9000 + m_idx
        year = 2023 if m_idx < 8 else (2024 if m_idx < 16 else 2025)
        cid = 1 if year == 2023 else (2 if year == 2024 else 3)
        is_dls = m_idx == 1
        rows_m.append(
            {
                "match_id": mid,
                "comp_id": cid,
                "match_number": m_idx + 1,
                "match_date": f"{year}-04-{(m_idx % 20) + 1:02d}",
                "stage": "Final" if is_dls else "League",
                "venue_id": 10 + (m_idx % 3),
                "venue_country": "India",
                "home_team_id": 1,
                "home_team": "Rajasthan Royals",
                "away_team_id": 2,
                "away_team": "Mumbai Indians",
                "first_inn_revised_overs": np.nan,
                "second_inn_revised_overs": 15.0 if is_dls else np.nan,
                "revised_target": 120.0 if is_dls else np.nan,
                "knock_out": 1 if is_dls else 0,
                "win_team_id": 1 if m_idx % 2 == 0 else 2,
                "result": "Won",
                "man_of_the_match": batters[0],
            }
        )
        for pid in batters[:8]:
            rows_p11.append({"match_id": mid, "team_id": 1, "player_id": pid, "batting_order": 1, "is_played": 1, "is_captain": 0, "is_wk": 0, "is_impact": 0, "is_impact_off": 0})
        for pid in bowlers[:6]:
            rows_p11.append({"match_id": mid, "team_id": 2, "player_id": pid, "batting_order": 8, "is_played": 1, "is_captain": 0, "is_wk": 0, "is_impact": 0, "is_impact_off": 0})

        inns = [1, 2, 3, 4] if m_idx == 2 else [1, 2]
        for inn in inns:
            b_team = 1 if inn in (1, 3) else 2
            l_team = 2 if b_team == 1 else 1
            n_balls = 6 if inn >= 3 else (42 if (is_dls and inn == 2) else 60)
            for b in range(1, n_balls + 1):
                overid = ((b - 1) // 6) + 1
                ob = ((b - 1) % 6) + 1
                st = int(batters[(b + m_idx + inn) % len(batters)])
                ns = int(batters[(b + m_idx + inn + 1) % len(batters)])
                bw = int(bowlers[(overid + m_idx) % len(bowlers)])
                ex_type = np.nan
                ex1 = 0
                ex2 = 0
                r = int(rng.choice([0, 1, 2, 4, 6], p=[0.35, 0.35, 0.10, 0.13, 0.07]))
                if m_idx == 3 and inn == 1:
                    if b == 5:
                        ex_type, ex1, r = "wide", 1, 0
                    elif b == 10:
                        ex_type, ex1, ex2, r = "Overstep", 1, 4, 1
                    elif b == 15:
                        ex_type, ex1, r = "other", 1, 2
                    elif b == 20:
                        ex_type, ex1, r = "byes", 2, 0
                    elif b == 25:
                        ex_type, ex1, r = "legbyes", 1, 0
                    elif b == 30:
                        ex_type, ex1, r = "Penalty", 5, 0

                is_w = 0
                d_id = np.nan
                d_type = np.nan
                d_pid = np.nan
                if b % 17 == 0:
                    is_w = 1
                    if m_idx == 4 and b == 17:
                        d_id, d_type, d_pid = 4.0, "RunOut", ns
                    else:
                        d_id, d_type, d_pid = 2.0, "Caught", st

                rows_bbb.append(
                    {
                        "match_id": mid,
                        "innings": inn,
                        "overid": overid,
                        "over_ball_num": ob,
                        "innings_ball_num": b,
                        "batting_team_id": b_team,
                        "bowling_team_id": l_team,
                        "striker_id": st,
                        "striker_name": f"Batter {st}",
                        "non_striker_id": ns,
                        "non_striker_name": f"Batter {ns}",
                        "bowler_id": bw,
                        "bowler_name": f"Bowler {bw}",
                        "batting_type": "RHB" if st % 2 == 0 else "LHB",
                        "bowling_type": "Fast" if bw % 2 == 0 else "Legbreak",
                        "batting_order": ((st - 101) % 7) + 1,
                        "runs": r,
                        "is_four": int(r == 4),
                        "is_six": int(r == 6),
                        "is_wicket": is_w,
                        "dismissal_type_id": d_id,
                        "dismissal_type": d_type,
                        "dismissed_player_id": d_pid,
                        "extras_type": ex_type,
                        "extras1_runs": ex1,
                        "extras_2_runs": ex2,
                        "pace_or_spin": "Pace" if bw % 2 == 0 else "Spin",
                        "bowl_speed": 138.0 if bw % 2 == 0 else 88.0,
                        "delivery_style": "Yorker" if b % 5 == 0 else "Good Length",
                        "delivery_ist_time": f"2024-04-01 20:{overid:02d}:{ob:02d}",
                    }
                )

    return {
        "competitions": comps,
        "matches": pd.DataFrame(rows_m),
        "playing_eleven": pd.DataFrame(rows_p11),
        "ball_by_ball": pd.DataFrame(rows_bbb),
    }


class TestFinalValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = _build_synthetic_t20_dataset(24)
        ms = reconstruct_match_situations(
            cls.raw["ball_by_ball"],
            cls.raw["matches"],
        )
        cls.deliv = apply_canonical_identity(compute_pressure_scores(ms, cls.raw["matches"], cls.raw["competitions"]))
        cls.cfg = PerformanceConfig(min_neff_elig=5.0, min_ipl_balls_bat=5, min_ipl_balls_bowl=5)
        cls.oof, _ = fit_predict_oof_baselines(
            cls.deliv, cls.raw["matches"], cls.raw["competitions"], include_suspected_dls=False, n_splits=3, exclude_emerging=True
        )
        cls.bat_df, _ = evaluate_batting_pressure(cls.oof, cls.cfg)
        cls.bowl_df, _ = evaluate_bowling_pressure(cls.oof, cls.cfg)

    def test_independent_sample_reconstruction_exact_match(self):
        res = reconstruct_and_audit_sample_matches(self.raw, self.deliv)
        self.assertTrue(res["all_cases_passed"], f"Mismatches: {res['mismatches_detail']}")
        self.assertEqual(res["mismatches_detail"], [])
        self.assertEqual(res["outcome_cols_in_processed_deliveries"], [])

    def test_pressure_intelligence_outputs(self):
        pi_res = analyze_pressure_intelligence(self.deliv, self.raw["matches"], self.raw["competitions"], self.cfg)
        self.assertFalse(pi_res["by_phase"].empty)
        self.assertFalse(pi_res["by_rrr"].empty)
        self.assertGreater(pi_res["super_over_summary"]["n_super_over_balls"], 0)

    def test_multiseed_bootstrap_2000_and_seed_determinism(self):
        res1 = run_multiseed_cluster_bootstrap(
            self.oof, self.raw["matches"], self.raw["competitions"], self.bat_df, self.bowl_df, self.cfg, seeds=(42, 101), n_boot_grid=(100, 200)
        )
        res2 = run_multiseed_cluster_bootstrap(
            self.oof, self.raw["matches"], self.raw["competitions"], self.bat_df, self.bowl_df, self.cfg, seeds=(42, 101), n_boot_grid=(100, 200)
        )
        pd.testing.assert_frame_equal(res1["convergence"], res2["convergence"])
        self.assertIn("deff_rsaa", res1["bowl_boot"].columns)
        self.assertIn("hp_post_ci_lo", res1["bat_boot"].columns)

    def test_enriched_roe_all_three_partitions_and_delta(self):
        roe_res = audit_enriched_roe_and_pressure_delta(
            self.oof, self.raw["matches"], self.raw["competitions"], self.bat_df, self.bowl_df, self.cfg
        )
        for col in ("enr_roe_ovr_post", "enr_roe_gen_post", "enr_roe_hp_post", "pressure_delta_100", "pressure_delta_ci_lo"):
            self.assertIn(col, roe_res["bat_enriched_delta"].columns)
        for col in ("enr_rsaa_ovr_post", "enr_rsaa_gen_post", "enr_rsaa_hp_post", "pressure_delta_100"):
            self.assertIn(col, roe_res["bowl_enriched_delta"].columns)

    def test_same_season_pdi_spin_hitting_and_7_archetypes(self):
        b_cards = self.bat_df[["canonical_player_id"]].copy()
        l_cards = self.bowl_df[["canonical_player_id"]].copy()
        res = evaluate_same_season_pdi_and_archetypes(
            self.oof, self.raw["matches"], self.raw["competitions"], self.bat_df, self.bowl_df, b_cards, l_cards, self.cfg
        )
        self.assertEqual(len(res["archetype_leaders"]), 7)
        self.assertIn("spin_pdi_post", res["bat_pdi_ss"].columns)

    def test_temporal_holdout_zero_leakage_structure(self):
        res = run_temporal_holdout_validation(
            self.deliv, self.raw["matches"], self.raw["competitions"], self.cfg, cutoff_year=2025
        )
        self.assertFalse(res["bat_delivery_holdout"].empty)
        self.assertFalse(res["bowl_delivery_holdout"].empty)
        self.assertEqual(res["cutoff_year"], 2025)


if __name__ == "__main__":
    unittest.main()

"""
Automated Unit & Integration Tests for src/performance.py (Phase 3 Engine).
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_datasets, DATA_PROCESSED
from src.performance import (
    COMP_TIER_IPL,
    COMP_TIER_SENIOR_T20I,
    COMP_TIER_EMERGING,
    PerformanceConfig,
    PRIMARY_DLS_STATUSES,
    SUSPECTED_DLS_STATUSES,
    classify_comp_tier,
    filter_dls_pool,
    fit_predict_oof_baselines,
    apply_normal_shrinkage,
    apply_beta_binomial_shrinkage,
    evaluate_batting_pressure,
    evaluate_bowling_pressure,
    evaluate_emerging_supplementary,
)


class TestPerformanceSynthetic(unittest.TestCase):
    def test_dls_policy_filtering_synthetic(self):
        df = pd.DataFrame([
            {"match_id": 1, "is_super_over": False, "dls_ambiguous": False, "dls_status": "standard", "pressure_bat": 40.0, "pressure_bowl": 45.0},
            {"match_id": 2, "is_super_over": False, "dls_ambiguous": False, "dls_status": "dls_pre_innings_confirmed", "pressure_bat": 50.0, "pressure_bowl": 55.0},
            {"match_id": 3, "is_super_over": False, "dls_ambiguous": False, "dls_status": "dls_innings_break_suspected", "pressure_bat": 50.0, "pressure_bowl": 55.0},
            {"match_id": 4, "is_super_over": False, "dls_ambiguous": False, "dls_status": "dls_post_stoppage_suspected", "pressure_bat": 50.0, "pressure_bowl": 55.0},
            {"match_id": 5, "is_super_over": False, "dls_ambiguous": True, "dls_status": "dls_pre_stoppage_suspected", "pressure_bat": np.nan, "pressure_bowl": np.nan},
            {"match_id": 6, "is_super_over": True, "dls_ambiguous": False, "dls_status": "standard", "pressure_bat": np.nan, "pressure_bowl": np.nan},
        ])
        pri = filter_dls_pool(df, include_suspected_dls=False)
        sens = filter_dls_pool(df, include_suspected_dls=True)
        self.assertEqual(pri["match_id"].tolist(), [1, 2])
        self.assertEqual(sens["match_id"].tolist(), [1, 2, 3, 4])

    def test_empirical_bayes_monotonicity_and_bounds(self):
        raw = pd.Series([30.0, 30.0, -30.0, 0.0, 5.0, -5.0, 12.0, -12.0])
        neff = pd.Series([10.0, 250.0, 10.0, 100.0, 120.0, 120.0, 80.0, 80.0])
        res, meta = apply_normal_shrinkage(raw, neff, sigma2_within=25000.0, min_neff_fit=10.0, lcb_z=0.5)

        # Higher N_eff gets strictly larger credibility B and smaller posterior SD
        self.assertGreater(res.loc[1, "B"], res.loc[0, "B"])
        self.assertLess(res.loc[1, "sd"], res.loc[0, "sd"])
        self.assertTrue(((res["B"] > 0.0) & (res["B"] < 1.0)).all())

        # Equal raw extremes (+30): tiny sample shrinks closer to mu0 than large sample
        mu0 = meta["mu0"]
        self.assertLess(abs(res.loc[0, "post"] - mu0), abs(res.loc[1, "post"] - mu0))

        # LCB <= post
        self.assertTrue((res["lcb"] <= res["post"]).all())

        # Beta-Binomial bounds
        cons_raw = pd.Series([1.0, 0.70, 0.0, 0.50, 0.45, 0.60])
        m_eff = pd.Series([2.0, 30.0, 3.0, 15.0, 20.0, 25.0])
        c_res, _ = apply_beta_binomial_shrinkage(cons_raw, m_eff, min_meff_fit=5.0, lcb_z=0.5)
        self.assertTrue(((c_res["post"] >= 0.0) & (c_res["post"] <= 1.0)).all())
        self.assertTrue(((c_res["lcb"] >= 0.0) & (c_res["lcb"] <= 1.0)).all())
        self.assertLess(c_res.loc[0, "post"], 1.0)  # 2/2 shrinks below 1.0

    def test_wides_penalize_bowler_and_groupkfold_zero_leakage(self):
        rng = np.random.default_rng(42)
        rows = []
        matches_rows = []
        for mid in range(1, 13):
            matches_rows.append({"match_id": mid, "venue_id": (mid % 3) + 1, "comp_id": 1})
            for b in range(1, 13):
                rows.append({
                    "match_id": mid, "innings": 1, "overid": (b - 1) // 6 + 1, "wickets_in_hand_before": 9,
                    "match_phase": "Powerplay", "is_ipl": True, "is_super_over": False, "dls_ambiguous": False,
                    "dls_status": "standard", "pressure_bat": 30.0 + b, "pressure_bowl": 35.0 + b,
                    "is_batter_ball_faced": True, "is_legal_ball": True, "runs": int(rng.choice([0, 1, 4])),
                    "bowler_runs_conceded": int(rng.choice([0, 1, 4])), "is_wicket": False,
                    "is_bowler_wicket": False, "canonical_striker_id": 100 + (b % 2),
                    "canonical_striker_name": f"Bat{b%2}", "canonical_dismissed_player_id": np.nan,
                    "canonical_bowler_id": 200 if mid != 1 else (201 if b <= 6 else 202),
                    "canonical_bowler_name": "Bowl",
                })
        # Give bowler 202 an extra wide in match 1 with identical legal balls
        df_syn = pd.DataFrame(rows)
        df_syn.loc[(df_syn["match_id"] == 1) & (df_syn["canonical_bowler_id"] == 202), "bowler_runs_conceded"] = 1
        df_syn.loc[(df_syn["match_id"] == 1) & (df_syn["canonical_bowler_id"] == 201), "bowler_runs_conceded"] = 1
        wide_row = df_syn.iloc[-1].copy()
        wide_row["match_id"] = 1
        wide_row["canonical_bowler_id"] = 202
        wide_row["is_legal_ball"] = False
        wide_row["is_batter_ball_faced"] = False
        wide_row["bowler_runs_conceded"] = 2
        df_syn = pd.concat([df_syn, pd.DataFrame([wide_row])], ignore_index=True)

        m_syn = pd.DataFrame(matches_rows)
        c_syn = pd.DataFrame([{"comp_id": 1, "season": "2025"}])

        oof1, _ = fit_predict_oof_baselines(df_syn, m_syn, c_syn, include_suspected_dls=False, n_splits=3)
        bowl_res, _ = evaluate_bowling_pressure(oof1, PerformanceConfig(w_t20i=0.5))
        r201 = bowl_res[bowl_res["canonical_player_id"] == 201].iloc[0]["rsaa_raw"]
        r202 = bowl_res[bowl_res["canonical_player_id"] == 202].iloc[0]["rsaa_raw"]
        self.assertLess(r202, r201)  # Wide reduces Runs Saved Above Expected

        # Mutate match 1 outcomes and confirm match 1 expected baselines are invariant (no self-leakage)
        df_mut = df_syn.copy()
        df_mut.loc[df_mut["match_id"] == 1, "runs"] = 6
        df_mut.loc[df_mut["match_id"] == 1, "bowler_runs_conceded"] = 6
        oof2, _ = fit_predict_oof_baselines(df_mut, m_syn, c_syn, include_suspected_dls=False, n_splits=3)

        m1_1 = oof1[oof1["match_id"] == 1].reset_index(drop=True)
        m1_2 = oof2[oof2["match_id"] == 1].reset_index(drop=True)
        np.testing.assert_allclose(m1_1["bat_exp_runs"].fillna(0), m1_2["bat_exp_runs"].fillna(0), atol=1e-9)
        np.testing.assert_allclose(m1_1["bowl_exp_runs"], m1_2["bowl_exp_runs"], atol=1e-9)

    def test_bowling_rsaa_excludes_byes_legbyes_penalty_and_secondary_extras(self):
        from src.match_state import reconstruct_match_situations

        matches = pd.DataFrame([
            {"match_id": m, "venue_id": 1, "comp_id": 1, "revised_target": np.nan, "first_inn_revised_overs": np.nan, "second_inn_revised_overs": np.nan, "knock_out": 0}
            for m in range(1, 7)
        ])
        comp = pd.DataFrame([{"comp_id": 1, "season": "2025"}])
        rows = []
        for m in range(1, 7):
            # Identical 6 legal deliveries for bowler 201 and 202, except bowler 202 concedes 4 byes + 2 legbyes + 5 Penalty + Overstep w/ 4 secondary byes
            for b, bid in [(1, 201), (2, 201), (3, 201), (4, 202), (5, 202), (6, 202)]:
                et = np.nan
                ex1 = 0
                if m == 1 and bid == 202 and b == 4:
                    et, ex1 = "byes", 4
                elif m == 1 and bid == 202 and b == 5:
                    et, ex1 = "legbyes", 2
                elif m == 1 and bid == 202 and b == 6:
                    et, ex1 = "Penalty", 5
                rows.append({
                    "match_id": m, "innings": 1, "overid": 1, "over_ball_num": b, "innings_ball_num": b,
                    "runs": 1, "extras_type": et, "extras1_runs": ex1, "extras_2_runs": 0,
                    "is_wicket": False, "dismissal_type": np.nan,
                    "canonical_striker_id": 101, "canonical_striker_name": "B1",
                    "canonical_dismissed_player_id": np.nan,
                    "canonical_bowler_id": bid, "canonical_bowler_name": f"W{bid}",
                    "is_ipl": True, "is_super_over": False, "dls_ambiguous": False, "dls_status": "standard",
                    "pressure_bat": 35.0, "pressure_bowl": 35.0,
                })
        st = reconstruct_match_situations(pd.DataFrame(rows), matches)
        oof, _ = fit_predict_oof_baselines(st, matches, comp, include_suspected_dls=False, n_splits=3)
        bowl_res, _ = evaluate_bowling_pressure(oof, PerformanceConfig(w_t20i=0.5))
        r201 = float(bowl_res.loc[bowl_res["canonical_player_id"] == 201, "rsaa_raw"].iloc[0])
        r202 = float(bowl_res.loc[bowl_res["canonical_player_id"] == 202, "rsaa_raw"].iloc[0])
        # Byes/legbyes/Penalty must NOT penalize bowler 202 relative to bowler 201
        self.assertAlmostEqual(r201, r202, places=9)

    def test_disjoint_hp_partition_and_bowling_weights(self):
        cfg = PerformanceConfig()
        self.assertTrue(cfg.disjoint_hp_split)
        self.assertAlmostEqual(cfg.bowl_w_rsaa, 0.55)
        self.assertAlmostEqual(cfg.bowl_w_waa, 0.15)
        self.assertAlmostEqual(cfg.bowl_w_cons, 0.15)
        self.assertAlmostEqual(cfg.bowl_w_hp, 0.15)

        # Synthetic dataset with both PI < 60 and PI >= 60 deliveries for same batter & bowler
        rows = []
        for idx in range(20):
            is_hp = idx >= 10
            rows.append({
                "match_id": (idx // 5) + 1, "innings": 1, "is_ipl": True,
                "is_batter_ball_faced": True, "is_legal_ball": True,
                "canonical_striker_id": 500, "canonical_striker_name": "BatA",
                "canonical_bowler_id": 600, "canonical_bowler_name": "BowlA",
                "pressure_bat": 75.0 if is_hp else 40.0,
                "pressure_bowl": 75.0 if is_hp else 40.0,
                "runs": 4 if is_hp else 1,
                "bowler_runs_conceded": 4 if is_hp else 1,
                "striker_out": 0, "bowler_wkt_int": 0,
                "bat_exp_runs": 1.3, "bat_exp_wkt": 0.05,
                "bowl_exp_runs": 1.3, "bowl_exp_wkt": 0.05,
            })
        df_base = pd.DataFrame(rows)
        b1, _ = evaluate_batting_pressure(df_base, cfg)
        w1, _ = evaluate_bowling_pressure(df_base, cfg)

        # Mutate ONLY PI >= 60 outcomes -> general & consistency metrics must stay 100% invariant
        df_mut_hp = df_base.copy()
        df_mut_hp.loc[df_mut_hp["pressure_bat"] >= 60.0, "runs"] = 0
        df_mut_hp.loc[df_mut_hp["pressure_bowl"] >= 60.0, "bowler_runs_conceded"] = 0
        b2, _ = evaluate_batting_pressure(df_mut_hp, cfg)
        w2, _ = evaluate_bowling_pressure(df_mut_hp, cfg)

        for col in ("raa_raw", "surv_raw", "cons_raw", "neff_gen"):
            self.assertAlmostEqual(float(b1.iloc[0][col]), float(b2.iloc[0][col]), places=10)
        for col in ("rsaa_raw", "waa_raw", "cons_raw", "neff_gen"):
            self.assertAlmostEqual(float(w1.iloc[0][col]), float(w2.iloc[0][col]), places=10)
        self.assertNotAlmostEqual(float(b1.iloc[0]["hp_raa_raw"]), float(b2.iloc[0]["hp_raa_raw"]))
        self.assertNotAlmostEqual(float(w1.iloc[0]["hp_impact_raw"]), float(w2.iloc[0]["hp_impact_raw"]))

        # Mutate ONLY PI < 60 outcomes -> dedicated HP metrics must stay 100% invariant
        df_mut_gen = df_base.copy()
        df_mut_gen.loc[df_mut_gen["pressure_bat"] < 60.0, "runs"] = 6
        df_mut_gen.loc[df_mut_gen["pressure_bowl"] < 60.0, "bowler_runs_conceded"] = 6
        b3, _ = evaluate_batting_pressure(df_mut_gen, cfg)
        w3, _ = evaluate_bowling_pressure(df_mut_gen, cfg)
        self.assertAlmostEqual(float(b1.iloc[0]["hp_raa_raw"]), float(b3.iloc[0]["hp_raa_raw"]), places=10)
        self.assertAlmostEqual(float(w1.iloc[0]["hp_impact_raw"]), float(w3.iloc[0]["hp_impact_raw"]), places=10)

    def test_comp_tier_classification_and_emerging_exclusion(self):
        cfg = PerformanceConfig()
        self.assertEqual(cfg.w_emerging, 0.0)
        self.assertTrue(cfg.exclude_emerging_from_primary)

        # Tier classification checks across explicit comp_name strings and fallback
        probe = pd.DataFrame([
            {"match_id": 1, "comp_name": "IPL 2025", "is_ipl": True},
            {"match_id": 2, "comp_name": "India in England T20I Series 2025", "is_ipl": False},
            {"match_id": 3, "comp_name": "Mens T20 Emerging Teams Asia Cup 2024", "is_ipl": False},
            {"match_id": 4, "comp_name": "Asia Cup Rising Stars 2025", "is_ipl": False},
            {"match_id": 5, "comp_name": None, "is_ipl": False},
        ])
        tiers = classify_comp_tier(probe).tolist()
        self.assertEqual(
            tiers,
            [
                COMP_TIER_IPL,
                COMP_TIER_SENIOR_T20I,
                COMP_TIER_EMERGING,
                COMP_TIER_EMERGING,
                COMP_TIER_SENIOR_T20I,
            ],
        )

        # Emerging deliveries must have ZERO effect on primary evaluations & t20i ball counts
        rows_pri = []
        for idx in range(20):
            is_ipl = idx < 10
            rows_pri.append({
                "match_id": 1 if is_ipl else 2,
                "innings": 1,
                "comp_name": "IPL 2025" if is_ipl else "India T20I Series 2025",
                "is_ipl": is_ipl,
                "is_batter_ball_faced": True,
                "is_legal_ball": True,
                "canonical_striker_id": 500,
                "canonical_striker_name": "BatA",
                "canonical_bowler_id": 600,
                "canonical_bowler_name": "BowlA",
                "pressure_bat": 65.0 if idx % 2 else 35.0,
                "pressure_bowl": 65.0 if idx % 2 else 35.0,
                "runs": 2,
                "bowler_runs_conceded": 1,
                "striker_out": 0,
                "bowler_wkt_int": 0,
                "bat_exp_runs": 1.2,
                "bat_exp_wkt": 0.05,
                "bowl_exp_runs": 1.2,
                "bowl_exp_wkt": 0.05,
            })
        df_pri = pd.DataFrame(rows_pri)

        rows_emg = []
        for idx in range(12):
            rows_emg.append({
                "match_id": 99,
                "innings": 1,
                "comp_name": "Mens T20 Emerging Teams Asia Cup 2024" if idx < 6 else "Asia Cup Rising Stars 2025",
                "is_ipl": False,
                "is_batter_ball_faced": True,
                "is_legal_ball": True,
                "canonical_striker_id": 500,
                "canonical_striker_name": "BatA",
                "canonical_bowler_id": 600,
                "canonical_bowler_name": "BowlA",
                "pressure_bat": 85.0,
                "pressure_bowl": 85.0,
                "runs": 6,
                "bowler_runs_conceded": 6,
                "striker_out": 1,
                "bowler_wkt_int": 1,
                "bat_exp_runs": 1.2,
                "bat_exp_wkt": 0.05,
                "bowl_exp_runs": 1.2,
                "bowl_exp_wkt": 0.05,
            })
        df_with_emg = pd.concat([df_pri, pd.DataFrame(rows_emg)], ignore_index=True)

        b_clean, _ = evaluate_batting_pressure(df_pri, cfg)
        b_mix, _ = evaluate_batting_pressure(df_with_emg, cfg)
        w_clean, _ = evaluate_bowling_pressure(df_pri, cfg)
        w_mix, _ = evaluate_bowling_pressure(df_with_emg, cfg)

        for col in ("total_balls", "ipl_balls", "t20i_balls", "neff", "raa_raw", "hp_raa_raw", "pri_score_lcb"):
            self.assertAlmostEqual(float(b_clean.iloc[0][col]), float(b_mix.iloc[0][col]), places=10)
        for col in ("total_legal_balls", "ipl_legal_balls", "t20i_legal_balls", "neff", "rsaa_raw", "waa_raw", "pri_score_lcb"):
            self.assertAlmostEqual(float(w_clean.iloc[0][col]), float(w_mix.iloc[0][col]), places=10)

        # Even if exclude_emerging_from_primary=False, w_emerging=0 keeps weighted Neff & t20i_balls invariant
        cfg_zero_w = PerformanceConfig(exclude_emerging_from_primary=False, w_emerging=0.0)
        b_zero, _ = evaluate_batting_pressure(df_with_emg, cfg_zero_w)
        w_zero, _ = evaluate_bowling_pressure(df_with_emg, cfg_zero_w)
        self.assertEqual(int(b_zero.iloc[0]["t20i_balls"]), int(b_clean.iloc[0]["t20i_balls"]))
        self.assertAlmostEqual(float(b_zero.iloc[0]["neff"]), float(b_clean.iloc[0]["neff"]), places=10)
        self.assertEqual(int(w_zero.iloc[0]["t20i_legal_balls"]), int(w_clean.iloc[0]["t20i_legal_balls"]))
        self.assertAlmostEqual(float(w_zero.iloc[0]["neff"]), float(w_clean.iloc[0]["neff"]), places=10)

    def test_evaluate_emerging_supplementary_no_leakage(self):
        comps = pd.DataFrame([
            {"comp_id": 1, "comp_name": "IPL 2024", "season": "2024"},
            {"comp_id": 2, "comp_name": "India T20I Series 2024", "season": "2024"},
            {"comp_id": 2670, "comp_name": "Mens T20 Emerging Teams Asia Cup 2024", "season": "2024"},
        ])
        matches = pd.DataFrame([
            {"match_id": 101, "comp_id": 1, "venue_id": 10},
            {"match_id": 102, "comp_id": 1, "venue_id": 11},
            {"match_id": 103, "comp_id": 2, "venue_id": 12},
            {"match_id": 104, "comp_id": 2670, "venue_id": 13},
        ])

        rng = np.random.default_rng(7)
        rows = []
        for mid in (101, 102, 103, 104):
            cid = 1 if mid in (101, 102) else (2 if mid == 103 else 2670)
            for ball in range(24):
                rows.append({
                    "match_id": mid,
                    "comp_id": cid,
                    "is_super_over": False,
                    "dls_ambiguous": False,
                    "dls_status": "standard",
                    "innings": 1 if ball < 12 else 2,
                    "match_phase": "powerplay" if ball % 12 < 6 else "middle",
                    "overid": (ball % 12) + 1,
                    "pressure_bat": float(25.0 + 4.0 * (ball % 12)),
                    "pressure_bowl": float(25.0 + 4.0 * (ball % 12)),
                    "wickets_in_hand_before": 9,
                    "is_batter_ball_faced": True,
                    "is_legal_ball": True,
                    "runs": int(rng.choice([0, 1, 2, 4])),
                    "bowler_runs_conceded": int(rng.choice([0, 1, 2, 4])),
                    "is_wicket": False,
                    "is_bowler_wicket": False,
                    "canonical_dismissed_player_id": np.nan,
                    "canonical_striker_id": 901 if mid == 104 else 801 + (ball % 2),
                    "canonical_striker_name": "EmgBat" if mid == 104 else f"PriBat{ball % 2}",
                    "canonical_bowler_id": 951 if mid == 104 else 851 + (ball % 2),
                    "canonical_bowler_name": "EmgBowl" if mid == 104 else f"PriBowl{ball % 2}",
                })
        df1 = pd.DataFrame(rows)
        df2 = df1.copy()
        df2.loc[df2["match_id"] == 104, "runs"] = 6
        df2.loc[df2["match_id"] == 104, "bowler_runs_conceded"] = 6

        oof1, _ = fit_predict_oof_baselines(df1, matches, comps, n_splits=3)
        oof2, _ = fit_predict_oof_baselines(df2, matches, comps, n_splits=3)
        self.assertTrue(np.allclose(oof1["bat_exp_runs"].values, oof2["bat_exp_runs"].values, atol=1e-10))
        self.assertTrue(np.allclose(oof1["bowl_exp_runs"].values, oof2["bowl_exp_runs"].values, atol=1e-10))
        self.assertNotIn(COMP_TIER_EMERGING, set(oof1["comp_tier"].unique()))

        emg_bat, emg_bowl, diag = evaluate_emerging_supplementary(df1, matches, comps)
        self.assertEqual(diag["n_emerging_rows"], 24)
        self.assertEqual(len(emg_bat), 1)
        self.assertEqual(len(emg_bowl), 1)
        self.assertEqual(int(emg_bat.iloc[0]["canonical_player_id"]), 901)
        self.assertEqual(int(emg_bat.iloc[0]["emerging_balls"]), 24)
        self.assertEqual(int(emg_bowl.iloc[0]["canonical_player_id"]), 951)
        self.assertEqual(int(emg_bowl.iloc[0]["emerging_legal_balls"]), 24)


DATA_RAW_EXISTS = (PROJECT_ROOT / "data" / "raw" / "competitions.csv").exists() and (DATA_PROCESSED / "deliveries_with_pressure.csv.gz").exists()


@unittest.skipUnless(DATA_RAW_EXISTS, "Proprietary raw competition dataset or full deliveries archive not present (kept local per challenge rules)")
class TestPerformanceFullDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_raw_datasets()
        cls.df = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz", low_memory=False)
        cls.pdir = pd.read_csv(DATA_PROCESSED / "canonical_player_directory.csv")

    def test_dls_counts_and_exclusions_full_dataset(self):
        pri = filter_dls_pool(self.df, include_suspected_dls=False)
        pri_no_emg = filter_dls_pool(
            self.df,
            include_suspected_dls=False,
            matches_df=self.ds["matches"],
            competitions_df=self.ds["competitions"],
            exclude_emerging=True,
        )
        sens = filter_dls_pool(self.df, include_suspected_dls=True)
        self.assertEqual(len(pri), 200117)
        self.assertEqual(len(pri_no_emg), 193144)
        self.assertEqual(len(sens), 200818)
        self.assertFalse(pri["dls_ambiguous"].any())
        self.assertFalse(pri["is_super_over"].any())
        self.assertFalse(sens["dls_ambiguous"].any())
        self.assertFalse(sens["is_super_over"].any())
        self.assertEqual(set(pri_no_emg["comp_tier"].unique()), {COMP_TIER_IPL, COMP_TIER_SENIOR_T20I})
        self.assertTrue(set(pri["dls_status"].unique()).issubset(set(PRIMARY_DLS_STATUSES)))
        self.assertTrue(set(sens["dls_status"].unique()).issubset(set(PRIMARY_DLS_STATUSES) | set(SUSPECTED_DLS_STATUSES)))

    def test_oof_conservation_and_canonical_ids_full_dataset(self):
        oof, diag = fit_predict_oof_baselines(
            self.df, self.ds["matches"], self.ds["competitions"], include_suspected_dls=False, n_splits=5
        )
        self.assertEqual(len(oof), 193144)
        self.assertEqual(set(oof["comp_tier"].unique()), {COMP_TIER_IPL, COMP_TIER_SENIOR_T20I})
        # Check tight zero-sum expectation conservation (< 0.5% drift)
        rel_bat_drift = abs(diag["bat_runs_actual_mean"] - diag["bat_runs_exp_mean"]) / diag["bat_runs_actual_mean"]
        rel_bowl_drift = abs(diag["bowl_runs_total_actual"] - diag["bowl_runs_total_exp"]) / diag["bowl_runs_total_actual"]
        self.assertLess(rel_bat_drift, 0.005)
        self.assertLess(rel_bowl_drift, 0.005)

        cfg = PerformanceConfig(w_t20i=0.50)
        bat_df, _ = evaluate_batting_pressure(oof, cfg, self.pdir)
        bowl_df, _ = evaluate_bowling_pressure(oof, cfg, self.pdir)

        # Canonical identity checks: 39444 merged -> 8638 (139 Senior T20I balls in primary);
        # 36153 and 8729 played ONLY in Emerging/Rising T20 so both are excluded from primary
        self.assertNotIn(39444, bat_df["canonical_player_id"].values)
        self.assertIn(8638, bat_df["canonical_player_id"].values)
        b_8638 = bat_df.loc[bat_df["canonical_player_id"] == 8638].iloc[0]
        self.assertEqual(int(b_8638["t20i_balls"]), 139)
        self.assertNotIn(8729, bat_df["canonical_player_id"].values)
        self.assertNotIn(36153, bat_df["canonical_player_id"].values)

        # Score bounds check
        for tbl in (bat_df, bowl_df):
            self.assertFalse(tbl["pri_score_lcb"].isna().any())
            self.assertTrue(((tbl["pri_score_lcb"] >= 0.0) & (tbl["pri_score_lcb"] <= 100.0)).all())
            self.assertTrue(((tbl["cons_post"] >= 0.0) & (tbl["cons_post"] <= 1.0)).all())

    def test_emerging_supplementary_full_dataset(self):
        emg_bat, emg_bowl, diag = evaluate_emerging_supplementary(
            self.df, self.ds["matches"], self.ds["competitions"], player_dir=self.pdir
        )
        self.assertEqual(diag["n_emerging_rows"], 6973)
        self.assertEqual(diag["n_emerging_matches"], 30)
        # Merged 39444->8638 and homonym pair (36153, 8729) all appear cleanly in Emerging supplementary
        self.assertNotIn(39444, emg_bat["canonical_player_id"].values)
        self.assertIn(8638, emg_bat["canonical_player_id"].values)
        self.assertIn(36153, emg_bat["canonical_player_id"].values)
        self.assertIn(8729, emg_bat["canonical_player_id"].values)
        self.assertEqual(int(emg_bat.loc[emg_bat["canonical_player_id"] == 8638, "emerging_balls"].iloc[0]), 34)
        self.assertFalse(emg_bat["emerging_ev_net_raw"].isna().any())
        self.assertFalse(emg_bowl["emerging_ev_net_po_raw"].isna().any())


if __name__ == "__main__":
    unittest.main(verbosity=2)

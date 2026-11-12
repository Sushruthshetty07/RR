"""
Automated Unit & Integration Tests for src/phase4_analysis.py (Phase 4 Evaluation Engine).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import json
from src.data_loader import DATA_PROCESSED, load_raw_datasets
from src.performance import PerformanceConfig, classify_comp_tier, filter_dls_pool, fit_predict_oof_baselines
from src.phase4_analysis import (
    RETIRED_FROM_IPL,
    _vectorized_loo_contextual_expected,
    build_canonical_id_map,
    build_detailed_player_profiles,
    compute_rr_2026_context,
    deduplicate_playing_eleven,
    select_shortlists_and_benchmarks,
    summarize_emerging_only_supplementary,
)


class TestPhase4Synthetic(unittest.TestCase):
    def test_retired_exclusion_preserves_benchmark_and_reranks_shortlist(self):
        # Create synthetic eligible batters & bowlers including one retired ID in top 3
        retired_bat_id = 143  # Dinesh Karthik in RETIRED_FROM_IPL
        retired_bowl_id = 305  # Ravichandran Ashwin in RETIRED_FROM_IPL

        bat_rows = []
        for idx, (pid, score) in enumerate(
            [(1001, 99.0), (retired_bat_id, 98.0), (1002, 97.0), (1003, 96.0), (1004, 95.0), (1005, 94.0)], start=1
        ):
            bat_rows.append({
                "canonical_player_id": pid,
                "player_name": f"Bat_{pid}",
                "elig_primary": True,
                "pri_score_lcb": score,
                "pri_score_post": score + 0.1,
            })
        bat_rows.append({
            "canonical_player_id": 9999,
            "player_name": "Inelig_Bat",
            "elig_primary": False,
            "pri_score_lcb": 99.9,
            "pri_score_post": 99.9,
        })

        bowl_rows = []
        for idx, (pid, score) in enumerate(
            [(2001, 99.5), (2002, 98.5), (retired_bowl_id, 97.5), (2003, 96.5), (2004, 95.5)], start=1
        ):
            bowl_rows.append({
                "canonical_player_id": pid,
                "player_name": f"Bowl_{pid}",
                "elig_primary": True,
                "pri_score_lcb": score,
                "pri_score_post": score + 0.1,
            })

        sel = select_shortlists_and_benchmarks(pd.DataFrame(bat_rows), pd.DataFrame(bowl_rows), top_n=3, watchlist_n=2)

        # Unfiltered benchmark retains retired ID at rank 2, and excludes ineligible rank 9999
        self.assertNotIn(9999, sel["bat_benchmark_top15"]["canonical_player_id"].tolist())
        self.assertIn(retired_bat_id, sel["bat_benchmark_top15"]["canonical_player_id"].tolist())
        # Shortlist excludes retired ID and assigns contiguous shortlist_rank 1..3
        self.assertEqual(sel["bat_shortlist"]["canonical_player_id"].tolist(), [1001, 1002, 1003])
        self.assertEqual(sel["bat_shortlist"]["shortlist_rank"].tolist(), [1, 2, 3])
        self.assertEqual(sel["bat_watchlist"]["canonical_player_id"].tolist(), [1004, 1005])
        self.assertEqual(sel["bowl_shortlist"]["canonical_player_id"].tolist(), [2001, 2002, 2003])
        self.assertNotIn(retired_bowl_id, sel["bowl_shortlist"]["canonical_player_id"].tolist())

    def test_canonical_id_map_handles_multi_ids(self):
        pdir = pd.DataFrame([
            {"canonical_player_id": 500, "canonical_name": "Player A", "original_player_ids": "500; 1500; 2500"},
            {"canonical_player_id": 600, "canonical_name": "Player B", "original_player_ids": "600"},
        ])
        id_map = build_canonical_id_map(pdir)
        self.assertEqual(id_map[500], 500)
        self.assertEqual(id_map[1500], 500)
        self.assertEqual(id_map[2500], 500)
        self.assertEqual(id_map[600], 600)

    def test_emerging_supplementary_excludes_primary_players(self):
        bat_pri = pd.DataFrame([{"canonical_player_id": 10}, {"canonical_player_id": 20}])
        bowl_pri = pd.DataFrame([{"canonical_player_id": 30}])
        emg_bat = pd.DataFrame([
            {"canonical_player_id": 10, "player_name": "Overlap", "emerging_balls": 120, "emerging_ev_net_raw": 50.0},
            {"canonical_player_id": 99, "player_name": "EmgOnly_A", "emerging_balls": 80, "emerging_ev_net_raw": 42.0},
            {"canonical_player_id": 98, "player_name": "EmgOnly_Small", "emerging_balls": 20, "emerging_ev_net_raw": 99.0},
        ])
        emg_bowl = pd.DataFrame([
            {"canonical_player_id": 30, "player_name": "OverlapW", "emerging_legal_balls": 90, "emerging_ev_net_po_raw": 5.0},
            {"canonical_player_id": 77, "player_name": "EmgOnly_W", "emerging_legal_balls": 60, "emerging_ev_net_po_raw": 3.5},
        ])
        out = summarize_emerging_only_supplementary(emg_bat, emg_bowl, bat_pri, bowl_pri)
        self.assertEqual(out["emerging_only_bat_top"]["canonical_player_id"].tolist(), [99])
        self.assertEqual(out["emerging_only_bowl_top"]["canonical_player_id"].tolist(), [77])

    def test_deduplicate_playing_eleven_preserves_earliest_order_and_flags(self):
        raw_p11 = pd.DataFrame([
            {"match_id": 100, "team_id": 1, "player_id": 55, "batting_order": 5, "is_played": True, "is_wk": False, "is_captain": False, "is_impact": True, "is_impact_off": False},
            {"match_id": 100, "team_id": 1, "player_id": 55, "batting_order": 2, "is_played": False, "is_wk": True, "is_captain": False, "is_impact": False, "is_impact_off": True},
            {"match_id": 100, "team_id": 1, "player_id": 56, "batting_order": 3, "is_played": True, "is_wk": False, "is_captain": True, "is_impact": False, "is_impact_off": False},
        ])
        dedup = deduplicate_playing_eleven(raw_p11)
        self.assertEqual(len(dedup), 2)
        p55 = dedup[dedup["player_id"] == 55].iloc[0]
        self.assertEqual(int(p55["batting_order"]), 2)
        self.assertTrue(bool(p55["is_played"]))
        self.assertTrue(bool(p55["is_wk"]))
        self.assertTrue(bool(p55["is_impact"]))
        self.assertTrue(bool(p55["is_impact_off"]))

    def test_loo_contextual_expected_has_zero_target_leakage(self):
        rows = []
        for idx in range(40):
            pid = 1 if idx < 10 else (2 if idx < 25 else 3)
            rows.append({
                "canonical_striker_id": pid,
                "comp_tier": "IPL",
                "season_yr": 2026,
                "innings": 1,
                "match_phase": "Powerplay",
                "venue_id": 6,
                "runs": 6.0 if pid == 1 else 1.0,
            })
        df_a = pd.DataFrame(rows)
        df_b = df_a.copy()
        df_b.loc[df_b["canonical_striker_id"] == 1, "runs"] = 0.0
        exp_a, _ = _vectorized_loo_contextual_expected(df_a, "canonical_striker_id", "runs", k_venue=120.0)
        exp_b, _ = _vectorized_loo_contextual_expected(df_b, "canonical_striker_id", "runs", k_venue=120.0)
        p1_mask = (df_a["canonical_striker_id"] == 1).values
        np.testing.assert_allclose(exp_a[p1_mask], exp_b[p1_mask], atol=1e-12)
        np.testing.assert_allclose(exp_a[p1_mask], 1.0, atol=1e-12)


class TestPhase4Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bat_df = pd.read_csv(DATA_PROCESSED / "phase3_batting_prelim.csv")
        cls.bowl_df = pd.read_csv(DATA_PROCESSED / "phase3_bowling_prelim.csv")
        cls.emg_bat = pd.read_csv(DATA_PROCESSED / "phase3_emerging_batting_supp.csv")
        cls.emg_bowl = pd.read_csv(DATA_PROCESSED / "phase3_emerging_bowling_supp.csv")
        cls.pdir = pd.read_csv(DATA_PROCESSED / "canonical_player_directory.csv")
        cls.ds = load_raw_datasets()
        cls.raw_df = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz", low_memory=False)

    def test_shortlist_and_pillar_reconstruction_real_data(self):
        sel = select_shortlists_and_benchmarks(self.bat_df, self.bowl_df, top_n=10, watchlist_n=5)
        self.assertEqual(len(sel["bat_shortlist"]), 10)
        self.assertEqual(len(sel["bowl_shortlist"]), 10)
        self.assertEqual(len(sel["bat_benchmark_top15"]), 15)
        self.assertEqual(len(sel["bowl_benchmark_top15"]), 15)
        # Ensure benchmark keeps Karthik (143) & Russell (1318) & Ashwin (305), while shortlists exclude them
        self.assertTrue({143, 1318}.issubset(set(sel["bat_benchmark_top15"]["canonical_player_id"])))
        self.assertIn(305, set(sel["bowl_benchmark_top15"]["canonical_player_id"]))
        for rid in RETIRED_FROM_IPL:
            self.assertNotIn(rid, set(sel["bat_shortlist"]["canonical_player_id"]))
            self.assertNotIn(rid, set(sel["bowl_shortlist"]["canonical_player_id"]))

        cfg = PerformanceConfig(w_t20i=0.50, w_emerging=0.00, exclude_emerging_from_primary=True)
        oof_pri, _ = fit_predict_oof_baselines(
            self.raw_df, self.ds["matches"], self.ds["competitions"],
            include_suspected_dls=False, n_splits=5, exclude_emerging=True,
        )
        m_meta = self.ds["matches"][["match_id", "comp_id", "match_date"]].merge(
            self.ds["competitions"][["comp_id", "season", "comp_name"]], on="comp_id", how="left"
        )
        raw_pri = filter_dls_pool(
            self.raw_df, include_suspected_dls=False, exclude_emerging=True,
            matches_df=self.ds["matches"], competitions_df=self.ds["competitions"],
        ).merge(m_meta, on="match_id", how="left")
        raw_pri["season_yr"] = pd.to_numeric(raw_pri["season"].astype(str).str.slice(0, 4), errors="coerce").fillna(2024).astype(int)

        rr_ctx = compute_rr_2026_context(
            raw_pri, self.ds["matches"], self.ds["competitions"], self.ds["teams"], self.ds["playing_eleven"], self.pdir
        )
        self.assertEqual(rr_ctx["rr_2026_matches"], 16)
        self.assertIn(50284, rr_ctx["rr_2026_played_ids"])  # Vaibhav Sooryavanshi played for RR in IPL 2026

        bat_cards = build_detailed_player_profiles(
            sel["bat_shortlist"], self.bat_df, self.bowl_df, oof_pri, raw_pri,
            self.ds["playing_eleven"], self.ds["teams"], self.pdir, self.emg_bat,
            rr_ctx["rr_2026_played_ids"], "bat", cfg,
        )
        bowl_cards = build_detailed_player_profiles(
            sel["bowl_shortlist"], self.bowl_df, self.bat_df, oof_pri, raw_pri,
            self.ds["playing_eleven"], self.ds["teams"], self.pdir, self.emg_bowl,
            rr_ctx["rr_2026_played_ids"], "bowl", cfg,
        )

        # Exact pillar contribution sum == pri_z_lcb
        bat_sum = bat_cards["c_raa"] + bat_cards["c_surv"] + bat_cards["c_cons"] + bat_cards["c_hp"]
        bowl_sum = bowl_cards["c_rsaa"] + bowl_cards["c_waa"] + bowl_cards["c_cons"] + bowl_cards["c_hp"]
        np.testing.assert_allclose(bat_sum.values, bat_cards["pri_z_lcb"].values, atol=1e-6)
        np.testing.assert_allclose(bowl_sum.values, bowl_cards["pri_z_lcb"].values, atol=1e-6)

        # Verified all-rounder check: Axar Patel (2876) & Sunil Narine (781) vs Jasprit Bumrah (2933)
        bowl_by_id = bowl_cards.set_index("canonical_player_id")
        self.assertEqual(bowl_by_id.loc[2876, "all_rounder_class"], "Genuine All-Rounder")
        self.assertEqual(bowl_by_id.loc[2933, "all_rounder_class"], "Specialist")

        # Vaibhav Sooryavanshi (50284) has 0 HP evidence and flagged accurately
        bat_by_id = bat_cards.set_index("canonical_player_id")
        self.assertIn("ZERO_HP_EVIDENCE", str(bat_by_id.loc[50284, "risk_flags"]))
        self.assertTrue(bool(bat_by_id.loc[50284, "on_rr_2026_squad"]))

    def test_raw_playing_eleven_dedup_and_phase4_csv_artifacts(self):
        dedup_p11 = deduplicate_playing_eleven(self.ds["playing_eleven"])
        self.assertEqual(dedup_p11.duplicated(["match_id", "team_id", "player_id"]).sum(), 0)
        self.assertLess(len(dedup_p11), len(self.ds["playing_eleven"]))

        bat_short = pd.read_csv(DATA_PROCESSED / "phase4_bat_shortlist_cards.csv")
        bowl_short = pd.read_csv(DATA_PROCESSED / "phase4_bowl_shortlist_cards.csv")
        bat_bench = pd.read_csv(DATA_PROCESSED / "phase4_bat_benchmark15_cards.csv")
        bat_ctx = pd.read_csv(DATA_PROCESSED / "phase4_bat_context_metrics.csv")
        bowl_ctx = pd.read_csv(DATA_PROCESSED / "phase4_bowl_context_metrics.csv")
        bat_rej = pd.read_csv(DATA_PROCESSED / "phase4_bat_rejected_audit.csv")
        bowl_rej = pd.read_csv(DATA_PROCESSED / "phase4_bowl_rejected_audit.csv")
        with open(DATA_PROCESSED / "phase4_rr_2026_context.json", "r", encoding="utf-8") as f:
            ctx_json = json.load(f)

        self.assertEqual(len(bat_short), 10)
        self.assertEqual(len(bowl_short), 10)
        self.assertGreaterEqual(len(bat_rej), 10)
        self.assertGreaterEqual(len(bowl_rej), 10)
        self.assertTrue(set(bat_short["canonical_player_id"]).issubset(set(bat_ctx["canonical_player_id"])))
        self.assertTrue(set(bowl_short["canonical_player_id"]).issubset(set(bowl_ctx["canonical_player_id"])))

        valid_tiers = {"Strong evidence", "Moderate evidence", "Limited evidence"}
        for df in (bat_short, bowl_short, bat_bench):
            self.assertTrue(set(df["confidence_tier"]).issubset(valid_tiers))
            self.assertTrue(((df["emp_pct_elig_primary"] > 0.0) & (df["emp_pct_elig_primary"] <= 100.0)).all())
            np.testing.assert_array_equal(df["neff_hp_lt_15"].values, (df["neff_hp"] < 15.0).values)

        venues = {v["venue_label"] for v in ctx_json["rr_2026_context"]["venue_conditions_2023_2026"]}
        self.assertEqual(venues, {"All IPL (2023-26)", "Jaipur (SMS, vid=6)", "Guwahati (ACA, vid=1241)"})


if __name__ == "__main__":
    unittest.main()

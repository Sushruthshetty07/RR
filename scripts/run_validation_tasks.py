"""
Complete Verification and Validation Runner for Tasks 1 to 7.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_datasets, DATA_PROCESSED
from src.performance import (
    PerformanceConfig,
    COMP_TIER_IPL,
    COMP_TIER_SENIOR_T20I,
    COMP_TIER_EMERGING,
    classify_comp_tier,
    filter_dls_pool,
    fit_predict_oof_baselines,
    evaluate_batting_pressure,
    evaluate_bowling_pressure,
    apply_normal_shrinkage,
    PRIMARY_DLS_STATUSES,
)

def run_all():
    print("Loading datasets...")
    raw = load_raw_datasets()
    matches_df = raw["matches"]
    competitions_df = raw["competitions"]
    player_dir_path = DATA_PROCESSED / "canonical_player_directory.csv"
    player_dir = pd.read_csv(player_dir_path) if player_dir_path.exists() else None
    
    deliv_path = DATA_PROCESSED / "deliveries_with_pressure.csv.gz"
    print(f"Loading deliveries from {deliv_path}...")
    deliv_df = pd.read_csv(deliv_path)
    
    print("\n" + "="*80)
    print("TASK 1: RECONCILING CONFLICTING NUMBERS")
    print("="*80)
    wp_path = DATA_PROCESSED / "wp_benchmark_summary.json"
    p4_path = DATA_PROCESSED / "phase4_ext_wp_calibration.csv"
    
    with open(wp_path, "r") as f:
        wp_data = json.load(f)
    p4_df = pd.read_csv(p4_path)
    
    print("A. Win Probability ROC-AUC:")
    print("   - README Section 13 claims: 0.8099 (Logistic Regression), 0.8006 (RF), 0.7963 (GBDT)")
    print("   - phase4_ext_wp_calibration.csv values:")
    for _, r in p4_df.iterrows():
        print(f"     {r['slice']:<25}: ROC-AUC={r['roc_auc']:.4f}, LogLoss={r['log_loss']:.4f}, Brier={r['brier_score']:.4f}")
        
    print("\nB. Correlation between Pressure and Leverage:")
    print(f"   - README Section 13 claims: Pearson r = {wp_data['correlation_pi_vs_leverage']['pearson']}, Spearman rho = {wp_data['correlation_pi_vs_leverage']['spearman']}")
    print("   - phase4_ext_wp_calibration.csv values:")
    for _, r in p4_df.iterrows():
        print(f"     {r['slice']:<25}: Pearson r = {r['corr_pi_li_pearson']:+.4f}, Spearman rho = {r['corr_pi_li_spearman']:+.4f}")
        
    print("\n" + "="*80)
    print("TASK 2: PRESSURE-THRESHOLD SENSITIVITY (PI = 50, 60, 70)")
    print("="*80)
    off_bat = pd.read_csv(DATA_PROCESSED / "official_top10_high_pressure_batters.csv")
    off_bowl = pd.read_csv(DATA_PROCESSED / "official_top10_high_pressure_bowlers.csv")
    off_bat_ids = off_bat["canonical_player_id"].tolist()
    off_bowl_ids = off_bowl["canonical_player_id"].tolist()
    
    oof_pri, _ = fit_predict_oof_baselines(
        deliv_df, matches_df, competitions_df, include_suspected_dls=False, n_splits=5, exclude_emerging=True
    )
    
    results_threshold = {}
    for t in [50.0, 60.0, 70.0]:
        cfg = PerformanceConfig(high_pressure_cutoff=t)
        b_res, _ = evaluate_batting_pressure(oof_pri, cfg, player_dir=player_dir)
        w_res, _ = evaluate_bowling_pressure(oof_pri, cfg, player_dir=player_dir)
        b_hp = b_res[b_res["elig_primary"] & (b_res["neff_hp"] >= 30.0)].sort_values("hp_raa_lcb", ascending=False).reset_index(drop=True)
        w_hp = w_res[w_res["elig_primary"] & (w_res["neff_hp"] >= 30.0)].sort_values("hp_impact_lcb", ascending=False).reset_index(drop=True)
        results_threshold[t] = {"bat": b_hp, "bowl": w_hp}
        
    base_b = results_threshold[60.0]["bat"]
    base_w = results_threshold[60.0]["bowl"]
    
    print("\n--- BATTING THRESHOLD SENSITIVITY ---")
    for t in [50.0, 60.0, 70.0]:
        df_t = results_threshold[t]["bat"]
        t10 = df_t.head(10)
        t10_ids = t10["canonical_player_id"].tolist()
        overlap = len(set(t10_ids).intersection(set(off_bat_ids)))
        common = base_b.merge(df_t, on="canonical_player_id", suffixes=("_base", f"_{int(t)}"))
        rho, _ = spearmanr(common["hp_raa_lcb_base"], common[f"hp_raa_lcb_{int(t)}"])
        enters = [f"{name} (#{i+1})" for i, (pid, name) in enumerate(zip(t10["canonical_player_id"], t10["player_name"])) if pid not in off_bat_ids]
        leaves = [name for pid, name in zip(off_bat["canonical_player_id"], off_bat["player_name"]) if pid not in t10_ids]
        print(f"\nThreshold PI >= {int(t)}:")
        print(f"  Eligible Pool Size (Neff_HP >= 30): {len(df_t)}")
        print(f"  Top-10 Overlap with Official:       {overlap}/10 ({overlap*10}%)")
        print(f"  Spearman rho across eligible pool:  {rho:.4f}")
        print(f"  Top 10: {t10['player_name'].tolist()}")
        if enters:
            print(f"  Entered Top 10: {enters}")
        if leaves:
            print(f"  Left Top 10:    {leaves}")

    print("\n--- BOWLING THRESHOLD SENSITIVITY ---")
    for t in [50.0, 60.0, 70.0]:
        df_t = results_threshold[t]["bowl"]
        t10 = df_t.head(10)
        t10_ids = t10["canonical_player_id"].tolist()
        overlap = len(set(t10_ids).intersection(set(off_bowl_ids)))
        common = base_w.merge(df_t, on="canonical_player_id", suffixes=("_base", f"_{int(t)}"))
        rho, _ = spearmanr(common["hp_impact_lcb_base"], common[f"hp_impact_lcb_{int(t)}"])
        enters = [f"{name} (#{i+1})" for i, (pid, name) in enumerate(zip(t10["canonical_player_id"], t10["player_name"])) if pid not in off_bowl_ids]
        leaves = [name for pid, name in zip(off_bowl["canonical_player_id"], off_bowl["player_name"]) if pid not in t10_ids]
        print(f"\nThreshold PI >= {int(t)}:")
        print(f"  Eligible Pool Size (Neff_HP >= 30): {len(df_t)}")
        print(f"  Top-10 Overlap with Official:       {overlap}/10 ({overlap*10}%)")
        print(f"  Spearman rho across eligible pool:  {rho:.4f}")
        print(f"  Top 10: {t10['player_name'].tolist()}")
        if enters:
            print(f"  Entered Top 10: {enters}")
        if leaves:
            print(f"  Left Top 10:    {leaves}")

    print("\n" + "="*80)
    print("TASK 3: UNCERTAINTY ON OFFICIAL HIGH-PRESSURE LISTS (CLUSTER BOOTSTRAP)")
    print("="*80)
    cfg = PerformanceConfig(high_pressure_cutoff=60.0)
    
    pri = oof_pri.copy()
    pri["comp_tier"] = classify_comp_tier(pri)
    pri = pri[pri["comp_tier"] != COMP_TIER_EMERGING].copy()
    pri["comp_w"] = np.where(pri["comp_tier"] == COMP_TIER_IPL, 1.0, float(cfg.w_t20i))
    
    # Batting HP
    bat_p = pri[pri["is_batter_ball_faced"] & (pri["pressure_bat"] >= 60.0)].copy()
    bat_p["w"] = bat_p["comp_w"] * ((bat_p["pressure_bat"] / 100.0) ** cfg.pressure_gamma)
    bat_p["w2"] = bat_p["w"] ** 2
    bat_p["d_raa"] = 100.0 * (bat_p["runs"] - bat_p["bat_exp_runs"])
    bat_p["wy_raa"] = bat_p["w"] * bat_p["d_raa"]
    
    unique_matches_b = np.sort(bat_p["match_id"].unique())
    M_b = len(unique_matches_b)
    m_to_i_b = {int(mid): i for i, mid in enumerate(unique_matches_b)}
    
    all_el_bat_pids = np.sort(base_b["canonical_player_id"].unique())
    p_to_j_b = {int(pid): j for j, pid in enumerate(all_el_bat_pids)}
    P_b = len(all_el_bat_pids)
    
    bat_pm = bat_p[bat_p["canonical_striker_id"].isin(p_to_j_b)].groupby(["canonical_striker_id", "match_id"], as_index=False).agg(
        w=("w", "sum"), w2=("w2", "sum"), wy=("wy_raa", "sum")
    )
    p_idx_b = bat_pm["canonical_striker_id"].map(p_to_j_b).values
    m_idx_b = bat_pm["match_id"].map(m_to_i_b).values
    w_vb = bat_pm["w"].values
    w2_vb = bat_pm["w2"].values
    wy_vb = bat_pm["wy"].values
    
    sig2_b = float(np.average(bat_p["d_raa"] ** 2, weights=bat_p["w"]))
    
    # Bowling HP
    bowl_p = pri[pri["is_legal_ball"] & (pri["pressure_bowl"] >= 60.0)].copy()
    bowl_p["w"] = bowl_p["comp_w"] * ((bowl_p["pressure_bowl"] / 100.0) ** cfg.pressure_gamma)
    bowl_p["wl"] = bowl_p["w"]
    bowl_p["wl2"] = bowl_p["wl"] ** 2
    bowl_p["d_rsaa"] = 6.0 * (bowl_p["bowl_exp_runs"] - bowl_p["bowler_runs_conceded"])
    bowl_p["d_waa"] = 24.0 * (bowl_p["bowler_wkt_int"].astype(float) - bowl_p["bowl_exp_wkt"])
    bowl_p["d_net"] = bowl_p["d_rsaa"] + (cfg.wicket_run_value / 4.0) * bowl_p["d_waa"]
    bowl_p["wy_net"] = bowl_p["wl"] * bowl_p["d_net"]
    
    unique_matches_l = np.sort(bowl_p["match_id"].unique())
    M_l = len(unique_matches_l)
    m_to_i_l = {int(mid): i for i, mid in enumerate(unique_matches_l)}
    
    all_el_bowl_pids = np.sort(base_w["canonical_player_id"].unique())
    p_to_j_l = {int(pid): j for j, pid in enumerate(all_el_bowl_pids)}
    P_l = len(all_el_bowl_pids)
    
    bowl_pm = bowl_p[bowl_p["canonical_bowler_id"].isin(p_to_j_l)].groupby(["canonical_bowler_id", "match_id"], as_index=False).agg(
        wl=("wl", "sum"), wl2=("wl2", "sum"), wy=("wy_net", "sum")
    )
    p_idx_l = bowl_pm["canonical_bowler_id"].map(p_to_j_l).values
    m_idx_l = bowl_pm["match_id"].map(m_to_i_l).values
    wl_vl = bowl_pm["wl"].values
    wl2_vl = bowl_pm["wl2"].values
    wy_vl = bowl_pm["wy"].values
    
    sig2_l = float(np.average(bowl_p["d_net"] ** 2, weights=bowl_p["wl"]))
    
    n_boot = 2000
    rng = np.random.default_rng(42)
    
    print(f"Running B={n_boot} match-cluster bootstrap for High-Pressure Top 10 uncertainty...")
    bat_ranks_boot = np.full((n_boot, P_b), np.nan)
    bowl_ranks_boot = np.full((n_boot, P_l), np.nan)
    
    sw_b0 = np.bincount(p_idx_b, weights=w_vb, minlength=P_b)
    sw2_b0 = np.bincount(p_idx_b, weights=w2_vb, minlength=P_b)
    swy_b0 = np.bincount(p_idx_b, weights=wy_vb, minlength=P_b)
    neff_b0 = np.where(sw2_b0 > 0, sw_b0**2 / sw2_b0, 0.0)
    raw_b0 = np.where(sw_b0 > 0, swy_b0 / sw_b0, 0.0)
    eb_b0, meta_b0 = apply_normal_shrinkage(pd.Series(raw_b0), pd.Series(neff_b0), sig2_b, min_neff_fit=10.0, lcb_z=0.5)
    mu0_b, tau2_b = meta_b0["mu0"], meta_b0["tau2"]
    
    sw_l0 = np.bincount(p_idx_l, weights=wl_vl, minlength=P_l)
    sw2_l0 = np.bincount(p_idx_l, weights=wl2_vl, minlength=P_l)
    swy_l0 = np.bincount(p_idx_l, weights=wy_vl, minlength=P_l)
    neff_l0 = np.where(sw2_l0 > 0, sw_l0**2 / sw2_l0, 0.0)
    raw_l0 = np.where(sw_l0 > 0, swy_l0 / sw_l0, 0.0)
    eb_l0, meta_l0 = apply_normal_shrinkage(pd.Series(raw_l0), pd.Series(neff_l0), sig2_l, min_neff_fit=10.0, lcb_z=0.5)
    mu0_l, tau2_l = meta_l0["mu0"], meta_l0["tau2"]
    
    for b in range(n_boot):
        drawn_b = rng.integers(0, M_b, size=M_b)
        m_mult_b = np.bincount(drawn_b, minlength=M_b)
        row_mult_b = m_mult_b[m_idx_b]
        b_sw = np.bincount(p_idx_b, weights=w_vb * row_mult_b, minlength=P_b)
        b_sw2 = np.bincount(p_idx_b, weights=w2_vb * row_mult_b, minlength=P_b)
        b_swy = np.bincount(p_idx_b, weights=wy_vb * row_mult_b, minlength=P_b)
        
        b_neff = np.where(b_sw2 > 0, (b_sw**2) / b_sw2, 0.0)
        b_raw = np.where(b_sw > 0, b_swy / b_sw, mu0_b)
        b_se2 = np.where(b_neff > 0, sig2_b / np.maximum(1e-9, b_neff), np.inf)
        b_B = np.where(b_neff > 0, tau2_b / (tau2_b + b_se2), 0.0)
        b_post = b_B * b_raw + (1.0 - b_B) * mu0_b
        b_sd = np.sqrt(np.maximum(0.0, tau2_b * (1.0 - b_B)))
        b_lcb = b_post - 0.50 * b_sd
        b_lcb = np.where(b_neff >= 30.0, b_lcb, -9999.0)
        ranks_b = pd.Series(b_lcb).rank(ascending=False, method="min").values
        bat_ranks_boot[b] = ranks_b
        
        drawn_l = rng.integers(0, M_l, size=M_l)
        m_mult_l = np.bincount(drawn_l, minlength=M_l)
        row_mult_l = m_mult_l[m_idx_l]
        l_sw = np.bincount(p_idx_l, weights=wl_vl * row_mult_l, minlength=P_l)
        l_sw2 = np.bincount(p_idx_l, weights=wl2_vl * row_mult_l, minlength=P_l)
        l_swy = np.bincount(p_idx_l, weights=wy_vl * row_mult_l, minlength=P_l)
        
        l_neff = np.where(l_sw2 > 0, (l_sw**2) / l_sw2, 0.0)
        l_raw = np.where(l_sw > 0, l_swy / l_sw, mu0_l)
        l_se2 = np.where(l_neff > 0, sig2_l / np.maximum(1e-9, l_neff), np.inf)
        l_B = np.where(l_neff > 0, tau2_l / (tau2_l + l_se2), 0.0)
        l_post = l_B * l_raw + (1.0 - l_B) * mu0_l
        l_sd = np.sqrt(np.maximum(0.0, tau2_l * (1.0 - l_B)))
        l_lcb = l_post - 0.50 * l_sd
        l_lcb = np.where(l_neff >= 30.0, l_lcb, -9999.0)
        ranks_l = pd.Series(l_lcb).rank(ascending=False, method="min").values
        bowl_ranks_boot[b] = ranks_l

    print("\n--- BATTING OFFICIAL TOP 10 BOOTSTRAP UNCERTAINTY ---")
    bat_boot_summary = []
    for rank_idx, row in off_bat.iterrows():
        pid = row["canonical_player_id"]
        pname = row["player_name"]
        j = p_to_j_b[pid]
        r_draws = bat_ranks_boot[:, j]
        med_r = float(np.median(r_draws))
        p05_r = float(np.percentile(r_draws, 5))
        p95_r = float(np.percentile(r_draws, 95))
        pct_t10 = float((r_draws <= 10).mean() * 100.0)
        bat_boot_summary.append({
            "official_rank": rank_idx + 1,
            "player_name": pname,
            "hp_lcb": row["hp_score_lcb_runs_per_100b"],
            "neff_hp": row["neff_hp_opportunities"],
            "median_rank": med_r,
            "rank_p05": p05_r,
            "rank_p95": p95_r,
            "top10_pct": pct_t10,
        })
    df_bat_boot = pd.DataFrame(bat_boot_summary)
    print(df_bat_boot.to_string(index=False))

    print("\n--- BOWLING OFFICIAL TOP 10 BOOTSTRAP UNCERTAINTY ---")
    bowl_boot_summary = []
    for rank_idx, row in off_bowl.iterrows():
        pid = row["canonical_player_id"]
        pname = row["player_name"]
        j = p_to_j_l[pid]
        r_draws = bowl_ranks_boot[:, j]
        med_r = float(np.median(r_draws))
        p05_r = float(np.percentile(r_draws, 5))
        p95_r = float(np.percentile(r_draws, 95))
        pct_t10 = float((r_draws <= 10).mean() * 100.0)
        bowl_boot_summary.append({
            "official_rank": rank_idx + 1,
            "player_name": pname,
            "hp_lcb": row["hp_impact_lcb_runs_saved_per_ov"],
            "neff_hp": row["neff_hp_opportunities"],
            "median_rank": med_r,
            "rank_p05": p05_r,
            "rank_p95": p95_r,
            "top10_pct": pct_t10,
        })
    df_bowl_boot = pd.DataFrame(bowl_boot_summary)
    print(df_bowl_boot.to_string(index=False))

    print("\n" + "="*80)
    print("TASK 4: PRESSURE DEFINITION ROBUSTNESS (LEVERAGE-BASED DEFINITION)")
    print("="*80)
    p4_li_path = DATA_PROCESSED / "phase4_ext_li_sensitivity.csv"
    if p4_li_path.exists():
        df_li = pd.read_csv(p4_li_path)
        print("Leverage Sensitivity Summary Table:")
        print(df_li[["li_regime", "pct_pool_balls", "bat_top10_vs_hp_raa", "bat_rho_vs_hp_raa", "bowl_top10_vs_hp_impact", "bowl_rho_vs_hp_impact"]].to_string(index=False))

    print("\n" + "="*80)
    print("TASK 5: VALIDATE PRESSURE AGAINST OUTCOMES (MONOTONICITY & PHASE)")
    print("="*80)
    df_deliv = oof_pri.copy()
    df_deliv = df_deliv[df_deliv["is_batter_ball_faced"] == 1].copy()
    df_deliv["pressure_band"] = pd.cut(
        df_deliv["pressure_bat"],
        bins=[-0.1, 20.0, 40.0, 60.0, 80.0, 100.1],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"]
    )
    
    agg_ovr = df_deliv.groupby("pressure_band", observed=False).agg(
        n_balls=("runs", "count"),
        total_runs=("runs", "sum"),
        mean_runs=("runs", "mean"),
        sr=("runs", lambda s: float(s.mean() * 100.0)),
        dot_pct=("runs", lambda s: float((s == 0).mean() * 100.0)),
        boundary_pct=("runs", lambda s: float(s.isin([4, 6]).mean() * 100.0)),
        dismissals=("striker_out", "sum"),
        dismissal_rate_pct=("striker_out", lambda s: float(s.mean() * 100.0)),
    ).reset_index()
    print("\n--- Overall Outcome Metrics Across Pressure Bands ---")
    print(agg_ovr.to_string(index=False))
    
    agg_phase = df_deliv.groupby(["match_phase", "pressure_band"], observed=False).agg(
        n_balls=("runs", "count"),
        sr=("runs", lambda s: float(s.mean() * 100.0)),
        dot_pct=("runs", lambda s: float((s == 0).mean() * 100.0)),
        boundary_pct=("runs", lambda s: float(s.isin([4, 6]).mean() * 100.0)),
        dismissal_rate_pct=("striker_out", lambda s: float(s.mean() * 100.0)),
    ).reset_index()
    print("\n--- Phase-Stratified Outcome Metrics Across Pressure Bands ---")
    print(agg_phase.to_string(index=False))

    print("\n" + "="*80)
    print("TASK 6: OUT-OF-SAMPLE VALIDATION (2022-24 to 2025-26 TEMPORAL HOLDOUT)")
    print("="*80)
    holdout_path = DATA_PROCESSED / "final_temporal_player_holdout.csv"
    if holdout_path.exists():
        df_h = pd.read_csv(holdout_path)
        print("Existing Temporal Player Holdout Results:")
        print(df_h.to_string(index=False))
        
    pri_h = oof_pri[
        oof_pri["dls_status"].isin(PRIMARY_DLS_STATUSES)
        & oof_pri["comp_tier"].isin([COMP_TIER_IPL, COMP_TIER_SENIOR_T20I])
        & oof_pri["pressure_bat"].notna()
        & oof_pri["pressure_bowl"].notna()
    ].copy()
    
    comp_w_map = {COMP_TIER_IPL: 1.0, COMP_TIER_SENIOR_T20I: float(cfg.w_t20i)}
    pri_h["comp_w"] = pri_h["comp_tier"].map(comp_w_map).fillna(0.0)
    
    tr_b_hp = pri_h[(pri_h["season_year"] < 2025) & pri_h["is_batter_ball_faced"] & (pri_h["pressure_bat"] >= 60.0)].copy()
    te_b_hp = pri_h[(pri_h["season_year"] >= 2025) & pri_h["is_batter_ball_faced"] & (pri_h["pressure_bat"] >= 60.0)].copy()
    
    tr_l_hp = pri_h[(pri_h["season_year"] < 2025) & pri_h["is_legal_ball"] & (pri_h["pressure_bowl"] >= 60.0)].copy()
    te_l_hp = pri_h[(pri_h["season_year"] >= 2025) & pri_h["is_legal_ball"] & (pri_h["pressure_bowl"] >= 60.0)].copy()
    
    print(f"\nSample Sizes (Deliveries):")
    print(f"  Batting HP Train (<=2024): {len(tr_b_hp):,} balls; Test (2025-26): {len(te_b_hp):,} balls")
    print(f"  Bowling HP Train (<=2024): {len(tr_l_hp):,} balls; Test (2025-26): {len(te_l_hp):,} balls")
    
    tr_b_hp["w"] = tr_b_hp["comp_w"] * ((tr_b_hp["pressure_bat"] / 100.0) ** cfg.pressure_gamma)
    tr_b_hp["w2"] = tr_b_hp["w"] ** 2
    tr_b_hp["d_raa"] = 100.0 * (tr_b_hp["runs"] - tr_b_hp["bat_exp_runs"])
    tr_b_hp["wd_raa"] = tr_b_hp["w"] * tr_b_hp["d_raa"]
    
    te_b_hp["w"] = te_b_hp["comp_w"] * ((te_b_hp["pressure_bat"] / 100.0) ** cfg.pressure_gamma)
    te_b_hp["w2"] = te_b_hp["w"] ** 2
    te_b_hp["d_raa"] = 100.0 * (te_b_hp["runs"] - te_b_hp["bat_exp_runs"])
    te_b_hp["wd_raa"] = te_b_hp["w"] * te_b_hp["d_raa"]
    
    agg_b_tr = tr_b_hp.groupby("canonical_striker_id").agg(sw=("w", "sum"), sw2=("w2", "sum"), swd=("wd_raa", "sum")).reset_index()
    agg_b_tr["neff_tr"] = agg_b_tr["sw"]**2 / agg_b_tr["sw2"]
    agg_b_tr["raw_tr"] = agg_b_tr["swd"] / agg_b_tr["sw"]
    
    agg_b_te = te_b_hp.groupby("canonical_striker_id").agg(sw=("w", "sum"), sw2=("w2", "sum"), swd=("wd_raa", "sum")).reset_index()
    agg_b_te["neff_te"] = agg_b_te["sw"]**2 / agg_b_te["sw2"]
    agg_b_te["raw_te"] = agg_b_te["swd"] / agg_b_te["sw"]
    
    m_b = agg_b_tr.merge(agg_b_te, on="canonical_striker_id", suffixes=("_tr", "_te"))
    print("\nBatting High-Pressure Temporal Holdout Generalization:")
    for min_n in [10.0, 15.0, 20.0, 30.0]:
        sub = m_b[(m_b["neff_tr"] >= min_n) & (m_b["neff_te"] >= min_n)]
        if len(sub) >= 5:
            rho, p_val = spearmanr(sub["raw_tr"], sub["raw_te"])
            r_val, _ = pearsonr(sub["raw_tr"], sub["raw_te"])
            print(f"  Batting HP Holdout (Neff >= {min_n:.0f}): N={len(sub):3d} players, Spearman rho={rho:+.4f} (p={p_val:.4f}), Pearson r={r_val:+.4f}")

    tr_l_hp["w"] = tr_l_hp["comp_w"] * ((tr_l_hp["pressure_bowl"] / 100.0) ** cfg.pressure_gamma)
    tr_l_hp["w2"] = tr_l_hp["w"] ** 2
    tr_l_hp["d_rsaa"] = 6.0 * (tr_l_hp["bowl_exp_runs"] - tr_l_hp["bowler_runs_conceded"])
    tr_l_hp["d_waa"] = 24.0 * (tr_l_hp["bowler_wkt_int"].astype(float) - tr_l_hp["bowl_exp_wkt"])
    tr_l_hp["d_net"] = tr_l_hp["d_rsaa"] + (cfg.wicket_run_value / 4.0) * tr_l_hp["d_waa"]
    tr_l_hp["wd_net"] = tr_l_hp["w"] * tr_l_hp["d_net"]
    
    te_l_hp["w"] = te_l_hp["comp_w"] * ((te_l_hp["pressure_bowl"] / 100.0) ** cfg.pressure_gamma)
    te_l_hp["w2"] = te_l_hp["w"] ** 2
    te_l_hp["d_rsaa"] = 6.0 * (te_l_hp["bowl_exp_runs"] - te_l_hp["bowler_runs_conceded"])
    te_l_hp["d_waa"] = 24.0 * (te_l_hp["bowler_wkt_int"].astype(float) - te_l_hp["bowl_exp_wkt"])
    te_l_hp["d_net"] = te_l_hp["d_rsaa"] + (cfg.wicket_run_value / 4.0) * te_l_hp["d_waa"]
    te_l_hp["wd_net"] = te_l_hp["w"] * te_l_hp["d_net"]
    
    agg_l_tr = tr_l_hp.groupby("canonical_bowler_id").agg(sw=("w", "sum"), sw2=("w2", "sum"), swd=("wd_net", "sum")).reset_index()
    agg_l_tr["neff_tr"] = agg_l_tr["sw"]**2 / agg_l_tr["sw2"]
    agg_l_tr["raw_tr"] = agg_l_tr["swd"] / agg_l_tr["sw"]
    
    agg_l_te = te_l_hp.groupby("canonical_bowler_id").agg(sw=("w", "sum"), sw2=("w2", "sum"), swd=("wd_net", "sum")).reset_index()
    agg_l_te["neff_te"] = agg_l_te["sw"]**2 / agg_l_te["sw2"]
    agg_l_te["raw_te"] = agg_l_te["swd"] / agg_l_te["sw"]
    
    m_l = agg_l_tr.merge(agg_l_te, on="canonical_bowler_id", suffixes=("_tr", "_te"))
    print("\nBowling High-Pressure Temporal Holdout Generalization:")
    for min_n in [10.0, 15.0, 20.0, 30.0]:
        sub = m_l[(m_l["neff_tr"] >= min_n) & (m_l["neff_te"] >= min_n)]
        if len(sub) >= 5:
            rho, p_val = spearmanr(sub["raw_tr"], sub["raw_te"])
            r_val, _ = pearsonr(sub["raw_tr"], sub["raw_te"])
            print(f"  Bowling HP Holdout (Neff >= {min_n:.0f}): N={len(sub):3d} players, Spearman rho={rho:+.4f} (p={p_val:.4f}), Pearson r={r_val:+.4f}")
        else:
            print(f"  Bowling HP Holdout (Neff >= {min_n:.0f}): N={len(sub):3d} players (insufficient sample size < 5)")

    print("\n" + "="*80)
    print("TASK 7: OPTIONAL VENUE & OPPOSITION-STRENGTH SENSITIVITY CHECK")
    print("="*80)
    bat_cmp_path = DATA_PROCESSED / "phase4_ext_bat_comparison.csv"
    bowl_cmp_path = DATA_PROCESSED / "phase4_ext_bowl_comparison.csv"
    if bat_cmp_path.exists() and bowl_cmp_path.exists():
        df_bc = pd.read_csv(bat_cmp_path)
        df_wc = pd.read_csv(bowl_cmp_path)
        
        # Check two-way matchup ranking vs approved HP rankings
        # Batters: tw_bat_eff_100
        el_b = df_bc[df_bc["elig_primary"] & (df_bc["neff_hp"] >= 30.0)].copy()
        if "tw_bat_eff_100" in el_b.columns:
            el_b_sorted = el_b.sort_values("tw_bat_eff_100", ascending=False).reset_index(drop=True)
            t10_tw_b = el_b_sorted.head(10)
            overlap_b = len(set(t10_tw_b["canonical_player_id"]).intersection(set(off_bat_ids)))
            rho_b, _ = spearmanr(el_b["hp_raa_post"], el_b["tw_bat_eff_100"])
            print(f"Batting: Two-Way Opponent-Adjusted Top 10 Overlap with Official HP: {overlap_b}/10 ({overlap_b*10}%)")
            print(f"Batting: Spearman rho between HP-RAA and Two-Way Ridge Effect:     {rho_b:.4f}")
            print(f"Batting Top 10 under Two-Way Ridge: {t10_tw_b['player_name'].tolist()}")
            
        el_w = df_wc[df_wc["elig_primary"] & (df_wc["neff_hp"] >= 30.0)].copy()
        if "tw_bowl_eff_po" in el_w.columns:
            el_w_sorted = el_w.sort_values("tw_bowl_eff_po", ascending=False).reset_index(drop=True)
            t10_tw_w = el_w_sorted.head(10)
            overlap_w = len(set(t10_tw_w["canonical_player_id"]).intersection(set(off_bowl_ids)))
            rho_w, _ = spearmanr(el_w["hp_impact_post"], el_w["tw_bowl_eff_po"])
            print(f"\nBowling: Two-Way Opponent-Adjusted Top 10 Overlap with Official HP: {overlap_w}/10 ({overlap_w*10}%)")
            print(f"Bowling: Spearman rho between HP-Impact and Two-Way Ridge Effect:  {rho_w:.4f}")
            print(f"Bowling Top 10 under Two-Way Ridge: {t10_tw_w['player_name'].tolist()}")

    print("\n" + "="*80)
    print("ALL VALIDATION TASKS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_all()

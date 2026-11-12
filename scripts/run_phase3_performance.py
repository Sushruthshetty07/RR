"""
Phase 3 Diagnostic & Sensitivity Runner (Pre-Ranking Methodology Approval):
- Enforces DLS confidence policy (Primary = standard + dls_pre_innings_confirmed;
  Sensitivity = + suspected DLS; always excludes 143 ambiguous DLS & 38 Super Overs).
- Fits & validates 5-fold GroupKFold (by match_id) Expected Runs & Expected Wickets baselines.
- Evaluates Kish effective sample sizes (N_eff) and eligibility cutoffs across IPL vs T20I.
- Fits Empirical Bayes Normal-Normal & Beta-Binomial shrinkage hyperparameters.
- Performs sensitivity analyses across competition weights (w_t20i in {0..1}), DLS inclusion,
  pressure weighting exponent gamma, LCB z-penalty, and composite vs EV weighting.
- Summarizes anonymous preliminary player performance distributions & saves Figures 13_..18_.
(Strictly NO Top 10 rankings or player leaderboard output generated in this stage.)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_datasets, DATA_PROCESSED
from src.performance import (
    PerformanceConfig,
    fit_predict_oof_baselines,
    evaluate_batting_pressure,
    evaluate_bowling_pressure,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "Segoe UI", "DejaVu Sans", "Arial"

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def _dist_table(df_in: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df_in[c].dropna()
        rows.append(
            {
                "Metric": c,
                "N": len(s),
                "Mean": s.mean(),
                "Std": s.std(),
                "Min": s.min(),
                "P10": s.quantile(0.10),
                "P25": s.quantile(0.25),
                "Median": s.median(),
                "P75": s.quantile(0.75),
                "P90": s.quantile(0.90),
                "P95": s.quantile(0.95),
                "Max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("=== 1. LOADING PROCESSED PRESSURE DATA & ENFORCING DLS POLICY ===")
    ds = load_raw_datasets()
    df = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz", low_memory=False)
    player_dir = pd.read_csv(DATA_PROCESSED / "canonical_player_directory.csv")

    print(f"Loaded deliveries: {len(df)} | Matches: {df['match_id'].nunique()}")
    print("DLS status partition in dataset:")
    print(df["dls_status"].value_counts().to_string())

    # Fit primary and suspected-DLS OOF baselines
    print("\n=== 2. ZERO-LEAKAGE OOF EXPECTED BASELINES (5-FOLD GROUPKFOLD BY MATCH_ID) ===")
    oof_pri, diag_pri = fit_predict_oof_baselines(
        df, ds["matches"], ds["competitions"], include_suspected_dls=False, n_splits=5
    )
    oof_sens, diag_sens = fit_predict_oof_baselines(
        df, ds["matches"], ds["competitions"], include_suspected_dls=True, n_splits=5
    )

    print(f"Primary pool deliveries: {diag_pri['n_pool_deliveries']} "
          f"(bat_balls_faced={diag_pri['n_bat_balls_faced']}, bowl_legal={diag_pri['n_bowl_legal_balls']})")
    print(f"Sensitivity pool (+suspected DLS): {diag_sens['n_pool_deliveries']} "
          f"(+{diag_sens['n_pool_deliveries'] - diag_pri['n_pool_deliveries']} deliveries)")

    print("\nOOF Baseline Performance Metrics (Primary Pool):")
    for k, v in diag_pri.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}")
        else:
            print(f"  {k}: {v}")

    # Group calibration checks
    bat_rows = oof_pri[oof_pri["is_batter_ball_faced"]].copy()
    bat_rows["pi_q"] = pd.qcut(bat_rows["pressure_bat"], 4)
    print("\nBatting Calibration by (is_ipl, match_phase):")
    print(bat_rows.groupby(["is_ipl", "match_phase"])[["runs", "bat_exp_runs", "striker_out", "bat_exp_wkt"]].mean().round(4).to_string())
    print("\nBatting Calibration by Pressure Quartile:")
    print(bat_rows.groupby("pi_q", observed=False)[["runs", "bat_exp_runs", "striker_out", "bat_exp_wkt"]].mean().round(4).to_string())

    bowl_rows = oof_pri.copy()
    bowl_rows["pi_q"] = pd.qcut(bowl_rows["pressure_bowl"], 4)
    print("\nBowling Calibration by (is_ipl, match_phase) [All Deliveries Conserving Extras]:")
    b_cal = bowl_rows.groupby(["is_ipl", "match_phase"]).agg(
        actual_conc_per_leg=("bowler_runs_conceded", lambda s: s.sum() / bowl_rows.loc[s.index, "is_legal_ball"].sum()),
        exp_conc_per_leg=("bowl_exp_runs", lambda s: s.sum() / bowl_rows.loc[s.index, "is_legal_ball"].sum()),
        actual_wkt_per_leg=("bowler_wkt_int", lambda s: s.sum() / bowl_rows.loc[s.index, "is_legal_ball"].sum()),
        exp_wkt_per_leg=("bowl_exp_wkt", lambda s: s.sum() / bowl_rows.loc[s.index, "is_legal_ball"].sum()),
    )
    print(b_cal.round(4).to_string())

    # Evaluate primary configuration (w_t20i = 0.50)
    print("\n=== 3. PRIMARY EVALUATION & BAYESIAN SHRINKAGE HYPERPARAMETERS ===")
    cfg_pri = PerformanceConfig(w_t20i=0.50)
    bat_df, bat_meta = evaluate_batting_pressure(oof_pri, cfg_pri, player_dir=player_dir)
    bowl_df, bowl_meta = evaluate_bowling_pressure(oof_pri, cfg_pri, player_dir=player_dir)

    # Save anonymous/preliminary tables (unsorted by rank -> sort strictly by canonical_player_id)
    bat_df.sort_values("canonical_player_id").to_csv(DATA_PROCESSED / "phase3_batting_prelim.csv", index=False)
    bowl_df.sort_values("canonical_player_id").to_csv(DATA_PROCESSED / "phase3_bowling_prelim.csv", index=False)

    print("Batting Empirical Bayes Parameters (gamma=2.0, w_t20i=0.50):")
    for m_name, m_dict in bat_meta.items():
        print(f"  [{m_name}] { {k: round(v, 4) if isinstance(v, float) else v for k, v in m_dict.items()} }")

    print("\nBowling Empirical Bayes Parameters (gamma=2.0, w_t20i=0.50):")
    for m_name, m_dict in bowl_meta.items():
        print(f"  [{m_name}] { {k: round(v, 4) if isinstance(v, float) else v for k, v in m_dict.items()} }")

    # Sample-Size & Eligibility Grid
    print("\n=== 4. SAMPLE-SIZE & ELIGIBILITY ANALYSIS (KISH N_EFF x IPL BALLS) ===")
    neff_cuts = [15, 30, 50, 75, 100]
    ipl_cuts = [0, 30, 60, 100]
    elig_rows = []
    for nc in neff_cuts:
        for ic in ipl_cuts:
            b_cnt = int(((bat_df["neff"] >= nc) & (bat_df["ipl_balls"] >= ic)).sum())
            w_cnt = int(((bowl_df["neff"] >= nc) & (bowl_df["ipl_legal_balls"] >= ic)).sum())
            elig_rows.append({"min_Neff": nc, "min_IPL_balls": ic, "Eligible_Batters": b_cnt, "Eligible_Bowlers": w_cnt})
    elig_grid = pd.DataFrame(elig_rows)
    print(elig_grid.to_string(index=False))

    print("\nSample Size Quantiles (N_eff, IPL balls, High-Pressure >=60 balls):")
    print("Batters (All N={}):".format(len(bat_df)))
    print(bat_df[["total_balls", "ipl_balls", "hp60_balls", "neff", "neff_hp", "meff_inn"]].describe(percentiles=[0.5, 0.75, 0.9, 0.95]).round(1).to_string())
    print("\nBowlers (All N={}):".format(len(bowl_df)))
    print(bowl_df[["total_legal_balls", "ipl_legal_balls", "hp60_legal_balls", "neff", "neff_hp", "meff_spell"]].describe(percentiles=[0.5, 0.75, 0.9, 0.95]).round(1).to_string())

    # Sensitivity Analyses
    print("\n=== 5. VALIDATION & SENSITIVITY ANALYSES ===")
    # 5a. Competition weighting sweep (0.00, 0.25, 0.50, 0.75, 1.00)
    wt_list = [0.00, 0.25, 0.50, 0.75, 1.00]
    bat_wt_scores = {}
    bowl_wt_scores = {}
    wt_summary = []

    for wt in wt_list:
        cfg_w = PerformanceConfig(w_t20i=wt)
        b_w, _ = evaluate_batting_pressure(oof_pri, cfg_w)
        w_w, _ = evaluate_bowling_pressure(oof_pri, cfg_w)
        bat_wt_scores[wt] = b_w.set_index("canonical_player_id")["pri_score_lcb"]
        bowl_wt_scores[wt] = w_w.set_index("canonical_player_id")["pri_score_lcb"]

        # Compare against primary w_t20i=0.50 on IPL-active eligible players (ipl>=30, total_balls>=60)
        b_ref = bat_df[bat_df["elig_primary"]]["canonical_player_id"]
        w_ref = bowl_df[bowl_df["elig_primary"]]["canonical_player_id"]
        sp_b_050 = spearmanr(bat_df.set_index("canonical_player_id").loc[b_ref, "pri_score_lcb"], bat_wt_scores[wt].loc[b_ref]).statistic
        sp_w_050 = spearmanr(bowl_df.set_index("canonical_player_id").loc[w_ref, "pri_score_lcb"], bowl_wt_scores[wt].loc[w_ref]).statistic

        wt_summary.append(
            {
                "w_t20i": wt,
                "Bat_Elig_Pri": int(b_w["elig_primary"].sum()),
                "Bat_Elig_Broad": int(b_w["elig_broad"].sum()),
                "Bat_Sp_vs_0.50": sp_b_050,
                "Bowl_Elig_Pri": int(w_w["elig_primary"].sum()),
                "Bowl_Elig_Broad": int(w_w["elig_broad"].sum()),
                "Bowl_Sp_vs_0.50": sp_w_050,
            }
        )
    wt_df = pd.DataFrame(wt_summary)
    print("Competition Weighting Sweep (Spearman r on primary eligible IPL pool vs w_t20i=0.50):")
    print(wt_df.round(4).to_string(index=False))

    # Pairwise Spearman matrix across w_t20i on IPL-exposed eligible pool
    b_elig_ids = bat_df[bat_df["elig_primary"]]["canonical_player_id"]
    w_elig_ids = bowl_df[bowl_df["elig_primary"]]["canonical_player_id"]
    b_mat = pd.DataFrame({f"w={wt:.2f}": bat_wt_scores[wt].loc[b_elig_ids] for wt in wt_list}).corr(method="spearman")
    w_mat = pd.DataFrame({f"w={wt:.2f}": bowl_wt_scores[wt].loc[w_elig_ids] for wt in wt_list}).corr(method="spearman")
    print("\nBatting Pairwise Spearman Matrix (w_t20i on Primary Eligible):")
    print(b_mat.round(4).to_string())
    print("\nBowling Pairwise Spearman Matrix (w_t20i on Primary Eligible):")
    print(w_mat.round(4).to_string())

    # 5b. DLS Policy, Gamma, LCB z, and Composite vs EV sensitivities
    bat_dls_sens, _ = evaluate_batting_pressure(oof_sens, cfg_pri)
    bowl_dls_sens, _ = evaluate_bowling_pressure(oof_sens, cfg_pri)

    sens_rows = []
    # Suspected DLS check
    m_b_dls = bat_df[bat_df["elig_primary"]][["canonical_player_id", "pri_score_lcb"]].merge(
        bat_dls_sens[["canonical_player_id", "pri_score_lcb"]], on="canonical_player_id", suffixes=("_pri", "_sens")
    )
    m_w_dls = bowl_df[bowl_df["elig_primary"]][["canonical_player_id", "pri_score_lcb"]].merge(
        bowl_dls_sens[["canonical_player_id", "pri_score_lcb"]], on="canonical_player_id", suffixes=("_pri", "_sens")
    )
    sens_rows.append(
        {
            "Sensitivity_Variant": "Include Suspected DLS (+701 balls)",
            "Bat_Spearman": spearmanr(m_b_dls["pri_score_lcb_pri"], m_b_dls["pri_score_lcb_sens"]).statistic,
            "Bat_MAE": np.mean(np.abs(m_b_dls["pri_score_lcb_pri"] - m_b_dls["pri_score_lcb_sens"])),
            "Bowl_Spearman": spearmanr(m_w_dls["pri_score_lcb_pri"], m_w_dls["pri_score_lcb_sens"]).statistic,
            "Bowl_MAE": np.mean(np.abs(m_w_dls["pri_score_lcb_pri"] - m_w_dls["pri_score_lcb_sens"])),
        }
    )

    alt_variants = [
        ("Previous v1 Model (Overlap + Bowl 40/30/15/15)", PerformanceConfig(w_t20i=0.50, disjoint_hp_split=False, bowl_w_rsaa=0.40, bowl_w_waa=0.30)),
        ("HP Weight = 20% (Prop. Gen 80%)", PerformanceConfig(w_t20i=0.50, bat_w_raa=0.45 * (0.80 / 0.85), bat_w_surv=0.25 * (0.80 / 0.85), bat_w_cons=0.15 * (0.80 / 0.85), bat_w_hp=0.20, bowl_w_rsaa=0.55 * (0.80 / 0.85), bowl_w_waa=0.15 * (0.80 / 0.85), bowl_w_cons=0.15 * (0.80 / 0.85), bowl_w_hp=0.20)),
        ("HP Weight = 25% (Prop. Gen 75%)", PerformanceConfig(w_t20i=0.50, bat_w_raa=0.45 * (0.75 / 0.85), bat_w_surv=0.25 * (0.75 / 0.85), bat_w_cons=0.15 * (0.75 / 0.85), bat_w_hp=0.25, bowl_w_rsaa=0.55 * (0.75 / 0.85), bowl_w_waa=0.15 * (0.75 / 0.85), bowl_w_cons=0.15 * (0.75 / 0.85), bowl_w_hp=0.25)),
        ("Pressure Gamma = 1.0 (Linear)", PerformanceConfig(w_t20i=0.50, pressure_gamma=1.0)),
        ("Pressure Gamma = 1.5 (Moderate)", PerformanceConfig(w_t20i=0.50, pressure_gamma=1.5)),
        ("Posterior Mean Only (lcb_z = 0.00)", PerformanceConfig(w_t20i=0.50, lcb_z=0.00)),
        ("Strict LCB Penalty (lcb_z = 1.00)", PerformanceConfig(w_t20i=0.50, lcb_z=1.00)),
        ("Bat Scoring-Heavy / Bowl Econ-Heavy", PerformanceConfig(w_t20i=0.50, bat_w_raa=0.55, bat_w_surv=0.15, bowl_w_rsaa=0.60, bowl_w_waa=0.10)),
        ("Bat Surv-Heavy / Bowl Wkt-Heavy", PerformanceConfig(w_t20i=0.50, bat_w_raa=0.35, bat_w_surv=0.35, bowl_w_rsaa=0.45, bowl_w_waa=0.25)),
    ]
    for vname, vcfg in alt_variants:
        bv, _ = evaluate_batting_pressure(oof_pri, vcfg)
        wv, _ = evaluate_bowling_pressure(oof_pri, vcfg)
        mb = bat_df[bat_df["elig_primary"]][["canonical_player_id", "pri_score_lcb"]].merge(
            bv[["canonical_player_id", "pri_score_lcb"]], on="canonical_player_id", suffixes=("_pri", "_alt")
        )
        mw = bowl_df[bowl_df["elig_primary"]][["canonical_player_id", "pri_score_lcb"]].merge(
            wv[["canonical_player_id", "pri_score_lcb"]], on="canonical_player_id", suffixes=("_pri", "_alt")
        )
        sens_rows.append(
            {
                "Sensitivity_Variant": vname,
                "Bat_Spearman": spearmanr(mb["pri_score_lcb_pri"], mb["pri_score_lcb_alt"]).statistic,
                "Bat_MAE": np.mean(np.abs(mb["pri_score_lcb_pri"] - mb["pri_score_lcb_alt"])),
                "Bowl_Spearman": spearmanr(mw["pri_score_lcb_pri"], mw["pri_score_lcb_alt"]).statistic,
                "Bowl_MAE": np.mean(np.abs(mw["pri_score_lcb_pri"] - mw["pri_score_lcb_alt"])),
            }
        )

    # Composite vs Unified Expected Value Run Equivalents (Full-Career ev_all_lcb & PI<60 ev_net)
    b_ev = bat_df[bat_df["elig_primary"]]
    w_ev = bowl_df[bowl_df["elig_primary"]]
    sens_rows.append(
        {
            "Sensitivity_Variant": "Composite vs Full-Career EV Run-Equiv (ev_all_lcb)",
            "Bat_Spearman": spearmanr(b_ev["pri_score_lcb"], b_ev["ev_all_lcb"]).statistic,
            "Bat_MAE": np.nan,
            "Bowl_Spearman": spearmanr(w_ev["pri_score_lcb"], w_ev["ev_all_lcb"]).statistic,
            "Bowl_MAE": np.nan,
        }
    )
    sens_rows.append(
        {
            "Sensitivity_Variant": "Composite vs PI<60 EV Run-Equiv (V_wkt=7.5)",
            "Bat_Spearman": spearmanr(b_ev["pri_score_lcb"], b_ev["ev_net_lcb"]).statistic,
            "Bat_MAE": np.nan,
            "Bowl_Spearman": spearmanr(w_ev["pri_score_lcb"], w_ev["ev_net_po_lcb"]).statistic,
            "Bowl_MAE": np.nan,
        }
    )

    sens_df = pd.DataFrame(sens_rows)
    print("\nModel Perturbation Sensitivity Summary (Primary Eligible Pool):")
    print(sens_df.round(4).to_string(index=False))

    # 6. Preliminary Distributions (Anonymous Summary Quantiles)
    print("\n=== 6. PRELIMINARY PLAYER PERFORMANCE DISTRIBUTIONS (NO RANKED LISTS) ===")
    bat_cols = ["raa_raw", "raa_post", "raa_lcb", "surv_raw", "surv_post", "cons_raw", "cons_post", "hp_raa_post", "ev_all_lcb", "ev_net_lcb", "pri_score_post", "pri_score_lcb"]
    bowl_cols = ["rsaa_raw", "rsaa_post", "rsaa_lcb", "waa_raw", "waa_post", "cons_raw", "cons_post", "hp_impact_post", "ev_all_lcb", "ev_net_po_lcb", "pri_score_post", "pri_score_lcb"]

    print(f"Batting Preliminary Distributions — Primary Eligible IPL Pool (N={bat_df['elig_primary'].sum()}):")
    print(_dist_table(bat_df[bat_df["elig_primary"]], bat_cols).round(2).to_string(index=False))

    print(f"\nBatting Preliminary Distributions — Full Pool (N={len(bat_df)}):")
    print(_dist_table(bat_df, ["raa_raw", "raa_post", "pri_score_lcb"]).round(2).to_string(index=False))

    print(f"\nBowling Preliminary Distributions — Primary Eligible IPL Pool (N={bowl_df['elig_primary'].sum()}):")
    print(_dist_table(bowl_df[bowl_df["elig_primary"]], bowl_cols).round(2).to_string(index=False))

    print(f"\nBowling Preliminary Distributions — Full Pool (N={len(bowl_df)}):")
    print(_dist_table(bowl_df, ["rsaa_raw", "rsaa_post", "pri_score_lcb"]).round(2).to_string(index=False))

    # Generate Plots (13_ to 18_)
    print("\n=== 7. GENERATING PHASE 3 VISUALIZATIONS (13_ to 18_) ===")
    generate_phase3_plots(bat_rows, bowl_rows, bat_df, bowl_df, elig_grid, b_mat, w_mat, m_b_dls, m_w_dls, bat_meta, bowl_meta)
    print("All Phase 3 methodology, validation & preliminary distribution visuals complete!")


def generate_phase3_plots(bat_rows, bowl_rows, bat_df, bowl_df, elig_grid, b_mat, w_mat, m_b_dls, m_w_dls, bat_meta, bowl_meta):
    # Fig 13: OOF Expected Baseline Calibration
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    b_p = bat_rows.groupby("pi_q", observed=False)[["runs", "bat_exp_runs", "striker_out", "bat_exp_wkt"]].mean().reset_index()
    b_p["q_lbl"] = ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    x = np.arange(4)
    w = 0.35
    axes[0, 0].bar(x - w / 2, b_p["runs"], width=w, label="Actual Runs/Ball", color="#E91E63", alpha=0.8)
    axes[0, 0].bar(x + w / 2, b_p["bat_exp_runs"], width=w, label="OOF Expected Runs/Ball", color="#3F51B5", alpha=0.8)
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(b_p["q_lbl"])
    axes[0, 0].set_title("Batting Runs Calibration Across Pressure Quartiles", fontweight="bold")
    axes[0, 0].legend()

    axes[0, 1].bar(x - w / 2, b_p["striker_out"] * 100, width=w, label="Actual Wicket %", color="#FF9800", alpha=0.85)
    axes[0, 1].bar(x + w / 2, b_p["bat_exp_wkt"] * 100, width=w, label="OOF Expected Wicket %", color="#009688", alpha=0.85)
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(b_p["q_lbl"])
    axes[0, 1].set_title("Batting Dismissal Calibration Across Pressure Quartiles", fontweight="bold")
    axes[0, 1].set_ylabel("Dismissal Prob (%)")
    axes[0, 1].legend()

    bw_ph = bowl_rows.groupby(["is_ipl", "match_phase"]).agg(
        act_r=("bowler_runs_conceded", lambda s: 6.0 * s.sum() / bowl_rows.loc[s.index, "is_legal_ball"].sum()),
        exp_r=("bowl_exp_runs", lambda s: 6.0 * s.sum() / bowl_rows.loc[s.index, "is_legal_ball"].sum()),
        act_w=("bowler_wkt_int", lambda s: 24.0 * s.sum() / bowl_rows.loc[s.index, "is_legal_ball"].sum()),
        exp_w=("bowl_exp_wkt", lambda s: 24.0 * s.sum() / bowl_rows.loc[s.index, "is_legal_ball"].sum()),
    ).reset_index()
    bw_ph["lbl"] = np.where(bw_ph["is_ipl"], "IPL ", "T20I ") + bw_ph["match_phase"]
    x2 = np.arange(len(bw_ph))
    axes[1, 0].bar(x2 - w / 2, bw_ph["act_r"], width=w, label="Actual RPO (incl. Extras)", color="#1E88E5", alpha=0.85)
    axes[1, 0].bar(x2 + w / 2, bw_ph["exp_r"], width=w, label="OOF Expected RPO", color="#26A69A", alpha=0.85)
    axes[1, 0].set_xticks(x2); axes[1, 0].set_xticklabels(bw_ph["lbl"], rotation=20, ha="right")
    axes[1, 0].set_title("Bowling Expected Runs Conservation (IPL vs T20I x Phase)", fontweight="bold")
    axes[1, 0].set_ylabel("Runs per Over (6 Legal Balls)")
    axes[1, 0].legend()

    axes[1, 1].bar(x2 - w / 2, bw_ph["act_w"], width=w, label="Actual Wkts/4 Ov", color="#7B1FA2", alpha=0.85)
    axes[1, 1].bar(x2 + w / 2, bw_ph["exp_w"], width=w, label="OOF Expected Wkts/4 Ov", color="#8D6E63", alpha=0.85)
    axes[1, 1].set_xticks(x2); axes[1, 1].set_xticklabels(bw_ph["lbl"], rotation=20, ha="right")
    axes[1, 1].set_title("Bowling Expected Wickets Calibration (IPL vs T20I x Phase)", fontweight="bold")
    axes[1, 1].set_ylabel("Wickets per 24 Legal Balls")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_phase3_oof_baseline_calibration.png", dpi=200)
    plt.close()

    # Fig 14: Sample Size & Eligibility Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.histplot(bat_df["neff"], bins=35, ax=axes[0, 0], color="#E91E63", alpha=0.7, label="All Batters")
    sns.histplot(bat_df[bat_df["ipl_balls"] >= 30]["neff"], bins=35, ax=axes[0, 0], color="#880E4F", alpha=0.75, label="IPL >= 30 Balls")
    axes[0, 0].axvline(30, color="black", linestyle="--", label="N_eff = 30")
    axes[0, 0].set_title("Batter Kish Effective Sample Size (N_eff, w_T20I=0.50)", fontweight="bold")
    axes[0, 0].set_xlabel("Effective Pressure-Weighted Balls (N_eff)")
    axes[0, 0].legend()

    sns.histplot(bowl_df["neff"], bins=35, ax=axes[0, 1], color="#1E88E5", alpha=0.7, label="All Bowlers")
    sns.histplot(bowl_df[bowl_df["ipl_legal_balls"] >= 30]["neff"], bins=35, ax=axes[0, 1], color="#0D47A1", alpha=0.75, label="IPL >= 30 Legal Balls")
    axes[0, 1].axvline(30, color="black", linestyle="--", label="N_eff = 30")
    axes[0, 1].set_title("Bowler Kish Effective Sample Size (N_eff, w_T20I=0.50)", fontweight="bold")
    axes[0, 1].set_xlabel("Effective Pressure-Weighted Legal Balls (N_eff)")
    axes[0, 1].legend()

    hm_b = elig_grid.pivot(index="min_Neff", columns="min_IPL_balls", values="Eligible_Batters")
    hm_w = elig_grid.pivot(index="min_Neff", columns="min_IPL_balls", values="Eligible_Bowlers")
    sns.heatmap(hm_b, annot=True, fmt="d", cmap="YlOrRd", ax=axes[1, 0])
    axes[1, 0].set_title("Eligible Batters Grid (min_N_eff x min_IPL_balls)", fontweight="bold")
    sns.heatmap(hm_w, annot=True, fmt="d", cmap="YlGnBu", ax=axes[1, 1])
    axes[1, 1].set_title("Eligible Bowlers Grid (min_N_eff x min_IPL_legal_balls)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "14_phase3_sample_size_eligibility.png", dpi=200)
    plt.close()

    # Fig 15: Empirical Bayes Shrinkage Behavior
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    b_vis = bat_df[bat_df["neff"] >= 5]
    sc1 = axes[0, 0].scatter(b_vis["raa_raw"], b_vis["raa_post"], c=np.minimum(300, b_vis["neff"]), cmap="viridis", alpha=0.7, s=22)
    axes[0, 0].axline((0, 0), slope=1, color="red", linestyle="--", linewidth=1, label="y = x (No Shrinkage)")
    axes[0, 0].axhline(bat_meta["raa"]["mu0"], color="gray", linestyle=":", label=f"Prior mu0={bat_meta['raa']['mu0']:.2f}")
    axes[0, 0].set_title("Batting RAA: Raw vs EB Posterior (Color = N_eff)", fontweight="bold")
    axes[0, 0].set_xlabel("Raw Pressure-Weighted RAA (/100 balls)")
    axes[0, 0].set_ylabel("Posterior RAA (/100 balls)")
    axes[0, 0].legend(fontsize=8)
    plt.colorbar(sc1, ax=axes[0, 0], label="N_eff (capped 300)")

    ne_grid = np.linspace(1, 500, 300)
    axes[0, 1].plot(ne_grid, ne_grid / (ne_grid + bat_meta["raa"]["k0_prior_neff"]), color="#E91E63", linewidth=2, label=f"Bat RAA (k0={bat_meta['raa']['k0_prior_neff']:.0f})")
    axes[0, 1].plot(ne_grid, ne_grid / (ne_grid + bat_meta["surv"]["k0_prior_neff"]), color="#FF9800", linewidth=2, label=f"Bat Surv (k0={bat_meta['surv']['k0_prior_neff']:.0f})")
    axes[0, 1].plot(ne_grid, ne_grid / (ne_grid + bowl_meta["rsaa"]["k0_prior_neff"]), color="#1E88E5", linewidth=2, label=f"Bowl RSAA (k0={bowl_meta['rsaa']['k0_prior_neff']:.0f})")
    axes[0, 1].plot(ne_grid, ne_grid / (ne_grid + bowl_meta["waa"]["k0_prior_neff"]), color="#00897B", linewidth=2, label=f"Bowl WAA (k0={bowl_meta['waa']['k0_prior_neff']:.0f})")
    axes[0, 1].axvline(30, color="black", linestyle="--", alpha=0.6)
    axes[0, 1].set_title("Empirical Bayes Credibility Weight B = N_eff / (N_eff + k0)", fontweight="bold")
    axes[0, 1].set_xlabel("Effective Sample Size (N_eff)")
    axes[0, 1].set_ylabel("Credibility Weight B")
    axes[0, 1].legend(fontsize=8)

    w_vis = bowl_df[bowl_df["neff"] >= 5]
    sc2 = axes[1, 0].scatter(w_vis["rsaa_raw"], w_vis["rsaa_post"], c=np.minimum(300, w_vis["neff"]), cmap="plasma", alpha=0.7, s=22)
    axes[1, 0].axline((0, 0), slope=1, color="red", linestyle="--", linewidth=1, label="y = x")
    axes[1, 0].axhline(bowl_meta["rsaa"]["mu0"], color="gray", linestyle=":", label=f"Prior mu0={bowl_meta['rsaa']['mu0']:.2f}")
    axes[1, 0].set_title("Bowling RSAA: Raw vs EB Posterior (Color = N_eff)", fontweight="bold")
    axes[1, 0].set_xlabel("Raw Runs Saved/Over")
    axes[1, 0].set_ylabel("Posterior Runs Saved/Over")
    axes[1, 0].legend(fontsize=8)
    plt.colorbar(sc2, ax=axes[1, 0], label="N_eff (capped 300)")

    axes[1, 1].scatter(b_vis["neff"], b_vis["raa_sd"], alpha=0.65, s=20, color="#E91E63", label="Bat RAA Post SD")
    axes[1, 1].scatter(w_vis["neff"], w_vis["rsaa_sd"] * 10.0, alpha=0.65, s=20, color="#1E88E5", label="Bowl RSAA Post SD (x10)")
    axes[1, 1].set_xlim(0, 450)
    axes[1, 1].set_title("Posterior Uncertainty Shrinks Monotonically With Sample Size", fontweight="bold")
    axes[1, 1].set_xlabel("Effective Sample Size (N_eff)")
    axes[1, 1].set_ylabel("Posterior Standard Deviation")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "15_phase3_bayesian_shrinkage.png", dpi=200)
    plt.close()

    # Fig 16: Competition Weight Matrix + Suspected DLS Sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.heatmap(b_mat, annot=True, fmt=".3f", cmap="RdPu", vmin=0.5, vmax=1.0, ax=axes[0, 0])
    axes[0, 0].set_title("Batting PRI Spearman Correlation Across w_T20I (IPL Eligible)", fontweight="bold")

    sns.heatmap(w_mat, annot=True, fmt=".3f", cmap="Blues", vmin=0.5, vmax=1.0, ax=axes[0, 1])
    axes[0, 1].set_title("Bowling PRI Spearman Correlation Across w_T20I (IPL Eligible)", fontweight="bold")

    axes[1, 0].scatter(m_b_dls["pri_score_lcb_pri"], m_b_dls["pri_score_lcb_sens"], alpha=0.75, color="#E91E63")
    axes[1, 0].axline((50, 50), slope=1, color="black", linestyle="--")
    axes[1, 0].set_title("Batting DLS Sensitivity: Primary vs +Suspected DLS", fontweight="bold")
    axes[1, 0].set_xlabel("Primary PRI (Standard + Confirmed DLS)")
    axes[1, 0].set_ylabel("Sensitivity PRI (+ Suspected DLS)")

    axes[1, 1].scatter(m_w_dls["pri_score_lcb_pri"], m_w_dls["pri_score_lcb_sens"], alpha=0.75, color="#1E88E5")
    axes[1, 1].axline((50, 50), slope=1, color="black", linestyle="--")
    axes[1, 1].set_title("Bowling DLS Sensitivity: Primary vs +Suspected DLS", fontweight="bold")
    axes[1, 1].set_xlabel("Primary PRI (Standard + Confirmed DLS)")
    axes[1, 1].set_ylabel("Sensitivity PRI (+ Suspected DLS)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "16_phase3_competition_weight_sensitivity.png", dpi=200)
    plt.close()

    # Fig 17: Preliminary Batting Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    b_el = bat_df[bat_df["elig_primary"]]
    sns.histplot(b_el["raa_raw"], bins=30, ax=axes[0, 0], color="gray", alpha=0.45, label="Raw")
    sns.histplot(b_el["raa_post"], bins=30, ax=axes[0, 0], color="#E91E63", alpha=0.65, label="EB Posterior")
    axes[0, 0].set_title("Eligible Batters: Pressure RAA (/100 balls)", fontweight="bold")
    axes[0, 0].legend()

    sns.histplot(b_el["surv_raw"], bins=30, ax=axes[0, 1], color="gray", alpha=0.45, label="Raw")
    sns.histplot(b_el["surv_post"], bins=30, ax=axes[0, 1], color="#FF9800", alpha=0.65, label="EB Posterior")
    axes[0, 1].set_title("Eligible Batters: Wicket Survival Above Exp (/100 balls)", fontweight="bold")
    axes[0, 1].legend()

    sns.histplot(b_el["cons_raw"], bins=25, ax=axes[1, 0], color="gray", alpha=0.45, label="Raw")
    sns.histplot(b_el["cons_post"], bins=25, ax=axes[1, 0], color="#8E24AA", alpha=0.65, label="Beta-Binomial Post")
    axes[1, 0].set_title("Eligible Batters: Positive-Impact Innings Rate", fontweight="bold")
    axes[1, 0].legend()

    sns.histplot(bat_df["pri_score_lcb"], bins=35, kde=True, ax=axes[1, 1], color="#B0BEC5", alpha=0.5, label="All Batters")
    sns.histplot(b_el["pri_score_lcb"], bins=25, kde=True, ax=axes[1, 1], color="#E91E63", alpha=0.7, label="Primary Eligible")
    axes[1, 1].set_title("Preliminary Batting Pressure Reliability Index (PRI LCB, 0-100)", fontweight="bold")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "17_phase3_prelim_distributions_batting.png", dpi=200)
    plt.close()

    # Fig 18: Preliminary Bowling Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    w_el = bowl_df[bowl_df["elig_primary"]]
    sns.histplot(w_el["rsaa_raw"], bins=30, ax=axes[0, 0], color="gray", alpha=0.45, label="Raw")
    sns.histplot(w_el["rsaa_post"], bins=30, ax=axes[0, 0], color="#1E88E5", alpha=0.65, label="EB Posterior")
    axes[0, 0].set_title("Eligible Bowlers: Pressure Runs Saved/Over (RSAA)", fontweight="bold")
    axes[0, 0].legend()

    sns.histplot(w_el["waa_raw"], bins=30, ax=axes[0, 1], color="gray", alpha=0.45, label="Raw")
    sns.histplot(w_el["waa_post"], bins=30, ax=axes[0, 1], color="#00897B", alpha=0.65, label="EB Posterior")
    axes[0, 1].set_title("Eligible Bowlers: Pressure Wickets/4-Over Spell (WAA)", fontweight="bold")
    axes[0, 1].legend()

    sns.histplot(w_el["cons_raw"], bins=25, ax=axes[1, 0], color="gray", alpha=0.45, label="Raw")
    sns.histplot(w_el["cons_post"], bins=25, ax=axes[1, 0], color="#3949AB", alpha=0.65, label="Beta-Binomial Post")
    axes[1, 0].set_title("Eligible Bowlers: Positive-Impact Spell Consistency", fontweight="bold")
    axes[1, 0].legend()

    sns.histplot(bowl_df["pri_score_lcb"], bins=35, kde=True, ax=axes[1, 1], color="#B0BEC5", alpha=0.5, label="All Bowlers")
    sns.histplot(w_el["pri_score_lcb"], bins=25, kde=True, ax=axes[1, 1], color="#1E88E5", alpha=0.7, label="Primary Eligible")
    axes[1, 1].set_title("Preliminary Bowling Pressure Reliability Index (PRI LCB, 0-100)", fontweight="bold")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "18_phase3_prelim_distributions_bowling.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

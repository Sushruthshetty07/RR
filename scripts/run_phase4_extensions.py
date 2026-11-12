"""
Execution & Diagnostics Script for Phase 4 Exploratory Extensions (12 Proposal Items).

Preserves all approved Phase 1-3 CSVs, reports, and Phase 4 Top 10 shortlists.
Outputs isolated `data/processed/phase4_ext_*` tables, `phase4_ext_summary.json`,
and Figures 29-32 in both `reports/figures/` and artifact `figures/`.
"""

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DATA_PROCESSED, load_raw_datasets
from src.performance import (
    PRIMARY_DLS_STATUSES,
    PerformanceConfig,
    fit_predict_oof_baselines,
)
from src.phase4_extensions import (
    audit_match_state_invariants,
    build_proposal_classification_table,
    evaluate_enriched_roe_and_pressure_delta,
    evaluate_match_bootstrap_and_stability,
    evaluate_phase_dominance_and_wicket_quality,
    evaluate_two_way_matchups_season_venue_comp_prog,
    evaluate_win_probability_and_leverage,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
_artifact_env = os.environ.get("ARTIFACT_FIG_DIR")
ARTIFACT_FIG_DIR = Path(_artifact_env) if _artifact_env else FIG_DIR
if ARTIFACT_FIG_DIR != FIG_DIR:
    ARTIFACT_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 12.5,
        "figure.titleweight": "bold",
    }
)


def _save_fig(fig: plt.Figure, filename: str) -> None:
    p1 = FIG_DIR / filename
    p2 = ARTIFACT_FIG_DIR / filename
    fig.savefig(p1, dpi=180, bbox_inches="tight", facecolor="white")
    shutil.copy2(p1, p2)
    plt.close(fig)


def _short_name(full_name: str) -> str:
    parts = str(full_name).strip().split()
    if len(parts) <= 1:
        return str(full_name)
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def make_figure_29(wp_res: dict, roe_res: dict, bat_short_ids: set, bowl_short_ids: set) -> None:
    """Figure 29: Win Probability Calibration, LI Sensitivity, RoE Fit & Top-10 Pressure Delta."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.suptitle(
        "Phase 4 Extensions (Experimental C): Win Probability, Leverage Index, Enriched RoE & Pressure Delta",
        y=0.99,
    )

    # Panel A: Chase WP Decile Calibration
    ax = axes[0, 0]
    dec = wp_res["chase_deciles"]
    cal = wp_res["calibration_summary"]
    inn2_cal = cal[cal["slice"] == "Innings 2 (Chasing)"].iloc[0]
    inn1_cal = cal[cal["slice"] == "Innings 1 (Setting)"].iloc[0]
    ax.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1.2, label="Perfect Calibration")
    ax.plot(dec["pred"], dec["actual"], "o-", color="#1B4F72", linewidth=2.0, markersize=6, label="Innings 2 Chase OOF WP")
    ax.set_xlabel("Predicted Batting Win Probability (OOF Deciles)")
    ax.set_ylabel("Observed Match Win Rate")
    ax.set_title("A. OOF Innings 2 Chase Win Probability Calibration")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    txt = (
        f"Inn 2 Chase: AUC={inn2_cal['roc_auc']:.3f}, LogLoss={inn2_cal['log_loss']:.3f}, Brier={inn2_cal['brier_score']:.3f}\n"
        f"Inn 1 Setting: AUC={inn1_cal['roc_auc']:.3f}, LogLoss={inn1_cal['log_loss']:.3f}, Brier={inn1_cal['brier_score']:.3f}\n"
        f"Corr(PI, LI) Chase: r={inn2_cal['corr_pi_li_pearson']:.3f}, rho={inn2_cal['corr_pi_li_spearman']:.3f}"
    )
    ax.text(0.03, 0.96, txt, transform=ax.transAxes, va="top", fontsize=8.0, bbox=dict(boxstyle="round", facecolor="#F4F6F7", alpha=0.92))
    ax.legend(loc="lower right")

    # Panel B: LI Regime Sensitivity vs Approved PI >= 60
    ax = axes[0, 1]
    sens = wp_res["li_sensitivity"].copy()
    x = np.arange(len(sens))
    w = 0.36
    ax.bar(x - w / 2, sens["bat_rho_vs_hp_raa"], width=w, color="#2E86C1", label="Bat rho vs HP-RAA (PI>=60)")
    ax.bar(x + w / 2, sens["bowl_rho_vs_hp_impact"], width=w, color="#D35400", label="Bowl rho vs HP-Impact (PI>=60)")
    for i, r in sens.iterrows():
        ax.text(i, max(r["bat_rho_vs_hp_raa"], r["bowl_rho_vs_hp_impact"]) + 0.025, f"{r['pct_pool_balls']:.0f}%", ha="center", fontsize=7.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(sens["li_regime"], rotation=26, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Spearman rho vs Approved PI>=60 HP Pillar")
    ax.set_title("B. High-LI Regime Sensitivity (% Pool Coverage Annotated)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    # Panel C: Phase 3 Baseline vs Enriched RoE (R2 & Rank Correlation)
    ax = axes[1, 0]
    fit_df = roe_res["roe_fit_summary"]
    cmp = roe_res["comparisons"]
    labels = ["Bat R^2\n(Approved)", "Bat R^2\n(Enriched RoE)", "Bowl R^2\n(Approved)", "Bowl R^2\n(Enriched RoE)"]
    vals = fit_df["r2"].values
    colors = ["#5D6D7E", "#2874A6", "#7D6608", "#D35400"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_ylabel("OOF Delivery-Level R^2")
    ax.set_ylim(0.0, max(vals) * 1.52)
    ax.set_title("C. Contextual OOF Baseline vs Enriched RoE Explanatory Power")
    ax.grid(axis="y", alpha=0.25)
    info = (
        f"Enriched vs Approved Post Rank Similarity (Eligible Pool):\n"
        f"Bat Enriched RoE vs RAA: rho={cmp['bat_enr_vs_raa_post']['spearman_rho']:.3f} (Top10: {cmp['bat_enr_vs_raa_post']['top10_overlap']}/10)\n"
        f"Bowl Enriched RoE vs RSAA: rho={cmp['bowl_enr_vs_rsaa_post']['spearman_rho']:.3f} (Top10: {cmp['bowl_enr_vs_rsaa_post']['top10_overlap']}/10)\n"
        f"Bowl Conceded vs Total-Runs Attr.: rho={cmp['bowl_extras_attribution_tot_vs_conc']['spearman_rho']:.3f}"
    )
    ax.text(0.03, 0.96, info, transform=ax.transAxes, va="top", fontsize=7.8, bbox=dict(boxstyle="round", facecolor="#FEF9E7", alpha=0.92))

    # Panel D: Formal Pressure Delta (+/- 1 SE) for Shortlisted Top 10 Batters & Bowlers
    ax = axes[1, 1]
    b_top = roe_res["bat_roe_delta"][roe_res["bat_roe_delta"]["canonical_player_id"].isin(bat_short_ids)].sort_values("delta_pi_post_100")
    l_top = roe_res["bowl_roe_delta"][roe_res["bowl_roe_delta"]["canonical_player_id"].isin(bowl_short_ids)].sort_values("delta_pi_post_100")
    comb = pd.concat(
        [
            b_top.assign(label=lambda d: d["player_name"].map(_short_name) + " (Bat)"),
            l_top.assign(label=lambda d: d["player_name"].map(_short_name) + " (Bowl)"),
        ],
        ignore_index=True,
    )
    y_pos = np.arange(len(comb))
    c_bars = ["#27AE60" if v >= 0 else "#C0392B" for v in comb["delta_pi_post_100"]]
    ax.barh(y_pos, comb["delta_pi_post_100"], xerr=comb["delta_pi_se_100"], color=c_bars, alpha=0.85, ecolor="#444444", capsize=2.5)
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comb["label"], fontsize=7.5)
    ax.set_xlabel("Pressure Delta = (PI>=60 RoE - PI<60 RoE) per 100 balls (+/- 1 SE)")
    ax.set_title("D. Top 10 Batters & Bowlers: Empirical Bayes Pressure Delta")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "29_phase4_ext_wp_leverage_roe.png")


def make_figure_30(bs_res: dict, bat_short_ids: set, bowl_short_ids: set) -> None:
    """Figure 30: Match-Cluster Bootstrap DEFF, LCB Stability & Split-Half Repeatability."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.suptitle(
        "Phase 4 Extensions (Experimental C): Match-Cluster Bootstrap SE/LCB & Split-Half Repeatability",
        y=0.99,
    )
    sm = bs_res["summary"]
    bat_df = bs_res["bat_boot_stab"]
    bowl_df = bs_res["bowl_boot_stab"]
    b_elig = bat_df[bat_df["elig_primary"]].copy()
    l_elig = bowl_df[bowl_df["elig_primary"]].copy()

    # Panel A: DEFF distribution
    ax = axes[0, 0]
    b_deff = np.clip(b_elig["deff_raa"].dropna(), 0.2, 4.5)
    l_deff = np.clip(l_elig["deff_rsaa"].dropna(), 0.2, 4.5)
    ax.hist(b_deff, bins=24, alpha=0.65, color="#2471A3", label=f"Batters (Med DEFF={sm['bat_median_deff_raa']:.2f})")
    ax.hist(l_deff, bins=24, alpha=0.60, color="#CA6F1E", label=f"Bowlers (Med DEFF={sm['bowl_median_deff_rsaa']:.2f})")
    ax.axvline(1.0, color="#111111", linestyle="--", linewidth=1.3, label="Kish Delivery Assumption (DEFF=1.0)")
    ax.set_xlabel("Match-Cluster Design Effect (Boot Var / Kish IID Var)")
    ax.set_ylabel("Eligible Player Count")
    ax.set_title("A. Intra-Match Clustering Design Effect (B=400 Matches)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    # Panel B: Approved PRI_LCB vs Cluster-Adjusted PRI_LCB
    ax = axes[0, 1]
    ax.plot([20, 99], [20, 99], "--", color="#777777", linewidth=1.1)
    ax.scatter(b_elig["pri_score_lcb"], b_elig["pri_score_lcb_cluster"], s=18, alpha=0.45, color="#2E86C1", label="Batters (Elig)")
    ax.scatter(l_elig["pri_score_lcb"], l_elig["pri_score_lcb_cluster"], s=18, alpha=0.45, color="#D35400", label="Bowlers (Elig)")
    b_top = b_elig[b_elig["canonical_player_id"].isin(bat_short_ids)]
    l_top = l_elig[l_elig["canonical_player_id"].isin(bowl_short_ids)]
    ax.scatter(b_top["pri_score_lcb"], b_top["pri_score_lcb_cluster"], s=52, color="#1B4F72", edgecolor="white", zorder=5, label="Shortlist Top 10 Bat")
    ax.scatter(l_top["pri_score_lcb"], l_top["pri_score_lcb_cluster"], s=52, color="#873600", edgecolor="white", zorder=5, label="Shortlist Top 10 Bowl")
    ax.set_xlabel("Approved Kish LCB PRI Score")
    ax.set_ylabel("Match-Cluster Adjusted LCB PRI Score")
    ax.set_title("B. PRI Score Stability Under Match-Cluster Uncertainty")
    ax.grid(alpha=0.25)
    txt = (
        f"Bat Cluster vs Approved: rho={sm['bat_cluster_pri_vs_approved']['spearman_rho']:.4f} "
        f"(Top10: {sm['bat_cluster_pri_vs_approved']['top10_overlap']}/10, mean |dRank|={sm['bat_cluster_pri_vs_approved']['mean_abs_rank_shift']:.1f})\n"
        f"Bowl Cluster vs Approved: rho={sm['bowl_cluster_pri_vs_approved']['spearman_rho']:.4f} "
        f"(Top10: {sm['bowl_cluster_pri_vs_approved']['top10_overlap']}/10, mean |dRank|={sm['bowl_cluster_pri_vs_approved']['mean_abs_rank_shift']:.1f})"
    )
    ax.text(0.03, 0.96, txt, transform=ax.transAxes, va="top", fontsize=7.8, bbox=dict(boxstyle="round", facecolor="#F4F6F7", alpha=0.92))
    ax.legend(loc="lower right", fontsize=7.5)

    # Panel C: Odd-Even vs Chronological Split-Half Reliability (Spearman-Brown)
    ax = axes[1, 0]
    br = sm["bat_repeatability"]
    lr = sm["bowl_repeatability"]
    cats = ["Bat\nOdd-Even r", "Bat\nOdd-Even SB", "Bat\nChrono r", "Bat\nChrono SB", "Bowl\nOdd-Even r", "Bowl\nOdd-Even SB", "Bowl\nChrono r", "Bowl\nChrono SB"]
    r_vals = [
        br["odd_even_repeatability"]["pearson_r"],
        br["odd_even_repeatability"]["spearman_brown"],
        br["chrono_split_half_repeatability"]["pearson_r"],
        br["chrono_split_half_repeatability"]["spearman_brown"],
        lr["odd_even_repeatability"]["pearson_r"],
        lr["odd_even_repeatability"]["spearman_brown"],
        lr["chrono_split_half_repeatability"]["pearson_r"],
        lr["chrono_split_half_repeatability"]["spearman_brown"],
    ]
    c_list = ["#2E86C1", "#1B4F72", "#5DADE2", "#21618C", "#E67E22", "#A04000", "#F5B041", "#B9770E"]
    bars = ax.bar(cats, r_vals, color=c_list, width=0.62)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    for b, v in zip(bars, r_vals):
        y_txt = v + 0.012 if v >= 0 else v - 0.014
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, y_txt, f"{v:.2f}", ha="center", va=va, fontsize=7.8)
    ax.set_ylim(min(min(r_vals) - 0.08, -0.14), max(max(r_vals) * 1.25, 0.35))
    ax.set_ylabel("Reliability Coefficient")
    ax.set_title("C. Player Impact Split-Half Repeatability (>=8 Matches)")
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Positive-Impact Appearance Rate (cons_post) vs Match-to-Match Volatility (SD)
    ax = axes[1, 1]
    b_sd = b_elig[b_elig["match_impact_sd"].notna() & (b_elig["n_inn_matches"] >= 6)]
    ax.scatter(b_sd["match_impact_sd"], b_sd["cons_post"], s=22, alpha=0.55, color="#2E86C1", label="Batters (>=6 inn)")
    b_top_sd = b_top[b_top["match_impact_sd"].notna()]
    ax.scatter(b_top_sd["match_impact_sd"], b_top_sd["cons_post"], s=58, color="#1B4F72", edgecolor="white", zorder=5, label="Top 10 Batters")
    offsets_30d = [(4, 4), (4, -8), (-34, 4), (4, 2), (-34, -8)]
    for idx_k, (_, r) in enumerate(b_top_sd.head(5).iterrows()):
        ox, oy = offsets_30d[idx_k % len(offsets_30d)]
        ax.annotate(_short_name(r["player_name"]), (r["match_impact_sd"], r["cons_post"]), xytext=(ox, oy), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Match-to-Match Impact SD (Runs/100b Equivalent)")
    ax.set_ylabel("Approved cons_post (Positive-Impact Rate)")
    ax.set_title("D. Appearance Consistency vs Match Volatility")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.05,
        f"Bat rho(cons_post, SD)={sm['bat_cons_vs_match_sd_rho']:.3f}\n"
        f"Inv-SD Swap vs Approved: Bat rho={sm['bat_invsd_pri_vs_approved']['spearman_rho']:.3f} (Top10: {sm['bat_invsd_pri_vs_approved']['top10_overlap']}/10)\n"
        f"Inv-SD Swap vs Approved: Bowl rho={sm['bowl_invsd_pri_vs_approved']['spearman_rho']:.3f} (Top10: {sm['bowl_invsd_pri_vs_approved']['top10_overlap']}/10)",
        transform=ax.transAxes,
        va="bottom",
        fontsize=7.6,
        bbox=dict(boxstyle="round", facecolor="#FEF9E7", alpha=0.92),
    )
    ax.legend(loc="upper right", fontsize=7.8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "30_phase4_ext_bootstrap_stability.png")


def make_figure_31(pdi_res: dict, bat_short_ids: set, bowl_short_ids: set) -> None:
    """Figure 31: Opportunity-Neutral PDI, Wicket Quality Weighting & Pace Execution Audit."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.suptitle(
        "Phase 4 Extensions (Supplementary B / Experimental C): Phase Dominance Index, Wicket Quality & Pace Execution",
        y=0.99,
    )
    sm = pdi_res["summary"]
    bat_pdi = pdi_res["bat_pdi"]
    bowl_wq = pdi_res["bowl_pdi_wq"]
    b_elig = bat_pdi[bat_pdi["elig_primary"]]
    l_elig = bowl_wq[bowl_wq["elig_primary"]]

    # Panel A: Opportunity-Neutral Equal-Phase PDI vs Approved PRI
    ax = axes[0, 0]
    ax.plot([20, 99], [20, 99], "--", color="#777777", linewidth=1.1)
    ax.scatter(b_elig["pri_score_lcb"], b_elig["pdi_equal_score"], s=20, alpha=0.45, color="#2E86C1", label="Batters")
    ax.scatter(l_elig["pri_score_lcb"], l_elig["pdi_equal_score"], s=20, alpha=0.45, color="#D35400", label="Bowlers")
    b_top = b_elig[b_elig["canonical_player_id"].isin(bat_short_ids)]
    l_top = l_elig[l_elig["canonical_player_id"].isin(bowl_short_ids)]
    ax.scatter(b_top["pri_score_lcb"], b_top["pdi_equal_score"], s=54, color="#1B4F72", edgecolor="white", zorder=5, label="Top 10 Bat")
    ax.scatter(l_top["pri_score_lcb"], l_top["pdi_equal_score"], s=54, color="#873600", edgecolor="white", zorder=5, label="Top 10 Bowl")
    ax.set_xlabel("Approved Overall PRI Score (LCB)")
    ax.set_ylabel("Opportunity-Neutral Equal-Phase PDI Score")
    ax.set_title("A. All-Phase PDI vs Overall PRI (Specialist vs Versatile)")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.96,
        f"Bat Equal-PDI vs PRI: rho={sm['bat_pdi_equal_vs_pri']['spearman_rho']:.3f} (Top10: {sm['bat_pdi_equal_vs_pri']['top10_overlap']}/10)\n"
        f"Bowl Equal-PDI vs PRI: rho={sm['bowl_pdi_equal_vs_pri']['spearman_rho']:.3f} (Top10: {sm['bowl_pdi_equal_vs_pri']['top10_overlap']}/10)\n"
        f"Peak-Phase Z vs PRI: Bat rho={sm['bat_pdi_peak_vs_pri']['spearman_rho']:.3f}, Bowl rho={sm['bowl_pdi_peak_vs_pri']['spearman_rho']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=7.8,
        bbox=dict(boxstyle="round", facecolor="#F4F6F7", alpha=0.92),
    )
    ax.legend(loc="lower right", fontsize=7.5)

    # Panel B: Heatmap of Z-PDI across PP/Mid/Death for Top 10 Batters
    ax = axes[0, 1]
    bt = b_top.sort_values("pri_score_lcb", ascending=False).head(10)
    mat_b = bt[["z_pdi_powerplay", "z_pdi_middle", "z_pdi_death"]].values
    im = ax.imshow(mat_b, cmap="RdYlGn", vmin=-1.5, vmax=2.2, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Powerplay", "Middle", "Death"])
    ax.set_yticks(np.arange(len(bt)))
    ax.set_yticklabels([_short_name(n) for n in bt["player_name"]], fontsize=8.0)
    for i in range(mat_b.shape[0]):
        for j in range(mat_b.shape[1]):
            ax.text(j, i, f"{mat_b[i, j]:+.2f}", ha="center", va="center", fontsize=7.5, color="#111111")
    ax.set_title("B. Top 10 Batters: EB-Shrunk Phase Dominance (z-scores)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel C: Wicket Quality Alternatives vs Approved Bowling PRI
    ax = axes[1, 0]
    wq_names = ["WQA:\nOrder+Set", "WQB:\n1 / 0.7 / 0.3", "WQC:\nBatter-PRI"]
    rhos = [
        sm["bowl_pri_wqa_vs_approved"]["spearman_rho"],
        sm["bowl_pri_wqb_vs_approved"]["spearman_rho"],
        sm["bowl_pri_wqc_vs_approved"]["spearman_rho"],
    ]
    t10s = [
        sm["bowl_pri_wqa_vs_approved"]["top10_overlap"],
        sm["bowl_pri_wqb_vs_approved"]["top10_overlap"],
        sm["bowl_pri_wqc_vs_approved"]["top10_overlap"],
    ]
    k0s = [sm["waa_meta_wqa_k0"], sm["waa_meta_wqb_k0"], sm["waa_meta_wqc_k0"]]
    bars = ax.bar(wq_names, rhos, color=["#2E86C1", "#28B463", "#8E44AD"], width=0.52)
    for b, r, t10, k0 in zip(bars, rhos, t10s, k0s):
        ax.text(b.get_x() + b.get_width() / 2, r - 0.04, f"rho={r:.4f}\nTop10={t10}/10\nk0={k0:.0f}", ha="center", va="top", color="white", fontsize=8.0, fontweight="bold")
    ax.set_ylim(0.80, 1.01)
    ax.set_ylabel("Spearman rho vs Approved Bowl PRI_LCB")
    ax.set_title(f"C. Wicket Quality Weighting Sensitivity ({sm['n_non_bowler_wkts_excluded_in_wq']} Non-Bowler Wkts Excluded)")
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Supplementary Pace Delivery-Style Audit (Yorker % vs Full-Toss Control Error %)
    ax = axes[1, 1]
    pace_e = l_elig[l_elig["pace_leg_balls"].fillna(0) >= 60].copy()
    sc = ax.scatter(pace_e["yorker_pct"], pace_e["full_toss_pct"], c=pace_e["pri_score_lcb"], cmap="viridis", s=34, alpha=0.82)
    pace_top = pace_e[pace_e["canonical_player_id"].isin(bowl_short_ids)]
    ax.scatter(pace_top["yorker_pct"], pace_top["full_toss_pct"], s=85, facecolors="none", edgecolors="#C0392B", linewidths=1.8, label="Shortlist Pace")
    offsets_31d = [(5, 4), (5, -8), (-42, 4), (5, -2), (-42, -8), (5, 7)]
    for idx_p, (_, r) in enumerate(pace_top.sort_values("yorker_pct").iterrows()):
        ox, oy = offsets_31d[idx_p % len(offsets_31d)]
        ax.annotate(_short_name(r["player_name"]), (r["yorker_pct"], r["full_toss_pct"]), xytext=(ox, oy), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Yorker + Wide Yorker % (Pace Legal Balls >= 60)")
    ax.set_ylabel("Full Toss % (Control Miss Proxy)")
    ax.set_title("D. Supplementary Pace Execution: Yorker vs Full Toss %")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Approved Bowl PRI")
    ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "31_phase4_ext_phase_dom_wicket_qual.png")


def make_figure_32(m_res: dict, bat_short_ids: set, bowl_short_ids: set) -> None:
    """Figure 32: Two-Way Matchup Ridge, Season x Phase Venue EB, Bridge Calibration & Tenure vs Age."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.suptitle(
        "Phase 4 Extensions (Experimental C / Audit): Two-Way Matchups, Venue/Season EB, Bridge Calibration & Tenure Proxy",
        y=0.99,
    )
    tw = m_res["two_way"]
    sv = m_res["season_venue"]
    cc = m_res["comp_calibration"]
    pr = m_res["progression_tenure"]

    # Panel A: Two-Way Matchup Bat Effect vs Approved RAA_post
    ax = axes[0, 0]
    bt = tw["bat_tw_table"]
    b_e = bt[bt["elig_primary"] & bt["tw_bat_eff_100"].notna()]
    ax.scatter(b_e["raa_post"], b_e["tw_bat_eff_100"], s=20, alpha=0.5, color="#2E86C1", label="Eligible Batters")
    b_top = b_e[b_e["canonical_player_id"].isin(bat_short_ids)]
    ax.scatter(b_top["raa_post"], b_top["tw_bat_eff_100"], s=54, color="#1B4F72", edgecolor="white", zorder=5, label="Shortlist Top 10")
    for _, r in b_top.head(4).iterrows():
        ax.annotate(_short_name(r["player_name"]), (r["raa_post"], r["tw_bat_eff_100"]), xytext=(4, 2), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Approved Disjoint Bat RAA_post (Runs/100b)")
    ax.set_ylabel("Two-Way Shrunk Ridge Batter Effect (Runs/100b)")
    ax.set_title("A. Two-Way Batter-Bowler Matchup vs Context-Only RAA")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.96,
        f"Bipartite Largest CC: {tw['largest_component_pct']:.1f}% ({tw['n_connected_components']} CCs)\n"
        f"Bridge Players (IPL+T20I): {tw['n_bridge_batters']} Bat, {tw['n_bridge_bowlers']} Bowl\n"
        f"OOF RMSE: {tw['oof_rmse_ctx_only']:.5f} -> {tw['oof_rmse_plus_twoway']:.5f}\n"
        f"Bat rho={tw['bat_tw_vs_raa_post']['spearman_rho']:.3f} | Bowl rho={tw['bowl_tw_vs_rsaa_post']['spearman_rho']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=7.7,
        bbox=dict(boxstyle="round", facecolor="#F4F6F7", alpha=0.92),
    )
    ax.legend(loc="lower right")

    # Panel B: Season x Phase Residual Drift (Approved vs Hierarchical Season x Phase + Venue EB)
    ax = axes[0, 1]
    sp = sv["season_phase_residuals"]
    for ph, col in [("Powerplay", "#2E86C1"), ("Middle", "#27AE60"), ("Death", "#C0392B")]:
        s_ph = sp[sp["match_phase"] == ph].sort_values("season_year")
        ax.plot(s_ph["season_year"], s_ph["base_resid_100"], "o-", color=col, linewidth=1.8, label=f"{ph} (Approved)")
        ax.plot(s_ph["season_year"], s_ph["hier_resid_100"], "s--", color=col, alpha=0.65, linewidth=1.2, label=f"{ph} (SeasxPh+HierVenue)")
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xlabel("Season Year")
    ax.set_ylabel("Mean OOF Residual (Runs per 100 balls)")
    ax.set_title(f"B. Season x Phase Residuals & Hierarchical Venue EB ({sv['venues_lt10_matches']}/{sv['n_venues']} Venues <10 Matches)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=7.0, loc="upper left")

    # Panel C: Cross-Competition Paired Bridge Deltas & Weight Sensitivity
    ax = axes[1, 0]
    bridges = [
        ("Bat: IPL vs\nAll T20I", cc["bat_bridge_ipl_vs_all_t20i"]["diff_a_minus_b"], cc["bat_bridge_ipl_vs_all_t20i"]["n_bridge"]),
        ("Bat: IPL vs\nTop-8 T20I", cc["bat_bridge_ipl_vs_top8_t20i"]["diff_a_minus_b"], cc["bat_bridge_ipl_vs_top8_t20i"]["n_bridge"]),
        ("Bat: IPL vs\nOther T20I", cc["bat_bridge_ipl_vs_other_t20i"]["diff_a_minus_b"], cc["bat_bridge_ipl_vs_other_t20i"]["n_bridge"]),
        ("Bowl: IPL vs\nAll T20I", cc["bowl_bridge_ipl_vs_all_t20i"]["diff_a_minus_b"], cc["bowl_bridge_ipl_vs_all_t20i"]["n_bridge"]),
        ("Bowl: IPL vs\nTop-8 T20I", cc["bowl_bridge_ipl_vs_top8_t20i"]["diff_a_minus_b"], cc["bowl_bridge_ipl_vs_top8_t20i"]["n_bridge"]),
        ("Bowl: IPL vs\nOther T20I", cc["bowl_bridge_ipl_vs_other_t20i"]["diff_a_minus_b"], cc["bowl_bridge_ipl_vs_other_t20i"]["n_bridge"]),
    ]
    x_l = [b[0] for b in bridges]
    y_v = [0.0 if np.isnan(b[1]) else b[1] for b in bridges]
    c_b = ["#2E86C1", "#2471A3", "#85C1E9", "#D35400", "#A04000", "#F5B041"]
    bars = ax.bar(x_l, y_v, color=c_b, width=0.56)
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    for b, (_, val, n_br) in zip(bars, bridges):
        y_off = val + (0.45 if val >= 0 else -0.45)
        va = "bottom" if val >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, y_off, f"{val:+.2f}\n(n={n_br})", ha="center", va=va, fontsize=7.4)
    ax.set_ylim(min(min(y_v) - 3.2, -2.0), max(max(y_v) + 2.0, 1.5))
    ax.set_ylabel("Mean Paired Delta (IPL - T20I; Bat=/100b, Bowl=/ov)")
    ax.set_title("C. Paired Bridge Gaps & T20I Opponent-Tier Audit")
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Observed-Window Tenure Proxy vs Verified Biological Age (Left-Censoring Proof)
    ax = axes[1, 1]
    bio = pr["bio_age_audit"]
    lc = bio[bio["left_censored_pre2022"]]
    nlc = bio[~bio["left_censored_pre2022"]]
    ax.scatter(lc["tenure_months"], lc["true_age_2026"], s=58, color="#C0392B", label=f"Left-Censored (<=2022-06, n={len(lc)})")
    ax.scatter(nlc["tenure_months"], nlc["true_age_2026"], s=58, color="#27AE60", marker="^", label=f"Uncensored Debut (n={len(nlc)})")
    offsets_32d = [(4, 4), (4, -8), (-42, 4), (-42, -8), (5, 0)]
    annot_idx = 0
    for _, r in bio.sort_values(["true_age_2026", "tenure_months"]).iterrows():
        if r["true_age_2026"] <= 22 or r["true_age_2026"] >= 37 or not r["left_censored_pre2022"]:
            ox, oy = offsets_32d[annot_idx % len(offsets_32d)]
            annot_idx += 1
            ax.annotate(_short_name(r["player_name"]), (r["tenure_months"], r["true_age_2026"]), xytext=(ox, oy), textcoords="offset points", fontsize=7.3)
    ax.set_xlabel("Observed-Window Tenure Proxy (Months in 2022-2026 Data)")
    ax.set_ylabel("Verified Biological Age (2026)")
    ax.set_title(f"D. Left-Censoring Bias: Tenure Proxy vs True Age (r={pr['bio_tenure_vs_true_age_r']:.3f})")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "32_phase4_ext_matchup_venue_comp_prog.png")


def main() -> None:
    print("Loading raw datasets & existing Phase 3/4 processed outputs...")
    raw = load_raw_datasets()
    matches_df = raw["matches"].copy()
    competitions_df = raw["competitions"].copy()

    deliv_df = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz", low_memory=False)
    bat_df = pd.read_csv(DATA_PROCESSED / "phase3_batting_prelim.csv")
    bowl_df = pd.read_csv(DATA_PROCESSED / "phase3_bowling_prelim.csv")
    emg_bat_df = pd.read_csv(DATA_PROCESSED / "phase3_emerging_batting_supp.csv")
    emg_bowl_df = pd.read_csv(DATA_PROCESSED / "phase3_emerging_bowling_supp.csv")
    bat_short = pd.read_csv(DATA_PROCESSED / "phase4_bat_shortlist_cards.csv")
    bowl_short = pd.read_csv(DATA_PROCESSED / "phase4_bowl_shortlist_cards.csv")
    bat_short_ids = set(bat_short["canonical_player_id"])
    bowl_short_ids = set(bowl_short["canonical_player_id"])

    cfg = PerformanceConfig()

    print("[1/7] Proposal Audit Classification & Item 12 Match-State Invariant Verification...")
    cls_df = build_proposal_classification_table()
    cls_df.to_csv(DATA_PROCESSED / "phase4_ext_classification_audit.csv", index=False)
    inv_audit = audit_match_state_invariants(deliv_df, matches_df)

    print("[2/7] Recomputing Approved Primary OOF Baselines (read-only comparator)...")
    oof_pri, _ = fit_predict_oof_baselines(
        deliv_df, matches_df, competitions_df, include_suspected_dls=False, n_splits=cfg.n_oof_folds, exclude_emerging=True
    )

    print("[3/7] Running Item 1: OOF Win Probability & Counterfactual Leverage Index...")
    wp_res = evaluate_win_probability_and_leverage(oof_pri, matches_df, bat_df, bowl_df, cfg)
    wp_res["calibration_summary"].to_csv(DATA_PROCESSED / "phase4_ext_wp_calibration.csv", index=False)
    wp_res["li_sensitivity"].to_csv(DATA_PROCESSED / "phase4_ext_li_sensitivity.csv", index=False)

    print("[4/7] Running Items 2 & 3: Enriched Contextual RoE & Formal Pressure Delta...")
    roe_res = evaluate_enriched_roe_and_pressure_delta(oof_pri, matches_df, competitions_df, wp_res, bat_df, bowl_df, cfg)
    roe_res["roe_fit_summary"].to_csv(DATA_PROCESSED / "phase4_ext_roe_fit.csv", index=False)

    print("[5/7] Running Items 4 & 5: Match-Cluster Bootstrap (B=400) & Split-Half Repeatability...")
    bs_res = evaluate_match_bootstrap_and_stability(oof_pri, matches_df, competitions_df, bat_df, bowl_df, cfg, n_boot=400)

    print("[6/7] Running Items 6 & 7: Opportunity-Neutral Phase Dominance & Wicket Quality + Pace Execution...")
    pdi_res = evaluate_phase_dominance_and_wicket_quality(oof_pri, matches_df, competitions_df, bat_df, bowl_df, cfg)

    print("[7/7] Running Items 8-11: Two-Way Matchup Ridge, Season/Venue EB, Bridge Calibration & Tenure Proxy...")
    m_res = evaluate_two_way_matchups_season_venue_comp_prog(
        oof_pri, matches_df, competitions_df, bat_df, bowl_df, cfg, emg_bat_df=emg_bat_df, emg_bowl_df=emg_bowl_df
    )

    # Assemble consolidated player comparison tables (strictly new extension tables)
    bat_ext = bat_df[
        ["canonical_player_id", "player_name", "elig_primary", "neff", "neff_hp", "ipl_balls", "t20i_balls", "pri_score_lcb", "raa_post", "raa_lcb", "hp_raa_post", "cons_post"]
    ].copy()
    bat_ext["is_phase4_shortlist"] = bat_ext["canonical_player_id"].isin(bat_short_ids)
    bat_ext = bat_ext.merge(wp_res["bat_li10"][["canonical_player_id", "neff_li10", "li10_raa_raw", "li10_raa_post"]], on="canonical_player_id", how="left")
    bat_ext = bat_ext.merge(roe_res["bat_roe_delta"][["canonical_player_id", "enr_roe_raw", "enr_roe_post", "delta_pi_raw_100", "delta_pi_post_100", "delta_pi_se_100", "delta_pi_z", "delta_li10_raw_100"]], on="canonical_player_id", how="left")
    bat_ext = bat_ext.merge(bs_res["bat_boot_stab"][["canonical_player_id", "deff_raa", "deff_hp", "raa_sd_cluster", "pri_score_lcb_cluster", "match_impact_sd", "odd_impact", "even_impact", "first_half_impact", "second_half_impact", "pri_score_lcb_invsd"]], on="canonical_player_id", how="left")
    bat_ext = bat_ext.merge(pdi_res["bat_pdi"][["canonical_player_id", "pdi_powerplay_post", "pdi_middle_post", "pdi_death_post", "pdi_equal_score", "pdi_peak_z", "n_pos_phases"]], on="canonical_player_id", how="left")
    bat_ext = bat_ext.merge(m_res["two_way"]["bat_tw_table"][["canonical_player_id", "tw_bat_eff_100"]], on="canonical_player_id", how="left")
    bat_ext = bat_ext.merge(m_res["progression_tenure"]["bat_tenure"][["canonical_player_id", "first_app_date", "last_app_date", "tenure_months", "left_censored_pre2022", "progression_delta"]], on="canonical_player_id", how="left")
    bat_ext.to_csv(DATA_PROCESSED / "phase4_ext_bat_comparison.csv", index=False)

    bowl_ext = bowl_df[
        ["canonical_player_id", "player_name", "elig_primary", "neff", "neff_hp", "ipl_legal_balls", "t20i_legal_balls", "pri_score_lcb", "rsaa_post", "waa_post", "hp_impact_post", "cons_post"]
    ].copy()
    bowl_ext["is_phase4_shortlist"] = bowl_ext["canonical_player_id"].isin(bowl_short_ids)
    bowl_ext = bowl_ext.merge(wp_res["bowl_li10"][["canonical_player_id", "neff_li10", "li10_rsaa_raw", "li10_rsaa_post"]], on="canonical_player_id", how="left")
    bowl_ext = bowl_ext.merge(roe_res["bowl_roe_delta"][["canonical_player_id", "enr_rsaa_raw", "enr_rsaa_post", "enr_tot_rsaa_raw", "delta_pi_raw_100", "delta_pi_post_100", "delta_pi_se_100", "delta_pi_z", "delta_li10_raw_100"]], on="canonical_player_id", how="left")
    bowl_ext = bowl_ext.merge(bs_res["bowl_boot_stab"][["canonical_player_id", "deff_rsaa", "deff_hp", "rsaa_sd_cluster", "pri_score_lcb_cluster", "match_impact_sd", "odd_impact", "even_impact", "first_half_impact", "second_half_impact", "pri_score_lcb_invsd"]], on="canonical_player_id", how="left")
    bowl_ext = bowl_ext.merge(pdi_res["bowl_pdi_wq"][["canonical_player_id", "pdi_powerplay_post", "pdi_middle_post", "pdi_death_post", "pdi_equal_score", "pdi_peak_z", "wqa_post", "pri_score_lcb_wqa", "pri_score_lcb_wqb", "pri_score_lcb_wqc", "pace_leg_balls", "yorker_pct", "death_yorker_pct", "bouncer_pct", "full_toss_pct"]], on="canonical_player_id", how="left")
    bowl_ext = bowl_ext.merge(m_res["two_way"]["bowl_tw_table"][["canonical_player_id", "tw_bowl_eff_po"]], on="canonical_player_id", how="left")
    bowl_ext = bowl_ext.merge(m_res["progression_tenure"]["bowl_tenure"][["canonical_player_id", "first_app_date", "last_app_date", "tenure_months", "left_censored_pre2022", "progression_delta"]], on="canonical_player_id", how="left")
    bowl_ext.to_csv(DATA_PROCESSED / "phase4_ext_bowl_comparison.csv", index=False)

    # Save supporting extension tables
    cc_df = pd.concat(
        [
            m_res["comp_calibration"]["bat_weight_sensitivity"].assign(discipline="Batting"),
            m_res["comp_calibration"]["bowl_weight_sensitivity"].assign(discipline="Bowling"),
        ],
        ignore_index=True,
    )
    cc_df.to_csv(DATA_PROCESSED / "phase4_ext_comp_weight_sensitivity.csv", index=False)
    m_res["season_venue"]["season_phase_residuals"].to_csv(DATA_PROCESSED / "phase4_ext_season_phase_residuals.csv", index=False)
    m_res["progression_tenure"]["bio_age_audit"].to_csv(DATA_PROCESSED / "phase4_ext_bio_tenure_audit.csv", index=False)

    # Save JSON summary
    summary = {
        "item12_invariant_audit": inv_audit,
        "item1_wp_calibration": wp_res["calibration_summary"].to_dict(orient="records"),
        "item1_li_sensitivity": wp_res["li_sensitivity"].to_dict(orient="records"),
        "item2_roe_fit": roe_res["roe_fit_summary"].to_dict(orient="records"),
        "item2_item3_comparisons": roe_res["comparisons"],
        "item4_item5_bootstrap_stability": bs_res["summary"],
        "item6_item7_pdi_wicket_quality": pdi_res["summary"],
        "item8_two_way_matchups": {k: v for k, v in m_res["two_way"].items() if k not in ("bat_tw_table", "bowl_tw_table")},
        "item9_season_venue": {k: v for k, v in m_res["season_venue"].items() if k not in ("season_phase_residuals", "venue_stats")},
        "item10_comp_calibration": {
            k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v)
            for k, v in m_res["comp_calibration"].items()
        },
        "item11_tenure_progression": {
            k: v for k, v in m_res["progression_tenure"].items() if k not in ("bio_age_audit", "bat_tenure", "bowl_tenure")
        },
    }
    with open(DATA_PROCESSED / "phase4_ext_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("Generating Figures 29-32...")
    make_figure_29(wp_res, roe_res, bat_short_ids, bowl_short_ids)
    make_figure_30(bs_res, bat_short_ids, bowl_short_ids)
    make_figure_31(pdi_res, bat_short_ids, bowl_short_ids)
    make_figure_32(m_res, bat_short_ids, bowl_short_ids)
    print("Done! All Phase 4 exploratory extension outputs saved cleanly.")


if __name__ == "__main__":
    main()

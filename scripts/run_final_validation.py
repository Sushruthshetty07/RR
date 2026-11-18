"""
Final Validation & Analytical Pipeline Runner (Parts C, D, F, G, H, I, L, M).

Preserves all frozen Phase 1-3 CSVs and Phase 4 cards untouched.
Outputs structured `data/processed/final_*.csv`, `final_validation_summary.json`,
and Figures 33-36 in both `reports/figures/` and artifact `figures/`.
"""

import json
import os
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
from src.final_validation import (
    analyze_pressure_intelligence,
    audit_enriched_roe_and_pressure_delta,
    build_part_l_validation_scorecard,
    evaluate_same_season_pdi_and_archetypes,
    reconstruct_and_audit_sample_matches,
    run_multiseed_cluster_bootstrap,
    run_temporal_holdout_validation,
)
from src.performance import PerformanceConfig, fit_predict_oof_baselines
from src.phase4_analysis import RETIRED_FROM_IPL, select_shortlists_and_benchmarks

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
        "legend.fontsize": 7.8,
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


def make_figure_33_pressure_methodology(pi_res: dict, bat_df: pd.DataFrame, bat_top10_ids: set) -> None:
    """Figure 33: Pressure Methodology Diagram (Distributions, RRR Achievability, Wickets/Comp, 4-Quadrant Map)."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.8))
    fig.suptitle(
        "Pressure Methodology & Calibration Audit (IPL + Senior T20I Primary Pool, PI Cutoff = 60)",
        y=0.99,
    )

    # Panel A: Phase x Innings Mean & % High-Pressure (PI >= 60)
    ax = axes[0, 0]
    ph = pi_res["by_phase"].copy()
    ph["label"] = ph["match_phase"].astype(str) + "\n(Inn " + ph["innings"].astype(str) + ")"
    bat_ph = ph[ph["role"] == "Batting"].copy()
    bowl_ph = ph[ph["role"] == "Bowling"].copy()
    x = np.arange(len(bat_ph))
    w = 0.36
    ax.bar(x - w / 2, bat_ph["mean_pi"], width=w, color="#1F618D", label="Batting Mean PI")
    ax.bar(x + w / 2, bowl_ph["mean_pi"], width=w, color="#CA6F1E", label="Bowling Mean PI")
    for i, r in enumerate(bat_ph.itertuples()):
        ax.text(i - w / 2, r.mean_pi + 1.2, f"{r.pct_hp:.0f}% HP", ha="center", fontsize=7.3, color="#1B4F72", fontweight="bold")
    ax.axhline(60.0, color="#922B21", linestyle="--", linewidth=1.2, label="HP Cutoff (PI = 60)")
    ax.set_xticks(x)
    ax.set_xticklabels(bat_ph["label"], fontsize=8.0)
    ax.set_ylabel("Mean Pressure Index (0-100)")
    ax.set_ylim(0, 82)
    ax.set_title("A. Pressure Escalation by Innings & Phase (% Deliveries >= 60)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    # Panel B: Chase RRR Achievability Regimes (Curbing Hopeless Chase Distortion)
    ax = axes[0, 1]
    rrr_df = pi_res["by_rrr"].copy()
    x2 = np.arange(len(rrr_df))
    ax.plot(x2, rrr_df["mean_pi"], "o-", color="#117A65", linewidth=2.2, markersize=7, label="Batting Mean PI (with Achievability Guard)")
    ax.plot(x2, rrr_df["mean_pi_bowl"], "s--", color="#B9770E", linewidth=2.0, markersize=6, label="Bowling Mean PI")
    ax.fill_between(x2, rrr_df["median_pi"], rrr_df["p90_pi"], color="#76D7C4", alpha=0.22, label="Batting Median-P90 Band")
    for i, r in rrr_df.iterrows():
        ax.annotate(f"n={r['n']:,}\n({r['pct_hp']:.0f}% HP)", (i, r["mean_pi"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.4)
    ax.axhline(60.0, color="#922B21", linestyle=":", linewidth=1.1)
    ax.set_xticks(x2)
    ax.set_xticklabels(["<8", "8-12", "12-16", "16-24", ">=24\n(Hopeless)"], fontsize=8.2)
    ax.set_xlabel("Chase Required Run Rate (RRR) Regime")
    ax.set_ylabel("Pressure Index (0-100)")
    ax.set_ylim(25, 108)
    ax.set_title("B. Innings-2 Achievability Guard Across Chase RRR Regimes")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.6)

    # Panel C: Wicket Brackets & Comp Tier Mean PI
    ax = axes[1, 0]
    wkt = pi_res["by_wicket"][pi_res["by_wicket"]["role"] == "Batting"].copy()
    wkt["label"] = wkt["wkt_bracket"].astype(str) + " | Inn " + wkt["innings"].astype(str)
    x3 = np.arange(len(wkt))
    colors = ["#2E86C1" if inn == 1 else "#D35400" for inn in wkt["innings"]]
    bars = ax.bar(x3, wkt["mean_pi"], color=colors, width=0.58)
    for b, r in zip(bars, wkt.itertuples()):
        ax.text(b.get_x() + b.get_width() / 2, r.mean_pi + 1.0, f"{r.mean_pi:.1f}\n({r.pct_hp:.0f}%)", ha="center", fontsize=7.6)
    ax.axhline(60.0, color="#922B21", linestyle="--", linewidth=1.1)
    ax.set_xticks(x3)
    ax.set_xticklabels(wkt["label"], rotation=15, ha="right", fontsize=8.0)
    ax.set_ylabel("Batting Mean PI (Blue=Inn1 Setting, Orange=Inn2 Chasing)")
    ax.set_ylim(0, 75)
    ax.set_title("C. Collapse Sensitivity: Batting Pressure by Wickets Lost")
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Four-Quadrant Diagram (General vs High-Pressure Disjoint Partition)
    ax = axes[1, 1]
    be = bat_df[bat_df["elig_primary"]].copy()
    ax.scatter(be["raa_post"], be["hp_raa_post"], c="#BDC3C7", alpha=0.55, s=28, edgecolors="none", label=f"Primary Eligible Pool (n={len(be)})")
    t10 = be[be["canonical_player_id"].isin(bat_top10_ids)].copy()
    ax.scatter(t10["raa_post"], t10["hp_raa_post"], c="#C0392B", s=65, edgecolors="black", linewidth=0.8, zorder=4, label="Official Top 10 Batters")
    for _, r in t10.iterrows():
        ax.annotate(_short_name(r["player_name"]), (r["raa_post"], r["hp_raa_post"]), xytext=(5, 3), textcoords="offset points", fontsize=7.8, fontweight="bold")
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.text(0.97, 0.96, "ALL-WEATHER ELITE\n(Gen > 0 & HP > 0)", transform=ax.transAxes, ha="right", va="top", fontsize=8.0, color="#145A32", fontweight="bold")
    ax.text(0.03, 0.96, "CLUTCH SPECIALISTS\n(Gen <= 0 & HP > 0)", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, color="#1B4F72", fontweight="bold")
    ax.text(0.97, 0.04, "FRONT-RUNNERS\n(Gen > 0 & HP <= 0)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, color="#7E5109", fontweight="bold")
    ax.set_xlabel("Disjoint General Performance: RAA_post (PI < 60, runs/100b)")
    ax.set_ylabel("Disjoint High-Pressure: HP-RAA_post (PI >= 60, runs/100b)")
    ax.set_title("D. Disjoint Four-Quadrant Tactical Framework (Official Top 10 Marked)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=7.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "33_final_pressure_methodology.png")


def make_figure_34_top10_and_uncertainty(
    bat_bench15: pd.DataFrame,
    bowl_bench15: pd.DataFrame,
    boot_res: dict,
    roe_res: dict,
) -> None:
    """Figure 34: Official Unrestricted Top 10 Batters & Bowlers + B=2,000 Bootstrap CIs + Bowling RoE Movers."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.2))
    fig.suptitle(
        "Official Unrestricted Top 10 Rankings, Cluster Bootstrap (B=2,000) CIs & Enriched RoE Audit",
        y=0.99,
    )

    # Panel A: Official Top 10 Batters (ordered 1..10, retired flagged cleanly)
    ax = axes[0, 0]
    b10 = bat_bench15.sort_values("raw_pri_lcb_rank").head(10).merge(
        boot_res["bat_boot"][["canonical_player_id", "pri_score_lcb_cluster", "deff_raa", "deff_hp", "hp_post_ci_lo", "hp_post_ci_hi"]],
        on="canonical_player_id",
        how="left",
    ).iloc[::-1]
    y = np.arange(len(b10))
    colors_b = ["#7F8C8D" if int(pid) in RETIRED_FROM_IPL else "#1B4F72" for pid in b10["canonical_player_id"]]
    ax.barh(y, b10["pri_score_lcb"], height=0.62, color=colors_b, alpha=0.9, label="Approved PRI_lcb")
    ax.scatter(b10["pri_score_lcb_cluster"], y, color="#E74C3C", marker="D", s=38, zorder=5, label="Cluster-Boot B=2000 PRI_lcb")
    labels_b = [
        f"#{int(r.raw_pri_lcb_rank)} {_short_name(r.player_name)}" + (" [RET]" if int(r.canonical_player_id) in RETIRED_FROM_IPL else "")
        for r in b10.itertuples()
    ]
    for i, r in enumerate(b10.itertuples()):
        ax.text(r.pri_score_lcb + 0.15, i, f"{r.pri_score_lcb:.2f} (HP={r.hp_raa_post:+.1f} [{r.hp_post_ci_lo:+.1f},{r.hp_post_ci_hi:+.1f}])", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_b, fontsize=8.3)
    ax.set_xlim(93.5, 106.5)
    ax.set_xlabel("PRI_lcb Score (Unrestricted Official Rank)")
    ax.set_title("A. Official Top 10 Batters (Grey = IPL Retired, Red Diamond = B=2000 Cluster PRI)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5)

    # Panel B: Official Top 10 Bowlers
    ax = axes[0, 1]
    l10 = bowl_bench15.sort_values("raw_pri_lcb_rank").head(10).merge(
        boot_res["bowl_boot"][["canonical_player_id", "pri_score_lcb_cluster", "deff_rsaa", "deff_hp", "hp_post_ci_lo", "hp_post_ci_hi"]],
        on="canonical_player_id",
        how="left",
    ).iloc[::-1]
    y2 = np.arange(len(l10))
    colors_l = ["#7F8C8D" if int(pid) in RETIRED_FROM_IPL else "#117A65" for pid in l10["canonical_player_id"]]
    ax.barh(y2, l10["pri_score_lcb"], height=0.62, color=colors_l, alpha=0.9, label="Approved PRI_lcb")
    ax.scatter(l10["pri_score_lcb_cluster"], y2, color="#D35400", marker="D", s=38, zorder=5, label="Cluster-Boot B=2000 PRI_lcb")
    labels_l = [f"#{int(r.raw_pri_lcb_rank)} {_short_name(r.player_name)}" for r in l10.itertuples()]
    for i, r in enumerate(l10.itertuples()):
        ax.text(r.pri_score_lcb + 0.15, i, f"{r.pri_score_lcb:.2f} (DEFF={r.deff_rsaa:.2f}, HP={r.hp_impact_post:+.2f})", va="center", fontsize=7.5)
    ax.set_yticks(y2)
    ax.set_yticklabels(labels_l, fontsize=8.3)
    ax.set_xlim(92.0, 105.5)
    ax.set_xlabel("PRI_lcb Score (Unrestricted Official Rank)")
    ax.set_title("B. Official Top 10 Bowlers & Exact Cluster DEFF Adjustment")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5)

    # Panel C: Bootstrap B=400 vs 1000 vs 2000 Convergence & Multi-Seed Agreement
    ax = axes[1, 0]
    conv = boot_res["convergence"]
    ax.plot(conv["n_boot"], conv["bat_rho_vs_approved"], "o-", color="#1F618D", linewidth=2.0, label="Bat Spearman rho vs Approved")
    ax.plot(conv["n_boot"], conv["bowl_rho_vs_approved"], "s-", color="#117A65", linewidth=2.0, label="Bowl Spearman rho vs Approved")
    ax.plot(conv["n_boot"], conv["bat_top10_overlap"] / 10.0, "^--", color="#2E86C1", linewidth=1.6, label="Bat Top-10 Overlap / 10")
    ax.plot(conv["n_boot"], conv["bowl_top10_overlap"] / 10.0, "d--", color="#CA6F1E", linewidth=1.6, label="Bowl Top-10 Overlap / 10")
    ax.set_xticks(conv["n_boot"])
    ax.set_ylim(0.85, 1.02)
    ax.set_xlabel("Cluster Bootstrap Replicates (B)")
    ax.set_ylabel("Rank Correlation / Top-10 Overlap Ratio")
    ax.set_title("C. Bootstrap Convergence (B=400 -> 1000 -> 2000) & Seed Stability")
    b_st = boot_res["bat_seed_stability"]
    l_st = boot_res["bowl_seed_stability"]
    deff_cmp = boot_res["bowling_deff_exact_vs_legal_only"]
    txt = (
        f"Seeds {boot_res['seeds']} @ B={boot_res['n_boot_primary']}:\n"
        f"Bat Top-10 Identical={b_st['top10_identical_across_seeds']} (Max Rank Diff={b_st['max_rank_diff_across_seeds']})\n"
        f"Bowl Top-10 Identical={l_st['top10_identical_across_seeds']} (Max Rank Diff={l_st['max_rank_diff_across_seeds']})\n"
        f"Bowl DEFF exact(incl W/NB)={deff_cmp['median_deff_exact_incl_wides_nb']:.3f} vs legal={deff_cmp['median_deff_legal_only']:.3f}"
    )
    ax.text(0.03, 0.05, txt, transform=ax.transAxes, va="bottom", fontsize=7.6, bbox=dict(boxstyle="round", facecolor="#F4F6F7", alpha=0.92))
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", fontsize=7.4)

    # Panel D: Exact Bowling Movers (Approved RSAA_post vs Enriched RSAA_post)
    ax = axes[1, 1]
    bm = roe_res["bowl_enriched_delta"]
    be = bm[bm["elig_primary"]].copy()
    ax.scatter(be["rsaa_post"], be["enr_rsaa_gen_post"], c="#BDC3C7", alpha=0.55, s=26, label="Primary Bowlers")
    mov = roe_res["bowl_top10_movers"].copy()
    for _, r in mov.iterrows():
        stay = (r["rank_rsaa_gen"] <= 10) and (r["rank_enr_gen"] <= 10)
        in_only = (r["rank_rsaa_gen"] > 10) and (r["rank_enr_gen"] <= 10)
        col = "#145A32" if stay else ("#1F618D" if in_only else "#C0392B")
        mrk = "o" if stay else ("^" if in_only else "X")
        ax.scatter(r["rsaa_post"], r["enr_rsaa_gen_post"], c=col, marker=mrk, s=65, edgecolors="black", linewidth=0.7, zorder=4)
        ax.annotate(
            f"{_short_name(r['player_name'])} ({int(r['rank_rsaa_gen'])}->{int(r['rank_enr_gen'])})",
            (r["rsaa_post"], r["enr_rsaa_gen_post"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.3,
            color=col,
            fontweight="bold",
        )
    lims = [min(be["rsaa_post"].min(), be["enr_rsaa_gen_post"].min()) - 0.1, max(be["rsaa_post"].max(), be["enr_rsaa_gen_post"].max()) + 0.1]
    ax.plot(lims, lims, "--", color="#555555", linewidth=1.0)
    ax.set_xlabel("Approved General RSAA_post (Runs Saved/Over, PI < 60)")
    ax.set_ylabel("Enriched General RSAA_post (+Style, Pace/Spin, Season*Phase)")
    ax.set_title("D. Exact Bowling Top-10 Movers: Context vs Style-Enriched RoE")
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "34_final_top10_validation_uncertainty.png")


def make_figure_35_temporal_and_delta(
    holdout_res: dict,
    ext_summary: dict,
    roe_res: dict,
    bat_bench15: pd.DataFrame,
    bowl_bench15: pd.DataFrame,
) -> None:
    """Figure 35: Temporal Holdout (<=2024 -> 2025-2026), WP Asymmetry & Formal Pressure Delta."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.8))
    fig.suptitle(
        "Independent Temporal Held-Out Validation (2022-24 Train -> 2025-26 Unseen Future) & Pressure Delta",
        y=0.99,
    )

    # Panel A: Delivery-level Temporal Holdout RMSE (Batting 2025-2026)
    ax = axes[0, 0]
    bd = holdout_res["bat_delivery_holdout"].copy()
    short_names = [
        "Approved Context\nBaseline",
        "Enriched RoE\n(+Style,S*P,Score)",
        "Hierarchical Venue\n+ Season Trend",
        "Context + Two-Way\n(alpha=100)",
        "Context + Two-Way\n(alpha=500)",
    ]
    x = np.arange(len(bd))
    cols = ["#1B4F72", "#D35400", "#117A65", "#7D3C98", "#A569BD"]
    bars = ax.bar(x, bd["future_rmse"], color=cols, width=0.58)
    for b, r in zip(bars, bd.itertuples()):
        ax.text(b.get_x() + b.get_width() / 2, r.future_rmse + 0.00012, f"{r.future_rmse:.5f}\n(R2={r.future_r2:.4f})", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7.8)
    ax.set_ylim(bd["future_rmse"].min() - 0.0015, bd["future_rmse"].max() + 0.0015)
    ax.set_ylabel("Future 2025-2026 Delivery RMSE (Lower = Better)")
    ax.set_title(f"A. Unseen Future Delivery Prediction (n_test={int(bd['n_test_future'].iloc[0]):,})")
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Player-level Future Generalization (Weighted r & Spearman rho)
    ax = axes[0, 1]
    ph = holdout_res["player_holdout_generalization"].copy()
    keep_idx = [0, 1, 2, 4, 6, 7, 8, 10, 11]
    ph_sub = ph.iloc[[i for i in keep_idx if i < len(ph)]].copy().iloc[::-1]
    y2 = np.arange(len(ph_sub))
    ax.barh(y2 - 0.16, ph_sub["weighted_r"], height=0.30, color="#1F618D", label="Exposure-Weighted Pearson r")
    ax.barh(y2 + 0.16, ph_sub["spearman_rho"], height=0.30, color="#D4AC0D", label="Spearman rho")
    lbls = [
        t.replace(" (<=2024 -> >=2025)", "").replace(" (<=2024 -> >=2025 WAA)", "").replace(" (<=2024 -> >=2025 HP)", " (HP)")
        for t in ph_sub["test"]
    ]
    for i, r in enumerate(ph_sub.itertuples()):
        ax.text(max(r.weighted_r, r.spearman_rho) + 0.015, i, f"r={r.weighted_r:.3f}, n={r.n_players}", va="center", fontsize=7.3)
    ax.set_yticks(y2)
    ax.set_yticklabels(lbls, fontsize=7.5)
    ax.set_xlim(0.0, 0.58)
    ax.set_xlabel("2022-24 Estimator Correlation with 2025-26 Unseen Performance")
    ax.set_title("B. Player-Level Future Year-Over-Year Generalization")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5)

    # Panel C: Pressure Delta with 95% CIs + Absolute PRI Guardrail (Top 10 Batters)
    ax = axes[1, 0]
    b10 = bat_bench15.sort_values("raw_pri_lcb_rank").head(10).merge(
        roe_res["bat_enriched_delta"][["canonical_player_id", "hp_roe_100", "norm_roe_100", "pressure_delta_100", "pressure_delta_ci_lo", "pressure_delta_ci_hi"]],
        on="canonical_player_id",
        how="left",
    ).iloc[::-1]
    y3 = np.arange(len(b10))
    err_lo = np.maximum(b10["pressure_delta_100"] - b10["pressure_delta_ci_lo"], 0.0)
    err_hi = np.maximum(b10["pressure_delta_ci_hi"] - b10["pressure_delta_100"], 0.0)
    c_delta = ["#117A65" if v >= 0 else "#C0392B" for v in b10["pressure_delta_100"]]
    ax.barh(y3, b10["pressure_delta_100"], height=0.55, color=c_delta, alpha=0.85)
    ax.errorbar(b10["pressure_delta_100"], y3, xerr=[err_lo, err_hi], fmt="none", ecolor="#2C3E50", elinewidth=1.2, capsize=3)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    for i, r in enumerate(b10.itertuples()):
        ax.text(r.pressure_delta_ci_hi + 0.8, i, f"PRI={r.pri_score_lcb:.1f} | HP={r.hp_roe_100:+.1f}", va="center", fontsize=7.3)
    ax.set_yticks(y3)
    ax.set_yticklabels([f"#{int(r.raw_pri_lcb_rank)} {_short_name(r.player_name)}" for r in b10.itertuples()], fontsize=8.1)
    ax.set_xlim(min(b10["pressure_delta_ci_lo"].min() - 4, -22), max(b10["pressure_delta_ci_hi"].max() + 26, 48))
    ax.set_xlabel("Pressure Delta = HP RoE (PI>=60) - Normal RoE (PI<60) [runs/100b, 95% CI]")
    ax.set_title("C. Top 10 Batters: Formal Pressure Delta & Absolute PRI Guardrail")
    ax.grid(axis="x", alpha=0.25)

    # Panel D: Top 10 Bowlers Pressure Delta with 95% CIs
    ax = axes[1, 1]
    l10 = bowl_bench15.sort_values("raw_pri_lcb_rank").head(10).merge(
        roe_res["bowl_enriched_delta"][["canonical_player_id", "hp_roe_100", "norm_roe_100", "pressure_delta_100", "pressure_delta_ci_lo", "pressure_delta_ci_hi"]],
        on="canonical_player_id",
        how="left",
    ).iloc[::-1]
    y4 = np.arange(len(l10))
    err_lo_l = np.maximum(l10["pressure_delta_100"] - l10["pressure_delta_ci_lo"], 0.0)
    err_hi_l = np.maximum(l10["pressure_delta_ci_hi"] - l10["pressure_delta_100"], 0.0)
    c_delta_l = ["#117A65" if v >= 0 else "#C0392B" for v in l10["pressure_delta_100"]]
    ax.barh(y4, l10["pressure_delta_100"], height=0.55, color=c_delta_l, alpha=0.85)
    ax.errorbar(l10["pressure_delta_100"], y4, xerr=[err_lo_l, err_hi_l], fmt="none", ecolor="#2C3E50", elinewidth=1.2, capsize=3)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    for i, r in enumerate(l10.itertuples()):
        ax.text(r.pressure_delta_ci_hi + 0.8, i, f"PRI={r.pri_score_lcb:.1f} | HP={r.hp_roe_100:+.1f}", va="center", fontsize=7.3)
    ax.set_yticks(y4)
    ax.set_yticklabels([f"#{int(r.raw_pri_lcb_rank)} {_short_name(r.player_name)}" for r in l10.itertuples()], fontsize=8.1)
    ax.set_xlim(min(l10["pressure_delta_ci_lo"].min() - 4, -22), max(l10["pressure_delta_ci_hi"].max() + 26, 48))
    ax.set_xlabel("Pressure Delta (HP Net Runs Saved/100b - Normal Net Runs Saved/100b, 95% CI)")
    ax.set_title("D. Top 10 Bowlers: Formal Pressure Delta & Absolute PRI Guardrail")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "35_final_temporal_holdout_validation.png")


def make_figure_36_archetypes_and_rr_fit(pdi_res: dict, bat_bench15: pd.DataFrame, bowl_bench15: pd.DataFrame) -> None:
    """Figure 36: Same-Season Same-Comp Phase PDI Heatmaps, Spin-Hitting & 7 Tactical Archetypes."""
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.2))
    fig.suptitle(
        "Same-Season Same-Comp Phase Dominance, Spin-Hitting & 7 Tactical Archetypes for RR 2027",
        y=0.99,
    )

    # Panel A: Official Top 10 + Watchlist Batters Same-Season Same-Comp PDI + Spin Hitting Heatmap
    ax = axes[0, 0]
    b_top = bat_bench15.sort_values("raw_pri_lcb_rank").head(12).merge(
        pdi_res["bat_pdi_ss"][["canonical_player_id", "pp_pdi_post", "mid_pdi_post", "death_pdi_post", "spin_pdi_post", "primary_archetype"]],
        on="canonical_player_id",
        how="left",
    )
    mat_b = b_top[["pp_pdi_post", "mid_pdi_post", "death_pdi_post", "spin_pdi_post"]].fillna(0.0).values
    im_b = ax.imshow(mat_b, cmap="RdYlGn", vmin=-15, vmax=30, aspect="auto")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Powerplay\nSS-PDI", "Middle\nSS-PDI", "Death\nSS-PDI", "Vs Spin\nSS-PDI"], fontsize=8.2)
    ax.set_yticks(np.arange(len(b_top)))
    ax.set_yticklabels([f"#{int(r.raw_pri_lcb_rank)} {_short_name(r.player_name)}" for r in b_top.itertuples()], fontsize=8.0)
    for i in range(mat_b.shape[0]):
        for j in range(mat_b.shape[1]):
            val = mat_b[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7.5, color="black" if abs(val) < 18 else "white", fontweight="bold")
    fig.colorbar(im_b, ax=ax, fraction=0.035, pad=0.02, label="Same-Season RAA / 100b")
    ax.set_title("A. Top 12 Batters: Same-Season Same-Comp Phase & Spin PDI")

    # Panel B: Official Top 12 Bowlers Same-Season Same-Comp PDI Heatmap
    ax = axes[0, 1]
    l_top = bowl_bench15.sort_values("raw_pri_lcb_rank").head(12).merge(
        pdi_res["bowl_pdi_ss"][["canonical_player_id", "pp_pdi_post", "mid_pdi_post", "death_pdi_post", "primary_archetype"]],
        on="canonical_player_id",
        how="left",
    )
    mat_l = l_top[["pp_pdi_post", "mid_pdi_post", "death_pdi_post"]].fillna(0.0).values
    im_l = ax.imshow(mat_l, cmap="RdYlGn", vmin=-0.6, vmax=1.6, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Powerplay\nSS-PDI", "Middle\nSS-PDI", "Death\nSS-PDI"], fontsize=8.2)
    ax.set_yticks(np.arange(len(l_top)))
    ax.set_yticklabels([f"#{int(r.raw_pri_lcb_rank)} {_short_name(r.player_name)} ({str(r.pace_or_spin)[:1]})" for r in l_top.itertuples()], fontsize=8.0)
    for i in range(mat_l.shape[0]):
        for j in range(mat_l.shape[1]):
            val = mat_l[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7.6, color="black" if abs(val) < 1.1 else "white", fontweight="bold")
    fig.colorbar(im_l, ax=ax, fraction=0.035, pad=0.02, label="Same-Season Net Saved/Over")
    ax.set_title("B. Top 12 Bowlers: Same-Season Same-Comp Phase PDI")

    # Panel C: Spin-Hitting Leaders Among Primary Eligible Batters
    ax = axes[1, 0]
    sp_top = pdi_res["archetype_leaders"]["3_Spin_Hitters"].head(10).iloc[::-1]
    y3 = np.arange(len(sp_top))
    cols_s = ["#D4AC0D" if r.is_ipl_retired else "#1B4F72" for r in sp_top.itertuples()]
    ax.barh(y3, sp_top["spin_pdi_post"], height=0.60, color=cols_s, alpha=0.9)
    for i, r in enumerate(sp_top.itertuples()):
        ax.text(r.spin_pdi_post + 0.4, i, f"{r.spin_pdi_post:+.1f} (n={int(r.spin_pdi_n)}, PRI={r.pri_score_lcb:.1f})", va="center", fontsize=7.5)
    ax.set_yticks(y3)
    ax.set_yticklabels([f"{_short_name(r.player_name)}" + (" [RET]" if r.is_ipl_retired else "") for r in sp_top.itertuples()], fontsize=8.1)
    ax.set_xlabel("Same-Season Same-Comp Runs Over Expected vs Spin (/100b)")
    ax.set_xlim(0, sp_top["spin_pdi_post"].max() + 14.0)
    ax.set_title("C. Top Spin Hitters (Crucial for Guwahati & Middle-Overs Acceleration)")
    ax.grid(axis="x", alpha=0.25)

    # Panel D: Top Active RR 2027 Tactical Acquisition Candidates by Archetype
    ax = axes[1, 1]
    active_b = pdi_res["bat_pdi_ss"][pdi_res["bat_pdi_ss"]["elig_primary"] & (~pdi_res["bat_pdi_ss"]["is_ipl_retired"].fillna(False))].copy()
    active_l = pdi_res["bowl_pdi_ss"][pdi_res["bowl_pdi_ss"]["elig_primary"] & (~pdi_res["bowl_pdi_ss"]["is_ipl_retired"].fillna(False))].copy()
    picks = []
    for lbl, df_src, mask_col, val_col in [
        ("Finisher", active_b, "is_finisher", "death_pdi_post"),
        ("PP Aggressor", active_b, "is_pp_aggressor", "pp_pdi_post"),
        ("Spin Hitter", active_b, "is_spin_hitter", "spin_pdi_post"),
        ("Mid Anchor", active_b, "is_mid_anchor", "mid_pdi_post"),
        ("Death Bowl", active_l, "is_death_specialist", "death_pdi_post"),
        ("Mid Spinner", active_l, "is_mid_spinner", "mid_pdi_post"),
        ("PP Bowler", active_l, "is_pp_bowler", "pp_pdi_post"),
    ]:
        sub = df_src[df_src[mask_col]].sort_values("pri_score_lcb", ascending=False).head(2)
        for _, r in sub.iterrows():
            picks.append(
                {
                    "tag": f"[{lbl}] {_short_name(r['player_name'])}",
                    "pri": float(r["pri_score_lcb"]),
                    "neff": float(r["neff"]),
                    "role": lbl,
                }
            )
    pk_df = pd.DataFrame(picks).drop_duplicates("tag").sort_values("pri", ascending=True)
    y4 = np.arange(len(pk_df))
    c_pk = ["#1B4F72" if any(k in r for k in ("Finisher", "PP Agg", "Spin H", "Mid Anc")) else "#117A65" for r in pk_df["role"]]
    ax.barh(y4, pk_df["pri"], height=0.62, color=c_pk, alpha=0.88)
    for i, r in enumerate(pk_df.itertuples()):
        ax.text(r.pri + 0.2, i, f"{r.pri:.1f} (Neff={r.neff:.0f})", va="center", fontsize=7.3)
    ax.set_yticks(y4)
    ax.set_yticklabels(pk_df["tag"], fontsize=7.6)
    ax.set_xlim(78, 106)
    ax.set_xlabel("Approved PRI_lcb Score")
    ax.set_title("D. Top Active Targets Across All 7 T20 Tactical Archetypes")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "36_final_role_archetypes_rr_fit.png")


def main() -> None:
    print("Loading raw datasets & frozen Phase 3/4 outputs...")
    raw = load_raw_datasets()
    matches_df = raw["matches"].copy()
    competitions_df = raw["competitions"].copy()

    deliv_df = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz", low_memory=False)
    bat_df = pd.read_csv(DATA_PROCESSED / "phase3_batting_prelim.csv")
    bowl_df = pd.read_csv(DATA_PROCESSED / "phase3_bowling_prelim.csv")
    bat_bench15 = pd.read_csv(DATA_PROCESSED / "phase4_bat_benchmark15_cards.csv")
    bowl_bench15 = pd.read_csv(DATA_PROCESSED / "phase4_bowl_benchmark15_cards.csv")
    with open(DATA_PROCESSED / "phase4_ext_summary.json", "r", encoding="utf-8") as f:
        ext_summary = json.load(f)

    cfg = PerformanceConfig()

    print("[1/6] Running Part C: Independent Match-State Reconstruction Audit (6 diverse real matches)...")
    recon_res = reconstruct_and_audit_sample_matches(raw, deliv_df)
    recon_res["sample_cases"].to_csv(DATA_PROCESSED / "final_match_reconstruction_audit.csv", index=False)
    recon_res["extras_audit"].to_csv(DATA_PROCESSED / "final_extras_audit.csv", index=False)
    print(f"      All 6 audit matches matched processed deliveries cleanly: {recon_res['all_cases_passed']} (mismatches={len(recon_res['mismatches_detail'])})")

    print("[2/6] Running Part D: Deep Pressure Intelligence & Achievability Audit...")
    pi_res = analyze_pressure_intelligence(deliv_df, matches_df, competitions_df, cfg)
    pi_res["by_phase"].to_csv(DATA_PROCESSED / "final_pressure_by_phase.csv", index=False)
    pi_res["by_wicket"].to_csv(DATA_PROCESSED / "final_pressure_by_wicket.csv", index=False)
    pi_res["by_comp"].to_csv(DATA_PROCESSED / "final_pressure_by_comp.csv", index=False)
    pi_res["by_progress"].to_csv(DATA_PROCESSED / "final_pressure_by_progress.csv", index=False)
    pi_res["by_rrr"].to_csv(DATA_PROCESSED / "final_pressure_by_rrr.csv", index=False)

    print("[3/6] Computing Approved OOF Baselines (read-only comparator)...")
    oof_pri, _ = fit_predict_oof_baselines(
        deliv_df, matches_df, competitions_df, include_suspected_dls=False, n_splits=cfg.n_oof_folds, exclude_emerging=True
    )

    print("[4/6] Running Part F.1: Multi-Seed Cluster Bootstrap B=400/1000/2000 & Exact Bowling DEFF...")
    boot_res = run_multiseed_cluster_bootstrap(oof_pri, matches_df, competitions_df, bat_df, bowl_df, cfg)
    boot_res["convergence"].to_csv(DATA_PROCESSED / "final_bootstrap_convergence.csv", index=False)
    boot_res["bat_boot"].to_csv(DATA_PROCESSED / "final_bat_bootstrap_2000.csv", index=False)
    boot_res["bowl_boot"].to_csv(DATA_PROCESSED / "final_bowl_bootstrap_2000.csv", index=False)

    print("[5/6] Running Part F.2, F.3 & F.6: Complete Enriched RoE, Movers, Pressure Delta & Same-Season PDI / 7 Archetypes...")
    roe_res = audit_enriched_roe_and_pressure_delta(oof_pri, matches_df, competitions_df, bat_df, bowl_df, cfg)
    roe_res["bowl_top10_movers"].to_csv(DATA_PROCESSED / "final_bowl_roe_top10_movers.csv", index=False)
    roe_res["bat_enriched_delta"].to_csv(DATA_PROCESSED / "final_bat_enriched_delta.csv", index=False)
    roe_res["bowl_enriched_delta"].to_csv(DATA_PROCESSED / "final_bowl_enriched_delta.csv", index=False)

    pdi_res = evaluate_same_season_pdi_and_archetypes(oof_pri, matches_df, competitions_df, bat_df, bowl_df, bat_bench15, bowl_bench15, cfg)
    pdi_res["bat_pdi_ss"].to_csv(DATA_PROCESSED / "final_bat_same_season_pdi.csv", index=False)
    pdi_res["bowl_pdi_ss"].to_csv(DATA_PROCESSED / "final_bowl_same_season_pdi.csv", index=False)
    arch_rows = []
    for arch_name, df_a in pdi_res["archetype_leaders"].items():
        tmp = df_a[["canonical_player_id", "player_name", "pri_score_lcb", "neff"]].copy()
        tmp["archetype"] = arch_name
        arch_rows.append(tmp)
    pd.concat(arch_rows, ignore_index=True).to_csv(DATA_PROCESSED / "final_7_archetype_leaders.csv", index=False)

    print("[6/6] Running Part G & L: Temporal Held-Out Validation (<=2024 Train -> 2025-2026 Test) & Scorecard...")
    holdout_res = run_temporal_holdout_validation(deliv_df, matches_df, competitions_df, cfg, cutoff_year=2025)
    holdout_res["bat_delivery_holdout"].to_csv(DATA_PROCESSED / "final_temporal_bat_delivery_holdout.csv", index=False)
    holdout_res["bowl_delivery_holdout"].to_csv(DATA_PROCESSED / "final_temporal_bowl_delivery_holdout.csv", index=False)
    holdout_res["player_holdout_generalization"].to_csv(DATA_PROCESSED / "final_temporal_player_holdout.csv", index=False)

    scorecard = build_part_l_validation_scorecard(boot_res, roe_res, pdi_res, holdout_res, ext_summary)
    scorecard.to_csv(DATA_PROCESSED / "final_part_l_validation_scorecard.csv", index=False)

    # Save master JSON summary
    summary = {
        "reconstruction_audit": {
            "all_cases_passed": recon_res["all_cases_passed"],
            "n_cases": int(len(recon_res["sample_cases"])),
            "total_raw_matches": recon_res["total_raw_matches"],
            "total_matches_with_bbb": recon_res["total_matches_with_bbb"],
            "total_raw_deliveries": recon_res["total_raw_deliveries"],
            "total_processed_deliveries": recon_res["total_processed_deliveries"],
            "playing_eleven_duplicate_keys": recon_res["playing_eleven_duplicate_keys"],
            "outcome_cols_in_processed_deliveries": recon_res["outcome_cols_in_processed_deliveries"],
            "mismatches_detail": recon_res["mismatches_detail"],
        },
        "pressure_audit": {
            "super_over_summary": pi_res["super_over_summary"],
            "rrr_regimes": pi_res["by_rrr"].to_dict(orient="records"),
        },
        "bootstrap_2000": {
            "n_boot_primary": boot_res["n_boot_primary"],
            "seeds": boot_res["seeds"],
            "convergence": boot_res["convergence"].to_dict(orient="records"),
            "bat_seed_stability": boot_res["bat_seed_stability"],
            "bowl_seed_stability": boot_res["bowl_seed_stability"],
            "bowling_deff_exact_vs_legal_only": boot_res["bowling_deff_exact_vs_legal_only"],
        },
        "enriched_roe_partitions": roe_res["partition_summary"],
        "same_season_pdi": {
            "bat_ss_pdi_vs_pri": pdi_res["bat_ss_pdi_vs_pri"],
            "bat_spin_vs_pri": pdi_res["bat_spin_vs_pri"],
            "bowl_ss_pdi_vs_pri": pdi_res["bowl_ss_pdi_vs_pri"],
        },
        "temporal_holdout": {
            "cutoff_year": holdout_res["cutoff_year"],
            "n_train_matches": holdout_res["n_train_matches"],
            "n_test_matches": holdout_res["n_test_matches"],
            "bat_delivery_holdout": holdout_res["bat_delivery_holdout"].to_dict(orient="records"),
            "bowl_delivery_holdout": holdout_res["bowl_delivery_holdout"].to_dict(orient="records"),
            "player_holdout_generalization": holdout_res["player_holdout_generalization"].to_dict(orient="records"),
        },
    }
    with open(DATA_PROCESSED / "final_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Render Figures 33-36
    bat_top10_ids = set(bat_bench15.sort_values("raw_pri_lcb_rank").head(10)["canonical_player_id"])
    print("Rendering Figures 33-36...")
    make_figure_33_pressure_methodology(pi_res, bat_df, bat_top10_ids)
    make_figure_34_top10_and_uncertainty(bat_bench15, bowl_bench15, boot_res, roe_res)
    make_figure_35_temporal_and_delta(holdout_res, ext_summary, roe_res, bat_bench15, bowl_bench15)
    make_figure_36_archetypes_and_rr_fit(pdi_res, bat_bench15, bowl_bench15)
    print("Done! Saved final validation CSVs, JSON summary, and Figures 33-36.")


if __name__ == "__main__":
    main()

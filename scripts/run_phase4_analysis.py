"""
Phase 4 Execution & Figure Generation Script.

Uses the corrected Phase 3 primary pool (IPL weight = 1.00, Senior_T20I weight = 0.50,
Emerging_Rising_T20 supplementary only) without modifying any Phase 1-3 CSVs or methodology.
Generates player evaluation cards, RR IPL 2026 verified baseline context, sensitivity
matrices across 9 model variants, Emerging-only supplementary summaries, and Figures 23-26.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DATA_PROCESSED, load_raw_datasets
from src.performance import (
    COMP_TIER_EMERGING,
    PerformanceConfig,
    classify_comp_tier,
    filter_dls_pool,
    fit_predict_oof_baselines,
)
from src.phase4_analysis import (
    RETIRED_FROM_IPL,
    build_detailed_player_profiles,
    build_portfolio_balance_summary,
    compute_contextual_adjusted_metrics,
    compute_rr_2026_context,
    compute_shortlist_sensitivity,
    compute_yoy_development_trajectories,
    extract_rejected_player_audit,
    select_shortlists_and_benchmarks,
    summarize_emerging_only_supplementary,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_FIG_DIR = Path(
    r"C:\Users\sushruth\.gemini\antigravity\brain\b6ea2651-73fc-4d7f-8e2e-0f0033d4425b\figures"
)
ARTIFACT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _enrich_raw_primary(
    raw_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Filters primary DLS + primary comp_tier pool and attaches season_yr & match_date."""
    m_meta = matches_df[["match_id", "comp_id", "match_date"]].merge(
        competitions_df[["comp_id", "season", "comp_name"]], on="comp_id", how="left"
    )
    pri = filter_dls_pool(
        raw_df,
        include_suspected_dls=False,
        exclude_emerging=True,
        matches_df=matches_df,
        competitions_df=competitions_df,
    )
    pri = pri.merge(m_meta, on="match_id", how="left")
    pri["season_yr"] = (
        pd.to_numeric(pri["season"].fillna("2024").astype(str).str.slice(0, 4), errors="coerce")
        .fillna(2024)
        .astype(int)
    )
    return pri


def _attach_emerging_comp_names(
    emg_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    id_col: str,
) -> pd.DataFrame:
    """Attaches Emerging competition names and match dates for Emerging-only supplementary context."""
    if len(emg_df) == 0:
        return emg_df
    m_meta = matches_df[["match_id", "comp_id", "match_date"]].merge(
        competitions_df[["comp_id", "comp_name"]], on="comp_id", how="left"
    )
    df = raw_df.copy()
    df["comp_tier"] = classify_comp_tier(df, matches_df=matches_df, competitions_df=competitions_df)
    emg_rows = df[df["comp_tier"] == COMP_TIER_EMERGING].merge(m_meta, on="match_id", how="left")
    if id_col == "canonical_striker_id":
        emg_rows = emg_rows[emg_rows["is_batter_ball_faced"]]
    else:
        emg_rows = emg_rows[emg_rows["is_legal_ball"]]
    agg = (
        emg_rows.groupby(id_col, as_index=False)
        .agg(
            emerging_competitions=("comp_name", lambda s: "; ".join(sorted(set(s.dropna().astype(str))))),
            last_emerging_date=("match_date", "max"),
        )
        .rename(columns={id_col: "canonical_player_id"})
    )
    return emg_df.merge(agg, on="canonical_player_id", how="left")


def _save_fig(fig: plt.Figure, filename: str) -> None:
    out_path = FIG_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    shutil.copy2(out_path, ARTIFACT_FIG_DIR / filename)
    plt.close(fig)


def _plot_batters_decomposition(bat_bench: pd.DataFrame, bat_short: pd.DataFrame, bat_watch: pd.DataFrame) -> None:
    """Figure 23: Benchmark Top-15 + Watchlist Batting Pillar Decomposition & Sample-Size Confidence."""
    active_rank_map = {
        int(r["canonical_player_id"]): int(r["shortlist_rank"])
        for _, r in pd.concat([bat_short, bat_watch], ignore_index=True).iterrows()
    }
    watch_extra = bat_watch[~bat_watch["canonical_player_id"].isin(bat_bench["canonical_player_id"])].copy()
    plot_df = pd.concat([bat_bench, watch_extra], ignore_index=True).iloc[::-1].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 8.8), gridspec_kw={"width_ratios": [1.30, 1.0]})

    y = np.arange(len(plot_df))
    labels = []
    is_active_top10 = []
    for _, r in plot_df.iterrows():
        pid = int(r["canonical_player_id"])
        raw_rk = int(r["raw_pri_lcb_rank"])
        if pid in RETIRED_FROM_IPL:
            labels.append(f"[RET raw #{raw_rk}] {r['player_name']}")
            is_active_top10.append(False)
        else:
            act_rk = active_rank_map[pid]
            if act_rk <= 10:
                tag = f"#{act_rk}" if act_rk == raw_rk else f"#{act_rk} (raw #{raw_rk})"
                labels.append(f"{tag} {r['player_name']}")
                is_active_top10.append(True)
            else:
                labels.append(f"W#{act_rk} (raw #{raw_rk}) {r['player_name']}")
                is_active_top10.append(False)

    c_cols = [("c_raa", "RAA LCB (45%)", "#1976D2"), ("c_surv", "Survival LCB (25%)", "#2E7D32"),
              ("c_cons", "Consistency LCB (15%)", "#F57C00"), ("c_hp", "High-Pressure LCB (15%)", "#C2185B")]

    pos_left = np.zeros(len(plot_df))
    neg_left = np.zeros(len(plot_df))
    for col, lbl, color in c_cols:
        vals = plot_df[col].values
        pos_v = np.maximum(vals, 0.0)
        neg_v = np.minimum(vals, 0.0)
        ax1.barh(y, pos_v, left=pos_left, height=0.68, color=color, alpha=0.88, label=lbl)
        ax1.barh(y, neg_v, left=neg_left, height=0.68, color=color, alpha=0.45)
        pos_left += pos_v
        neg_left += neg_v

    ax1.axvline(0, color="black", linewidth=0.9)
    for i, r in plot_df.iterrows():
        x_txt = max(pos_left[i], 0.15) + 0.06
        ax1.text(x_txt, y[i], f"PRI {r['pri_score_lcb']:.2f} (z={r['pri_z_lcb']:+.2f})",
                 va="center", fontsize=8.3, fontweight="bold" if is_active_top10[i] else "normal")

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.8)
    ax1.set_xlim(min(neg_left.min() - 0.3, -0.8), pos_left.max() + 1.50)
    ax1.set_xlabel("Weighted Z-Score Contribution to PRI_z_lcb", fontsize=10)
    ax1.set_title("A. Batting Pillar Contributions (Benchmark Top 15 & Active Watchlist)", fontweight="bold", fontsize=11)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax1.grid(axis="x", alpha=0.25)

    # Panel B: Effective sample size vs High-Pressure Neff, colored by Confidence Tier
    tier_colors = {
        "Tier A (High-Confidence Multi-Regime)": "#1B5E20",
        "Tier B (Strong Core / Targeted Gap)": "#F57C00",
        "Tier C (Low IPL Depth / Caveated Profile)": "#C62828",
    }
    for tier, clr in tier_colors.items():
        sub = plot_df[plot_df["confidence_tier"] == tier]
        if len(sub) > 0:
            sizes = np.clip(sub["ipl_balls"].values * 0.28, 55, 420)
            ax2.scatter(sub["neff"], sub["neff_hp"], s=sizes, color=clr, alpha=0.82, edgecolors="black", linewidth=0.8, label=tier)

    bat_offsets = {
        "Vaibhav Sooryavanshi": (12, 8),
        "Priyansh Arya": (-95, 11),
        "Finn Allen": (12, -9),
        "Rajat Patidar": (12, -9),
        "Phil Salt": (12, 5),
        "Pathum Nissanka": (12, -4),
        "Abhishek Sharma": (-18, 12),
        "Shreyas Iyer": (12, 3),
        "Suryakumar Yadav": (-50, 14),
    }
    for _, r in plot_df.iterrows():
        dx, dy = bat_offsets.get(r["player_name"], (12, 2))
        ax2.annotate(r["player_name"], (r["neff"], r["neff_hp"]), xytext=(dx, dy),
                     textcoords="offset points", fontsize=8.0, alpha=0.92)

    ax2.axvline(300, color="#1B5E20", linestyle="--", linewidth=1.0, alpha=0.65, label="Tier A Neff=300")
    ax2.axhline(35, color="#C2185B", linestyle=":", linewidth=1.1, alpha=0.75, label="HP Depth Threshold (Neff_HP=35)")
    ax2.set_xlabel("Primary Weighted Effective Sample Size (Neff)", fontsize=10)
    ax2.set_ylabel("High-Pressure Effective Sample Size (Neff_HP, PI >= 60)", fontsize=10)
    ax2.set_title("B. Evidence Depth & HP Exposure (Bubble Size = IPL Balls Faced)", fontweight="bold", fontsize=11)
    ax2.set_xlim(-40, plot_df["neff"].max() * 1.18)
    ax2.set_ylim(-18, plot_df["neff_hp"].max() * 1.16)
    ax2.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
    ax2.grid(alpha=0.25)

    fig.suptitle("Phase 4 Batting Shortlist & Benchmark Diagnostics (Corrected IPL 1.00 / Senior T20I 0.50 Pool)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "23_phase4_top10_batters_decomposition.png")


def _plot_bowlers_decomposition(bowl_short: pd.DataFrame, bowl_watch: pd.DataFrame) -> None:
    """Figure 24: Bowler Shortlist Top-10 + Watchlist Pillar Decomposition & Phase/Role Comparison."""
    plot_df = pd.concat([bowl_short, bowl_watch], ignore_index=True).iloc[::-1].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 8.2), gridspec_kw={"width_ratios": [1.22, 1.0]})

    y = np.arange(len(plot_df))
    labels = [
        f"#{int(r['shortlist_rank'])} {r['player_name']}" if int(r["shortlist_rank"]) <= 10
        else f"W#{int(r['shortlist_rank'])} {r['player_name']}"
        for _, r in plot_df.iterrows()
    ]

    c_cols = [("c_rsaa", "RSAA LCB (55%)", "#1565C0"), ("c_waa", "WAA LCB (15%)", "#2E7D32"),
              ("c_cons", "Consistency LCB (15%)", "#EF6C00"), ("c_hp", "High-Pressure LCB (15%)", "#AD1457")]

    pos_left = np.zeros(len(plot_df))
    neg_left = np.zeros(len(plot_df))
    for col, lbl, color in c_cols:
        vals = plot_df[col].values
        pos_v = np.maximum(vals, 0.0)
        neg_v = np.minimum(vals, 0.0)
        ax1.barh(y, pos_v, left=pos_left, height=0.68, color=color, alpha=0.88, label=lbl)
        ax1.barh(y, neg_v, left=neg_left, height=0.68, color=color, alpha=0.45)
        pos_left += pos_v
        neg_left += neg_v

    ax1.axvline(0, color="black", linewidth=0.9)
    for i, r in plot_df.iterrows():
        x_txt = max(pos_left[i], 0.15) + 0.06
        ax1.text(x_txt, y[i], f"PRI {r['pri_score_lcb']:.2f} (B_waa={r['waa_B']:.2f})",
                 va="center", fontsize=8.2, fontweight="bold" if r["shortlist_rank"] <= 10 else "normal")

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlim(min(neg_left.min() - 0.3, -0.7), pos_left.max() + 1.55)
    ax1.set_xlabel("Weighted Z-Score Contribution to PRI_z_lcb", fontsize=10)
    ax1.set_title("A. Bowling Pillar Contributions (Active Top 10 & Watchlist #11-15)", fontweight="bold", fontsize=11)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax1.grid(axis="x", alpha=0.25)

    # Panel B: Career Econ vs Bowling Strike Rate, shaped by Pace vs Spin, annotated with IPL Balls
    for ptype, marker, color in (("Pace", "o", "#1E88E5"), ("Spin", "D", "#8E24AA")):
        sub = plot_df[plot_df["pace_or_spin"] == ptype]
        if len(sub) > 0:
            sizes = np.clip(sub["ipl_legal_balls"].values * 0.25, 55, 420)
            ax2.scatter(sub["career_econ"], sub["career_bowl_sr"], s=sizes, marker=marker, color=color,
                        alpha=0.82, edgecolors="black", linewidth=0.8, label=f"{ptype} (Bubble = IPL Legal Balls)")

    bowl_offsets = {
        "Josh Hazlewood": (-110, 8),
        "Nathan Ellis": (-108, -10),
        "Varun Chakravarthy": (10, 4),
        "Kuldeep Yadav": (10, -8),
        "Sunil Narine": (-85, 8),
        "Jasprit Bumrah": (-45, -14),
    }
    for _, r in plot_df.iterrows():
        dx, dy = bowl_offsets.get(r["player_name"], (6, 4))
        ax2.annotate(f"{r['player_name']} ({int(r['ipl_legal_balls'])} IPL)",
                     (r["career_econ"], r["career_bowl_sr"]),
                     xytext=(dx, dy), textcoords="offset points", fontsize=7.8)

    ax2.set_xlabel("Career Primary Economy (Runs Conceded / 6 Legal Balls) [Lower Better]", fontsize=10)
    ax2.set_ylabel("Career Primary Strike Rate (Legal Balls / Bowler Wkt) [Lower Better]", fontsize=10)
    ax2.set_title("B. Run Containment vs Wicket-Taking Rate (Top 15 Active Bowlers)", fontweight="bold", fontsize=11)
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax2.grid(alpha=0.25)

    fig.suptitle("Phase 4 Bowling Shortlist & Watchlist Diagnostics (Active IPL Pool; #14 Ashwin Retired)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "24_phase4_top10_bowlers_decomposition.png")


def _plot_sensitivity_and_recency(
    bat_cards: pd.DataFrame,
    bowl_cards: pd.DataFrame,
    bat_sens: pd.DataFrame,
    bowl_sens: pd.DataFrame,
) -> None:
    """Figure 25: Multi-Variant Rank Stability & Recency / IPL Relevance Audit."""
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 11.5))

    # Top row: Rank range across variants for Batters and Bowlers
    for ax, sens_df, title in (
        (axes[0, 0], bat_sens, "A. Batter Rank Range Across 9 Methodology Variants"),
        (axes[0, 1], bowl_sens, "B. Bowler Rank Range Across 9 Methodology Variants"),
    ):
        s_plot = sens_df.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(s_plot))
        for i, r in s_plot.iterrows():
            ax.plot([r["min_rank_across_variants"], r["max_rank_across_variants"]], [y[i], y[i]],
                    color="#78909C", linewidth=2.5, alpha=0.75, zorder=1)
            ax.scatter(r["rank_base_lcb"], y[i], color="#1565C0", s=70, zorder=3, label="Base PRI LCB" if i == 0 else None)
            ax.scatter(r["rank_ev_all_lcb"], y[i], color="#E91E63", marker="x", s=55, zorder=3, label="EV_All LCB" if i == 0 else None)
            ax.scatter(r["rank_ipl_only"], y[i], color="#2E7D32", marker="^", s=55, zorder=3, label="IPL-Only (w_t20i=0)" if i == 0 else None)

        ax.axvline(10.5, color="#C62828", linestyle="--", linewidth=1.2, alpha=0.8, label="Top-10 Cutoff")
        ax.set_yticks(y)
        ax.set_yticklabels([f"#{int(r['rank_base_lcb'])} {r['player_name']}" for _, r in s_plot.iterrows()], fontsize=8.8)
        ax.invert_xaxis()
        ax.set_xlabel("Eligible Active Rank (Lower / Right = Higher Rank)", fontsize=9.5)
        ax.set_title(title, fontweight="bold", fontsize=10.8)
        ax.legend(loc="lower left", fontsize=7.8, framealpha=0.9)
        ax.grid(axis="x", alpha=0.25)

    # Bottom row: Recency shift (2025-2026 EV - 2022-2024 EV) vs IPL Share %
    rec_offsets = {
        "Rajat Patidar": (-55, -14),
        "Heinrich Klaasen": (-92, 6),
        "Sunil Narine": (-58, 6),
        "Wanindu Hasaranga": (6, -8),
        "Phil Salt": (-35, 8),
        "Nicholas Pooran": (6, 4),
    }
    for ax, cards, title, unit in (
        (axes[1, 0], bat_cards, "C. Batting Recency Shift (2025-26 vs 2022-24 Net EV) & IPL Share", "EV Delta (Runs/100b)"),
        (axes[1, 1], bowl_cards, "D. Bowling Recency Shift (2025-26 vs 2022-24 Net EV/Over) & IPL Share", "EV Delta (Runs/Over)"),
    ):
        valid = cards.copy()
        valid["ev_delta"] = valid["ev_recent_25_26_raw"] - valid["ev_earlier_22_24_raw"]
        has_both = valid[valid["ev_delta"].notna()]
        rec_only = valid[valid["ev_delta"].isna()]

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(50.0, color="#616161", linestyle=":", linewidth=1.0, alpha=0.7)

        if len(has_both) > 0:
            sc = ax.scatter(has_both["ipl_share_pct"], has_both["ev_delta"],
                            c=has_both["ipl_2026_balls"], cmap="viridis", s=110, edgecolors="black", linewidth=0.8)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("IPL 2026 Balls", fontsize=8.5)
            for _, r in has_both.iterrows():
                dx, dy = rec_offsets.get(r["player_name"], (6, 4))
                ax.annotate(r["player_name"], (r["ipl_share_pct"], r["ev_delta"]),
                            xytext=(dx, dy), textcoords="offset points", fontsize=7.8)

        if len(rec_only) > 0:
            for idx_r, (_, r) in enumerate(rec_only.iterrows()):
                ax.scatter(r["ipl_share_pct"], 0.0, marker="*", s=180, color="#E91E63", edgecolors="black", zorder=4)
                ax.annotate(f"{r['player_name']} (2025-26 only; EV={r['ev_recent_25_26_raw']:+.1f})",
                            (r["ipl_share_pct"], 0.0), xytext=(-148, 14 + 12 * idx_r), textcoords="offset points",
                            fontsize=7.8, fontweight="bold", color="#AD1457",
                            arrowprops=dict(arrowstyle="->", color="#AD1457", lw=0.8))

        ax.set_xlabel("IPL Share of Primary Weighted Balls (%)", fontsize=9.5)
        ax.set_ylabel(unit, fontsize=9.5)
        ax.set_title(title, fontweight="bold", fontsize=10.8)
        ax.set_xlim(-5, 110)
        ax.grid(alpha=0.25)

    fig.suptitle("Phase 4 Robustness Audit: Rank Sensitivity & Recency / Competition-Balance Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "25_phase4_sensitivity_and_recency.png")


def _plot_rr_squad_fit(rr_ctx: Dict, bat_short: pd.DataFrame, bowl_short: pd.DataFrame) -> None:
    """Figure 26: Verified RR IPL 2026 Phase Baselines vs Target Shortlist Coverage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 7.6), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Panel A: RR IPL 2026 vs League Phase Deltas
    bp = pd.DataFrame(rr_ctx["bat_phases_2026"])
    wp = pd.DataFrame(rr_ctx["bowl_phases_2026"])
    phases = ["Powerplay", "Middle", "Death"]
    x = np.arange(len(phases))
    w = 0.34

    bat_delta = bp["rr_sr"].values - bp["lg_sr"].values
    bowl_econ_diff = wp["lg_econ"].values - wp["rr_econ"].values  # positive = RR saved runs vs league

    b1 = ax1.bar(x - w / 2, bat_delta, width=w, color=["#2E7D32" if v >= 0 else "#C62828" for v in bat_delta],
                 alpha=0.86)
    ax1_twin = ax1.twinx()
    b2 = ax1_twin.bar(x + w / 2, bowl_econ_diff, width=w, color=["#1565C0" if v >= 0 else "#EF6C00" for v in bowl_econ_diff],
                      alpha=0.78)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2E7D32", alpha=0.86, label="RR Bat SR > League (Left Axis)"),
        Patch(facecolor="#C62828", alpha=0.86, label="RR Bat SR < League (Left Axis)"),
        Patch(facecolor="#1565C0", alpha=0.78, label="RR Bowl Tighter Than League (Right Axis)"),
        Patch(facecolor="#EF6C00", alpha=0.78, label="RR Bowl More Expensive (Right Axis)"),
    ]
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=8.0, framealpha=0.92)

    ax1.axhline(0, color="black", linewidth=0.9)
    for bar, v in zip(b1, bat_delta):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + (1.2 if v >= 0 else -2.8), f"{v:+.1f}", ha="center", fontsize=9, fontweight="bold")
    for bar, v in zip(b2, bowl_econ_diff):
        ax1_twin.text(bar.get_x() + bar.get_width() / 2, v + (0.04 if v >= 0 else -0.07), f"{v:+.2f}", ha="center", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Batting Strike Rate Delta vs League (SR pts)", fontsize=9.5)
    ax1_twin.set_ylabel("Bowling Economy Advantage vs League (RPO)", fontsize=9.5)
    ax1.set_ylim(-15, 38)
    ax1_twin.set_ylim(-0.6, 0.6)
    ax1.set_title("A. Verified RR IPL 2026 Phase Profile vs League (16 Matches)", fontweight="bold", fontsize=11)
    ax1.grid(axis="y", alpha=0.25)

    # Panel B: Heatmap of Death / Middle / PP Exposure (%) & HP_Post for Shortlisted Targets
    b_rows = []
    for _, r in bat_short.iterrows():
        b_rows.append({
            "label": f"BAT #{int(r['shortlist_rank'])} {r['player_name']} ({r['role_bucket'].split()[0]})",
            "PP %": r["powerplay_share_pct"],
            "Mid %": r["middle_share_pct"],
            "Death %": r["death_share_pct"],
            "IPL 2026 %": 100.0 * min(1.0, r["ipl_2026_balls"] / 250.0),
        })
    for _, r in bowl_short.iterrows():
        b_rows.append({
            "label": f"BOWL #{int(r['shortlist_rank'])} {r['player_name']} ({r['pace_or_spin']})",
            "PP %": r["powerplay_share_pct"],
            "Mid %": r["middle_share_pct"],
            "Death %": r["death_share_pct"],
            "IPL 2026 %": 100.0 * min(1.0, r["ipl_2026_balls"] / 250.0),
        })

    hm_df = pd.DataFrame(b_rows)
    mat = hm_df[["PP %", "Mid %", "Death %", "IPL 2026 %"]].values
    im = ax2.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=80)
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(["Powerplay %", "Middle %", "Death %", "IPL26 Active"], fontsize=9.5, fontweight="bold")
    ax2.set_yticks(np.arange(len(hm_df)))
    ax2.set_yticklabels(hm_df["label"], fontsize=8.3)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = f"{val:.0f}%" if j < 3 else (f"{int((bat_short if i < 10 else bowl_short).iloc[i if i < 10 else i - 10]['ipl_2026_balls'])}b")
            ax2.text(j, i, txt, ha="center", va="center", fontsize=7.8,
                     color="white" if val > 48 else "black", fontweight="bold" if j == 2 and val >= 25 else "normal")
    ax2.axhline(9.5, color="#C62828", linewidth=1.8)
    ax2.set_title("B. Phase Usage & IPL 2026 Activity Across All 20 Shortlisted Players", fontweight="bold", fontsize=11)

    fig.suptitle("Rajasthan Royals IPL 2026 Empirical Squad Context & Shortlist Role/Phase Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "26_phase4_rr_2026_fit_matrix.png")


def _plot_context_and_phase_heatmaps(
    bat_short: pd.DataFrame,
    bowl_short: pd.DataFrame,
    bat_ctx: pd.DataFrame,
    bowl_ctx: pd.DataFrame,
) -> None:
    """Figure 27: Phase-Wise Performance Heatmaps & Venue-Adjusted True SR / True Economy."""
    fig, axes = plt.subplots(2, 2, figsize=(17.2, 12.2))
    ax_bhm, ax_whm, ax_btrue, ax_wtrue = axes.flat

    # Panel A: Batting Phase SR Heatmap (PP / Middle / Death + Death HP EV)
    bs = bat_short.copy()
    mat_b = bs[["powerplay_sr", "middle_sr", "death_sr"]].fillna(0.0).values
    im1 = ax_bhm.imshow(mat_b, aspect="auto", cmap="YlOrRd", vmin=100, vmax=230)
    ax_bhm.set_xticks([0, 1, 2])
    ax_bhm.set_xticklabels(["Powerplay SR", "Middle SR", "Death SR"], fontweight="bold", fontsize=9.5)
    ax_bhm.set_yticks(np.arange(len(bs)))
    ax_bhm.set_yticklabels([f"#{int(r['shortlist_rank'])} {r['player_name']} ({r['batting_hand']})" for _, r in bs.iterrows()], fontsize=8.8)
    for i in range(mat_b.shape[0]):
        balls_cols = ["powerplay_balls", "middle_balls", "death_balls"]
        for j in range(3):
            nb = int(bs.iloc[i][balls_cols[j]])
            v = bs.iloc[i][["powerplay_sr", "middle_sr", "death_sr"][j]]
            txt = f"{v:.1f}\n({nb}b)" if pd.notna(v) and nb > 0 else "—"
            ax_bhm.text(j, i, txt, ha="center", va="center", fontsize=7.6,
                        color="white" if (pd.notna(v) and v > 180) else "black", fontweight="bold" if j == 2 and pd.notna(v) and v >= 195 else "normal")
    ax_bhm.set_title("A. Active Top 10 Batters: Phase Strike Rates & Ball Volume", fontweight="bold", fontsize=11)
    fig.colorbar(im1, ax=ax_bhm, fraction=0.04, pad=0.03, label="Strike Rate")

    # Panel B: Bowling Phase Economy Heatmap
    ws = bowl_short.copy()
    mat_w = ws[["powerplay_econ", "middle_econ", "death_econ"]].fillna(9.0).values
    im2 = ax_whm.imshow(mat_w, aspect="auto", cmap="RdYlGn_r", vmin=6.0, vmax=11.0)
    ax_whm.set_xticks([0, 1, 2])
    ax_whm.set_xticklabels(["Powerplay Econ", "Middle Econ", "Death Econ"], fontweight="bold", fontsize=9.5)
    ax_whm.set_yticks(np.arange(len(ws)))
    ax_whm.set_yticklabels([f"#{int(r['shortlist_rank'])} {r['player_name']} ({r['pace_or_spin']})" for _, r in ws.iterrows()], fontsize=8.8)
    for i in range(mat_w.shape[0]):
        balls_cols = ["powerplay_balls", "middle_balls", "death_balls"]
        for j in range(3):
            nb = int(ws.iloc[i][balls_cols[j]])
            v = ws.iloc[i][["powerplay_econ", "middle_econ", "death_econ"][j]]
            txt = f"{v:.2f}\n({nb}b)" if pd.notna(v) and nb > 0 else "—"
            ax_whm.text(j, i, txt, ha="center", va="center", fontsize=7.6,
                        color="white" if (pd.notna(v) and (v < 7.1 or v > 9.9)) else "black",
                        fontweight="bold" if pd.notna(v) and v < 7.6 else "normal")
    ax_whm.set_title("B. Active Top 10 Bowlers: Phase Economy & Legal Balls", fontweight="bold", fontsize=11)
    fig.colorbar(im2, ax=ax_whm, fraction=0.04, pad=0.03, label="Economy (RPO)")

    # Panel C: Exploratory Batting Venue-Adjusted True SR vs Raw SR
    bc = bs[["shortlist_rank", "canonical_player_id", "player_name"]].merge(bat_ctx, on="canonical_player_id", how="left").iloc[::-1].reset_index(drop=True)
    y = np.arange(len(bc))
    ax_btrue.barh(y, bc["true_sr"], color="#1565C0", alpha=0.88, label="Venue-Season-Phase True SR (Explore Only)")
    ax_btrue.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax_btrue.set_yticks(y)
    ax_btrue.set_yticklabels([f"#{int(r['shortlist_rank'])} {r['player_name']}" for _, r in bc.iterrows()], fontsize=8.8)
    for idx, r in bc.iterrows():
        off = 1.2 if r["true_sr"] >= 0 else -1.2
        ha = "left" if r["true_sr"] >= 0 else "right"
        ax_btrue.text(r["true_sr"] + off, idx, f"{r['true_sr']:+.1f} (vs {r['ctx_exp_sr']:.0f} exp)", va="center", ha=ha, fontsize=7.8)
    ax_btrue.set_xlim(min(-10, bc["true_sr"].min() - 12), bc["true_sr"].max() + 30)
    ax_btrue.set_xlabel("True Strike Rate = Weighted Player SR − LOO Venue/Season/Innings/Phase Expected SR")
    ax_btrue.set_title("C. Exploratory Venue-Adjusted True SR (Batters Top 10)", fontweight="bold", fontsize=11)
    ax_btrue.grid(axis="x", alpha=0.25)

    # Panel D: Exploratory Bowling Venue-Adjusted True Economy
    wc = ws[["shortlist_rank", "canonical_player_id", "player_name"]].merge(bowl_ctx, on="canonical_player_id", how="left").iloc[::-1].reset_index(drop=True)
    y2 = np.arange(len(wc))
    ax_wtrue.barh(y2, wc["true_econ"], color="#2E7D32", alpha=0.88, label="True Economy (Exp Econ − Player Econ)")
    ax_wtrue.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax_wtrue.set_yticks(y2)
    ax_wtrue.set_yticklabels([f"#{int(r['shortlist_rank'])} {r['player_name']}" for _, r in wc.iterrows()], fontsize=8.8)
    for idx, r in wc.iterrows():
        ax_wtrue.text(r["true_econ"] + 0.04, idx, f"{r['true_econ']:+.2f}/ov (vs {r['ctx_exp_econ']:.2f} exp)", va="center", ha="left", fontsize=7.8)
    ax_wtrue.set_xlim(0.0, wc["true_econ"].max() + 0.95)
    ax_wtrue.set_xlabel("True Economy = LOO Venue/Season/Innings/Phase Expected RPO − Player RPO (+ = Saved/Over)")
    ax_wtrue.set_title("D. Exploratory Venue-Adjusted True Economy (Bowlers Top 10)", fontweight="bold", fontsize=11)
    ax_wtrue.grid(axis="x", alpha=0.25)

    fig.suptitle("Phase 4 Exploratory Phase-Wise Heatmaps & Venue-Adjusted Context Metrics (Separate from PRI)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "27_phase4_context_and_phase_heatmaps.png")


def _plot_hp_neff_and_yoy_trends(
    bat_short: pd.DataFrame,
    bowl_short: pd.DataFrame,
    bat_rej: pd.DataFrame,
    bowl_rej: pd.DataFrame,
    bat_yoy: pd.DataFrame,
    bowl_yoy: pd.DataFrame,
    rr_ctx: Dict,
) -> None:
    """Figure 28: General vs High-Pressure Bubble Plot (with Rejections), YoY Trends, and Jaipur/Guwahati Profile."""
    fig, axes = plt.subplots(2, 2, figsize=(17.2, 12.2))
    ax_bat, ax_bowl, ax_yoy, ax_ven = axes.flat

    # Panel A: Bat General RAA vs HP RAA (Top 10 + Key Rejections)
    b_top = bat_short.copy()
    ax_bat.axhline(0.0, color="#666", linestyle="--", linewidth=1)
    ax_bat.axvline(0.0, color="#666", linestyle="--", linewidth=1)
    ax_bat.scatter(b_top["raa_post"], b_top["hp_raa_post"], s=np.clip(b_top["neff"], 80, 1400) * 0.35,
                   c="#1565C0", alpha=0.85, edgecolors="white", linewidth=1.2, label="Active Top 10 Batters (size ~ Neff)")
    for _, r in b_top.iterrows():
        dx, dy = 0.8, 0.8
        if "Vaibhav" in r["player_name"]:
            dx, dy = -8.5, 1.8
        elif "Finn" in r["player_name"]:
            dx, dy = 0.8, -2.5
        elif "Pooran" in r["player_name"]:
            dx, dy = -6.0, 1.5
        elif "Klaasen" in r["player_name"]:
            dx, dy = 1.0, -2.2
        tag = f"#{int(r['shortlist_rank'])} {r['player_name'].split()[-1]}" + (" (0 HP)" if r["neff_hp"] < 1 else "")
        ax_bat.text(r["raa_post"] + dx, r["hp_raa_post"] + dy, tag, fontsize=7.8, fontweight="bold", color="#0D47A1")

    br_sub = bat_rej[bat_rej["canonical_player_id"].isin([47, 4117, 953, 126, 6819])].copy()
    ax_bat.scatter(br_sub["raa_raw"].clip(-12, 30), br_sub["hp_raa_post"], s=110, marker="X", c="#C62828", alpha=0.9, label="Rejected / Outside Top 10")
    for _, r in br_sub.iterrows():
        x_val = float(np.clip(r["raa_raw"], -12, 30))
        dx, dy = 0.7, -1.3
        if "Rahul" in r["player_name"]:
            dx, dy = -4.2, 1.3
        elif "Rohit" in r["player_name"]:
            dx, dy = 0.8, -1.7
        ax_bat.text(x_val + dx, r["hp_raa_post"] + dy, r["player_name"], fontsize=7.6, color="#B71C1C")
    ax_bat.set_xlim(-6.5, 45.5)
    ax_bat.set_xlabel("General Pressure RAA (PI < 60, Runs / 100 balls; Post Mean for Top 10, Raw for Ref)")
    ax_bat.set_ylabel("High-Pressure RAA Post Mean (PI >= 60, Runs / 100 balls)")
    ax_bat.set_title("A. Batters: General vs High-Pressure Performance & Rejections", fontweight="bold", fontsize=11)
    ax_bat.legend(loc="upper left", fontsize=8.3)
    ax_bat.grid(alpha=0.25)

    # Panel B: Bowl General RSAA vs HP Impact (Top 10 + Key Rejections)
    w_top = bowl_short.copy()
    ax_bowl.axhline(0.0, color="#666", linestyle="--", linewidth=1)
    ax_bowl.axvline(0.0, color="#666", linestyle="--", linewidth=1)
    ax_bowl.scatter(w_top["rsaa_post"], w_top["hp_impact_post"], s=np.clip(w_top["neff"], 80, 1800) * 0.25,
                    c="#2E7D32", alpha=0.85, edgecolors="white", linewidth=1.2, label="Active Top 10 Bowlers (size ~ Neff)")
    for _, r in w_top.iterrows():
        dx, dy = 0.03, 0.03
        if "Bumrah" in r["player_name"]:
            dx, dy = -0.35, -0.09
        elif "Narine" in r["player_name"]:
            dx, dy = -0.38, 0.04
        elif "Varun" in r["player_name"]:
            dx, dy = -0.40, 0.06
        elif "Rashid Khan" in r["player_name"]:
            dx, dy = -0.25, -0.09
        elif "Hasaranga" in r["player_name"]:
            dx, dy = -0.18, 0.055
        elif "Axar" in r["player_name"]:
            dx, dy = 0.04, -0.07
        elif "Hazlewood" in r["player_name"]:
            dx, dy = 0.03, 0.03
        ax_bowl.text(r["rsaa_post"] + dx, r["hp_impact_post"] + dy, f"#{int(r['shortlist_rank'])} {r['player_name']}", fontsize=7.6, fontweight="bold", color="#1B5E20")

    wr_sub = bowl_rej[bowl_rej["canonical_player_id"].isin([5944, 3628, 1612, 7503, 6806])].copy()
    ax_bowl.scatter(wr_sub["rsaa_raw"], wr_sub["hp_impact_post"], s=110, marker="X", c="#C62828", alpha=0.9, label="Rejected / Outside Top 10")
    for _, r in wr_sub.iterrows():
        ax_bowl.text(r["rsaa_raw"] + 0.03, r["hp_impact_post"] - 0.06, r["player_name"], fontsize=7.6, color="#B71C1C")
    ax_bowl.set_ylim(-0.96, 0.72)
    ax_bowl.set_xlabel("General Pressure RSAA (PI < 60, Runs Saved/Over; Post Mean Top 10)")
    ax_bowl.set_ylabel("High-Pressure Impact Post Mean (PI >= 60, Net Runs/Over)")
    ax_bowl.set_title("B. Bowlers: General vs High-Pressure Impact & Rejections", fontweight="bold", fontsize=11)
    ax_bowl.legend(loc="lower left", fontsize=8.3)
    ax_bowl.grid(alpha=0.25)

    # Panel C: YoY Development Trends (Selected Key Batters 2022-2026, >=30 balls/yr)
    focus_bat_ids = [1699, 3647, 6396, 4260, 3125]
    colors_yoy = ["#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A", "#C62828"]
    for pid, col in zip(focus_bat_ids, colors_yoy):
        sub_y = bat_yoy[bat_yoy["canonical_player_id"] == pid].sort_values("season_yr")
        if len(sub_y) >= 2:
            ax_yoy.plot(sub_y["season_yr"], sub_y["ev_net_raw"], marker="o", linewidth=2.1, color=col, label=sub_y["player_name"].iloc[0])
    ax_yoy.axhline(0.0, color="#444", linestyle="--", linewidth=1)
    ax_yoy.set_xticks([2022, 2023, 2024, 2025, 2026])
    ax_yoy.set_xlabel("Season")
    ax_yoy.set_ylabel("Pressure-Weighted Net EV (/100 balls)")
    ax_yoy.set_title("C. Year-over-Year Net EV Progression (Batters with >=30b/Season)", fontweight="bold", fontsize=11)
    ax_yoy.legend(loc="best", fontsize=8.3)
    ax_yoy.grid(alpha=0.25)

    # Panel D: Jaipur (vid=6) vs Guwahati (vid=1241) vs All IPL (2023-2026) Scoring Conditions
    vc_df = pd.DataFrame(rr_ctx["venue_conditions_2023_2026"])
    metrics_v = ["pp_rpo", "mid_rpo", "dth_rpo", "pace_econ", "spin_econ"]
    labels_v = ["PP RPO", "Middle RPO", "Death RPO", "Pace Econ", "Spin Econ"]
    x = np.arange(len(metrics_v))
    w = 0.25
    bar_cols = ["#78909C", "#E91E63", "#1565C0"]
    for i, (_, r) in enumerate(vc_df.iterrows()):
        vals = [r[m] for m in metrics_v]
        ax_ven.bar(x + (i - 1) * w, vals, w, color=bar_cols[i], alpha=0.88,
                   label=f"{r['venue_label']} (M={int(r['matches'])}, Inn1={r['inn1_avg_runs']:.0f})")
        for xi, v in zip(x + (i - 1) * w, vals):
            ax_ven.text(xi, v + 0.14, f"{v:.1f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold")
    ax_ven.set_xticks(x)
    ax_ven.set_xticklabels(labels_v, fontweight="bold", fontsize=9.2)
    ax_ven.set_ylim(0, 12.8)
    ax_ven.set_ylabel("Runs Per Over (IPL 2023–2026)")
    ax_ven.set_title("D. Verified Home Conditions: Jaipur (SMS) vs Guwahati (ACA) vs All IPL", fontweight="bold", fontsize=11)
    ax_ven.legend(loc="upper left", fontsize=8.0)
    ax_ven.grid(axis="y", alpha=0.25)

    fig.suptitle("Phase 4 HP-vs-General Validation, Year-over-Year Trajectories & RR Venue Profiles", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "28_phase4_hp_vs_neff_and_yoy_trends.png")


def main():
    ds = load_raw_datasets()
    raw_df = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz", low_memory=False)
    pdir = pd.read_csv(DATA_PROCESSED / "canonical_player_directory.csv")

    # Preserve and read approved Phase 3 outputs directly
    bat_df = pd.read_csv(DATA_PROCESSED / "phase3_batting_prelim.csv")
    bowl_df = pd.read_csv(DATA_PROCESSED / "phase3_bowling_prelim.csv")
    emg_bat = pd.read_csv(DATA_PROCESSED / "phase3_emerging_batting_supp.csv")
    emg_bowl = pd.read_csv(DATA_PROCESSED / "phase3_emerging_bowling_supp.csv")

    emg_bat = _attach_emerging_comp_names(emg_bat, raw_df, ds["matches"], ds["competitions"], "canonical_striker_id")
    emg_bowl = _attach_emerging_comp_names(emg_bowl, raw_df, ds["matches"], ds["competitions"], "canonical_bowler_id")

    cfg = PerformanceConfig(w_t20i=0.50, w_emerging=0.00, exclude_emerging_from_primary=True)

    # Compute OOF baseline predictions on exact primary pool (read-only relative to Phase 3 CSVs)
    oof_pri, _ = fit_predict_oof_baselines(
        raw_df, ds["matches"], ds["competitions"], include_suspected_dls=False, n_splits=5, exclude_emerging=True
    )
    raw_pri_enriched = _enrich_raw_primary(raw_df, ds["matches"], ds["competitions"])

    # 1. Unrestricted & Active Benchmarks / Shortlists / Watchlists
    sel = select_shortlists_and_benchmarks(bat_df, bowl_df, top_n=10, watchlist_n=5)

    # 2. RR Verified IPL 2026 Context + Jaipur/Guwahati Conditions
    rr_ctx = compute_rr_2026_context(
        raw_pri_enriched, ds["matches"], ds["competitions"], ds["teams"], ds["playing_eleven"], pdir
    )
    rr_played_26 = rr_ctx["rr_2026_played_ids"]

    # 3. Detailed Player Evaluation Cards
    bat_bench_cards = build_detailed_player_profiles(
        sel["bat_benchmark_top15"], bat_df, bowl_df, oof_pri, raw_pri_enriched,
        ds["playing_eleven"], ds["teams"], pdir, emg_bat, rr_played_26, "bat", cfg
    )
    bat_short_cards = build_detailed_player_profiles(
        sel["bat_shortlist"], bat_df, bowl_df, oof_pri, raw_pri_enriched,
        ds["playing_eleven"], ds["teams"], pdir, emg_bat, rr_played_26, "bat", cfg
    )
    bat_watch_cards = build_detailed_player_profiles(
        sel["bat_watchlist"], bat_df, bowl_df, oof_pri, raw_pri_enriched,
        ds["playing_eleven"], ds["teams"], pdir, emg_bat, rr_played_26, "bat", cfg
    )

    bowl_bench_cards = build_detailed_player_profiles(
        sel["bowl_benchmark_top15"], bowl_df, bat_df, oof_pri, raw_pri_enriched,
        ds["playing_eleven"], ds["teams"], pdir, emg_bowl, rr_played_26, "bowl", cfg
    )
    bowl_short_cards = build_detailed_player_profiles(
        sel["bowl_shortlist"], bowl_df, bat_df, oof_pri, raw_pri_enriched,
        ds["playing_eleven"], ds["teams"], pdir, emg_bowl, rr_played_26, "bowl", cfg
    )
    bowl_watch_cards = build_detailed_player_profiles(
        sel["bowl_watchlist"], bowl_df, bat_df, oof_pri, raw_pri_enriched,
        ds["playing_eleven"], ds["teams"], pdir, emg_bowl, rr_played_26, "bowl", cfg
    )

    # 4. Exploratory Context-Adjusted Metrics, Opposition Quality Splits, YoY & Rejected Audit
    rej = extract_rejected_player_audit(bat_df, bowl_df, sel["bat_elig_all"], sel["bowl_elig_all"])
    bat_target_ids = sorted(set(
        list(bat_bench_cards["canonical_player_id"].astype(int))
        + list(bat_short_cards["canonical_player_id"].astype(int))
        + list(bat_watch_cards["canonical_player_id"].astype(int))
        + list(rej["bat_rejected"]["canonical_player_id"].astype(int))
    ))
    bowl_target_ids = sorted(set(
        list(bowl_bench_cards["canonical_player_id"].astype(int))
        + list(bowl_short_cards["canonical_player_id"].astype(int))
        + list(bowl_watch_cards["canonical_player_id"].astype(int))
        + list(rej["bowl_rejected"]["canonical_player_id"].astype(int))
    ))

    ctx_metrics = compute_contextual_adjusted_metrics(
        oof_pri, ds["matches"], ds["teams"], bat_df, bowl_df, bat_target_ids, bowl_target_ids
    )
    yoy = compute_yoy_development_trajectories(
        oof_pri,
        list(bat_short_cards["canonical_player_id"].astype(int)) + list(bat_watch_cards["canonical_player_id"].astype(int)),
        list(bowl_short_cards["canonical_player_id"].astype(int)) + list(bowl_watch_cards["canonical_player_id"].astype(int)),
    )

    # 5. Sensitivity Across 9 Variants for Active Top-15 Batters & Bowlers
    bat_sens_ids = list(bat_short_cards["canonical_player_id"].astype(int)) + list(bat_watch_cards["canonical_player_id"].astype(int))
    bowl_sens_ids = list(bowl_short_cards["canonical_player_id"].astype(int)) + list(bowl_watch_cards["canonical_player_id"].astype(int))
    sens = compute_shortlist_sensitivity(
        oof_pri, raw_df, ds["matches"], ds["competitions"], pdir, bat_df, bowl_df, bat_sens_ids, bowl_sens_ids
    )

    # 6. Enriched Emerging-only Supplementary Summary
    emg_raw = raw_df.copy()
    emg_raw["comp_tier"] = classify_comp_tier(emg_raw, matches_df=ds["matches"], competitions_df=ds["competitions"])
    emg_delivs = emg_raw[emg_raw["comp_tier"] == COMP_TIER_EMERGING].merge(
        ds["matches"][["match_id", "comp_id"]], on="match_id", how="left"
    )
    emg_supp = summarize_emerging_only_supplementary(
        emg_bat, emg_bowl, bat_df, bowl_df, emg_deliveries=emg_delivs, teams_df=ds["teams"], comps_df=ds["competitions"]
    )
    portfolio = build_portfolio_balance_summary(bat_short_cards, bowl_short_cards)

    # Save Phase 4 outputs (never overwriting phase3_*.csv)
    bat_bench_cards.to_csv(DATA_PROCESSED / "phase4_bat_benchmark15_cards.csv", index=False)
    bat_short_cards.to_csv(DATA_PROCESSED / "phase4_bat_shortlist_cards.csv", index=False)
    bat_watch_cards.to_csv(DATA_PROCESSED / "phase4_bat_watchlist_cards.csv", index=False)
    bowl_bench_cards.to_csv(DATA_PROCESSED / "phase4_bowl_benchmark15_cards.csv", index=False)
    bowl_short_cards.to_csv(DATA_PROCESSED / "phase4_bowl_shortlist_cards.csv", index=False)
    bowl_watch_cards.to_csv(DATA_PROCESSED / "phase4_bowl_watchlist_cards.csv", index=False)
    sens["bat_sensitivity"].to_csv(DATA_PROCESSED / "phase4_bat_sensitivity.csv", index=False)
    sens["bowl_sensitivity"].to_csv(DATA_PROCESSED / "phase4_bowl_sensitivity.csv", index=False)
    ctx_metrics["bat_context"].to_csv(DATA_PROCESSED / "phase4_bat_context_metrics.csv", index=False)
    ctx_metrics["bowl_context"].to_csv(DATA_PROCESSED / "phase4_bowl_context_metrics.csv", index=False)
    yoy["bat_yoy"].to_csv(DATA_PROCESSED / "phase4_bat_yoy.csv", index=False)
    yoy["bowl_yoy"].to_csv(DATA_PROCESSED / "phase4_bowl_yoy.csv", index=False)
    rej["bat_rejected"].to_csv(DATA_PROCESSED / "phase4_bat_rejected_audit.csv", index=False)
    rej["bowl_rejected"].to_csv(DATA_PROCESSED / "phase4_bowl_rejected_audit.csv", index=False)
    emg_supp["emerging_only_bat_top"].to_csv(DATA_PROCESSED / "phase4_emerging_only_bat_supp.csv", index=False)
    emg_supp["emerging_only_bowl_top"].to_csv(DATA_PROCESSED / "phase4_emerging_only_bowl_supp.csv", index=False)
    rr_ctx["rr_2026_roster"].to_csv(DATA_PROCESSED / "phase4_rr_2026_roster.csv", index=False)

    bat_el = bat_df[bat_df["elig_primary"]]
    bowl_el = bowl_df[bowl_df["elig_primary"]]
    summary_payload: Dict = {
        "eligible_counts": {
            "bat_elig_broad": int(bat_df["elig_broad"].sum()),
            "bat_elig_primary": int(len(bat_el)),
            "bowl_elig_broad": int(bowl_df["elig_broad"].sum()),
            "bowl_elig_primary": int(len(bowl_el)),
        },
        "pri_distribution_notes": {
            "bat_z_lcb_mean_elig": float(bat_el["pri_z_lcb"].mean()),
            "bat_z_lcb_sd_elig": float(bat_el["pri_z_lcb"].std(ddof=1)),
            "bat_pri_score_lcb_q50_q75_q90_max": [float(bat_el["pri_score_lcb"].quantile(q)) for q in (0.50, 0.75, 0.90, 1.00)],
            "bowl_z_lcb_mean_elig": float(bowl_el["pri_z_lcb"].mean()),
            "bowl_z_lcb_sd_elig": float(bowl_el["pri_z_lcb"].std(ddof=1)),
            "bowl_pri_score_lcb_q50_q75_q90_max": [float(bowl_el["pri_score_lcb"].quantile(q)) for q in (0.50, 0.75, 0.90, 1.00)],
        },
        "rr_2026_context": {
            "rr_team_id": rr_ctx["rr_team_id"],
            "rr_2026_matches": rr_ctx["rr_2026_matches"],
            "rr_2026_record": rr_ctx["rr_2026_record"],
            "bat_phases_2026": rr_ctx["bat_phases_2026"],
            "bowl_phases_2026": rr_ctx["bowl_phases_2026"],
            "bowl_pace_spin_2026": rr_ctx["bowl_pace_spin_2026"],
            "venue_conditions_2023_2026": rr_ctx["venue_conditions_2023_2026"],
        },
        "portfolio_balance_summary": portfolio,
        "retired_excluded_from_active_shortlist": RETIRED_FROM_IPL,
    }
    with open(DATA_PROCESSED / "phase4_rr_2026_context.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # 7. Figures 23-28
    _plot_batters_decomposition(bat_bench_cards, bat_short_cards, bat_watch_cards)
    _plot_bowlers_decomposition(bowl_short_cards, bowl_watch_cards)
    _plot_sensitivity_and_recency(bat_short_cards, bowl_short_cards, sens["bat_sensitivity"], sens["bowl_sensitivity"])
    _plot_rr_squad_fit(rr_ctx, bat_short_cards, bowl_short_cards)
    _plot_context_and_phase_heatmaps(bat_short_cards, bowl_short_cards, ctx_metrics["bat_context"], ctx_metrics["bowl_context"])
    _plot_hp_neff_and_yoy_trends(
        bat_short_cards, bowl_short_cards, rej["bat_rejected"], rej["bowl_rejected"],
        yoy["bat_yoy"], yoy["bowl_yoy"], rr_ctx
    )

    print("=== PHASE 4 ANALYSIS COMPLETE ===")
    print("Bat Unrestricted Top 10:", sel["bat_unfiltered_top10"]["player_name"].tolist())
    print("Bat Active Top 10:", bat_short_cards["player_name"].tolist())
    print("Bowl Top 10:", bowl_short_cards["player_name"].tolist())
    print("Files written to data/processed/phase4_* and reports/figures/23..28_*.png")


if __name__ == "__main__":
    main()


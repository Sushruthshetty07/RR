"""
Phase 3 Competition-Tier Correction Re-Run & Pre/Post Comparison Script.

Implements the approved targeted comp_tier correction before Phase 4:
  - IPL: weight 1.00
  - Senior_T20I: weight 0.50
  - Emerging_Rising_T20: supplementary only; excluded from primary OOF baselines,
    EB shrinkage priors, eligibility, and PRI rankings.

Preserves pre-correction snapshots in data/processed/archive_pre_comp_tier/ and
existing reports/figures 13..21, saves updated primary Phase 3 tables and
separate Emerging supplementary tables, and emits strictly anonymous comparison
diagnostics and Figure 22.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DATA_PROCESSED, load_raw_datasets
from src.performance import (
    COMP_TIER_EMERGING,
    COMP_TIER_IPL,
    COMP_TIER_SENIOR_T20I,
    PerformanceConfig,
    evaluate_batting_pressure,
    evaluate_bowling_pressure,
    evaluate_emerging_supplementary,
    fit_predict_oof_baselines,
)

ARCHIVE_DIR = DATA_PROCESSED / "archive_pre_comp_tier"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _rank_series(df: pd.DataFrame, score_col: str = "pri_score_lcb") -> pd.Series:
    return df[score_col].rank(ascending=False, method="min")


def _compare_tables(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    ball_col: str,
    t20i_col: str,
    ipl_col: str,
) -> Dict:
    """Computes strictly anonymous comparison metrics between pre-correction and corrected Phase 3 tables."""
    # Pre-merge outer on canonical_player_id to capture players who dropped completely (Emerging-only)
    m = old_df.merge(
        new_df,
        on="canonical_player_id",
        how="outer",
        suffixes=("_old", "_new"),
    )

    dropped_mask = m["pri_score_lcb_new"].isna()
    added_mask = m["pri_score_lcb_old"].isna()

    def _elig_grid(df: pd.DataFrame, suffix: str = "") -> Dict[str, int]:
        n_col = f"neff{suffix}"
        i_col = f"{ipl_col}{suffix}"
        valid = df[n_col].notna()
        sub = df.loc[valid]
        return {
            "neff15_ipl0": int((sub[n_col] >= 15).sum()),
            "neff30_ipl0": int((sub[n_col] >= 30).sum()),
            "neff30_ipl30": int(((sub[n_col] >= 30) & (sub[i_col] >= 30)).sum()),
            "neff50_ipl60": int(((sub[n_col] >= 50) & (sub[i_col] >= 60)).sum()),
            "neff75_ipl100": int(((sub[n_col] >= 75) & (sub[i_col] >= 100)).sum()),
        }

    elig_old_grid = _elig_grid(m, "_old")
    elig_new_grid = _elig_grid(m, "_new")

    old_elig_ids = set(old_df.loc[old_df["elig_primary"], "canonical_player_id"].astype(int))
    new_elig_ids = set(new_df.loc[new_df["elig_primary"], "canonical_player_id"].astype(int))
    old_broad_ids = set(old_df.loc[old_df["elig_broad"], "canonical_player_id"].astype(int))
    new_broad_ids = set(new_df.loc[new_df["elig_broad"], "canonical_player_id"].astype(int))

    # Eligible comparison (on union or intersection)
    common_elig = m[m["elig_primary_old"].fillna(False) & m["elig_primary_new"].fillna(False)].copy()
    common_elig["rank_lcb_old"] = _rank_series(
        old_df.loc[old_df["elig_primary"]].set_index("canonical_player_id"), "pri_score_lcb"
    ).reindex(common_elig["canonical_player_id"]).values
    common_elig["rank_lcb_new"] = _rank_series(
        new_df.loc[new_df["elig_primary"]].set_index("canonical_player_id"), "pri_score_lcb"
    ).reindex(common_elig["canonical_player_id"]).values
    common_elig["rank_post_old"] = _rank_series(
        old_df.loc[old_df["elig_primary"]].set_index("canonical_player_id"), "pri_score_post"
    ).reindex(common_elig["canonical_player_id"]).values
    common_elig["rank_post_new"] = _rank_series(
        new_df.loc[new_df["elig_primary"]].set_index("canonical_player_id"), "pri_score_post"
    ).reindex(common_elig["canonical_player_id"]).values

    common_elig["d_pri_lcb"] = common_elig["pri_score_lcb_new"] - common_elig["pri_score_lcb_old"]
    common_elig["d_pri_post"] = common_elig["pri_score_post_new"] - common_elig["pri_score_post_old"]
    common_elig["d_rank_lcb"] = common_elig["rank_lcb_new"] - common_elig["rank_lcb_old"]
    common_elig["d_t20i"] = common_elig[f"{t20i_col}_new"] - common_elig[f"{t20i_col}_old"]
    common_elig["d_neff"] = common_elig["neff_new"] - common_elig["neff_old"]

    top10_lcb_old = set(old_df.loc[old_df["elig_primary"]].nlargest(10, "pri_score_lcb")["canonical_player_id"].astype(int))
    top10_lcb_new = set(new_df.loc[new_df["elig_primary"]].nlargest(10, "pri_score_lcb")["canonical_player_id"].astype(int))
    top15_lcb_old = set(old_df.loc[old_df["elig_primary"]].nlargest(15, "pri_score_lcb")["canonical_player_id"].astype(int))
    top15_lcb_new = set(new_df.loc[new_df["elig_primary"]].nlargest(15, "pri_score_lcb")["canonical_player_id"].astype(int))
    top20_lcb_old = set(old_df.loc[old_df["elig_primary"]].nlargest(20, "pri_score_lcb")["canonical_player_id"].astype(int))
    top20_lcb_new = set(new_df.loc[new_df["elig_primary"]].nlargest(20, "pri_score_lcb")["canonical_player_id"].astype(int))
    top10_post_old = set(old_df.loc[old_df["elig_primary"]].nlargest(10, "pri_score_post")["canonical_player_id"].astype(int))
    top10_post_new = set(new_df.loc[new_df["elig_primary"]].nlargest(10, "pri_score_post")["canonical_player_id"].astype(int))

    rho_elig, _ = spearmanr(common_elig["pri_score_lcb_old"], common_elig["pri_score_lcb_new"])
    tau_elig, _ = kendalltau(common_elig["pri_score_lcb_old"], common_elig["pri_score_lcb_new"])
    rho_post_elig, _ = spearmanr(common_elig["pri_score_post_old"], common_elig["pri_score_post_new"])

    # Exposure shift across ALL old players vs primary-eligible players
    all_common = m[m["pri_score_lcb_old"].notna() & m["pri_score_lcb_new"].notna()].copy()
    all_common["d_t20i"] = all_common[f"{t20i_col}_new"] - all_common[f"{t20i_col}_old"]
    all_common["d_neff"] = all_common["neff_new"] - all_common["neff_old"]
    all_common["d_pri_lcb"] = all_common["pri_score_lcb_new"] - all_common["pri_score_lcb_old"]

    aff_all = all_common[all_common["d_t20i"] < 0]
    aff_elig = common_elig[common_elig["d_t20i"] < 0]
    unaff_elig = common_elig[common_elig["d_t20i"] == 0]

    # Broad eligible comparison
    common_broad = m[m["elig_broad_old"].fillna(False) & m["elig_broad_new"].fillna(False)].copy()
    rho_broad, _ = spearmanr(common_broad["pri_score_lcb_old"], common_broad["pri_score_lcb_new"])
    top10_broad_old = set(old_df.loc[old_df["elig_broad"]].nlargest(10, "pri_score_lcb")["canonical_player_id"].astype(int))
    top10_broad_new = set(new_df.loc[new_df["elig_broad"]].nlargest(10, "pri_score_lcb")["canonical_player_id"].astype(int))

    return {
        "n_players_old": int(len(old_df)),
        "n_players_new": int(len(new_df)),
        "n_emerging_only_dropped": int(dropped_mask.sum()),
        "n_added": int(added_mask.sum()),
        "elig_grid_old": elig_old_grid,
        "elig_grid_new": elig_new_grid,
        "primary_elig_gained_ids": sorted(int(x) for x in (new_elig_ids - old_elig_ids)),
        "primary_elig_lost_ids": sorted(int(x) for x in (old_elig_ids - new_elig_ids)),
        "broad_elig_gained_ids": sorted(int(x) for x in (new_broad_ids - old_broad_ids)),
        "broad_elig_lost_ids": sorted(int(x) for x in (old_broad_ids - new_broad_ids)),
        "n_common_elig": int(len(common_elig)),
        "top10_lcb_overlap": int(len(top10_lcb_old & top10_lcb_new)),
        "top15_lcb_overlap": int(len(top15_lcb_old & top15_lcb_new)),
        "top20_lcb_overlap": int(len(top20_lcb_old & top20_lcb_new)),
        "top10_post_overlap": int(len(top10_post_old & top10_post_new)),
        "top10_broad_overlap": int(len(top10_broad_old & top10_broad_new)),
        "spearman_elig_lcb": float(rho_elig),
        "kendall_elig_lcb": float(tau_elig),
        "spearman_elig_post": float(rho_post_elig),
        "spearman_broad_lcb": float(rho_broad),
        "elig_abs_dpri_mean": float(common_elig["d_pri_lcb"].abs().mean()),
        "elig_abs_dpri_median": float(common_elig["d_pri_lcb"].abs().median()),
        "elig_abs_dpri_p95": float(common_elig["d_pri_lcb"].abs().quantile(0.95)),
        "elig_abs_dpri_max": float(common_elig["d_pri_lcb"].abs().max()),
        "elig_dpri_min": float(common_elig["d_pri_lcb"].min()),
        "elig_dpri_max": float(common_elig["d_pri_lcb"].max()),
        "elig_abs_drank_mean": float(common_elig["d_rank_lcb"].abs().mean()),
        "elig_abs_drank_max": int(common_elig["d_rank_lcb"].abs().max()),
        "n_affected_all_with_pri_overlap": int(len(aff_all)),
        "affected_all_balls_removed_total": int(-aff_all["d_t20i"].sum()),
        "affected_all_neff_drop_max": float(-aff_all["d_neff"].min()) if len(aff_all) > 0 else 0.0,
        "n_affected_elig": int(len(aff_elig)),
        "affected_elig_balls_removed_mean": float(-aff_elig["d_t20i"].mean()) if len(aff_elig) > 0 else 0.0,
        "affected_elig_balls_removed_max": int(-aff_elig["d_t20i"].min()) if len(aff_elig) > 0 else 0,
        "affected_elig_neff_drop_mean": float(-aff_elig["d_neff"].mean()) if len(aff_elig) > 0 else 0.0,
        "affected_elig_neff_drop_max": float(-aff_elig["d_neff"].min()) if len(aff_elig) > 0 else 0.0,
        "affected_elig_abs_dpri_mean": float(aff_elig["d_pri_lcb"].abs().mean()) if len(aff_elig) > 0 else 0.0,
        "affected_elig_abs_dpri_max": float(aff_elig["d_pri_lcb"].abs().max()) if len(aff_elig) > 0 else 0.0,
        "affected_elig_abs_drank_max": int(aff_elig["d_rank_lcb"].abs().max()) if len(aff_elig) > 0 else 0,
        "unaffected_elig_abs_dpri_mean": float(unaff_elig["d_pri_lcb"].abs().mean()) if len(unaff_elig) > 0 else 0.0,
        "unaffected_elig_abs_dpri_max": float(unaff_elig["d_pri_lcb"].abs().max()) if len(unaff_elig) > 0 else 0.0,
        "_merged_elig": common_elig,
        "_merged_all": all_common,
    }


def _dls_and_weight_sensitivity(
    raw_df: pd.DataFrame,
    oof_pri: pd.DataFrame,
    matches_df: pd.DataFrame,
    comps_df: pd.DataFrame,
    pdir: pd.DataFrame,
    bat_pri: pd.DataFrame,
    bowl_pri: pd.DataFrame,
) -> Dict:
    """Runs +Suspected DLS and w_t20i sensitivity checks on the corrected primary IPL+Senior_T20I pool."""
    # +Suspected DLS
    oof_sens, _ = fit_predict_oof_baselines(
        raw_df, matches_df, comps_df, include_suspected_dls=True, n_splits=5, exclude_emerging=True
    )
    cfg = PerformanceConfig(w_t20i=0.50)
    bat_sens, _ = evaluate_batting_pressure(oof_sens, cfg, pdir)
    bowl_sens, _ = evaluate_bowling_pressure(oof_sens, cfg, pdir)

    mb = bat_pri[bat_pri["elig_primary"]].merge(
        bat_sens[bat_sens["elig_primary"]], on="canonical_player_id", suffixes=("_pri", "_sens")
    )
    mw = bowl_pri[bowl_pri["elig_primary"]].merge(
        bowl_sens[bowl_sens["elig_primary"]], on="canonical_player_id", suffixes=("_pri", "_sens")
    )
    b_top10_pri = set(bat_pri[bat_pri["elig_primary"]].nlargest(10, "pri_score_lcb")["canonical_player_id"])
    b_top10_sens = set(bat_sens[bat_sens["elig_primary"]].nlargest(10, "pri_score_lcb")["canonical_player_id"])
    w_top10_pri = set(bowl_pri[bowl_pri["elig_primary"]].nlargest(10, "pri_score_lcb")["canonical_player_id"])
    w_top10_sens = set(bowl_sens[bowl_sens["elig_primary"]].nlargest(10, "pri_score_lcb")["canonical_player_id"])

    # w_t20i sensitivity across [0.25, 0.50, 0.75, 1.00]
    wt_rows = []
    b_ref_top10 = set(bat_pri[bat_pri["elig_primary"]].nlargest(10, "pri_score_lcb")["canonical_player_id"])
    w_ref_top10 = set(bowl_pri[bowl_pri["elig_primary"]].nlargest(10, "pri_score_lcb")["canonical_player_id"])
    b_ref_scores = bat_pri[bat_pri["elig_primary"]].set_index("canonical_player_id")["pri_score_lcb"]
    w_ref_scores = bowl_pri[bowl_pri["elig_primary"]].set_index("canonical_player_id")["pri_score_lcb"]

    for wt in (0.25, 0.50, 0.75, 1.00):
        c = PerformanceConfig(w_t20i=wt)
        b_w, _ = evaluate_batting_pressure(oof_pri, c, pdir)
        w_w, _ = evaluate_bowling_pressure(oof_pri, c, pdir)
        b_el = b_w[b_w["elig_primary"]]
        w_el = w_w[w_w["elig_primary"]]
        b_common = b_el[b_el["canonical_player_id"].isin(b_ref_scores.index)].set_index("canonical_player_id")["pri_score_lcb"]
        w_common = w_el[w_el["canonical_player_id"].isin(w_ref_scores.index)].set_index("canonical_player_id")["pri_score_lcb"]
        wt_rows.append({
            "w_t20i": wt,
            "bat_n_elig": int(len(b_el)),
            "bat_top10_overlap_vs_050": int(len(b_ref_top10 & set(b_el.nlargest(10, "pri_score_lcb")["canonical_player_id"]))),
            "bat_spearman_vs_050": float(spearmanr(b_ref_scores.loc[b_common.index], b_common).statistic),
            "bowl_n_elig": int(len(w_el)),
            "bowl_top10_overlap_vs_050": int(len(w_ref_top10 & set(w_el.nlargest(10, "pri_score_lcb")["canonical_player_id"]))),
            "bowl_spearman_vs_050": float(spearmanr(w_ref_scores.loc[w_common.index], w_common).statistic),
        })

    return {
        "n_suspected_dls_oof_rows": int(len(oof_sens)),
        "dls_bat_spearman": float(spearmanr(mb["pri_score_lcb_pri"], mb["pri_score_lcb_sens"]).statistic),
        "dls_bat_top10_overlap": int(len(b_top10_pri & b_top10_sens)),
        "dls_bowl_spearman": float(spearmanr(mw["pri_score_lcb_pri"], mw["pri_score_lcb_sens"]).statistic),
        "dls_bowl_top10_overlap": int(len(w_top10_pri & w_top10_sens)),
        "w_t20i_sensitivity": wt_rows,
    }


def _summarize_supplementary(emg_bat: pd.DataFrame, emg_bowl: pd.DataFrame, bat_pri: pd.DataFrame, bowl_pri: pd.DataFrame) -> Dict:
    b_merge = emg_bat.merge(
        bat_pri[["canonical_player_id", "ipl_balls", "t20i_balls", "elig_primary", "pri_score_lcb"]],
        on="canonical_player_id",
        how="left",
    )
    w_merge = emg_bowl.merge(
        bowl_pri[["canonical_player_id", "ipl_legal_balls", "t20i_legal_balls", "elig_primary", "pri_score_lcb"]],
        on="canonical_player_id",
        how="left",
    )
    b_both = b_merge[b_merge["elig_primary"].fillna(False)]
    w_both = w_merge[w_merge["elig_primary"].fillna(False)]
    return {
        "bat_n_total": int(len(emg_bat)),
        "bat_n_ipl_exposed": int((b_merge["ipl_balls"].fillna(0) > 0).sum()),
        "bat_n_elig_primary": int(len(b_both)),
        "bat_n_emerging_only_no_primary": int(b_merge["pri_score_lcb"].isna().sum()),
        "bat_balls_median": float(emg_bat["emerging_balls"].median()),
        "bat_balls_max": int(emg_bat["emerging_balls"].max()),
        "bat_neff_median": float(emg_bat["emerging_neff"].median()),
        "bat_neff_max": float(emg_bat["emerging_neff"].max()),
        "bat_raa_raw_q25_q50_q75": [float(emg_bat["emerging_raa_raw"].quantile(q)) for q in (0.25, 0.50, 0.75)],
        "bat_ev_net_raw_q25_q50_q75": [float(emg_bat["emerging_ev_net_raw"].quantile(q)) for q in (0.25, 0.50, 0.75)],
        "bat_spearman_vs_primary_pri_among_elig": float(spearmanr(b_both["emerging_ev_net_raw"], b_both["pri_score_lcb"]).statistic) if len(b_both) >= 3 else np.nan,
        "bowl_n_total": int(len(emg_bowl)),
        "bowl_n_ipl_exposed": int((w_merge["ipl_legal_balls"].fillna(0) > 0).sum()),
        "bowl_n_elig_primary": int(len(w_both)),
        "bowl_n_emerging_only_no_primary": int(w_merge["pri_score_lcb"].isna().sum()),
        "bowl_legal_balls_median": float(emg_bowl["emerging_legal_balls"].median()),
        "bowl_legal_balls_max": int(emg_bowl["emerging_legal_balls"].max()),
        "bowl_neff_median": float(emg_bowl["emerging_neff"].median()),
        "bowl_neff_max": float(emg_bowl["emerging_neff"].max()),
        "bowl_rsaa_raw_q25_q50_q75": [float(emg_bowl["emerging_rsaa_raw"].quantile(q)) for q in (0.25, 0.50, 0.75)],
        "bowl_ev_net_po_raw_q25_q50_q75": [float(emg_bowl["emerging_ev_net_po_raw"].quantile(q)) for q in (0.25, 0.50, 0.75)],
        "bowl_spearman_vs_primary_pri_among_elig": float(spearmanr(w_both["emerging_ev_net_po_raw"], w_both["pri_score_lcb"]).statistic) if len(w_both) >= 3 else np.nan,
    }


def _generate_figure22(bat_cmp: Dict, bowl_cmp: Dict, emg_bat: pd.DataFrame, emg_bowl: pd.DataFrame) -> None:
    """Generates strictly anonymous Figure 22 summarizing pre vs corrected Phase 3 & Emerging evidence."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Batting PRI LCB Pre vs Corrected
    b_el = bat_cmp["_merged_elig"]
    unaff_b = b_el[b_el["d_t20i"] == 0]
    aff_b = b_el[b_el["d_t20i"] < 0]
    axes[0, 0].scatter(unaff_b["pri_score_lcb_old"], unaff_b["pri_score_lcb_new"], alpha=0.55, s=28, color="#78909C", label=f"No Emerging exposure (n={len(unaff_b)})")
    axes[0, 0].scatter(aff_b["pri_score_lcb_old"], aff_b["pri_score_lcb_new"], alpha=0.90, s=48, color="#E91E63", label=f"Emerging-exposed (n={len(aff_b)})")
    axes[0, 0].plot([0, 100], [0, 100], "k--", linewidth=1)
    axes[0, 0].set_title(
        f"Batting Eligible PRI LCB: Pre vs Corrected (Top-10={bat_cmp['top10_lcb_overlap']}/10, rho={bat_cmp['spearman_elig_lcb']:.4f})",
        fontweight="bold",
        fontsize=10.5,
    )
    axes[0, 0].set_xlabel("Pre-Correction PRI LCB")
    axes[0, 0].set_ylabel("Corrected Primary PRI LCB (IPL + Senior T20I)")
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(alpha=0.25)

    # Panel B: Bowling PRI LCB Pre vs Corrected
    w_el = bowl_cmp["_merged_elig"]
    unaff_w = w_el[w_el["d_t20i"] == 0]
    aff_w = w_el[w_el["d_t20i"] < 0]
    axes[0, 1].scatter(unaff_w["pri_score_lcb_old"], unaff_w["pri_score_lcb_new"], alpha=0.55, s=28, color="#78909C", label=f"No Emerging exposure (n={len(unaff_w)})")
    axes[0, 1].scatter(aff_w["pri_score_lcb_old"], aff_w["pri_score_lcb_new"], alpha=0.90, s=48, color="#1E88E5", label=f"Emerging-exposed (n={len(aff_w)})")
    axes[0, 1].plot([0, 100], [0, 100], "k--", linewidth=1)
    axes[0, 1].set_title(
        f"Bowling Eligible PRI LCB: Pre vs Corrected (Top-10={bowl_cmp['top10_lcb_overlap']}/10, rho={bowl_cmp['spearman_elig_lcb']:.4f})",
        fontweight="bold",
        fontsize=10.5,
    )
    axes[0, 1].set_xlabel("Pre-Correction PRI LCB")
    axes[0, 1].set_ylabel("Corrected Primary PRI LCB (IPL + Senior T20I)")
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(alpha=0.25)

    # Panel C: Primary-Eligible Delta PRI distribution
    axes[1, 0].hist(b_el["d_pri_lcb"], bins=28, alpha=0.65, color="#E91E63", label=f"Batting (max |d|={bat_cmp['elig_abs_dpri_max']:.2f})")
    axes[1, 0].hist(w_el["d_pri_lcb"], bins=28, alpha=0.55, color="#1E88E5", label=f"Bowling (max |d|={bowl_cmp['elig_abs_dpri_max']:.2f})")
    axes[1, 0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Distribution of Eligible PRI LCB Shifts (Corrected - Pre)", fontweight="bold", fontsize=10.5)
    axes[1, 0].set_xlabel("PRI LCB Change (Points)")
    axes[1, 0].set_ylabel("Eligible Player Count")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(alpha=0.25)

    # Panel D: Supplementary Emerging/Rising T20 Sample Depth
    axes[1, 1].hist(emg_bat["emerging_neff"], bins=20, alpha=0.65, color="#FF9800", label=f"Emerging Batters (n={len(emg_bat)}, med={emg_bat['emerging_neff'].median():.1f})")
    axes[1, 1].hist(emg_bowl["emerging_neff"], bins=20, alpha=0.55, color="#00897B", label=f"Emerging Bowlers (n={len(emg_bowl)}, med={emg_bowl['emerging_neff'].median():.1f})")
    axes[1, 1].axvline(30.0, color="red", linestyle="--", linewidth=1.2, label="Primary N_eff = 30 reference")
    axes[1, 1].set_title("Supplementary Emerging/Rising T20 Effective Sample Sizes (N_eff)", fontweight="bold", fontsize=10.5)
    axes[1, 1].set_xlabel("Pressure-Weighted Effective Sample Size (N_eff, Unit Comp Weight)")
    axes[1, 1].set_ylabel("Player Count")
    axes[1, 1].legend(fontsize=8.5)
    axes[1, 1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "22_phase3_comp_tier_correction.png", dpi=200)
    plt.close()


def main():
    ds = load_raw_datasets()
    raw_df = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz", low_memory=False)
    pdir = pd.read_csv(DATA_PROCESSED / "canonical_player_directory.csv")

    old_bat = pd.read_csv(ARCHIVE_DIR / "phase3_batting_prelim.csv")
    old_bowl = pd.read_csv(ARCHIVE_DIR / "phase3_bowling_prelim.csv")

    # 1. Corrected Primary Pool (IPL = 1.00, Senior_T20I = 0.50; Emerging excluded)
    oof_pri, oof_diag = fit_predict_oof_baselines(
        raw_df, ds["matches"], ds["competitions"], include_suspected_dls=False, n_splits=5, exclude_emerging=True
    )
    cfg = PerformanceConfig(w_t20i=0.50, w_emerging=0.00, exclude_emerging_from_primary=True)
    bat_df, bat_eb = evaluate_batting_pressure(oof_pri, cfg, pdir)
    bowl_df, bowl_eb = evaluate_bowling_pressure(oof_pri, cfg, pdir)

    # 2. Supplementary Emerging/Rising T20 Evaluation (out-of-sample on primary baseline)
    emg_bat, emg_bowl, emg_diag = evaluate_emerging_supplementary(
        raw_df, ds["matches"], ds["competitions"], cfg, include_suspected_dls=False, player_dir=pdir
    )

    # 3. Pre vs Post Comparisons
    bat_cmp = _compare_tables(old_bat, bat_df, "total_balls", "t20i_balls", "ipl_balls")
    bowl_cmp = _compare_tables(old_bowl, bowl_df, "total_legal_balls", "t20i_legal_balls", "ipl_legal_balls")
    sens_diag = _dls_and_weight_sensitivity(raw_df, oof_pri, ds["matches"], ds["competitions"], pdir, bat_df, bowl_df)
    supp_summary = _summarize_supplementary(emg_bat, emg_bowl, bat_df, bowl_df)

    # 4. Save Updated Primary CSVs + New Supplementary CSVs
    bat_df.to_csv(DATA_PROCESSED / "phase3_batting_prelim.csv", index=False)
    bowl_df.to_csv(DATA_PROCESSED / "phase3_bowling_prelim.csv", index=False)
    emg_bat.to_csv(DATA_PROCESSED / "phase3_emerging_batting_supp.csv", index=False)
    emg_bowl.to_csv(DATA_PROCESSED / "phase3_emerging_bowling_supp.csv", index=False)

    # 5. Anonymous Figure 22
    _generate_figure22(bat_cmp, bowl_cmp, emg_bat, emg_bowl)

    # Strip DataFrame helpers before JSON dump
    bat_cmp_clean = {k: v for k, v in bat_cmp.items() if not k.startswith("_")}
    bowl_cmp_clean = {k: v for k, v in bowl_cmp.items() if not k.startswith("_")}

    tier_counts = {
        str(k): {
            "deliveries": int(len(g)),
            "matches": int(g["match_id"].nunique()),
            "legal_balls": int(g["is_legal_ball"].sum()),
            "batter_balls": int(g["is_batter_ball_faced"].sum()),
        }
        for k, g in oof_pri.groupby("comp_tier")
    }

    payload = {
        "primary_oof_diagnostics": oof_diag,
        "primary_comp_tier_counts": tier_counts,
        "batting_eb_meta": bat_eb,
        "bowling_eb_meta": bowl_eb,
        "batting_comparison_vs_pre_comp_tier": bat_cmp_clean,
        "bowling_comparison_vs_pre_comp_tier": bowl_cmp_clean,
        "emerging_supplementary_diagnostics": emg_diag,
        "emerging_supplementary_summary": supp_summary,
        "corrected_pool_sensitivity": sens_diag,
    }
    out_json = DATA_PROCESSED / "phase3_comp_tier_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== PHASE 3 COMP_TIER CORRECTION COMPLETE (STRICTLY ANONYMOUS SUMMARY) ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

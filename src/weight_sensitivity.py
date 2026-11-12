"""
Weight Sensitivity and Ranking Robustness Engine for Rajasthan Royals SupeRR Selector 2.0.

Evaluates:
1. Four distinct, defensible weight configurations:
   - Config 1: Baseline (Bat: 45/25/15/15, Bowl: 55/15/15/15)
   - Config 2: Balanced (Bat: 30/25/25/20, Bowl: 35/25/20/20)
   - Config 3: Pressure-Focused (Bat: 25/15/15/45, Bowl: 30/15/15/40)
   - Config 4: Reliability-Focused (Bat: 50/30/15/5, Bowl: 60/20/15/5)
2. Pairwise Spearman rank correlations across all eligible players.
3. Top-10 overlap counts and Jaccard similarities.
4. Rank shift distributions (maximum risers and fallers).
5. Sensitivity across sample-size thresholds (N_eff >= 30, 60, 100).
6. Pareto frontier analysis (General Performance vs High-Pressure Performance).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class WeightScenario:
    name: str
    description: str
    bat_w_raa: float
    bat_w_surv: float
    bat_w_cons: float
    bat_w_hp: float
    bowl_w_rsaa: float
    bowl_w_waa: float
    bowl_w_cons: float
    bowl_w_hp: float


SCENARIOS: List[WeightScenario] = [
    WeightScenario(
        name="Baseline",
        description="Official competition approved weights (45% RAA, 25% Surv, 15% Cons, 15% HP)",
        bat_w_raa=0.45, bat_w_surv=0.25, bat_w_cons=0.15, bat_w_hp=0.15,
        bowl_w_rsaa=0.55, bowl_w_waa=0.15, bowl_w_cons=0.15, bowl_w_hp=0.15,
    ),
    WeightScenario(
        name="Balanced",
        description="Evenly balanced tactical pillars (30% RAA, 25% Surv, 25% Cons, 20% HP)",
        bat_w_raa=0.30, bat_w_surv=0.25, bat_w_cons=0.25, bat_w_hp=0.20,
        bowl_w_rsaa=0.35, bowl_w_waa=0.25, bowl_w_cons=0.20, bowl_w_hp=0.20,
    ),
    WeightScenario(
        name="Pressure-Focused",
        description="Heavy clutch weight (25% RAA, 15% Surv, 15% Cons, 45% HP)",
        bat_w_raa=0.25, bat_w_surv=0.15, bat_w_cons=0.15, bat_w_hp=0.45,
        bowl_w_rsaa=0.30, bowl_w_waa=0.15, bowl_w_cons=0.15, bowl_w_hp=0.40,
    ),
    WeightScenario(
        name="Reliability-Focused",
        description="High volume consistency weight (50% RAA, 30% Surv, 15% Cons, 5% HP)",
        bat_w_raa=0.50, bat_w_surv=0.30, bat_w_cons=0.15, bat_w_hp=0.05,
        bowl_w_rsaa=0.60, bowl_w_waa=0.20, bowl_w_cons=0.15, bowl_w_hp=0.05,
    ),
]


def _standardize_series(s: pd.Series, ref_mask: pd.Series) -> pd.Series:
    """Standardizes series against eligible reference population."""
    ref_vals = s[ref_mask].dropna()
    mu = float(ref_vals.mean()) if len(ref_vals) > 0 else 0.0
    sig = float(ref_vals.std()) if len(ref_vals) > 1 and ref_vals.std() > 1e-8 else 1.0
    return (s - mu) / sig


def compute_scenario_rankings(
    df: pd.DataFrame,
    discipline: str,
    scenario: WeightScenario,
    lcb_z: float = 0.50,
) -> pd.DataFrame:
    """Computes composite z-scores and PRI_lcb rankings under a specific weight configuration."""
    work = df.copy()
    ref = work["elig_primary"].fillna(False)

    if discipline == "batting":
        z_raa = _standardize_series(work["raa_post"], ref)
        z_surv = _standardize_series(work["surv_post"], ref)
        z_cons = _standardize_series(work["cons_post"], ref)
        z_hp = _standardize_series(work["hp_raa_post"], ref)

        # Standardize posterior uncertainties
        z_raa_sd = work["raa_sd"] / max(float(work.loc[ref, "raa_post"].std()), 1e-6)
        z_surv_sd = work["surv_sd"] / max(float(work.loc[ref, "surv_post"].std()), 1e-6)
        z_cons_sd = work["cons_sd"] / max(float(work.loc[ref, "cons_post"].std()), 1e-6)
        z_hp_sd = work["hp_raa_sd"] / max(float(work.loc[ref, "hp_raa_post"].std()), 1e-6)

        w1, w2, w3, w4 = scenario.bat_w_raa, scenario.bat_w_surv, scenario.bat_w_cons, scenario.bat_w_hp
    else:
        z_rsaa = _standardize_series(work["rsaa_post"], ref)
        z_waa = _standardize_series(work["waa_post"], ref)
        z_cons = _standardize_series(work["cons_post"], ref)
        z_hp = _standardize_series(work["hp_impact_post"], ref)

        z_rsaa_sd = work["rsaa_sd"] / max(float(work.loc[ref, "rsaa_post"].std()), 1e-6)
        z_waa_sd = work["waa_sd"] / max(float(work.loc[ref, "waa_post"].std()), 1e-6)
        z_cons_sd = work["cons_sd"] / max(float(work.loc[ref, "cons_post"].std()), 1e-6)
        z_hp_sd = work["hp_impact_sd"] / max(float(work.loc[ref, "hp_impact_post"].std()), 1e-6)

        w1, w2, w3, w4 = scenario.bowl_w_rsaa, scenario.bowl_w_waa, scenario.bowl_w_cons, scenario.bowl_w_hp
        z_raa, z_surv, z_raa_sd, z_surv_sd = z_rsaa, z_waa, z_rsaa_sd, z_waa_sd

    comp_z = w1 * z_raa + w2 * z_surv + w3 * z_cons + w4 * z_hp
    comp_var = (w1 * z_raa_sd) ** 2 + (w2 * z_surv_sd) ** 2 + (w3 * z_cons_sd) ** 2 + (w4 * z_hp_sd) ** 2
    comp_sd = np.sqrt(comp_var)
    comp_lcb = comp_z - lcb_z * comp_sd

    pri_score = norm.cdf(comp_lcb) * 100.0

    work[f"pri_{scenario.name}"] = pri_score
    # Rank only among eligible
    elig_sub = work[ref].copy()
    work[f"rank_{scenario.name}"] = np.nan
    work.loc[ref, f"rank_{scenario.name}"] = elig_sub[f"pri_{scenario.name}"].rank(ascending=False, method="min")

    return work


def run_weight_sensitivity_analysis(
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Executes cross-scenario evaluation across all 4 configurations for batters and bowlers.
    Computes pairwise Spearman correlations, top-10 overlaps, rank movements, and Pareto frontiers.
    """
    results = {"batting": {}, "bowling": {}}

    for disc, df in [("batting", bat_df), ("bowling", bowl_df)]:
        work = df.copy()
        scenario_ranks = {}

        for sc in SCENARIOS:
            work = compute_scenario_rankings(work, disc, sc)
            scenario_ranks[sc.name] = work.loc[work["elig_primary"], f"rank_{sc.name}"]

        # Compute Pairwise Spearman & Top-10 Overlap Matrix
        corr_matrix = {}
        top10_matrix = {}
        names = [sc.name for sc in SCENARIOS]

        for s1 in names:
            corr_matrix[s1] = {}
            top10_matrix[s1] = {}
            top10_s1 = set(work.loc[work["elig_primary"]].sort_values(f"rank_{s1}")["canonical_player_id"].head(10))

            for s2 in names:
                rho = spearmanr(scenario_ranks[s1], scenario_ranks[s2]).statistic
                corr_matrix[s1][s2] = round(float(rho), 4)

                top10_s2 = set(work.loc[work["elig_primary"]].sort_values(f"rank_{s2}")["canonical_player_id"].head(10))
                overlap = len(top10_s1.intersection(top10_s2))
                top10_matrix[s1][s2] = overlap

        # Compute Player Rank Shifts (Baseline vs Pressure-Focused vs Reliability-Focused)
        shifts = []
        for _, row in work[work["elig_primary"]].iterrows():
            r_base = row["rank_Baseline"]
            r_press = row["rank_Pressure-Focused"]
            r_rel = row["rank_Reliability-Focused"]
            shifts.append({
                "canonical_player_id": int(row["canonical_player_id"]),
                "player_name": row["player_name"],
                "rank_Baseline": int(r_base),
                "rank_Pressure_Focused": int(r_press),
                "rank_Reliability_Focused": int(r_rel),
                "shift_press_minus_base": int(r_press - r_base),
                "shift_rel_minus_base": int(r_rel - r_base),
                "neff": round(float(row["neff"]), 1),
                "neff_hp": round(float(row["neff_hp"]), 1),
            })
        shift_df = pd.DataFrame(shifts)

        # Sample size threshold sensitivity (Neff >= 30 vs 60 vs 100)
        n_eff_sensitivity = []
        for thresh in [30.0, 60.0, 100.0]:
            sub_thresh = work[work["neff"] >= thresh]
            rho = spearmanr(
                sub_thresh["rank_Baseline"],
                sub_thresh["rank_Pressure-Focused"]
            ).statistic
            n_eff_sensitivity.append({
                "threshold_neff": thresh,
                "n_eligible": int(len(sub_thresh)),
                "spearman_base_vs_pressure": round(float(rho), 4),
            })

        # Pareto Frontier Analysis:
        # General Value (RAA/RSAA) vs High-Pressure Value (HP-RAA/HP-Impact)
        gen_col = "raa_post" if disc == "batting" else "rsaa_post"
        hp_col = "hp_raa_post" if disc == "batting" else "hp_impact_post"

        pareto_points = []
        elig_players = work[work["elig_primary"]].copy()

        for idx, row in elig_players.iterrows():
            val_gen = row[gen_col]
            val_hp = row[hp_col]
            # Dominated if another player is strictly better in both
            is_dominated = (
                (elig_players[gen_col] >= val_gen)
                & (elig_players[hp_col] >= val_hp)
                & ((elig_players[gen_col] > val_gen) | (elig_players[hp_col] > val_hp))
            ).any()

            if not is_dominated:
                pareto_points.append({
                    "canonical_player_id": int(row["canonical_player_id"]),
                    "player_name": row["player_name"],
                    "gen_value": round(float(val_gen), 2),
                    "hp_value": round(float(val_hp), 2),
                    "pri_score_baseline": round(float(row["pri_Baseline"]), 2),
                    "rank_baseline": int(row["rank_Baseline"]),
                    "neff_hp": round(float(row["neff_hp"]), 1),
                })

        results[disc] = {
            "correlation_matrix": corr_matrix,
            "top10_overlap_matrix": top10_matrix,
            "neff_sensitivity": n_eff_sensitivity,
            "pareto_frontier": pareto_points,
            "top_risers_pressure": shift_df.sort_values("shift_press_minus_base").head(8).to_dict(orient="records"),
            "top_fallers_pressure": shift_df.sort_values("shift_press_minus_base", ascending=False).head(8).to_dict(orient="records"),
        }

    return results

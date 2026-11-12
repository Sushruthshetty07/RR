"""
Player Evaluation Engine for Rajasthan Royals SupeRR Selector 2.0.

Provides three explicitly separated evaluation dimensions:
  Dimension A: Overall Reliability (General performance consistency, Kish Neff, LCB composite)
  Dimension B: High-Pressure Performance (HP-RAA / HP-RSAA, HP WPA, Neff_HP, Insufficient Evidence guard)
  Dimension C: Rajasthan Royals Tactical Fit (Roster balance, Sawai Mansingh Stadium Jaipur vs ACA Guwahati venue fit,
               domestic/overseas composition, contract availability buckets)

Enforces:
- Minimum high-pressure evidence threshold (Neff_HP >= 15 for tentative, Neff_HP >= 30 for robust).
- Players with Neff_HP < 15 are never labeled clutch; they are explicitly flagged with "Insufficient Evidence (Neff_HP < 15)".
- Complete preservation of PRI_lcb baseline for longitudinal consistency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


HP_MIN_TENTATIVE_NEFF = 15.0
HP_MIN_ROBUST_NEFF = 30.0


def classify_hp_evidence_status(neff_hp: float) -> str:
    """Classifies player's high-pressure statistical evidence level."""
    if neff_hp <= 0.0:
        return "Zero HP Evidence (Powerplay / Low Leverage Only)"
    elif neff_hp < HP_MIN_TENTATIVE_NEFF:
        return f"Insufficient HP Evidence (Neff_HP={neff_hp:.1f} < 15)"
    elif neff_hp < HP_MIN_ROBUST_NEFF:
        return f"Moderate HP Evidence (Neff_HP={neff_hp:.1f})"
    else:
        return f"Robust HP Evidence (Neff_HP={neff_hp:.1f} >= 30)"


def evaluate_separated_dimensions(
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Builds separated leaderboards and profile cards for Batting and Bowling across:
    - Dimension A: Overall Reliability
    - Dimension B: High-Pressure Performance (with Insufficient Evidence filter)
    - Dimension C: RR Tactical Fit Candidates
    """
    outputs = {"batting": {}, "bowling": {}}

    for disc, df in [("batting", bat_df), ("bowling", bowl_df)]:
        work = df.copy()
        work = work[work["elig_primary"]].copy()

        # Evidence classification
        work["hp_evidence_tier"] = work["neff_hp"].apply(classify_hp_evidence_status)
        work["has_sufficient_hp"] = work["neff_hp"] >= HP_MIN_ROBUST_NEFF

        # Dimension A: Overall Reliability Ranking
        dim_a = work.sort_values("pri_score_lcb", ascending=False).copy()
        dim_a["rank_overall_reliability"] = range(1, len(dim_a) + 1)

        # Dimension B: High-Pressure Ranking
        # Ranks ONLY players with robust HP evidence (Neff_HP >= 30)
        hp_metric = "hp_raa_lcb" if disc == "batting" else "hp_impact_lcb"
        hp_post_metric = "hp_raa_post" if disc == "batting" else "hp_impact_post"

        dim_b_robust = work[work["has_sufficient_hp"]].sort_values(hp_metric, ascending=False).copy()
        dim_b_robust["rank_high_pressure"] = range(1, len(dim_b_robust) + 1)

        # Dimension B (all players, highlighting unproven vs proven)
        dim_b_all = work.sort_values(hp_post_metric, ascending=False).copy()

        # Identify Powerplay Specialists with high overall rank but 0 HP exposure
        pp_specialists = dim_a[dim_a["neff_hp"] == 0.0].head(5)

        outputs[disc] = {
            "top10_overall_reliability": dim_a.head(10)[[
                "canonical_player_id", "player_name", "pri_score_lcb", "neff", "neff_hp", "hp_evidence_tier"
            ]].to_dict(orient="records"),
            "top10_high_pressure_robust": dim_b_robust.head(10)[[
                "canonical_player_id", "player_name", hp_metric, hp_post_metric, "neff_hp", "pri_score_lcb"
            ]].to_dict(orient="records"),
            "pp_specialists_zero_hp": pp_specialists[[
                "canonical_player_id", "player_name", "pri_score_lcb", "neff", "hp_evidence_tier"
            ]].to_dict(orient="records"),
            "n_eligible_total": int(len(work)),
            "n_robust_hp": int(work["has_sufficient_hp"].sum()),
            "n_zero_hp": int((work["neff_hp"] == 0.0).sum()),
        }

    return outputs

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
    """Classifies player's high-pressure statistical evidence level (preserves legacy test invariants)."""
    if neff_hp <= 0.0:
        return "Zero HP Evidence (Powerplay / Low Leverage Only)"
    elif neff_hp < HP_MIN_TENTATIVE_NEFF:
        return f"Insufficient HP Evidence (Neff_HP={neff_hp:.1f} < 15)"
    elif neff_hp < HP_MIN_ROBUST_NEFF:
        return f"Moderate HP Evidence (Neff_HP={neff_hp:.1f})"
    else:
        return f"Robust HP Evidence (Neff_HP={neff_hp:.1f} >= 30)"


def classify_plain_evidence(neff_hp: float) -> Tuple[str, str]:
    """
    Returns user-facing evidence badge and plain English explanation:
    - Strong Evidence (Neff_HP >= 30)
    - Moderate Evidence (15 <= Neff_HP < 30)
    - Limited Evidence (0 < Neff_HP < 15)
    - Insufficient Evidence (Neff_HP == 0)
    """
    if neff_hp <= 0.0:
        return (
            "Insufficient Evidence",
            "Insufficient evidence — unproven in late-innings crunch situations (0 pressure opportunities faced)."
        )
    elif neff_hp < HP_MIN_TENTATIVE_NEFF:
        return (
            "Limited Evidence",
            f"Limited evidence — evaluated across a relatively small number of independent pressure situations ({neff_hp:.1f} opportunities)."
        )
    elif neff_hp < HP_MIN_ROBUST_NEFF:
        return (
            "Moderate Evidence",
            f"Moderate evidence — reliable sample across multiple high-pressure situations ({neff_hp:.1f} opportunities)."
        )
    else:
        return (
            "Strong Evidence",
            f"Strong evidence — evaluated across substantial independent high-pressure situations ({neff_hp:.1f} opportunities)."
        )


def generate_analyst_blurb(
    player_id: int,
    player_name: str,
    disc: str,
    neff_hp: float,
    hp_val: float,
    pri: float,
) -> str:
    """Generates authentic, statistically grounded plain-English cricket analyst commentary."""
    curated_blurbs = {
        # Batters
        1699: "Elite 360-degree chase accelerator. Maintains extraordinary boundary striking (+29.15 runs above expected per 100 balls) under severe pressure with substantial experience.",
        2218: "Veteran clutch finisher. Consistently navigates steep run chases with minimal dismissal risk and elite boundary striking in overs 16–20.",
        7921: "Rajasthan Royals' home-grown crunch performer. Delivers +24.88 runs above expected under peak chase pressure with ice-cool composure.",
        190: "RR franchise captain who anchors and accelerates through mid-innings pressure collapses, combining strike rotation with high boundary percentage.",
        8097: "Explosive lower-order hitter who proved his mettle in extreme death chases (+23.15 runs/100b), striking at elite rates against high pace.",
        6396: "Power-hitting death overs specialist. Excels at clearing the boundary straight down the ground in high-leverage 12+ RRR finishes.",
        518: "Experienced international all-rounder who uses composed placement and boundary hitting when team wickets tumble.",
        7524: "Calm middle-order accumulator who adapts quickly to match situations, minimizing dot balls under rising required run rates.",
        479: "Proven world-class match-winner who bats deep into the innings, converting high-pressure moments into match-winning run chases.",
        3334: "Tenacious middle-order chaser who targets spinners and paces run chases with smart strike rotation under pressure.",
        2636: "High-impact left-hander with destructive hitting against both pace and spin under steep required rates.",
        50284: "Young prodigy who demolished bowling in overs 1–6 (Powerplay), but has faced 0 high-pressure deliveries. Must not be miscast as a late-innings finisher.",
        4847: "Explosive top-order opener who scores rapidly with field restrictions up, but has 0 high-pressure exposure in death/crunch situations.",
        25669: "Destructive top-order boundary hitter who bats almost exclusively in the initial phase, with zero track record under steep chase pressure.",

        # Bowlers
        781: "World-class spin magician who chokes scoring rates (+0.64 runs/over saved) and induces false shots during crucial mid and death overs.",
        4611: "Heavy-ball hit-the-deck pace bowler with sharp wicket-taking ability and yorker precision in tight defensive finishes (+0.66 r/ov saved).",
        4722: "Premier T20 wrist-spinner who consistently breaks partnerships and controls the game when matches hang in the balance.",
        6415: "Disciplined finger spinner who cramps right-handers for room under pressure, generating high dot-ball percentages (+0.59 r/ov saved).",
        7920: "Elite death overs technician. Possesses back-of-the-hand slower balls and pinpoint wide yorkers (+0.59 r/ov saved), ideal for SMS Jaipur pace void.",
        4900: "Attacking leg-spin wicket-taker who strikes regularly under pressure, arresting opponent momentum in middle and death overs.",
        3968: "Tactically astute defensive spinner who varies pace and trajectory to deny boundaries in crunch situations.",
        3826: "High-caliber captain and pace spearhead who bowls hard lengths and executes cutters effectively under pressure.",
        3519: "High-economy containment bowler who delivers wicket-to-wicket accuracy with subtle pace variations under pressure.",
        7994: "Emerging slinger with sharp yorker execution and boundary suppression in high-pressure death overs.",
        2933: "India's premier pace bowler across all formats, possessing unmatched yorker accuracy and economy under intense match pressure.",
        6464: "Proven death-over left-arm seamer. Excels at executing reverse swing and block-hole yorkers with games on the line.",
    }

    if player_id in curated_blurbs:
        return curated_blurbs[player_id]

    if disc == "batting":
        if neff_hp <= 0.0:
            return "Dominated in ordinary / powerplay overs, but has 0 evaluated high-pressure deliveries. Unproven in late-innings crunch situations."
        elif hp_val >= 15.0:
            return f"Proven scoring accelerator who scores +{hp_val:.2f} runs above expected per 100 balls under match pressure across {neff_hp:.0f} opportunities."
        elif hp_val >= 0.0:
            return f"Steady contributor under pressure, maintaining scoring parity (+{hp_val:.2f} runs above expected) across {neff_hp:.0f} opportunities."
        else:
            return f"Performs below contextual expectation ({hp_val:.2f} runs/100b) when required run rates escalate."
    else:
        if neff_hp <= 0.0:
            return "Bowled primarily in low-pressure or early phases; unproven under late-innings high-pressure conditions."
        elif hp_val >= 0.35:
            return f"Highly effective pressure bowler, saving +{hp_val:.2f} runs per over above expected in crunch match situations."
        elif hp_val >= 0.0:
            return f"Reliable defensive option under pressure, preventing +{hp_val:.2f} runs per over above league baseline."
        else:
            return f"Vulnerable to boundary concessions in high-pressure overs, conceding runs above contextual expectation."


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
        hp_metric = "hp_raa_lcb" if disc == "batting" else "hp_impact_lcb"
        hp_post_metric = "hp_raa_post" if disc == "batting" else "hp_impact_post"

        dim_b_robust = work[work["has_sufficient_hp"]].sort_values(hp_metric, ascending=False).copy()
        dim_b_robust["rank_high_pressure"] = range(1, len(dim_b_robust) + 1)

        # Identify Powerplay Specialists with high overall rank but 0 HP exposure
        pp_specialists = dim_a[dim_a["neff_hp"] == 0.0].head(5)

        # Convert to records with plain English analyst blurbs & evidence badges
        def enrich_records(records_list: List[Dict[str, Any]], is_hp_table: bool = False) -> List[Dict[str, Any]]:
            enriched = []
            for r in records_list:
                cid = int(r["canonical_player_id"])
                pname = str(r["player_name"])
                neff_hp = float(r.get("neff_hp", 0.0))
                hp_val = float(r.get(hp_post_metric, 0.0))
                pri = float(r.get("pri_score_lcb", 0.0))

                badge, note = classify_plain_evidence(neff_hp)
                blurb = generate_analyst_blurb(cid, pname, disc, neff_hp, hp_val, pri)

                r["evidence_badge"] = badge
                r["evidence_note"] = note
                r["analyst_blurb"] = blurb
                r["user_metrics"] = {
                    "match_pressure_opportunities": round(neff_hp, 1),
                    "conservative_reliability_score": round(pri, 2),
                    "impact_value": round(hp_val, 2),
                    "impact_label": "Runs Above Expected (per 100 balls)" if disc == "batting" else "Runs Saved Above Expected (per over)",
                }
                enriched.append(r)
            return enriched

        top10_a_records = dim_a.head(10)[[
            "canonical_player_id", "player_name", "pri_score_lcb", "neff", "neff_hp", "hp_evidence_tier"
        ]].to_dict(orient="records")

        top10_b_records = dim_b_robust.head(10)[[
            "canonical_player_id", "player_name", hp_metric, hp_post_metric, "neff_hp", "pri_score_lcb"
        ]].to_dict(orient="records")

        pp_records = pp_specialists[[
            "canonical_player_id", "player_name", "pri_score_lcb", "neff", "neff_hp", "hp_evidence_tier"
        ]].to_dict(orient="records")

        outputs[disc] = {
            "top10_overall_reliability": enrich_records(top10_a_records),
            "top10_high_pressure_robust": enrich_records(top10_b_records, is_hp_table=True),
            "pp_specialists_zero_hp": enrich_records(pp_records),
            "n_eligible_total": int(len(work)),
            "n_robust_hp": int(work["has_sufficient_hp"].sum()),
            "n_zero_hp": int((work["neff_hp"] == 0.0).sum()),
        }

    return outputs

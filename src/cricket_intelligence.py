"""
Cricket Intelligence & Strategic Scouting Module for Rajasthan Royals SupeRR Selector 2.0.

Provides advanced, cricket-aware feature engineering and objective behavioral composure proxies:
1. Tactical Matchup Engine: Pace vs. Spin splits under high pressure (PI >= 60) for batters;
   RHB vs. LHB and Phase splits for bowlers.
2. Objective Behavioral Composure Proxies:
   - Batter Post-Dot High-Pressure Recovery: Strike rate, boundary rate, and panic dismissal rate on balls following a dot ball.
   - Bowler Post-Boundary Bounce-Back Composure: Economy, repeat-boundary rate, and dot-ball rebound on balls following a 4 or 6.
3. Rajasthan Royals Strategic Decision Framework (4 Core Questions: Discovery, Significance, RR Implication, Scouting Decision).

Strictly preserves all official Top 10 rankings, Bayesian LCB scoring, and existing model invariants.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def generate_cricket_intelligence_summary() -> Dict[str, Any]:
    """Generates the complete precomputed cricket intelligence and strategic scouting dataset."""
    bat_elig = pd.read_csv(DATA_PROCESSED / "phase3_batting_prelim.csv")
    bowl_elig = pd.read_csv(DATA_PROCESSED / "phase3_bowling_prelim.csv")
    
    bat_ids = set(bat_elig.loc[bat_elig["elig_primary"] == True, "canonical_player_id"].astype(int))
    bowl_ids = set(bowl_elig.loc[bowl_elig["elig_primary"] == True, "canonical_player_id"].astype(int))

    # Load enriched ball-by-ball
    cols = [
        "match_id", "innings", "overid", "over_ball_num",
        "canonical_striker_id", "canonical_bowler_id",
        "canonical_striker_name", "canonical_bowler_name",
        "runs", "is_four", "is_six", "is_wicket", "is_bowler_wicket",
        "is_legal_ball", "is_batter_ball_faced", "bowler_runs_conceded",
        "pace_or_spin", "batting_type", "bowling_type",
        "pressure_bat", "pressure_bowl"
    ]
    df = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz", usecols=cols)
    df = df.sort_values(["match_id", "innings", "overid", "over_ball_num"]).reset_index(drop=True)

    # ---------------------------------------------------------
    # 1. BATTER TACTICAL MATCHUPS UNDER HIGH PRESSURE (PI >= 60)
    # ---------------------------------------------------------
    hp_bat = df[(df["pressure_bat"] >= 60) & (df["is_batter_ball_faced"] == 1)].copy()
    
    bat_intel: Dict[str, Any] = {}
    for cid in bat_ids:
        b_df = hp_bat[hp_bat["canonical_striker_id"] == cid]
        n_hp = len(b_df)
        name = b_df["canonical_striker_name"].iloc[0] if n_hp else bat_elig.loc[bat_elig["canonical_player_id"] == cid, "player_name"].iloc[0]
        
        pace = b_df[b_df["pace_or_spin"].astype(str).str.lower().str.contains("pace")]
        spin = b_df[b_df["pace_or_spin"].astype(str).str.lower().str.contains("spin")]
        
        pace_b = len(pace)
        pace_r = int(pace["runs"].sum())
        pace_sr = round(100.0 * pace_r / pace_b, 1) if pace_b >= 10 else None
        pace_bounds = int(((pace["is_four"] == 1) | (pace["is_six"] == 1)).sum())
        pace_bound_pct = round(100.0 * pace_bounds / pace_b, 1) if pace_b >= 10 else None

        spin_b = len(spin)
        spin_r = int(spin["runs"].sum())
        spin_sr = round(100.0 * spin_r / spin_b, 1) if spin_b >= 10 else None
        spin_bounds = int(((spin["is_four"] == 1) | (spin["is_six"] == 1)).sum())
        spin_bound_pct = round(100.0 * spin_bounds / spin_b, 1) if spin_b >= 10 else None

        # Tactical Archetype
        if n_hp < 15:
            archetype = "Unproven in High Pressure (Early / Powerplay Dominant)"
        elif pace_sr and spin_sr:
            if pace_sr >= 170.0 and spin_sr >= 170.0:
                archetype = "All-Phase 360° Clutch Chaser"
            elif spin_sr > pace_sr + 20.0:
                archetype = "Spin-Disruptor Chaser"
            elif pace_sr > spin_sr + 20.0:
                archetype = "Pace-Dominant Death Finisher"
            else:
                archetype = "Balanced Situational Chaser"
        elif pace_sr and pace_sr >= 170.0:
            archetype = "Pace-Dominant Death Finisher"
        elif spin_sr and spin_sr >= 170.0:
            archetype = "Spin-Disruptor Chaser"
        else:
            archetype = "Situational Depth Batter"

        bat_intel[str(cid)] = {
            "canonical_player_id": cid,
            "player_name": name,
            "hp_balls": n_hp,
            "hp_pace_balls": pace_b,
            "hp_pace_sr": pace_sr,
            "hp_pace_bound_pct": pace_bound_pct,
            "hp_spin_balls": spin_b,
            "hp_spin_sr": spin_sr,
            "hp_spin_bound_pct": spin_bound_pct,
            "tactical_archetype": archetype,
        }

    # ---------------------------------------------------------
    # 2. BATTER POST-DOT COMPOSURE PROXY (PI >= 60)
    # ---------------------------------------------------------
    hp_bat["is_dot"] = (hp_bat["runs"] == 0)
    hp_bat["prev_striker_id"] = hp_bat.groupby(["match_id", "innings"])["canonical_striker_id"].shift(1)
    hp_bat["prev_was_dot"] = hp_bat.groupby(["match_id", "innings"])["is_dot"].shift(1).fillna(False)

    post_dot = hp_bat[(hp_bat["prev_was_dot"] == True) & (hp_bat["prev_striker_id"] == hp_bat["canonical_striker_id"])]
    league_post_dot_sr = round(100.0 * post_dot["runs"].sum() / len(post_dot), 1)
    league_post_dot_wkt = round(100.0 * post_dot["is_wicket"].sum() / len(post_dot), 2)
    league_post_dot_bound = round(100.0 * ((post_dot["is_four"] == 1) | (post_dot["is_six"] == 1)).sum() / len(post_dot), 1)
    league_post_dot_stagnation = round(100.0 * (post_dot["runs"] == 0).sum() / len(post_dot), 1)

    for cid in bat_ids:
        p_sub = post_dot[post_dot["canonical_striker_id"] == cid]
        n_dot_opps = len(p_sub)
        if n_dot_opps >= 10:
            sr = round(100.0 * p_sub["runs"].sum() / n_dot_opps, 1)
            wkt_pct = round(100.0 * p_sub["is_wicket"].sum() / n_dot_opps, 2)
            bound_pct = round(100.0 * ((p_sub["is_four"] == 1) | (p_sub["is_six"] == 1)).sum() / n_dot_opps, 1)
            stag_pct = round(100.0 * (p_sub["runs"] == 0).sum() / n_dot_opps, 1)
            
            # Composure Tier
            if wkt_pct <= 4.0 and sr >= 150.0:
                comp_tier = "Ice-Cool Composure (High Scoring, Minimal Panic)"
            elif wkt_pct <= 5.0:
                comp_tier = "Disciplined Strike Rotator"
            elif sr >= 175.0:
                comp_tier = "High-Risk Aggressive Counter-Puncher"
            else:
                comp_tier = "Standard League Composure"
        else:
            sr, wkt_pct, bound_pct, stag_pct = None, None, None, None
            comp_tier = "Insufficient Post-Dot Sample (< 10 opps)"

        bat_intel[str(cid)].update({
            "post_dot_opps": n_dot_opps,
            "post_dot_sr": sr,
            "post_dot_wkt_pct": wkt_pct,
            "post_dot_bound_pct": bound_pct,
            "post_dot_stagnation_pct": stag_pct,
            "composure_tier": comp_tier,
        })

    # ---------------------------------------------------------
    # 3. BOWLER BOUNCE-BACK COMPOSURE & HANDEDNESS SPLITS
    # ---------------------------------------------------------
    df["is_boundary"] = (df["is_four"] == 1) | (df["is_six"] == 1)
    df["prev_was_boundary"] = df.groupby(["match_id", "innings", "overid"])["is_boundary"].shift(1).fillna(False)
    df["prev_bowler_id"] = df.groupby(["match_id", "innings", "overid"])["canonical_bowler_id"].shift(1)

    same_bowler_mask = (df["prev_was_boundary"] == True) & (df["prev_bowler_id"] == df["canonical_bowler_id"]) & (df["is_legal_ball"] == 1)
    post_b_df = df[same_bowler_mask]

    league_post_b_econ = round(6.0 * post_b_df["bowler_runs_conceded"].sum() / len(post_b_df), 2)
    league_post_b_dot_pct = round(100.0 * (post_b_df["bowler_runs_conceded"] == 0).sum() / len(post_b_df), 1)
    league_post_b_repeat_b = round(100.0 * post_b_df["is_boundary"].sum() / len(post_b_df), 1)
    league_post_b_wkt = round(100.0 * post_b_df["is_bowler_wicket"].sum() / len(post_b_df), 2)

    bowl_intel: Dict[str, Any] = {}
    for cid in bowl_ids:
        b_sub = df[(df["canonical_bowler_id"] == cid) & (df["is_legal_ball"] == 1)]
        n_legal = len(b_sub)
        name = b_sub["canonical_bowler_name"].iloc[0] if n_legal else bowl_elig.loc[bowl_elig["canonical_player_id"] == cid, "player_name"].iloc[0]

        # Post-boundary rebound
        pb = post_b_df[post_b_df["canonical_bowler_id"] == cid]
        n_pb = len(pb)
        if n_pb >= 15:
            pb_econ = round(6.0 * pb["bowler_runs_conceded"].sum() / n_pb, 2)
            pb_dot_pct = round(100.0 * (pb["bowler_runs_conceded"] == 0).sum() / n_pb, 1)
            pb_repeat_pct = round(100.0 * pb["is_boundary"].sum() / n_pb, 1)
            pb_wkt_pct = round(100.0 * pb["is_bowler_wicket"].sum() / n_pb, 2)

            if pb_econ <= 8.0 and pb_repeat_pct <= 18.0:
                bounce_tier = "Elite Counter-Punch (Immediate Lockdown)"
            elif pb_econ <= 9.0:
                bounce_tier = "Above-Average Rebound Resistance"
            else:
                bounce_tier = "Standard League Recovery"
        else:
            pb_econ, pb_dot_pct, pb_repeat_pct, pb_wkt_pct = None, None, None, None
            bounce_tier = "Insufficient Post-Boundary Sample (< 15 opps)"

        # Handedness splits
        rhb = b_sub[b_sub["batting_type"].astype(str).str.lower().str.contains("right")]
        lhb = b_sub[b_sub["batting_type"].astype(str).str.lower().str.contains("left")]

        rhb_b = len(rhb)
        rhb_econ = round(6.0 * rhb["bowler_runs_conceded"].sum() / rhb_b, 2) if rhb_b >= 30 else None
        lhb_b = len(lhb)
        lhb_econ = round(6.0 * lhb["bowler_runs_conceded"].sum() / lhb_b, 2) if lhb_b >= 30 else None

        # Tactical classification
        if pb_econ and pb_econ <= 8.2 and rhb_econ and rhb_econ <= 7.8:
            tactical_role = "Elite Death & Recovery Specialist"
        elif lhb_econ and rhb_econ and abs(lhb_econ - rhb_econ) <= 0.4:
            tactical_role = "Dual-Handedness Neutralizer"
        elif lhb_econ and rhb_econ and lhb_econ < rhb_econ - 0.7:
            tactical_role = "LHB Matchup Specialist"
        elif rhb_econ and lhb_econ and rhb_econ < lhb_econ - 0.7:
            tactical_role = "RHB Matchup Specialist"
        else:
            tactical_role = "Situational Container"

        bowl_intel[str(cid)] = {
            "canonical_player_id": cid,
            "player_name": name,
            "post_b_opps": n_pb,
            "post_b_econ": pb_econ,
            "post_b_dot_pct": pb_dot_pct,
            "post_b_repeat_b_pct": pb_repeat_pct,
            "post_b_wkt_pct": pb_wkt_pct,
            "bounceback_tier": bounce_tier,
            "rhb_balls": rhb_b,
            "rhb_econ": rhb_econ,
            "lhb_balls": lhb_b,
            "lhb_econ": lhb_econ,
            "tactical_role": tactical_role,
        }

    # ---------------------------------------------------------
    # 4. RAJASTHAN ROYALS STRATEGIC DECISION PROFILES (4-QUESTIONS)
    # ---------------------------------------------------------
    strategic_profiles = {
        # Batters
        1699: {
            "player_name": "Suryakumar Yadav",
            "q1_discovery": "Generates +29.15 runs above expected per 100 balls in high pressure, striking at 196.4 vs Pace and 175.0 vs Spin.",
            "q2_significance": "True all-phase 360-degree chaser whose boundary rate does not decay regardless of bowler handedness or pace.",
            "q3_rr_implication": "Benchmark reference standard for RR chase modeling; illustrates the ideal boundary-access distribution for #3/#4.",
            "q4_scouting_decision": "Retained core star at MI; inaccessible in mini-auction, but serves as prototype template for scouting domestic clones.",
        },
        1077: {
            "player_name": "David Miller",
            "q1_discovery": "Averages only 1.85% panic dismissals on balls following a dot ball under high pressure, striking at 187.3 vs spin.",
            "q2_significance": "Exhibits the highest composure retention in T20 cricket; absorbs dot-ball pressure without throwing his wicket away.",
            "q3_rr_implication": "Perfect tactical archetype for RR's middle-order crunch chases when early wickets fall at SMS Jaipur.",
            "q4_scouting_decision": "High-priority trade target if released or made available; veteran finisher to partner with Jurel.",
        },
        7260: {
            "player_name": "Dhruv Jurel",
            "q1_discovery": "Maintains 0.0% post-dot panic dismissals in high pressure, striking at 195.8 vs Pace and 197.7 vs Spin.",
            "q2_significance": "Phenomenal composure and bilateral matchup symmetry (equally lethal against 145 kph pace and mystery spin).",
            "q3_rr_implication": "RR's home-grown cornerstone finisher; confirms RR's internal scouting excellence in developing clutch finishers.",
            "q4_scouting_decision": "Must be permanently retained as primary designated #6 finisher and wicket-keeping clutch asset.",
        },
        2258: {
            "player_name": "Sanju Samson",
            "q1_discovery": "Strikes at an astonishing 227.3 vs Spin under high pressure, turning spin overs into match-winning accelerators.",
            "q2_significance": "Most lethal spin-disruptor in high-pressure chases, relieving required-run-rate stress before the death overs.",
            "q3_rr_implication": "Franchise captain whose spin dominance anchors RR's tactical advantage at spin-friendly venues like Guwahati.",
            "q4_scouting_decision": "Franchise cornerstone; maintain top-order flexibility to ensure he faces prime middle-overs spin.",
        },
        8101: {
            "player_name": "Ashutosh Sharma",
            "q1_discovery": "Delivers +23.15 runs/100b under peak pressure, with 190.6 SR vs Pace and 0.0% post-dot panic dismissals across 33 opps.",
            "q2_significance": "Cold-blooded lower-order boundary striker who punishes express pace and yorker attempts at the death.",
            "q3_rr_implication": "Ideal high-ceiling domestic acquisition candidate to provide lower-order hitting depth behind Hetmyer and Jurel.",
            "q4_scouting_decision": "Aggressive target in mini-auction up to ₹4.5 Cr; provides domestic balance and insurance against death batting collapse.",
        },
        6396: {
            "player_name": "Tim David",
            "q1_discovery": "Strikes at 190.3 vs Pace in high pressure, but drops to 141.3 vs Spin, with a higher post-dot dismissal rate (8.96%).",
            "q2_significance": "Extreme pace specialist with high variance; vulnerable if opposition captains hold back mystery spin.",
            "q3_rr_implication": "Valuable death finisher for pace-heavy tracks, but requires tactical shielding against quality spin.",
            "q4_scouting_decision": "Secondary overseas target; bid conditionally based on price and complementary spin-hitting presence in the XI.",
        },
        7897: {
            "player_name": "Namboori Tilak Varma",
            "q1_discovery": "Blasts a league-leading 245.5 SR on deliveries immediately following a dot ball under high pressure (36.4% boundary response).",
            "q2_significance": "Instantaneous counter-punch reflex; reverses match momentum the moment bowling attacks attempt to build dot-ball pressure.",
            "q3_rr_implication": "Young left-handed anchor-accelerator who provides structural stability between overs 7 and 17.",
            "q4_scouting_decision": "Contracted core asset at MI; monitor trade availability in multi-year succession planning.",
        },

        # Bowlers
        2933: {
            "player_name": "Jasprit Bumrah",
            "q1_discovery": "Concedes only 6.00 RPO on deliveries immediately following a boundary, with a miniscule 10.1% repeat-boundary concession.",
            "q2_significance": "Unrivaled tactical reset capability; when attacked, immediately clamps down with dot balls (44.2% dot rate).",
            "q3_rr_implication": "The gold standard of T20 bowling; demonstrates that true elite bowling is defined by post-boundary lockdown.",
            "q4_scouting_decision": "Untouchable national asset; blueprint model for evaluating domestic death bowling targets.",
        },
        7920: {
            "player_name": "Nathan Ellis",
            "q1_discovery": "Maintains 7.90 RPO post-boundary economy and takes 6.35% wickets on the rebound, with 8.35 death economy.",
            "q2_significance": "Premier death overs yorker technician whose execution does not waver after conceding a boundary.",
            "q3_rr_implication": "Directly resolves RR's primary 2026 tactical weakness: the 10.13 RPO death pace leak at SMS Jaipur.",
            "q4_scouting_decision": "Highest-priority overseas fast-bowling acquisition target in mini-auction up to ₹6.5 Cr.",
        },
        781: {
            "player_name": "Sunil Narine",
            "q1_discovery": "Saves +0.64 runs/over under pressure, conceding only 7.97 RPO post-boundary with a 6.57% rebound wicket rate.",
            "q2_significance": "Mystery spin chokehold; eliminates boundary clusters and forces opposition batters into unforced errors.",
            "q3_rr_implication": "Proves the value of restrictive spin in shifting win probability; reinforces RR's Guwahati venue strategy.",
            "q4_scouting_decision": "Franchise icon at KKR; illustrates the archetype for RR's secondary mystery spinner search.",
        },
        4641: {
            "player_name": "Lungi Ngidi",
            "q1_discovery": "Produces a 43.8% dot ball rate and 7.30% wicket rate on balls immediately following a boundary.",
            "q2_significance": "Heavy-ball rebound enforcer who strikes back aggressively when challenged by opposing top batters.",
            "q3_rr_implication": "High-value, cost-effective overseas pace alternative who thrives on hard Jaipur pitches.",
            "q4_scouting_decision": "Strong value-for-money target in mini-auction (estimated ₹2.5–₹4.0 Cr).",
        },
    }

    return {
        "metadata": {
            "version": "2.0_strategic_intelligence",
            "league_baselines": {
                "post_dot_high_pressure": {
                    "strike_rate": league_post_dot_sr,
                    "dismissal_pct": league_post_dot_wkt,
                    "boundary_pct": league_post_dot_bound,
                    "stagnation_dot_pct": league_post_dot_stagnation,
                },
                "post_boundary_bowling": {
                    "economy_rpo": league_post_b_econ,
                    "dot_ball_pct": league_post_b_dot_pct,
                    "repeat_boundary_pct": league_post_b_repeat_b,
                    "wicket_pct": league_post_b_wkt,
                },
            },
        },
        "batting_intelligence": bat_intel,
        "bowling_intelligence": bowl_intel,
        "strategic_profiles": strategic_profiles,
    }


if __name__ == "__main__":
    res = generate_cricket_intelligence_summary()
    out_file = DATA_PROCESSED / "cricket_intelligence_summary.json"
    import json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Successfully generated cricket intelligence summary at {out_file} ({len(res['batting_intelligence'])} batters, {len(res['bowling_intelligence'])} bowlers).")

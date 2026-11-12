"""
Phase 4 Player Evaluation, Role Profiling, Recency Audit & RR Squad-Fit Engine.

Builds strictly on the corrected Phase 3 primary pool (IPL weight = 1.00,
Senior_T20I weight = 0.50, Emerging_Rising_T20 supplementary only) while
keeping all approved pressure formulas, OOF baselines, EB shrinkage, disjoint
pressure partitions, eligibility thresholds (Neff >= 30, IPL balls >= 30),
component weights, LCB penalty (lcb_z = 0.50), and PRI scoring unchanged.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.performance import (
    COMP_TIER_EMERGING,
    COMP_TIER_IPL,
    COMP_TIER_SENIOR_T20I,
    PerformanceConfig,
    classify_comp_tier,
    evaluate_batting_pressure,
    evaluate_bowling_pressure,
    filter_dls_pool,
    fit_predict_oof_baselines,
)

JAIPUR_VENUE_ID = 6
GUWAHATI_VENUE_ID = 1241

T20I_TOP8_TEAMS = {
    "India", "Australia", "England", "South Africa",
    "New Zealand", "Pakistan", "West Indies", "Sri Lanka",
}
T20I_OTHER_FM_TEAMS = {"Afghanistan", "Bangladesh", "Ireland", "Zimbabwe"}

# Verified IPL-retired players (cross-checked against IPL 2026 match logs & official announcements)
RETIRED_FROM_IPL: Dict[int, Dict[str, str]] = {
    143: {
        "name": "Dinesh Karthik",
        "reason": "Retired from IPL after IPL 2024 (0 deliveries in 2025-2026; last IPL match 2024-05-22).",
    },
    1318: {
        "name": "Andre Russell",
        "reason": "Retired from IPL ahead of IPL 2026 (0 IPL 2026 deliveries; joined KKR coaching staff).",
    },
    305: {
        "name": "Ravichandran Ashwin",
        "reason": "Retired from IPL in August 2025 after IPL 2025 (0 deliveries in 2026).",
    },
}

# Independently verified supporting auction & bio context (checked 2026-11-12; NEVER used as filters or weights)
PLAYER_AUCTION_CONTEXT: Dict[int, Dict[str, str]] = {
    # Batting Benchmark / Shortlist / Watchlist
    1699: {"age_2026": "36 (DOB 1990-09-14)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Mumbai Indians", "ipl_2026_price": "INR 16.35 Cr (Retained; Sportskeeda 2026-11-12)", "availability_note": "Active IPL 2026"},
    50284: {"age_2026": "15 (DOB 2011-03-27)", "nationality": "Indian", "cap_status": "Capped (India, 2026 debut)", "ipl_2026_team": "Rajasthan Royals", "ipl_2026_price": "INR 1.10 Cr (Retained; iplprice.com 2026-11-12)", "availability_note": "Active RR IPL 2026 starter (16/16 apps)"},
    1318: {"age_2026": "38 (DOB 1988-04-29)", "nationality": "West Indian", "cap_status": "Capped (WI) / IPL Retired", "ipl_2026_team": "None (Retired pre-2026)", "ipl_2026_price": "N/A (IPL Retired)", "availability_note": "Retired pre-IPL 2026"},
    143: {"age_2026": "41 (DOB 1985-06-01)", "nationality": "Indian", "cap_status": "Capped (India) / IPL Retired", "ipl_2026_team": "None (Retired post-2024)", "ipl_2026_price": "N/A (IPL Retired)", "availability_note": "Retired post-IPL 2024"},
    4847: {"age_2026": "27 (DOB 1999-04-22)", "nationality": "New Zealander", "cap_status": "Capped (NZ)", "ipl_2026_team": "Kolkata Knight Riders", "ipl_2026_price": "INR 2.00 Cr (Auction; mrcricketuae 2026-11-12)", "availability_note": "Active IPL 2026"},
    3125: {"age_2026": "31 (DOB 1995-10-02)", "nationality": "West Indian", "cap_status": "Capped (WI; T20I retired 2025)", "ipl_2026_team": "Lucknow Super Giants", "ipl_2026_price": "INR 21.00 Cr (Retained; iplsalary.com.in 2026-11-12)", "availability_note": "Active IPL 2026; retired from T20Is"},
    3647: {"age_2026": "31 (DOB 1994-12-06)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Punjab Kings", "ipl_2026_price": "INR 26.75 Cr (Retained; iplsalary.com.in 2026-11-12)", "availability_note": "Active IPL 2026 PBKS Captain"},
    4260: {"age_2026": "35 (DOB 1991-07-30)", "nationality": "South African", "cap_status": "Capped (SA)", "ipl_2026_team": "Sunrisers Hyderabad", "ipl_2026_price": "INR 23.00 Cr (Retained; cricroster.com 2026-11-12)", "availability_note": "Active IPL 2026"},
    6396: {"age_2026": "30 (DOB 1996-03-16)", "nationality": "Australian", "cap_status": "Capped (AUS)", "ipl_2026_team": "Royal Challengers Bengaluru", "ipl_2026_price": "INR 3.00 Cr (Retained; Sportskeeda 2026-11-12)", "availability_note": "Active IPL 2026"},
    5553: {"age_2026": "30 (DOB 1996-08-28)", "nationality": "English", "cap_status": "Capped (ENG)", "ipl_2026_team": "Royal Challengers Bengaluru", "ipl_2026_price": "INR 11.50 Cr (Retained; InsideSport 2026-11-12)", "availability_note": "Active IPL 2026"},
    6931: {"age_2026": "28 (DOB 1998-05-18)", "nationality": "Sri Lankan", "cap_status": "Capped (SL)", "ipl_2026_team": "Delhi Capitals", "ipl_2026_price": "INR 4.00 Cr (Auction; Times of India 2026-11-12)", "availability_note": "Active IPL 2026"},
    5039: {"age_2026": "33 (DOB 1993-06-01)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Royal Challengers Bengaluru", "ipl_2026_price": "INR 11.00 Cr (Retained; NDTV 2026-11-12)", "availability_note": "Active IPL 2026 RCB Captain"},
    2756: {"age_2026": "34 (DOB 1991-11-21)", "nationality": "Indian", "cap_status": "Uncapped (India)", "ipl_2026_team": "Punjab Kings", "ipl_2026_price": "INR 5.50 Cr (Retained; InsideSport 2026-11-12)", "availability_note": "Active IPL 2026"},
    25669: {"age_2026": "25 (DOB 2001-01-18)", "nationality": "Indian", "cap_status": "Uncapped (India)", "ipl_2026_team": "Punjab Kings", "ipl_2026_price": "INR 3.80 Cr (Retained; CricBoost 2026-11-12)", "availability_note": "Active IPL 2026"},
    2258: {"age_2026": "32 (DOB 1994-11-11)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Chennai Super Kings", "ipl_2026_price": "INR 18.00 Cr (Traded RR->CSK; Times of India 2026-11-12)", "availability_note": "Active IPL 2026"},
    37052: {"age_2026": "26 (DOB 1999-12-31)", "nationality": "Indian", "cap_status": "Capped (India, 2026 debut)", "ipl_2026_team": "Mumbai Indians", "ipl_2026_price": "INR 5.25 Cr (Retained; InsideSport 2026-11-12)", "availability_note": "Active IPL 2026"},
    6281: {"age_2026": "26 (DOB 2000-09-04)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Sunrisers Hyderabad", "ipl_2026_price": "INR 14.00 Cr (Retained; iplsalary.com.in 2026-11-12)", "availability_note": "Active IPL 2026"},
    # Bowling Benchmark / Shortlist / Watchlist
    2933: {"age_2026": "32 (DOB 1993-12-06)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Mumbai Indians", "ipl_2026_price": "INR 18.00 Cr (Retained; Sportskeeda 2026-11-12)", "availability_note": "Active IPL 2026"},
    781: {"age_2026": "38 (DOB 1988-05-26)", "nationality": "West Indian", "cap_status": "Capped (WI; T20I retired 2023)", "ipl_2026_team": "Kolkata Knight Riders", "ipl_2026_price": "INR 12.00 Cr (Retained; CricketArmy 2026-11-12)", "availability_note": "Active IPL 2026; T20I retired"},
    6256: {"age_2026": "35 (DOB 1991-08-29)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Kolkata Knight Riders", "ipl_2026_price": "INR 12.00 Cr (Retained; NDTV 2026-11-12)", "availability_note": "Active IPL 2026"},
    4722: {"age_2026": "28 (DOB 1998-09-20)", "nationality": "Afghan", "cap_status": "Capped (AFG)", "ipl_2026_team": "Gujarat Titans", "ipl_2026_price": "INR 18.00 Cr (Retained; iplsalary.com.in 2026-11-12)", "availability_note": "Active IPL 2026"},
    2876: {"age_2026": "32 (DOB 1994-01-20)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Delhi Capitals", "ipl_2026_price": "INR 16.50 Cr (Retained; InsideSport 2026-11-12)", "availability_note": "Active IPL 2026"},
    360: {"age_2026": "36 (DOB 1990-02-05)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Royal Challengers Bengaluru", "ipl_2026_price": "INR 10.75 Cr (Retained; Sportskeeda 2026-11-12)", "availability_note": "Active IPL 2026"},
    7920: {"age_2026": "32 (DOB 1994-09-22)", "nationality": "Australian", "cap_status": "Capped (AUS)", "ipl_2026_team": "CSK squad (0 played; traded to SRH pre-2027)", "ipl_2026_price": "Retained CSK pre-2026; missed IPL 2026 (iplt20.com 2026-11-12)", "availability_note": "Ruled out of IPL 2026 with hamstring injury (Spencer Johnson replaced him); verify fitness"},
    830: {"age_2026": "35 (DOB 1991-01-08)", "nationality": "Australian", "cap_status": "Capped (AUS)", "ipl_2026_team": "Royal Challengers Bengaluru", "ipl_2026_price": "INR 12.50 Cr (Retained; Sportskeeda 2026-11-12)", "availability_note": "Active IPL 2026"},
    479: {"age_2026": "38 (DOB 1988-02-17)", "nationality": "English", "cap_status": "Capped (ENG)", "ipl_2026_team": "None (Unsold IPL 2026; last 2023 SRH)", "ipl_2026_price": "Unsold in recent IPL auctions (Wikipedia 2026-11-12)", "availability_note": "Active T20I; zero IPL since 2023"},
    4900: {"age_2026": "29 (DOB 1997-07-29)", "nationality": "Sri Lankan", "cap_status": "Capped (SL)", "ipl_2026_team": "LSG squad (0 played; 2025 RR)", "ipl_2026_price": "INR 2.00 Cr (LSG IPL 2026 auction; Wikipedia 2026-11-12)", "availability_note": "Released by RR pre-2026; ruled out of IPL 2026 (hamstring tear at T20 WC; HT 2026-11-12)"},
    2636: {"age_2026": "34 (DOB 1992-03-31)", "nationality": "Australian", "cap_status": "Capped (AUS)", "ipl_2026_team": "None (Withdrew/opted out IPL 2026)", "ipl_2026_price": "INR 2.40 Cr (2025 SRH); opted out 2026 (Livemint 2026-11-12)", "availability_note": "Skipped IPL 2026 (played PSL)"},
    3644: {"age_2026": "31 (DOB 1994-12-14)", "nationality": "Indian", "cap_status": "Capped (India)", "ipl_2026_team": "Delhi Capitals", "ipl_2026_price": "INR 13.25 Cr (Retained; InsideSport 2026-11-12)", "availability_note": "Active IPL 2026"},
    4101: {"age_2026": "34 (DOB 1992-02-05)", "nationality": "New Zealander", "cap_status": "Capped (NZ)", "ipl_2026_team": "Mumbai Indians", "ipl_2026_price": "INR 2.00 Cr (Retained; InsideSport 2026-11-12)", "availability_note": "Active IPL 2026"},
    305: {"age_2026": "40 (DOB 1986-09-17)", "nationality": "Indian", "cap_status": "Capped (India) / IPL Retired", "ipl_2026_team": "None (Retired Aug 2025)", "ipl_2026_price": "N/A (IPL Retired)", "availability_note": "Retired Aug 2025"},
    35318: {"age_2026": "21 (DOB 2004-12-14)", "nationality": "Indian", "cap_status": "Uncapped (India)", "ipl_2026_team": "Sunrisers Hyderabad", "ipl_2026_price": "INR 30 Lakh (Auction; iplprice.com 2026-11-12)", "availability_note": "Active IPL 2026"},
    7967: {"age_2026": "21 (DOB 2005-01-03)", "nationality": "Afghan", "cap_status": "Capped (AFG)", "ipl_2026_team": "Chennai Super Kings", "ipl_2026_price": "INR 10.00 Cr (Retained; iplsalary.com.in 2026-11-12)", "availability_note": "Active IPL 2026"},
}


def deduplicate_playing_eleven(playing_eleven_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministically resolves duplicate (match_id, team_id, player_id) rows in playing_eleven.csv
    (observed on 35 match-team-player keys in raw data) by preserving earliest batting_order and
    taking logical OR (max) across played/wk/captain/impact boolean indicators.
    """
    pe = playing_eleven_df.copy()
    for col in ("is_played", "is_wk", "is_captain", "is_impact", "is_impact_off"):
        if col in pe.columns:
            pe[col] = pe[col].fillna(False).astype(int)
        else:
            pe[col] = 0
    pe = pe.sort_values(
        ["match_id", "team_id", "player_id", "batting_order"],
        na_position="last",
        kind="mergesort",
    )
    dedup = (
        pe.groupby(["match_id", "team_id", "player_id"], as_index=False)
        .agg(
            batting_order=("batting_order", "min"),
            is_played=("is_played", "max"),
            is_wk=("is_wk", "max"),
            is_captain=("is_captain", "max"),
            is_impact=("is_impact", "max"),
            is_impact_off=("is_impact_off", "max"),
        )
    )
    for col in ("is_played", "is_wk", "is_captain"):
        dedup[col] = dedup[col].astype(bool)
    return dedup


def _safe_z_against_ref(series: pd.Series, ref_mask: pd.Series) -> pd.Series:
    sub = series[ref_mask]
    mu = sub.mean() if len(sub) > 1 else series.mean()
    sd = sub.std(ddof=1) if len(sub) > 1 else series.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-9:
        sd = 1.0
    return (series - mu) / sd


def build_canonical_id_map(pdir: pd.DataFrame) -> Dict[int, int]:
    """Maps every raw player_id in canonical_player_directory.csv to its canonical_player_id."""
    id_map: Dict[int, int] = {}
    for _, r in pdir.iterrows():
        cid = int(r["canonical_player_id"])
        for tok in str(r["original_player_ids"]).split(";"):
            tok = tok.strip()
            if tok.isdigit():
                id_map[int(tok)] = cid
    return id_map


def select_shortlists_and_benchmarks(
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    top_n: int = 10,
    watchlist_n: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Produces BOTH:
    - Unrestricted statistical PRI_lcb rankings (official Top 10 + Top 15 benchmarks) with empirical percentiles
    - Operational IPL-active Top 10 shortlists + Watchlist #11-15 (moving verified IPL-retired players out cleanly).
    """
    out: Dict[str, pd.DataFrame] = {}
    for key, df in (("bat", bat_df), ("bowl", bowl_df)):
        df_copy = df.copy()
        broad_mask = df_copy["elig_broad"] if "elig_broad" in df_copy.columns else df_copy["elig_primary"]
        broad_scores = df_copy.loc[broad_mask, "pri_score_lcb"].dropna().values

        elig = df_copy[df_copy["elig_primary"]].sort_values("pri_score_lcb", ascending=False).copy()
        n_prim = len(elig)
        elig["raw_pri_lcb_rank"] = np.arange(1, n_prim + 1)
        elig["raw_pri_post_rank"] = elig["pri_score_post"].rank(ascending=False, method="min").astype(int)
        # Empirical percentile within primary-eligible cohort: 100 = highest rank
        elig["emp_pct_elig_primary"] = 100.0 * (n_prim - elig["raw_pri_lcb_rank"] + 1) / max(1, n_prim)
        elig["emp_pct_elig_broad"] = [
            float(100.0 * np.mean(broad_scores <= s)) if len(broad_scores) > 0 else np.nan
            for s in elig["pri_score_lcb"].values
        ]
        elig["is_ipl_retired"] = elig["canonical_player_id"].astype(int).isin(RETIRED_FROM_IPL)

        unfiltered_top10 = elig.head(top_n).copy()
        unfiltered_top10["shortlist_rank"] = unfiltered_top10["raw_pri_lcb_rank"]
        benchmark = elig.head(15).copy()

        active = elig[~elig["is_ipl_retired"]].copy()
        active["shortlist_rank"] = np.arange(1, len(active) + 1)

        out[f"{key}_elig_all"] = elig
        out[f"{key}_unfiltered_top10"] = unfiltered_top10
        out[f"{key}_benchmark_top15"] = benchmark
        out[f"{key}_shortlist"] = active.head(top_n).copy()
        out[f"{key}_watchlist"] = active.iloc[top_n : top_n + watchlist_n].copy()
    return out


def compute_rr_2026_context(
    pri_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    playing_eleven_df: pd.DataFrame,
    pdir: pd.DataFrame,
) -> Dict:
    """
    Extracts Rajasthan Royals' verified IPL 2026 roster (with deterministic playing_eleven
    deduplication + Impact Player counts), phase baselines vs league, and Jaipur/Guwahati
    2023-2026 venue scoring profiles.
    """
    id_map = build_canonical_id_map(pdir)
    name_map = dict(zip(pdir["canonical_player_id"].astype(int), pdir["canonical_name"]))
    rr_tid = int(teams_df.loc[teams_df["team_name"] == "Rajasthan Royals", "team_id"].iloc[0])

    pe_clean = deduplicate_playing_eleven(playing_eleven_df)
    m = matches_df.merge(competitions_df[["comp_id", "comp_name", "season"]], on="comp_id", how="left")
    m["season_yr"] = pd.to_numeric(m["season"].astype(str).str[:4], errors="coerce").fillna(0).astype(int)

    rr_m26 = m[
        (m["comp_name"] == "IPL 2026")
        & ((m["home_team"] == "Rajasthan Royals") | (m["away_team"] == "Rajasthan Royals"))
    ].copy()
    rr_wins = int((rr_m26["win_team_id"] == rr_tid).sum())
    rr_losses = int((rr_m26["win_team_id"].notna() & (rr_m26["win_team_id"] != rr_tid)).sum())
    guw_wins = int(((rr_m26["venue_id"] == GUWAHATI_VENUE_ID) & (rr_m26["win_team_id"] == rr_tid)).sum())
    guw_total = int((rr_m26["venue_id"] == GUWAHATI_VENUE_ID).sum())
    jai_wins = int(((rr_m26["venue_id"] == JAIPUR_VENUE_ID) & (rr_m26["win_team_id"] == rr_tid)).sum())
    jai_total = int((rr_m26["venue_id"] == JAIPUR_VENUE_ID).sum())

    pe_rr = pe_clean.merge(rr_m26[["match_id", "stage"]], on="match_id", how="inner")
    pe_rr = pe_rr[pe_rr["team_id"] == rr_tid].copy()
    pe_rr["canonical_player_id"] = pe_rr["player_id"].map(id_map).fillna(pe_rr["player_id"]).astype(int)
    pe_rr["canonical_name"] = pe_rr["canonical_player_id"].map(name_map)

    roster = (
        pe_rr.groupby(["canonical_player_id", "canonical_name"], as_index=False)
        .agg(
            squad_sheets=("match_id", "nunique"),
            apps_2026=("is_played", "sum"),
            wk_apps_2026=("is_wk", "sum"),
            capt_apps_2026=("is_captain", "sum"),
            impact_in_2026=("is_impact", "sum"),
            impact_out_2026=("is_impact_off", "sum"),
            med_bat_order=("batting_order", "median"),
        )
        .sort_values(["apps_2026", "squad_sheets"], ascending=False)
    )

    ipl26 = pri_df[(pri_df["comp_tier"] == COMP_TIER_IPL) & (pri_df["season_yr"] == 2026)].copy()
    rr_bat26 = ipl26[(ipl26["batting_team_id"] == rr_tid) & (ipl26["is_batter_ball_faced"])]
    lg_bat26 = ipl26[ipl26["is_batter_ball_faced"]]
    rr_bowl26 = ipl26[ipl26["bowling_team_id"] == rr_tid]

    bat_phases = []
    bowl_phases = []
    for ph in ("Powerplay", "Middle", "Death"):
        rb = rr_bat26[rr_bat26["match_phase"] == ph]
        lb = lg_bat26[lg_bat26["match_phase"] == ph]
        rb_w = (rb["is_wicket"].fillna(False) & (rb["canonical_dismissed_player_id"] == rb["canonical_striker_id"])).mean()
        lb_w = (lb["is_wicket"].fillna(False) & (lb["canonical_dismissed_player_id"] == lb["canonical_striker_id"])).mean()
        bat_phases.append({
            "phase": ph,
            "rr_balls": int(len(rb)),
            "rr_sr": float(100.0 * rb["runs"].mean()),
            "lg_sr": float(100.0 * lb["runs"].mean()),
            "rr_wkt_pct": float(100.0 * rb_w),
            "lg_wkt_pct": float(100.0 * lb_w),
        })

        rw = rr_bowl26[rr_bowl26["match_phase"] == ph]
        lw = ipl26[ipl26["match_phase"] == ph]
        rw_leg = max(1, int(rw["is_legal_ball"].sum()))
        lw_leg = max(1, int(lw["is_legal_ball"].sum()))
        bowl_phases.append({
            "phase": ph,
            "rr_legal_balls": int(rw_leg),
            "rr_econ": float(6.0 * rw["bowler_runs_conceded"].sum() / rw_leg),
            "lg_econ": float(6.0 * lw["bowler_runs_conceded"].sum() / lw_leg),
            "rr_sr": float(rw_leg / max(1, int(rw["is_bowler_wicket"].sum()))),
            "lg_sr": float(lw_leg / max(1, int(lw["is_bowler_wicket"].sum()))),
        })

    pace_spin = []
    for ps in ("Pace", "Spin"):
        rw = rr_bowl26[rr_bowl26["pace_or_spin"] == ps]
        lw = ipl26[ipl26["pace_or_spin"] == ps]
        rw_leg = max(1, int(rw["is_legal_ball"].sum()))
        lw_leg = max(1, int(lw["is_legal_ball"].sum()))
        pace_spin.append({
            "type": ps,
            "rr_legal_balls": int(rw_leg),
            "rr_wkts": int(rw["is_bowler_wicket"].sum()),
            "rr_econ": float(6.0 * rw["bowler_runs_conceded"].sum() / rw_leg),
            "lg_econ": float(6.0 * lw["bowler_runs_conceded"].sum() / lw_leg),
        })

    venue_conditions = compute_rr_jaipur_guwahati_conditions(pri_df, matches_df)

    return {
        "rr_team_id": rr_tid,
        "rr_2026_matches": int(len(rr_m26)),
        "rr_2026_record": f"{rr_wins}-{rr_losses} (Guwahati {guw_wins}/{guw_total}, Jaipur {jai_wins}/{jai_total})",
        "rr_2026_roster": roster,
        "rr_2026_played_ids": set(roster.loc[roster["apps_2026"] > 0, "canonical_player_id"].astype(int)),
        "bat_phases_2026": bat_phases,
        "bowl_phases_2026": bowl_phases,
        "bowl_pace_spin_2026": pace_spin,
        "venue_conditions_2023_2026": venue_conditions,
    }


def compute_rr_jaipur_guwahati_conditions(
    pri_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> List[Dict]:
    """
    Computes verified IPL 2023-2026 primary-pool scoring, phase, pace/spin, boundary,
    and chase profiles for Jaipur (venue_id=6), Guwahati (venue_id=1241), and All IPL.
    """
    ipl_pri = pri_df[(pri_df["comp_tier"] == COMP_TIER_IPL) & (pri_df["season_yr"] >= 2023)].copy()
    if "venue_id" not in ipl_pri.columns:
        ipl_pri = ipl_pri.merge(
            matches_df[["match_id", "venue_id", "win_team_id", "home_team_id", "away_team_id", "toss_win_team_id", "toss_opted"]],
            on="match_id",
            how="left",
        )

    out: List[Dict] = []
    for label, vid in (("All IPL (2023-26)", None), ("Jaipur (SMS, vid=6)", JAIPUR_VENUE_ID), ("Guwahati (ACA, vid=1241)", GUWAHATI_VENUE_ID)):
        sub = ipl_pri if vid is None else ipl_pri[ipl_pri["venue_id"] == vid]
        if len(sub) == 0:
            continue
        leg = max(1, int(sub["is_legal_ball"].sum()))
        inn1 = sub[sub["innings"] == 1].groupby("match_id")["ball_total_runs"].sum()
        mids = sub["match_id"].unique()
        m_sub = matches_df[matches_df["match_id"].isin(mids)].copy()
        bat_first_id = m_sub.apply(
            lambda r: r["home_team_id"]
            if (
                (r["toss_win_team_id"] == r["home_team_id"] and str(r["toss_opted"]).lower() == "bat")
                or (r["toss_win_team_id"] == r["away_team_id"] and str(r["toss_opted"]).lower() == "field")
            )
            else r["away_team_id"],
            axis=1,
        )
        chase_win_pct = float(100.0 * (m_sub["win_team_id"].notna() & (m_sub["win_team_id"] != bat_first_id)).mean())

        pp = sub[sub["match_phase"] == "Powerplay"]
        mid = sub[sub["match_phase"] == "Middle"]
        dth = sub[sub["match_phase"] == "Death"]
        pace = sub[sub["pace_or_spin"] == "Pace"]
        spin = sub[sub["pace_or_spin"] == "Spin"]
        bf = sub[sub["is_batter_ball_faced"]]
        out.append({
            "venue_label": label,
            "matches": int(len(mids)),
            "inn1_avg_runs": float(inn1.mean()) if len(inn1) > 0 else np.nan,
            "overall_rpo": float(6.0 * sub["ball_total_runs"].sum() / leg),
            "pp_rpo": float(6.0 * pp["ball_total_runs"].sum() / max(1, int(pp["is_legal_ball"].sum()))),
            "mid_rpo": float(6.0 * mid["ball_total_runs"].sum() / max(1, int(mid["is_legal_ball"].sum()))),
            "dth_rpo": float(6.0 * dth["ball_total_runs"].sum() / max(1, int(dth["is_legal_ball"].sum()))),
            "pace_econ": float(6.0 * pace["bowler_runs_conceded"].sum() / max(1, int(pace["is_legal_ball"].sum()))),
            "spin_econ": float(6.0 * spin["bowler_runs_conceded"].sum() / max(1, int(spin["is_legal_ball"].sum()))),
            "spin_share_pct": float(100.0 * spin["is_legal_ball"].sum() / leg),
            "boundary_pct": float(100.0 * (bf["is_four"].fillna(False) | bf["is_six"].fillna(False)).mean()),
            "chase_win_pct": chase_win_pct,
        })
    return out


def _classify_all_rounder(
    other_row: Optional[pd.Series],
    primary_discipline: str,
) -> Tuple[str, Dict]:
    """Determines whether a player offers verified two-way contributions across both Phase 3 tables."""
    if other_row is None:
        return "Specialist", {
            "other_ipl_balls": 0,
            "other_t20i_balls": 0,
            "other_neff": 0.0,
            "other_elig_primary": False,
            "other_pri_lcb": np.nan,
            "other_ev_all_lcb": np.nan,
        }

    ball_key = "ipl_legal_balls" if primary_discipline == "bat" else "ipl_balls"
    t20i_key = "t20i_legal_balls" if primary_discipline == "bat" else "t20i_balls"
    o_ipl = int(other_row[ball_key])
    o_t20i = int(other_row[t20i_key])
    o_neff = float(other_row["neff"])
    o_elig = bool(other_row["elig_primary"])
    o_pri = float(other_row["pri_score_lcb"])
    o_ev = float(other_row["ev_all_lcb"])

    if o_elig and o_neff >= 150.0 and o_pri >= 45.0:
        label = "Genuine All-Rounder"
    elif o_elig or (o_neff >= 100.0 and o_ipl >= 25):
        label = "Secondary / Part-Time Two-Way"
    else:
        label = "Specialist"

    return label, {
        "other_ipl_balls": o_ipl,
        "other_t20i_balls": o_t20i,
        "other_neff": o_neff,
        "other_elig_primary": o_elig,
        "other_pri_lcb": o_pri,
        "other_ev_all_lcb": o_ev,
    }


def _assign_confidence_and_flags(
    row: pd.Series,
    discipline: str,
    ipl_25_26: int,
    hp_count: int,
    on_rr_2026: bool,
) -> Tuple[str, str]:
    """
    Assigns objective 3-level evidence labels ('Strong evidence', 'Moderate evidence',
    'Limited evidence') alongside explicit risk flags including NEFF_HP_LT_15.
    """
    neff = float(row["neff"])
    neff_hp = float(row["neff_hp"])
    ipl_b = int(row["ipl_balls"] if discipline == "bat" else row["ipl_legal_balls"])
    t20i_b = int(row["t20i_balls"] if discipline == "bat" else row["t20i_legal_balls"])
    main_b_shrink = float(row["raa_B"] if discipline == "bat" else row["rsaa_B"])
    hp_post = float(row["hp_raa_post"] if discipline == "bat" else row["hp_impact_post"])

    flags: List[str] = []
    if hp_count == 0 or neff_hp < 1.0:
        flags.append("NEFF_HP_LT_15")
        flags.append("ZERO_HP_EVIDENCE")
    elif neff_hp < 15.0:
        flags.append("NEFF_HP_LT_15")
    elif neff_hp < 35.0:
        flags.append("LOW_HP_DEPTH")
    if hp_count > 0 and hp_post < -0.50:
        flags.append("NEGATIVE_HP_OFFSET_BY_GENERAL")

    if ipl_b < 100:
        flags.append("VERY_LOW_IPL_EXPOSURE")
    elif t20i_b > 3.0 * max(1, ipl_b):
        flags.append("T20I_HEAVY_PROFILE")
    elif t20i_b == 0:
        flags.append("IPL_ONLY_SAMPLE")

    if ipl_25_26 == 0:
        flags.append("NO_RECENT_IPL_25_26")
    elif ipl_25_26 < 60:
        flags.append("LOW_RECENT_IPL_25_26")

    if main_b_shrink < 0.45:
        flags.append("MODERATE_HIGH_SHRINKAGE")

    if on_rr_2026:
        flags.append("ON_RR_IPL_2026_SQUAD")

    if (
        neff >= 500.0
        and ipl_b >= 300
        and neff_hp >= 40.0
        and ipl_25_26 >= 100
        and main_b_shrink >= 0.65
    ):
        tier = "Strong evidence"
    elif (
        neff >= 250.0
        and ipl_b >= 150
        and ipl_25_26 >= 60
        and (neff_hp >= 15.0 or ipl_25_26 >= 150)
    ):
        tier = "Moderate evidence"
    else:
        tier = "Limited evidence"

    return tier, ", ".join(flags) if flags else "CLEAN"


def _assign_tactical_role(discipline: str, role_info: Dict, ph_info: Dict) -> str:
    """Assigns a descriptive T20 cricketing tactical role from verified phase & technical metrics."""
    if discipline == "bat":
        order = role_info.get("med_bat_order", np.nan)
        pp_s = ph_info.get("powerplay_share_pct", 0.0)
        mid_s = ph_info.get("middle_share_pct", 0.0)
        dth_s = ph_info.get("death_share_pct", 0.0)
        sr = role_info.get("career_sr", 0.0)
        if pd.notna(order) and order <= 2.5 and pp_s >= 50.0:
            return "Powerplay Aggressor (Opener)" if sr >= 150.0 else "Top-Order Accumulator (Opener)"
        if pd.notna(order) and order >= 5.5 and dth_s >= 30.0:
            return "Death-Over Finisher"
        if mid_s >= 50.0 and ph_info.get("middle_sr", 0.0) >= 155.0 and ph_info.get("death_sr", 0.0) >= 190.0:
            return "Middle-Over & Death Enforcer"
        return "Middle-Order Anchor / Accelerator"

    ps = str(role_info.get("pace_or_spin", "?"))
    sub = str(role_info.get("bowling_subtype", ""))
    pp_s = ph_info.get("powerplay_share_pct", 0.0)
    mid_s = ph_info.get("middle_share_pct", 0.0)
    dth_s = ph_info.get("death_share_pct", 0.0)
    spd = role_info.get("mean_speed_kph", np.nan)
    if ps == "Pace":
        if pp_s >= 40.0 and dth_s >= 28.0:
            return "All-Phase Pace Spearhead" if (pd.notna(spd) and spd >= 135.0) else "Powerplay Swing & Death Seamer"
        if dth_s >= 35.0:
            return "Death-Over Pace Specialist"
        if pp_s >= 45.0:
            return "Powerplay Swing / Hit-the-Deck Pacer"
        return "Multi-Phase Seam Bowler"
    if "Leg-Break" in sub or "Chinaman" in sub or "LAC" in sub:
        if "6256" in str(role_info.get("_pid", "")):
            return "Middle-Over Mystery Wrist-Spinner"
        return "Attacking Wrist-Spinner"
    if "orthodox" in sub.lower() or "SLAO" in sub:
        return "Left-Arm Orthodox Containment Spinner"
    if dth_s >= 12.0 and mid_s >= 65.0:
        return "All-Phase Finger/Mystery Spinner"
    return "Middle-Over Spin Controller"


def build_detailed_player_profiles(
    selected_df: pd.DataFrame,
    full_disc_df: pd.DataFrame,
    other_disc_df: pd.DataFrame,
    oof_pri: pd.DataFrame,
    raw_pri_enriched: pd.DataFrame,
    playing_eleven_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    pdir: pd.DataFrame,
    emg_supp_df: pd.DataFrame,
    rr_played_ids_2026: set,
    discipline: str,
    config: PerformanceConfig = PerformanceConfig(),
) -> pd.DataFrame:
    """
    Constructs comprehensive evaluation profiles for shortlisted / benchmark / watchlist players.
    Computes pillar contributions, empirical percentiles, phase/role/HP splits, recency trends
    (including IPL vs T20I recency & role shifts), all-rounder cross-checks, Impact Player apps,
    Strong/Moderate/Limited evidence labels, and separate Emerging supplementary metadata.
    """
    id_map = build_canonical_id_map(pdir)
    pe_clean = deduplicate_playing_eleven(playing_eleven_df)
    pe_clean["canonical_player_id"] = pe_clean["player_id"].map(id_map).fillna(pe_clean["player_id"]).astype(int)
    # Collapse rare edge case where multiple raw IDs map to the same canonical ID within a match
    pe_canon = (
        pe_clean.groupby(["match_id", "canonical_player_id"], as_index=False)
        .agg(
            is_played=("is_played", "max"),
            is_wk=("is_wk", "max"),
            is_captain=("is_captain", "max"),
            is_impact=("is_impact", "max"),
            is_impact_off=("is_impact_off", "max"),
        )
    )
    pe_played = pe_canon[pe_canon["is_played"]]
    wk_counts = pe_played.groupby("canonical_player_id")["is_wk"].sum().to_dict()
    capt_counts = pe_played.groupby("canonical_player_id")["is_captain"].sum().to_dict()
    imp_in_counts = pe_played.groupby("canonical_player_id")["is_impact"].sum().to_dict()
    imp_out_counts = pe_played.groupby("canonical_player_id")["is_impact_off"].sum().to_dict()

    team_map = teams_df.groupby("team_id")["team_name"].first().to_dict()
    ref_mask = full_disc_df["elig_broad"]

    if discipline == "bat":
        z_cols = [("raa_lcb", config.bat_w_raa, "c_raa"), ("surv_lcb", config.bat_w_surv, "c_surv"),
                  ("cons_lcb", config.bat_w_cons, "c_cons"), ("hp_raa_lcb", config.bat_w_hp, "c_hp")]
        id_col = "canonical_striker_id"
        team_col = "batting_team_id"
    else:
        z_cols = [("rsaa_lcb", config.bowl_w_rsaa, "c_rsaa"), ("waa_lcb", config.bowl_w_waa, "c_waa"),
                  ("cons_lcb", config.bowl_w_cons, "c_cons"), ("hp_impact_lcb", config.bowl_w_hp, "c_hp")]
        id_col = "canonical_bowler_id"
        team_col = "bowling_team_id"

    contrib_df = pd.DataFrame({"canonical_player_id": full_disc_df["canonical_player_id"]})
    for col, weight, out_c in z_cols:
        contrib_df[out_c] = weight * _safe_z_against_ref(full_disc_df[col], ref_mask)

    other_lookup = other_disc_df.set_index("canonical_player_id")
    emg_lookup = emg_supp_df.set_index("canonical_player_id") if len(emg_supp_df) > 0 else pd.DataFrame()
    oof_sub_all = oof_pri[oof_pri["is_batter_ball_faced"]] if discipline == "bat" else oof_pri

    def _weighted_ev(slice_df: pd.DataFrame) -> float:
        if len(slice_df) == 0:
            return np.nan
        w_comp = np.where(slice_df["comp_tier"] == COMP_TIER_IPL, 1.0, config.w_t20i)
        p_col = "pressure_bat" if discipline == "bat" else "pressure_bowl"
        w = w_comp * ((slice_df[p_col].values / 100.0) ** config.pressure_gamma)
        if discipline == "bat":
            s_out = (slice_df["is_wicket"].fillna(False) & (slice_df["canonical_dismissed_player_id"] == slice_df["canonical_striker_id"])).astype(float)
            d_net = 100.0 * (slice_df["runs"] - slice_df["bat_exp_runs"]) + config.wicket_run_value * 100.0 * (slice_df["bat_exp_wkt"] - s_out)
            return float(np.average(d_net, weights=w)) if w.sum() > 0 else np.nan
        leg_f = slice_df["is_legal_ball"].astype(float)
        wl = w * leg_f
        d_net_po = 6.0 * (leg_f * slice_df["bowl_exp_runs"] - slice_df["bowler_runs_conceded"]) + (config.wicket_run_value / 4.0) * 24.0 * (slice_df["is_bowler_wicket"].fillna(False).astype(float) - leg_f * slice_df["bowl_exp_wkt"])
        return float((w * d_net_po).sum() / wl.sum()) if wl.sum() > 0 else np.nan

    rows = []
    for _, r in selected_df.iterrows():
        pid = int(r["canonical_player_id"])
        sub = raw_pri_enriched[raw_pri_enriched[id_col] == pid]
        oof_sub = oof_sub_all[oof_sub_all[id_col] == pid]
        if discipline == "bat":
            sub = sub[sub["is_batter_ball_faced"]]
        sub_leg = sub[sub["is_legal_ball"]] if discipline == "bowl" else sub

        p_col = "pressure_bat" if discipline == "bat" else "pressure_bowl"
        n_inn = int(sub_leg[["match_id", "innings"]].drop_duplicates().shape[0])
        hp_mask = sub_leg[p_col] >= 60.0
        n_hp_opps = int(sub_leg.loc[hp_mask, ["match_id", "innings"]].drop_duplicates().shape[0])

        sub_ipl = sub_leg[sub_leg["comp_tier"] == COMP_TIER_IPL]
        sub_t20i = sub_leg[sub_leg["comp_tier"] == COMP_TIER_SENIOR_T20I]
        rec_mask = sub_leg["season_yr"] >= 2025
        rec_ipl = int(((sub_leg["season_yr"] >= 2025) & (sub_leg["comp_tier"] == COMP_TIER_IPL)).sum())
        rec_25_ipl = int(((sub_leg["season_yr"] == 2025) & (sub_leg["comp_tier"] == COMP_TIER_IPL)).sum())
        rec_26_ipl = int(((sub_leg["season_yr"] == 2026) & (sub_leg["comp_tier"] == COMP_TIER_IPL)).sum())
        rec_t20i = int(((sub_leg["season_yr"] >= 2025) & (sub_leg["comp_tier"] == COMP_TIER_SENIOR_T20I)).sum())
        last_ipl_yr = int(sub_ipl["season_yr"].max()) if len(sub_ipl) > 0 else 0
        last_ipl_date = str(sub_ipl["match_date"].max()) if len(sub_ipl) > 0 else "None"
        last_ipl_team = (
            team_map.get(int(sub_ipl[sub_ipl["season_yr"] == last_ipl_yr][team_col].mode().iloc[0]), "Unknown")
            if len(sub_ipl) > 0 else "None"
        )

        yr_col = "season_yr" if "season_yr" in oof_sub.columns else ("season_year" if "season_year" in oof_sub.columns else None)
        oof_rec = oof_sub[oof_sub[yr_col] >= 2025] if yr_col else oof_sub.iloc[0:0]
        oof_old = oof_sub[oof_sub[yr_col] <= 2024] if yr_col else oof_sub.iloc[0:0]
        ev_recent = _weighted_ev(oof_rec)
        ev_earlier = _weighted_ev(oof_old)
        ev_rec_ipl = _weighted_ev(oof_rec[oof_rec["comp_tier"] == COMP_TIER_IPL])
        ev_old_ipl = _weighted_ev(oof_old[oof_old["comp_tier"] == COMP_TIER_IPL])
        ev_rec_t20i = _weighted_ev(oof_rec[oof_rec["comp_tier"] == COMP_TIER_SENIOR_T20I])
        ev_old_t20i = _weighted_ev(oof_old[oof_old["comp_tier"] == COMP_TIER_SENIOR_T20I])

        # Phase splits (including detailed Bnd%, Dot%, Wkt%, and within-phase HP PI>=60 EV)
        ph_info: Dict[str, object] = {}
        for ph in ("Powerplay", "Middle", "Death"):
            pk = ph.lower()
            p_sub = sub_leg[sub_leg["match_phase"] == ph]
            oof_ph = oof_sub[oof_sub["match_phase"] == ph]
            oof_ph_hp = oof_ph[oof_ph[p_col] >= 60.0]
            ph_info[f"{pk}_balls"] = int(len(p_sub))
            ph_info[f"{pk}_share_pct"] = float(100.0 * len(p_sub) / max(1, len(sub_leg)))
            ph_info[f"{pk}_hp_balls"] = int((p_sub[p_col] >= 60.0).sum())
            ph_info[f"{pk}_hp_ev"] = _weighted_ev(oof_ph_hp)
            if discipline == "bat":
                ph_info[f"{pk}_sr"] = float(100.0 * p_sub["runs"].mean()) if len(p_sub) > 0 else np.nan
                ph_info[f"{pk}_bnd_pct"] = float(100.0 * (p_sub["is_four"].fillna(False) | p_sub["is_six"].fillna(False)).mean()) if len(p_sub) > 0 else np.nan
                ph_info[f"{pk}_dot_pct"] = float(100.0 * (p_sub["runs"] == 0).mean()) if len(p_sub) > 0 else np.nan
                ph_outs = (p_sub["is_wicket"].fillna(False) & (p_sub["canonical_dismissed_player_id"] == pid)).mean() if len(p_sub) > 0 else np.nan
                ph_info[f"{pk}_wkt_pct"] = float(100.0 * ph_outs) if pd.notna(ph_outs) else np.nan
            else:
                p_all = sub[sub["match_phase"] == ph]
                wk_ph = int(p_all["is_bowler_wicket"].fillna(False).sum())
                ph_info[f"{pk}_wkts"] = wk_ph
                ph_info[f"{pk}_econ"] = float(6.0 * p_all["bowler_runs_conceded"].sum() / len(p_sub)) if len(p_sub) > 0 else np.nan
                ph_info[f"{pk}_sr"] = float(len(p_sub) / wk_ph) if wk_ph > 0 else np.nan
                ph_info[f"{pk}_dot_pct"] = float(100.0 * ((p_sub["runs"] == 0) & (p_sub["extras1_runs"] == 0)).mean()) if len(p_sub) > 0 else np.nan
                ph_info[f"{pk}_bnd_pct"] = float(100.0 * (p_sub["is_four"].fillna(False) | p_sub["is_six"].fillna(False)).mean()) if len(p_sub) > 0 else np.nan

        role_info: Dict[str, object] = {"_pid": pid}
        if discipline == "bat":
            hand = sub["batting_type"].dropna().mode().iloc[0] if len(sub["batting_type"].dropna()) > 0 else "?"
            med_ord = float(sub["batting_order"].median()) if len(sub["batting_order"].dropna()) > 0 else np.nan
            med_ord_old = float(sub.loc[sub["season_yr"] <= 2024, "batting_order"].median()) if len(sub.loc[sub["season_yr"] <= 2024, "batting_order"].dropna()) > 0 else np.nan
            med_ord_rec = float(sub.loc[sub["season_yr"] >= 2025, "batting_order"].median()) if len(sub.loc[sub["season_yr"] >= 2025, "batting_order"].dropna()) > 0 else np.nan
            if np.isnan(med_ord):
                pos_bucket = "Unknown"
            elif med_ord <= 2.5:
                pos_bucket = "Opener (1-2)"
            elif med_ord <= 3.5:
                pos_bucket = "Top Order (3)"
            elif med_ord <= 5.5:
                pos_bucket = "Middle Order (4-5)"
            else:
                pos_bucket = "Finisher / Lower-Mid (6+)"
            total_runs = int(sub["runs"].sum())
            fours = int(sub["is_four"].fillna(False).sum())
            sixes = int(sub["is_six"].fillna(False).sum())
            outs = int((sub["is_wicket"].fillna(False) & (sub["canonical_dismissed_player_id"] == pid)).sum())
            role_info.update({
                "batting_hand": hand,
                "med_bat_order": med_ord,
                "med_order_22_24": med_ord_old,
                "med_order_25_26": med_ord_rec,
                "role_bucket": pos_bucket,
                "wk_apps": int(wk_counts.get(pid, 0)),
                "capt_apps": int(capt_counts.get(pid, 0)),
                "impact_in_apps": int(imp_in_counts.get(pid, 0)),
                "impact_out_apps": int(imp_out_counts.get(pid, 0)),
                "runs_scored": total_runs,
                "career_sr": float(100.0 * total_runs / max(1, len(sub))),
                "career_avg": float(total_runs / max(1, outs)) if outs > 0 else float(total_runs),
                "boundary_pct": float(100.0 * (fours + sixes) / max(1, len(sub))),
                "dot_pct": float(100.0 * (sub["runs"] == 0).mean()),
            })
        else:
            pos_type = sub["pace_or_spin"].dropna().mode().iloc[0] if len(sub["pace_or_spin"].dropna()) > 0 else "?"
            subtype = sub["bowling_type"].dropna().mode().iloc[0] if len(sub["bowling_type"].dropna()) > 0 else "?"
            spd_pos = sub["bowl_speed"].dropna()
            spd_pos = spd_pos[spd_pos > 0]
            spd = float(spd_pos.median()) if len(spd_pos) > 0 else np.nan
            valid_spd_pct = float(100.0 * len(spd_pos) / max(1, len(sub)))
            wkts = int(sub["is_bowler_wicket"].fillna(False).sum())
            conc = int(sub["bowler_runs_conceded"].sum())
            leg_n = max(1, len(sub_leg))
            old_leg = sub_leg[sub_leg["season_yr"] <= 2024]
            rec_leg = sub_leg[sub_leg["season_yr"] >= 2025]
            role_info.update({
                "pace_or_spin": pos_type,
                "bowling_subtype": subtype,
                "mean_speed_kph": spd,
                "valid_speed_pct": valid_spd_pct,
                "impact_in_apps": int(imp_in_counts.get(pid, 0)),
                "impact_out_apps": int(imp_out_counts.get(pid, 0)),
                "dth_share_22_24": float(100.0 * (old_leg["match_phase"] == "Death").mean()) if len(old_leg) > 0 else np.nan,
                "dth_share_25_26": float(100.0 * (rec_leg["match_phase"] == "Death").mean()) if len(rec_leg) > 0 else np.nan,
                "wkts_taken": wkts,
                "career_econ": float(6.0 * conc / leg_n),
                "career_bowl_sr": float(leg_n / max(1, wkts)),
                "dot_pct": float(100.0 * ((sub_leg["runs"] == 0) & (sub_leg["extras1_runs"] == 0)).mean()),
                "bnd_conc_pct": float(100.0 * (sub_leg["is_four"].fillna(False) | sub_leg["is_six"].fillna(False)).mean()),
            })

        tactical_role = _assign_tactical_role(discipline, role_info, ph_info)
        role_info.pop("_pid", None)
        role_info["tactical_role"] = tactical_role

        other_row = other_lookup.loc[pid] if pid in other_lookup.index else None
        ar_label, ar_info = _classify_all_rounder(other_row, discipline)

        hp_cnt = int(r["hp60_balls"] if discipline == "bat" else r["hp60_legal_balls"])
        on_rr = pid in rr_played_ids_2026
        conf_tier, flags = _assign_confidence_and_flags(r, discipline, rec_ipl, hp_cnt, on_rr)

        if pid in emg_lookup.index:
            er = emg_lookup.loc[pid]
            emg_info = {
                "emg_balls": int(er["emerging_balls"] if discipline == "bat" else er["emerging_legal_balls"]),
                "emg_ev_raw": float(er["emerging_ev_net_raw"] if discipline == "bat" else er["emerging_ev_net_po_raw"]),
            }
        else:
            emg_info = {"emg_balls": 0, "emg_ev_raw": np.nan}

        bio = PLAYER_AUCTION_CONTEXT.get(pid, {
            "age_2026": "Unverified",
            "nationality": "Unverified",
            "cap_status": "Unverified",
            "ipl_2026_team": "Unverified",
            "ipl_2026_price": "Unverified",
            "availability_note": "Unverified",
        })
        c_row = contrib_df.loc[contrib_df["canonical_player_id"] == pid].iloc[0]

        base = {
            "shortlist_rank": int(r.get("shortlist_rank", r.get("raw_pri_lcb_rank", 0))),
            "raw_pri_lcb_rank": int(r.get("raw_pri_lcb_rank", 0)),
            "raw_pri_post_rank": int(r.get("raw_pri_post_rank", 0)),
            "canonical_player_id": pid,
            "player_name": str(r["player_name"]),
            "pri_score_lcb": float(r["pri_score_lcb"]),
            "pri_score_post": float(r["pri_score_post"]),
            "pri_z_lcb": float(r["pri_z_lcb"]),
            "emp_pct_elig_primary": float(r.get("emp_pct_elig_primary", np.nan)),
            "emp_pct_elig_broad": float(r.get("emp_pct_elig_broad", np.nan)),
            "neff": float(r["neff"]),
            "neff_hp": float(r["neff_hp"]),
            "neff_hp_lt_15": bool(float(r["neff_hp"]) < 15.0),
            "n_matches": int(r["n_matches"]),
            "n_innings_spells": n_inn,
            "n_hp_opps": n_hp_opps,
            "last_match_date": str(sub_leg["match_date"].max()),
            "last_ipl_date": last_ipl_date,
            "last_ipl_season": last_ipl_yr,
            "last_ipl_team": last_ipl_team,
            "on_rr_2026_squad": on_rr,
            "recent_25_26_balls": int(rec_mask.sum()),
            "recent_25_26_ipl_balls": rec_ipl,
            "ipl_2025_balls": rec_25_ipl,
            "ipl_2026_balls": rec_26_ipl,
            "recent_25_26_t20i_balls": rec_t20i,
            "recent_share_pct": float(100.0 * rec_mask.sum() / max(1, len(sub_leg))),
            "ipl_share_pct": float(100.0 * len(sub_ipl) / max(1, len(sub_leg))),
            "ev_recent_25_26_raw": ev_recent,
            "ev_earlier_22_24_raw": ev_earlier,
            "ev_recent_ipl_raw": ev_rec_ipl,
            "ev_earlier_ipl_raw": ev_old_ipl,
            "ev_recent_t20i_raw": ev_rec_t20i,
            "ev_earlier_t20i_raw": ev_old_t20i,
            "all_rounder_class": ar_label,
            "confidence_tier": conf_tier,
            "risk_flags": flags,
            **bio,
            **role_info,
            **ph_info,
            **ar_info,
            **emg_info,
        }
        if discipline == "bat":
            for col in (
                "ipl_balls", "t20i_balls", "hp60_balls", "ipl_hp60_balls", "meff_inn",
                "raa_raw", "raa_post", "raa_lcb", "raa_sd", "raa_B",
                "surv_raw", "surv_post", "surv_lcb", "surv_sd", "surv_B",
                "cons_raw", "cons_post", "cons_lcb", "cons_sd", "cons_B",
                "hp_raa_raw", "hp_raa_post", "hp_raa_lcb", "hp_raa_sd", "hp_raa_B",
                "ev_all_raw", "ev_all_post", "ev_all_lcb",
            ):
                base[col] = float(r[col]) if col in r else np.nan
            for k in ("c_raa", "c_surv", "c_cons", "c_hp"):
                base[k] = float(c_row[k])
        else:
            for col in (
                "ipl_legal_balls", "t20i_legal_balls", "hp60_legal_balls", "ipl_hp60_legal_balls", "meff_spell",
                "rsaa_raw", "rsaa_post", "rsaa_lcb", "rsaa_sd", "rsaa_B",
                "waa_raw", "waa_post", "waa_lcb", "waa_sd", "waa_B",
                "cons_raw", "cons_post", "cons_lcb", "cons_sd", "cons_B",
                "hp_impact_raw", "hp_impact_post", "hp_impact_lcb", "hp_impact_sd", "hp_impact_B",
                "ev_all_raw", "ev_all_post", "ev_all_lcb",
            ):
                base[col] = float(r[col]) if col in r else np.nan
            for k in ("c_rsaa", "c_waa", "c_cons", "c_hp"):
                base[k] = float(c_row[k])
        rows.append(base)

    return pd.DataFrame(rows)


def compute_shortlist_sensitivity(
    oof_pri: pd.DataFrame,
    raw_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    comps_df: pd.DataFrame,
    pdir: pd.DataFrame,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    bat_target_ids: List[int],
    bowl_target_ids: List[int],
) -> Dict[str, pd.DataFrame]:
    """Computes exact eligible ranks across methodology sensitivity variants for shortlisted players."""
    oof_sens_dls, _ = fit_predict_oof_baselines(
        raw_df, matches_df, comps_df, include_suspected_dls=True, n_splits=5, exclude_emerging=True
    )

    variants = [
        ("base_lcb", PerformanceConfig(w_t20i=0.50, lcb_z=0.50), oof_pri, "pri_score_lcb"),
        ("post_mean", PerformanceConfig(w_t20i=0.50, lcb_z=0.00), oof_pri, "pri_score_post"),
        ("strict_lcb", PerformanceConfig(w_t20i=0.50, lcb_z=1.00), oof_pri, "pri_score_lcb"),
        ("ipl_only", PerformanceConfig(w_t20i=0.00, lcb_z=0.50), oof_pri, "pri_score_lcb"),
        ("t20i_025", PerformanceConfig(w_t20i=0.25, lcb_z=0.50), oof_pri, "pri_score_lcb"),
        ("t20i_100", PerformanceConfig(w_t20i=1.00, lcb_z=0.50), oof_pri, "pri_score_lcb"),
        ("hp_25pct", PerformanceConfig(w_t20i=0.50, lcb_z=0.50, bat_w_raa=0.40, bat_w_surv=0.22, bat_w_cons=0.13, bat_w_hp=0.25,
                                       bowl_w_rsaa=0.48, bowl_w_waa=0.14, bowl_w_cons=0.13, bowl_w_hp=0.25), oof_pri, "pri_score_lcb"),
        ("plus_susp_dls", PerformanceConfig(w_t20i=0.50, lcb_z=0.50), oof_sens_dls, "pri_score_lcb"),
    ]

    bat_ranks: Dict[str, pd.Series] = {}
    bowl_ranks: Dict[str, pd.Series] = {}

    for label, cfg, src_oof, score_col in variants:
        if label == "base_lcb":
            b_tbl, w_tbl = bat_df, bowl_df
        else:
            b_tbl, _ = evaluate_batting_pressure(src_oof, cfg, pdir)
            w_tbl, _ = evaluate_bowling_pressure(src_oof, cfg, pdir)

        for tbl, r_dict in ((b_tbl, bat_ranks), (w_tbl, bowl_ranks)):
            active = tbl[tbl["elig_primary"] & (~tbl["canonical_player_id"].astype(int).isin(RETIRED_FROM_IPL))].copy()
            r_dict[label] = active.set_index("canonical_player_id")[score_col].rank(ascending=False, method="min")

    # Also add EV_All rank (excluding retired)
    for tbl, r_dict in ((bat_df, bat_ranks), (bowl_df, bowl_ranks)):
        active = tbl[tbl["elig_primary"] & (~tbl["canonical_player_id"].astype(int).isin(RETIRED_FROM_IPL))].copy()
        r_dict["ev_all_lcb"] = active.set_index("canonical_player_id")["ev_all_lcb"].rank(ascending=False, method="min")

    def _assemble(target_ids: List[int], base_tbl: pd.DataFrame, r_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        name_lookup = base_tbl.set_index("canonical_player_id")["player_name"].to_dict()
        records = []
        v_cols = list(r_dict.keys())
        for pid in target_ids:
            rec = {"canonical_player_id": pid, "player_name": name_lookup.get(pid, "?")}
            ranks_vals = []
            for v in v_cols:
                val = r_dict[v].get(pid, np.nan)
                rec[f"rank_{v}"] = int(val) if pd.notna(val) else np.nan
                if pd.notna(val):
                    ranks_vals.append(int(val))
            rec["min_rank_across_variants"] = min(ranks_vals) if ranks_vals else np.nan
            rec["max_rank_across_variants"] = max(ranks_vals) if ranks_vals else np.nan
            rec["rank_spread"] = (max(ranks_vals) - min(ranks_vals)) if ranks_vals else np.nan
            rec["top10_hits"] = sum(1 for x in ranks_vals if x <= 10)
            rec["n_variants"] = len(v_cols)
            records.append(rec)
        return pd.DataFrame(records)

    return {
        "bat_sensitivity": _assemble(bat_target_ids, bat_df, bat_ranks),
        "bowl_sensitivity": _assemble(bowl_target_ids, bowl_df, bowl_ranks),
    }


def summarize_emerging_only_supplementary(
    emg_bat: pd.DataFrame,
    emg_bowl: pd.DataFrame,
    bat_pri: pd.DataFrame,
    bowl_pri: pd.DataFrame,
    emg_deliveries: Optional[pd.DataFrame] = None,
    teams_df: Optional[pd.DataFrame] = None,
    comps_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Returns separate supplementary tables of notable Emerging-only prospects (>= 40 balls / 36 legal),
    enriched with competition, team, tactical role, and HP sample context when deliveries are supplied.
    """
    b_only = emg_bat[~emg_bat["canonical_player_id"].isin(bat_pri["canonical_player_id"])].copy()
    w_only = emg_bowl[~emg_bowl["canonical_player_id"].isin(bowl_pri["canonical_player_id"])].copy()
    b_top = b_only[b_only["emerging_balls"] >= 40].sort_values("emerging_ev_net_raw", ascending=False).head(10).copy()
    w_top = w_only[w_only["emerging_legal_balls"] >= 36].sort_values("emerging_ev_net_po_raw", ascending=False).head(10).copy()

    if emg_deliveries is not None and len(emg_deliveries) > 0:
        team_map = teams_df.groupby("team_id")["team_name"].first().to_dict() if teams_df is not None else {}
        comp_map = comps_df.groupby("comp_id")["comp_name"].first().to_dict() if comps_df is not None else {}

        def _enrich(df_top: pd.DataFrame, disc: str) -> pd.DataFrame:
            id_col = "canonical_striker_id" if disc == "bat" else "canonical_bowler_id"
            tm_col = "batting_team_id" if disc == "bat" else "bowling_team_id"
            extra_rows = []
            for _, r in df_top.iterrows():
                pid = int(r["canonical_player_id"])
                sub = emg_deliveries[emg_deliveries[id_col] == pid]
                if disc == "bat":
                    sub = sub[sub["is_batter_ball_faced"]]
                    hp_cnt = int((sub["pressure_bat"] >= 60.0).sum()) if "pressure_bat" in sub.columns else 0
                    hand = sub["batting_type"].dropna().mode().iloc[0] if len(sub["batting_type"].dropna()) > 0 else "?"
                    ord_m = sub["batting_order"].median() if len(sub["batting_order"].dropna()) > 0 else np.nan
                    role_str = f"{hand} (Med #{ord_m:.0f})" if pd.notna(ord_m) else str(hand)
                else:
                    sub = sub[sub["is_legal_ball"]]
                    hp_cnt = int((sub["pressure_bowl"] >= 60.0).sum()) if "pressure_bowl" in sub.columns else 0
                    ps = sub["pace_or_spin"].dropna().mode().iloc[0] if len(sub["pace_or_spin"].dropna()) > 0 else "?"
                    bt = sub["bowling_type"].dropna().mode().iloc[0] if len(sub["bowling_type"].dropna()) > 0 else "?"
                    role_str = f"{ps} ({bt})"
                tm_id = int(sub[tm_col].mode().iloc[0]) if len(sub) > 0 else -1
                c_id = int(sub["comp_id"].mode().iloc[0]) if ("comp_id" in sub.columns and len(sub) > 0) else -1
                extra_rows.append({
                    "canonical_player_id": pid,
                    "emg_team": team_map.get(tm_id, "Unknown"),
                    "emg_comp": comp_map.get(c_id, "Emerging/Rising T20"),
                    "emg_role": role_str,
                    "emg_hp60_balls": hp_cnt,
                    "needs_external_verification": True,
                })
            if not extra_rows:
                return df_top
            return df_top.merge(pd.DataFrame(extra_rows), on="canonical_player_id", how="left")

        b_top = _enrich(b_top, "bat")
        w_top = _enrich(w_top, "bowl")

    return {
        "emerging_only_bat_top": b_top,
        "emerging_only_bowl_top": w_top,
    }


def _vectorized_loo_contextual_expected(
    df: pd.DataFrame,
    player_col: str,
    target_col: str,
    k_venue: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes zero-leakage Leave-One-Player-Out hierarchical venue-season-innings-phase
    expected value per delivery:
      ctx_exp = cell0_loo + (n_vp_loo / (n_vp_loo + k_venue)) * (vp_mean_loo - vp_base_loo)
    where cell0 = (comp_tier, season_yr, innings, match_phase) with fallback to
    (comp_tier, innings, match_phase), and vp = (venue_id, comp_tier, match_phase).
    Returns (ctx_exp_array, venue_shrinkage_weight_array).
    """
    work = df[[player_col, "comp_tier", "season_yr", "innings", "match_phase", "venue_id", target_col]].copy()
    work["_y"] = pd.to_numeric(work[target_col], errors="coerce").fillna(0.0).astype(float)
    work["_one"] = 1.0

    g0 = ["comp_tier", "season_yr", "innings", "match_phase"]
    gf = ["comp_tier", "innings", "match_phase"]
    gv = ["venue_id", "comp_tier", "match_phase"]
    gvb = ["comp_tier", "match_phase"]

    def _loo_mean(grp_cols: List[str], global_fallback: float) -> Tuple[np.ndarray, np.ndarray]:
        tot_s = work.groupby(grp_cols)["_y"].transform("sum").values
        tot_n = work.groupby(grp_cols)["_one"].transform("sum").values
        p_s = work.groupby(grp_cols + [player_col])["_y"].transform("sum").values
        p_n = work.groupby(grp_cols + [player_col])["_one"].transform("sum").values
        n_other = np.maximum(0.0, tot_n - p_n)
        s_other = tot_s - p_s
        mean_other = np.where(n_other > 0, s_other / np.maximum(1.0, n_other), global_fallback)
        return mean_other, n_other

    glob_mean = float(work["_y"].mean()) if len(work) > 0 else 0.0
    m_fb, n_fb = _loo_mean(gf, glob_mean)
    m_0, n_0 = _loo_mean(g0, glob_mean)
    base_0 = np.where(n_0 >= 25.0, m_0, m_fb)

    m_vp, n_vp = _loo_mean(gv, glob_mean)
    m_vpb, _ = _loo_mean(gvb, glob_mean)
    w_v = n_vp / (n_vp + k_venue)
    ctx_exp = base_0 + w_v * (m_vp - m_vpb)
    return ctx_exp, w_v


def compute_contextual_adjusted_metrics(
    oof_pri: pd.DataFrame,
    matches_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    target_bat_ids: List[int],
    target_bowl_ids: List[int],
    k_venue: float = 120.0,
) -> Dict[str, pd.DataFrame]:
    """
    Computes exploratory (STRICTLY SEPARATE FROM PRI) zero-leakage context-adjusted metrics:
      A. Venue-Season-Innings-Phase adjusted True Strike Rate / True Economy / True Bnd% / True Dot%
      B. Opposition-quality splits: IPL vs T20I Top-8 Full Members vs Other FM/Associates,
         plus performance vs Established IPL-Exposed opponents (opp_ipl_balls >= 100).
    """
    df = oof_pri.copy()
    if "season_yr" not in df.columns:
        if "season_year" in df.columns:
            df["season_yr"] = pd.to_numeric(df["season_year"], errors="coerce").fillna(2024).astype(int)
        elif "season" in df.columns:
            df["season_yr"] = pd.to_numeric(df["season"].astype(str).str.slice(0, 4), errors="coerce").fillna(2024).astype(int)
        else:
            df = df.merge(matches_df[["match_id", "match_date"]], on="match_id", how="left")
            df["season_yr"] = pd.to_datetime(df["match_date"], errors="coerce").dt.year.fillna(2024).astype(int)
    if "venue_id" not in df.columns:
        df = df.merge(matches_df[["match_id", "venue_id"]], on="match_id", how="left")
    df["venue_id"] = df["venue_id"].fillna(-1)
    df["is_bnd"] = (df["is_four"].fillna(False) | df["is_six"].fillna(False)).astype(float)
    df["is_dot_bat"] = (df["runs"] == 0).astype(float)
    df["is_dot_bowl"] = ((df["runs"] == 0) & (df["extras1_runs"] == 0)).astype(float)

    team_map = teams_df.groupby("team_id")["team_name"].first().to_dict()
    opp_bat_team = df["bowling_team_id"].map(team_map).fillna("Unknown")
    opp_bowl_team = df["batting_team_id"].map(team_map).fillna("Unknown")

    def _opp_bucket(comp_tier_s: pd.Series, opp_name_s: pd.Series) -> pd.Series:
        out = np.where(
            comp_tier_s == COMP_TIER_IPL,
            "IPL",
            np.where(
                opp_name_s.isin(T20I_TOP8_TEAMS),
                "T20I_Top8_FM",
                np.where(opp_name_s.isin(T20I_OTHER_FM_TEAMS), "T20I_Other_FM", "T20I_Assoc"),
            ),
        )
        return pd.Series(out, index=comp_tier_s.index)

    df["bat_opp_tier"] = _opp_bucket(df["comp_tier"], opp_bat_team)
    df["bowl_opp_tier"] = _opp_bucket(df["comp_tier"], opp_bowl_team)

    est_bowl_ids = set(bowl_df.loc[bowl_df["ipl_legal_balls"] >= 100, "canonical_player_id"].astype(int))
    est_bat_ids = set(bat_df.loc[bat_df["ipl_balls"] >= 100, "canonical_player_id"].astype(int))
    df["opp_bowler_est_ipl"] = df["canonical_bowler_id"].isin(est_bowl_ids)
    df["opp_batter_est_ipl"] = df["canonical_striker_id"].isin(est_bat_ids)

    # Batting contextual baselines (on batter balls faced)
    bf = df[df["is_batter_ball_faced"]].copy()
    bf["ctx_exp_rpb"], bf["venue_rel_w"] = _vectorized_loo_contextual_expected(bf, "canonical_striker_id", "runs", k_venue=k_venue)
    bf["ctx_exp_bnd"], _ = _vectorized_loo_contextual_expected(bf, "canonical_striker_id", "is_bnd", k_venue=k_venue)
    bf["ctx_exp_dot"], _ = _vectorized_loo_contextual_expected(bf, "canonical_striker_id", "is_dot_bat", k_venue=k_venue)

    bat_records = []
    for pid in target_bat_ids:
        sub = bf[bf["canonical_striker_id"] == int(pid)]
        if len(sub) == 0:
            continue
        w = np.where(sub["comp_tier"] == COMP_TIER_IPL, 1.0, 0.50)
        raw_sr = float(100.0 * np.average(sub["runs"], weights=w))
        exp_sr = float(100.0 * np.average(sub["ctx_exp_rpb"], weights=w))
        raw_bnd = float(100.0 * np.average(sub["is_bnd"], weights=w))
        exp_bnd = float(100.0 * np.average(sub["ctx_exp_bnd"], weights=w))
        raw_dot = float(100.0 * np.average(sub["is_dot_bat"], weights=w))
        exp_dot = float(100.0 * np.average(sub["ctx_exp_dot"], weights=w))

        top8 = sub[sub["bat_opp_tier"] == "T20I_Top8_FM"]
        oth_t20i = sub[sub["bat_opp_tier"].isin(["T20I_Other_FM", "T20I_Assoc"])]
        ipl_s = sub[sub["bat_opp_tier"] == "IPL"]
        est_s = sub[sub["opp_bowler_est_ipl"]]
        non_est_s = sub[~sub["opp_bowler_est_ipl"]]

        bat_records.append({
            "canonical_player_id": int(pid),
            "n_venues": int(sub["venue_id"].nunique()),
            "venue_rel_weight_mean": float(np.average(sub["venue_rel_w"], weights=w)),
            "ctx_raw_sr": raw_sr,
            "ctx_exp_sr": exp_sr,
            "true_sr": raw_sr - exp_sr,
            "ctx_raw_bnd_pct": raw_bnd,
            "ctx_exp_bnd_pct": exp_bnd,
            "true_bnd_pct": raw_bnd - exp_bnd,
            "ctx_raw_dot_pct": raw_dot,
            "ctx_exp_dot_pct": exp_dot,
            "true_dot_pct": raw_dot - exp_dot,
            "ipl_true_sr": float(100.0 * (ipl_s["runs"].mean() - ipl_s["ctx_exp_rpb"].mean())) if len(ipl_s) > 0 else np.nan,
            "t20i_top8_balls": int(len(top8)),
            "t20i_top8_true_sr": float(100.0 * (top8["runs"].mean() - top8["ctx_exp_rpb"].mean())) if len(top8) > 0 else np.nan,
            "t20i_other_balls": int(len(oth_t20i)),
            "t20i_other_true_sr": float(100.0 * (oth_t20i["runs"].mean() - oth_t20i["ctx_exp_rpb"].mean())) if len(oth_t20i) > 0 else np.nan,
            "vs_est_ipl_bowlers_balls": int(len(est_s)),
            "vs_est_ipl_true_sr": float(100.0 * (est_s["runs"].mean() - est_s["ctx_exp_rpb"].mean())) if len(est_s) > 0 else np.nan,
            "vs_non_est_ipl_balls": int(len(non_est_s)),
            "vs_non_est_true_sr": float(100.0 * (non_est_s["runs"].mean() - non_est_s["ctx_exp_rpb"].mean())) if len(non_est_s) > 0 else np.nan,
        })

    # Bowling contextual baselines (per delivery: runs conceded scaled cleanly to legal over)
    bw = df.copy()
    # Map legal-ball expectation using runs_conceded on all balls scaled by legal ball share within context
    bw["ctx_exp_rpb_all"], bw["venue_rel_w"] = _vectorized_loo_contextual_expected(bw, "canonical_bowler_id", "bowler_runs_conceded", k_venue=k_venue)
    bw["ctx_exp_leg_share"], _ = _vectorized_loo_contextual_expected(bw, "canonical_bowler_id", "is_legal_ball", k_venue=k_venue)
    bw["ctx_exp_rpo"] = 6.0 * bw["ctx_exp_rpb_all"] / np.maximum(0.50, bw["ctx_exp_leg_share"])

    bw_leg = bw[bw["is_legal_ball"]].copy()
    bw_leg["ctx_exp_bnd"], _ = _vectorized_loo_contextual_expected(bw_leg, "canonical_bowler_id", "is_bnd", k_venue=k_venue)
    bw_leg["ctx_exp_dot"], _ = _vectorized_loo_contextual_expected(bw_leg, "canonical_bowler_id", "is_dot_bowl", k_venue=k_venue)

    bowl_records = []
    for pid in target_bowl_ids:
        sub_all = bw[bw["canonical_bowler_id"] == int(pid)]
        sub_l = bw_leg[bw_leg["canonical_bowler_id"] == int(pid)]
        if len(sub_l) == 0:
            continue
        w_l = np.where(sub_l["comp_tier"] == COMP_TIER_IPL, 1.0, 0.50)
        w_a = np.where(sub_all["comp_tier"] == COMP_TIER_IPL, 1.0, 0.50)

        def _w_econ(s_all: pd.DataFrame, weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
            if len(s_all) == 0 or s_all["is_legal_ball"].sum() == 0:
                return np.nan, np.nan
            wt = np.ones(len(s_all)) if weights is None else weights
            leg_wt = float((wt * s_all["is_legal_ball"].astype(float)).sum())
            raw_e = float(6.0 * (wt * s_all["bowler_runs_conceded"]).sum() / max(1e-9, leg_wt))
            exp_e = float(np.average(s_all["ctx_exp_rpo"], weights=wt * s_all["is_legal_ball"].astype(float)))
            return raw_e, exp_e

        raw_econ, exp_econ = _w_econ(sub_all, w_a)
        raw_bnd = float(100.0 * np.average(sub_l["is_bnd"], weights=w_l))
        exp_bnd = float(100.0 * np.average(sub_l["ctx_exp_bnd"], weights=w_l))
        raw_dot = float(100.0 * np.average(sub_l["is_dot_bowl"], weights=w_l))
        exp_dot = float(100.0 * np.average(sub_l["ctx_exp_dot"], weights=w_l))

        ipl_a = sub_all[sub_all["bowl_opp_tier"] == "IPL"]
        top8_a = sub_all[sub_all["bowl_opp_tier"] == "T20I_Top8_FM"]
        oth_a = sub_all[sub_all["bowl_opp_tier"].isin(["T20I_Other_FM", "T20I_Assoc"])]
        est_a = sub_all[sub_all["opp_batter_est_ipl"]]
        non_est_a = sub_all[~sub_all["opp_batter_est_ipl"]]

        ir, ie = _w_econ(ipl_a)
        tr, te = _w_econ(top8_a)
        orr, oe = _w_econ(oth_a)
        er_e, ee_e = _w_econ(est_a)
        nr_e, ne_e = _w_econ(non_est_a)

        bowl_records.append({
            "canonical_player_id": int(pid),
            "n_venues": int(sub_l["venue_id"].nunique()),
            "venue_rel_weight_mean": float(np.average(sub_l["venue_rel_w"], weights=w_l)),
            "ctx_raw_econ": raw_econ,
            "ctx_exp_econ": exp_econ,
            "true_econ": exp_econ - raw_econ,  # positive = runs/over SAVED vs context
            "ctx_raw_bnd_pct": raw_bnd,
            "ctx_exp_bnd_pct": exp_bnd,
            "true_bnd_conc_pct": exp_bnd - raw_bnd,  # positive = fewer boundaries conceded
            "ctx_raw_dot_pct": raw_dot,
            "ctx_exp_dot_pct": exp_dot,
            "true_dot_pct": raw_dot - exp_dot,  # positive = more dots bowled
            "ipl_true_econ": ie - ir if pd.notna(ir) else np.nan,
            "t20i_top8_leg_balls": int(top8_a["is_legal_ball"].sum()),
            "t20i_top8_true_econ": te - tr if pd.notna(tr) else np.nan,
            "t20i_other_leg_balls": int(oth_a["is_legal_ball"].sum()),
            "t20i_other_true_econ": oe - orr if pd.notna(orr) else np.nan,
            "vs_est_ipl_batters_leg_balls": int(est_a["is_legal_ball"].sum()),
            "vs_est_ipl_true_econ": ee_e - er_e if pd.notna(er_e) else np.nan,
            "vs_non_est_ipl_leg_balls": int(non_est_a["is_legal_ball"].sum()),
            "vs_non_est_true_econ": ne_e - nr_e if pd.notna(nr_e) else np.nan,
        })

    return {
        "bat_context": pd.DataFrame(bat_records),
        "bowl_context": pd.DataFrame(bowl_records),
    }


def compute_yoy_development_trajectories(
    oof_pri: pd.DataFrame,
    target_bat_ids: List[int],
    target_bowl_ids: List[int],
    min_season_balls: int = 30,
    config: PerformanceConfig = PerformanceConfig(),
) -> Dict[str, pd.DataFrame]:
    """Computes season-by-season (2022-2026) metrics for shortlisted batters & bowlers."""
    yr_col = "season_yr" if "season_yr" in oof_pri.columns else "season_year"
    bf = oof_pri[oof_pri["is_batter_ball_faced"]].copy()
    bat_rows = []
    for pid in target_bat_ids:
        sub = bf[bf["canonical_striker_id"] == int(pid)]
        for yr, grp in sub.groupby(yr_col):
            if len(grp) < min_season_balls:
                continue
            w = np.where(grp["comp_tier"] == COMP_TIER_IPL, 1.0, config.w_t20i) * ((grp["pressure_bat"].values / 100.0) ** config.pressure_gamma)
            s_out = (grp["is_wicket"].fillna(False) & (grp["canonical_dismissed_player_id"] == int(pid))).astype(float)
            d_net = 100.0 * (grp["runs"] - grp["bat_exp_runs"]) + config.wicket_run_value * 100.0 * (grp["bat_exp_wkt"] - s_out)
            bat_rows.append({
                "canonical_player_id": int(pid),
                "player_name": str(grp["canonical_striker_name"].mode().iloc[0]),
                "season_yr": int(yr),
                "balls": int(len(grp)),
                "ipl_balls": int((grp["comp_tier"] == COMP_TIER_IPL).sum()),
                "sr": float(100.0 * grp["runs"].mean()),
                "bnd_pct": float(100.0 * (grp["is_four"].fillna(False) | grp["is_six"].fillna(False)).mean()),
                "dot_pct": float(100.0 * (grp["runs"] == 0).mean()),
                "ev_net_raw": float(np.average(d_net, weights=w)) if w.sum() > 0 else np.nan,
            })

    bowl_rows = []
    for pid in target_bowl_ids:
        sub = oof_pri[oof_pri["canonical_bowler_id"] == int(pid)]
        for yr, grp in sub.groupby(yr_col):
            leg = int(grp["is_legal_ball"].sum())
            if leg < min_season_balls:
                continue
            grp_l = grp[grp["is_legal_ball"]]
            w = np.where(grp["comp_tier"] == COMP_TIER_IPL, 1.0, config.w_t20i) * ((grp["pressure_bowl"].values / 100.0) ** config.pressure_gamma)
            leg_f = grp["is_legal_ball"].astype(float)
            wl = w * leg_f
            d_net_po = 6.0 * (leg_f * grp["bowl_exp_runs"] - grp["bowler_runs_conceded"]) + (config.wicket_run_value / 4.0) * 24.0 * (grp["is_bowler_wicket"].fillna(False).astype(float) - leg_f * grp["bowl_exp_wkt"])
            wkts = int(grp["is_bowler_wicket"].fillna(False).sum())
            bowl_rows.append({
                "canonical_player_id": int(pid),
                "player_name": str(grp["canonical_bowler_name"].mode().iloc[0]),
                "season_yr": int(yr),
                "legal_balls": leg,
                "ipl_legal_balls": int(grp_l[grp_l["comp_tier"] == COMP_TIER_IPL].shape[0]),
                "econ": float(6.0 * grp["bowler_runs_conceded"].sum() / max(1, leg)),
                "bowl_sr": float(leg / wkts) if wkts > 0 else np.nan,
                "dot_pct": float(100.0 * ((grp_l["runs"] == 0) & (grp_l["extras1_runs"] == 0)).mean()),
                "ev_net_po_raw": float((w * d_net_po).sum() / wl.sum()) if wl.sum() > 0 else np.nan,
            })

    return {
        "bat_yoy": pd.DataFrame(bat_rows),
        "bowl_yoy": pd.DataFrame(bowl_rows),
    }


def extract_rejected_player_audit(
    bat_full: pd.DataFrame,
    bowl_full: pd.DataFrame,
    bat_elig: pd.DataFrame,
    bowl_elig: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Programmatically documents concrete metric-backed rejected / non-shortlisted players across
    distinct rejection archetypes (retired high-rankers, broad-eligible non-IPL failing IPL >= 30,
    high raw RAA/low survival, and high-reputation IPL stars with HP or context deficits).
    """
    bat_focus = [
        (1318, "Retired pre-IPL 2026 (Raw #3 benchmark)"),
        (143, "Retired post-IPL 2024 (Raw #4 benchmark)"),
        (2921, "Failed Primary IPL >= 30 gate (0 IPL balls; Senior T20I only)"),
        (2485, "Failed Primary IPL >= 30 gate (0 IPL balls; Senior T20I only)"),
        (7341, "High raw scoring (RAA +71.2) offset by severe dismissal cost (Surv -3.88) & 0 HP"),
        (47, "High raw Avg/survival (Surv +1.64), neutral scoring (RAA -0.17) & Neff_hp=6.98 < 15"),
        (4117, "High IPL volume (1,443b), severe HP collapse (HP RAA post -11.63, raw -60.94)"),
        (953, "High IPL volume (1,778b), negative RAA (-2.20) & negative HP (-6.05 post)"),
        (126, "High reputation (1,211 IPL balls), negative survival (-1.02) & HP collapse (-6.19 post)"),
        (6819, "Incumbent RR star (PRI 90.90, RAA +17.51), Neff_hp=1.00 < 15 & negative survival (-0.50)"),
    ]
    bowl_focus = [
        (305, "Retired Aug 2025 (Raw #14 benchmark; 0 IPL 2026 balls)"),
        (2490, "Failed Primary IPL >= 30 gate (0 IPL balls despite Neff 782.6)"),
        (7364, "Failed Primary IPL >= 30 gate (0 IPL balls despite Neff 770.2)"),
        (5944, "High IPL volume (1,644b), severe HP collapse (HP Impact post -0.86, raw -1.91)"),
        (3628, "Strong general WAA (+0.29), negative HP Impact (post -0.63, raw -1.24)"),
        (1612, "High IPL volume (1,589b), negative general RSAA (-0.11/ov) keeps rank #26"),
        (7503, "Incumbent RR spinner (1,266 IPL balls), modest RSAA (+0.26) & neutral WAA (-0.001) rank #44"),
        (152, "Incumbent RR SLAO (1,247 IPL balls), negative HP Impact (-0.21 post) vs Axar (#5)"),
        (3653, "High IPL volume (1,221b), negative general RSAA (-0.25) & negative HP post (-0.08)"),
        (6806, "RR 2026 pacer (1,001 IPL balls), negative RSAA (-0.52/ov) & WAA (-0.15)"),
    ]

    def _build_rej(specs: List[Tuple[int, str]], full_df: pd.DataFrame, elig_df: pd.DataFrame, disc: str) -> pd.DataFrame:
        rank_map = elig_df.set_index("canonical_player_id")["raw_pri_lcb_rank"].to_dict()
        pct_map = elig_df.set_index("canonical_player_id")["emp_pct_elig_primary"].to_dict()
        lookup = full_df.set_index("canonical_player_id")
        rows = []
        for pid, reason in specs:
            if pid not in lookup.index:
                continue
            r = lookup.loc[pid]
            if disc == "bat":
                rows.append({
                    "canonical_player_id": pid,
                    "player_name": r["player_name"],
                    "elig_primary": bool(r["elig_primary"]),
                    "raw_pri_lcb_rank": rank_map.get(pid, np.nan),
                    "emp_pct_elig_primary": pct_map.get(pid, np.nan),
                    "pri_score_lcb": float(r["pri_score_lcb"]),
                    "neff": float(r["neff"]),
                    "neff_hp": float(r["neff_hp"]),
                    "ipl_balls": int(r["ipl_balls"]),
                    "t20i_balls": int(r["t20i_balls"]),
                    "raa_raw": float(r["raa_raw"]),
                    "surv_raw": float(r["surv_raw"]),
                    "hp_raa_post": float(r["hp_raa_post"]),
                    "rejection_rationale": reason,
                })
            else:
                rows.append({
                    "canonical_player_id": pid,
                    "player_name": r["player_name"],
                    "elig_primary": bool(r["elig_primary"]),
                    "raw_pri_lcb_rank": rank_map.get(pid, np.nan),
                    "emp_pct_elig_primary": pct_map.get(pid, np.nan),
                    "pri_score_lcb": float(r["pri_score_lcb"]),
                    "neff": float(r["neff"]),
                    "neff_hp": float(r["neff_hp"]),
                    "ipl_legal_balls": int(r["ipl_legal_balls"]),
                    "t20i_legal_balls": int(r["t20i_legal_balls"]),
                    "rsaa_raw": float(r["rsaa_raw"]),
                    "waa_raw": float(r["waa_raw"]),
                    "hp_impact_post": float(r["hp_impact_post"]),
                    "rejection_rationale": reason,
                })
        return pd.DataFrame(rows)

    return {
        "bat_rejected": _build_rej(bat_focus, bat_full, bat_elig, "bat"),
        "bowl_rejected": _build_rej(bowl_focus, bowl_full, bowl_elig, "bowl"),
    }


def build_portfolio_balance_summary(
    bat_shortlist_profiles: pd.DataFrame,
    bowl_shortlist_profiles: pd.DataFrame,
) -> Dict[str, object]:
    """Quantifies tactical role balance, overlaps, and structural gaps across the two Top 10 shortlists."""
    b = bat_shortlist_profiles
    w = bowl_shortlist_profiles
    return {
        "bat_n": int(len(b)),
        "bat_rhb": int((b["batting_hand"] == "RHB").sum()),
        "bat_lhb": int((b["batting_hand"] == "LHB").sum()),
        "bat_openers": int((b["role_bucket"] == "Opener (1-2)").sum()),
        "bat_middle_4_5": int((b["role_bucket"] == "Middle Order (4-5)").sum()),
        "bat_finishers_6plus": int((b["role_bucket"] == "Finisher / Lower-Mid (6+)").sum()),
        "bat_wk_options": int((b["wk_apps"] > 0).sum()),
        "bat_neff_hp_lt_15": int(b["neff_hp_lt_15"].sum()),
        "bowl_n": int(len(w)),
        "bowl_pace": int((w["pace_or_spin"] == "Pace").sum()),
        "bowl_spin": int((w["pace_or_spin"] == "Spin").sum()),
        "bowl_left_arm_pace": int(w["bowling_subtype"].astype(str).str.contains("Left-arm", case=False).mul(w["pace_or_spin"] == "Pace").sum()),
        "bowl_wrist_or_mystery_spin": int(w["bowling_subtype"].astype(str).str.contains("Leg-Break|Chinaman", case=False).sum()),
        "bowl_left_arm_orthodox": int(w["bowling_subtype"].astype(str).str.contains("orthodox", case=False).sum()),
        "bowl_genuine_all_rounders": int((w["all_rounder_class"] == "Genuine All-Rounder").sum()),
        "bowl_sec_two_way": int((w["all_rounder_class"] == "Secondary / Part-Time Two-Way").sum()),
        "bat_role_counts": b["tactical_role"].value_counts().to_dict(),
        "bowl_role_counts": w["tactical_role"].value_counts().to_dict(),
    }


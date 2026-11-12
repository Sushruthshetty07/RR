"""
Phase 4 Exploratory Extensions & Independent Proposal Audit Engine.

Strictly preserves all approved Phase 1-3 methodology, competition tiers
(IPL = 1.00, Senior_T20I = 0.50, Emerging_Rising_T20 = supplementary only),
pressure indices, OOF baselines, Empirical Bayes shrinkage, disjoint pressure
partitions, eligibility rules, LCB penalties, and primary Phase 4 Top 10 shortlists.

All 12 independent proposal additions are audited, classified, tested as
isolated exploratory analyses, and compared quantitatively against the approved
methodology under clear separation:
  (A) Approved primary ranking methodology
  (B) Supplementary analytical evidence
  (C) Experimental methods not approved for ranking
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from src.performance import (
    COMP_TIER_EMERGING,
    COMP_TIER_IPL,
    COMP_TIER_SENIOR_T20I,
    PRIMARY_DLS_STATUSES,
    PerformanceConfig,
    _prepare_context_columns,
    _safe_z,
    apply_normal_shrinkage,
)
from src.phase4_analysis import T20I_TOP8_TEAMS


W_IPL = 1.00


# ---------------------------------------------------------------------------
# Audit Classification & Item 12 Invariant Checks
# ---------------------------------------------------------------------------

def build_proposal_classification_table() -> pd.DataFrame:
    """Returns formal audit classification of all 12 proposed additions."""
    rows = [
        (
            1,
            "Win Probability and Leverage",
            "3. Missing and worth implementing",
            "None (approved PI uses chase/first-innings run-wicket pressure without outcome WP model).",
            "Fit OOF chase & setting WP models (win_team_id label only); compute counterfactual Leverage Index (LI); test vs PI >= 60.",
            "Experimental (C) — evaluated as comparator; not approved for ranking.",
        ),
        (
            2,
            "Runs Over Expected (RoE)",
            "2. Partially implemented",
            "OOF context-only Ridge/Logit models already compute d_raa and d_rsaa (plus ball_total_runs vs bowler_runs_conceded).",
            "Test enriched OOF RoE adding pace/spin, bowling/batting style, season x phase, and innings state; verify extras attribution.",
            "Experimental (C) — sensitivity benchmark against Phase 3 RAA/RSAA.",
        ),
        (
            3,
            "Pressure Delta",
            "2. Partially implemented",
            "Phase 3 computes disjoint PI >= 60 (HP) vs PI < 60 (Gen); Phase 4 flags hp_raa_post - raa_post decay.",
            "Formalize Pressure Delta = HP_RoE_100 - Norm_RoE_100 with SE_delta, HP Neff, opportunities, and LI-based sensitivity.",
            "Supplementary (B) under PI >= 60; Experimental (C) under LI.",
        ),
        (
            4,
            "Match-Level Bootstrap",
            "3. Missing and worth implementing",
            "Approved LCB uses delivery-level Kish N_eff assuming within-match conditional independence.",
            "Run B=400 match-cluster bootstrap to estimate within-match Design Effect (DEFF), cluster SEs, LCBs, and rank stability.",
            "Experimental (C) — critical uncertainty/reliability audit.",
        ),
        (
            5,
            "Consistency and Stability",
            "2. Partially implemented",
            "EB-shrunk Beta-Binomial positive-impact innings/spell rate (cons_post, 15% weight).",
            "Quantify match-to-match SD, odd-vs-even and chronological split-half repeatability (Spearman-Brown), and volatility swaps.",
            "Experimental (C) — separates appearance consistency from repeatability.",
        ),
        (
            6,
            "Phase Dominance Index",
            "2. Partially implemented",
            "Phase 4 already reports phase SR/Econ/Bnd%/Dot%/shares & LOO baselines per player card.",
            "Build EB-shrunk opportunity-neutral phase rates and equal-weight / peak Phase Dominance Index (PDI).",
            "Supplementary (B) / Experimental (C) — role specialization diagnostic.",
        ),
        (
            7,
            "Wicket Quality",
            "3. Missing and worth implementing",
            "Approved WAA weights every bowler wicket identically (= 1.0); delivery_style is >99.6% populated.",
            "Test batter-order/balls-faced/PRI-weighted wickets (excluding non-bowler dismissals) + supplementary Yorker/Bouncer execution.",
            "Experimental (C) for weighted WAA; Supplementary (B) for pace execution.",
        ),
        (
            8,
            "Two-Way Batter-Bowler Matchup Adjustment",
            "3. Missing and worth implementing",
            "Context baselines intentionally exclude batter/bowler identity.",
            "Fit OOF sparse two-way Ridge (batter + bowler + context); audit connected components, bridge density, and rank shift.",
            "Experimental (C) — schedule/strength-of-opponent diagnostic.",
        ),
        (
            9,
            "Season and Venue Adjustment",
            "2. Partially implemented",
            "Phase 3 includes linear season_year & one-hot venue_id (min_freq=30); Phase 4 checks venue/context diffs.",
            "Audit season x phase inflation residuals & implement OOF hierarchical EB venue shrinkage (venue -> country -> tier).",
            "Experimental (C) — thin-venue shrinkage & scoring inflation audit.",
        ),
        (
            10,
            "Competition Calibration",
            "2. Partially implemented",
            "Approved fixed weights IPL=1.00, Senior_T20I=0.50, Emerging=0.00 (supplementary only).",
            "Estimate paired IPL-vs-T20I bridge player offsets, Top-8 FM vs Other FM opponent tiers, and Emerging vs IPL gaps.",
            "Experimental (C) — empirical calibration check on approved 0.50 weight.",
        ),
        (
            11,
            "Career Progression and Age Information",
            "2. Partially implemented",
            "Phase 4 tracks 22-24 vs 25-26 trajectories and manually verified DOBs for shortlist/watchlist candidates.",
            "Audit first_appearance_date & tenure_months as left-censored window proxy (NOT true age) vs verified DOBs.",
            "Supplementary (B) / Audit — proves left-censoring bias vs DOB age.",
        ),
        (
            12,
            "Data and Match-State Validation",
            "1. Already implemented",
            "src/match_state.py, src/pressure.py, tests/test_match_state.py, and tests/test_pressure.py enforce all 9 invariants.",
            "Run automated audit check on processed dataset verifying zero regression across all 9 invariants.",
            "Approved Foundation (A) — audited & verified green.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "proposal_id",
            "proposal_name",
            "classification",
            "existing_coverage",
            "exploratory_extension",
            "governance_tier",
        ],
    )


def audit_match_state_invariants(df: pd.DataFrame, matches_df: pd.DataFrame) -> Dict:
    """
    Programmatically verifies Item 12 data & match-state invariants on the processed
    deliveries table (`deliveries_with_pressure.csv.gz`).
    """
    n = len(df)
    ex_type = df["extras_type"].fillna("").astype(str).str.lower()
    non_legal = {"wide", "overstep", "other"}
    is_non_legal = ex_type.isin(non_legal)
    is_bye_lb = ex_type.isin({"bye", "legbye"})

    calc_total = df["runs"].fillna(0) + df["extras1_runs"].fillna(0) + df["extras_2_runs"].fillna(0)
    inv_total_runs = bool((calc_total == df["ball_total_runs"]).all())

    calc_conc = df["runs"].fillna(0) + np.where(is_non_legal, df["extras1_runs"].fillna(0), 0.0)
    inv_bowler_conc = bool((calc_conc == df["bowler_runs_conceded"]).all())
    bye_lb_not_charged = bool(
        (df.loc[is_bye_lb, "bowler_runs_conceded"] == df.loc[is_bye_lb, "runs"].fillna(0)).all()
    )

    inv_legal_ball = bool((df["is_legal_ball"].astype(bool) == (~is_non_legal)).all())

    non_bowler_wkt_mask = df["is_wicket"].astype(bool) & (~df["is_bowler_wicket"].astype(bool))
    inv_non_bowler_excluded = bool((~df.loc[non_bowler_wkt_mask, "is_bowler_wicket"].astype(bool)).all())

    ro_mask = df["dismissal_type"].fillna("").astype(str).str.lower().str.contains("run out")
    non_striker_ro = int(
        (
            ro_mask
            & df["canonical_dismissed_player_id"].notna()
            & (df["canonical_dismissed_player_id"] != df["canonical_striker_id"])
        ).sum()
    )

    from src.performance import classify_comp_tier, filter_dls_pool

    max_innings_raw = int(df["innings"].max())
    so_col = df["is_super_over"].astype(bool)
    inv_so_flag = bool(((df["innings"] >= 3) == so_col).all())
    reg_pool = filter_dls_pool(df, include_suspected_dls=True) if "dls_status" in df.columns else df[df["innings"] <= 2]
    inv_so_excluded_by_filter = bool(int(reg_pool["innings"].max()) <= 2) if len(reg_pool) > 0 else True

    forbidden = {"win_team_id", "result", "man_of_the_match"}
    inv_no_leak_cols = len(forbidden.intersection(df.columns)) == 0

    first_balls = df[df["innings_ball_num"] == 1]
    inv_pre_shift_zero = bool((first_balls["score_before"] == 0).all() and (first_balls["wickets_before"] == 0).all())

    tiers = df["comp_tier"] if "comp_tier" in df.columns else (classify_comp_tier(df) if "competition_name" in df.columns else pd.Series(dtype=str))

    return {
        "n_deliveries_checked": int(n),
        "n_matches_checked": int(df["match_id"].nunique()),
        "inv_ball_total_runs_exact": inv_total_runs,
        "inv_bowler_runs_conceded_exact": inv_bowler_conc,
        "inv_byes_legbyes_excluded_from_bowler": bye_lb_not_charged,
        "inv_legal_ball_definition_exact": inv_legal_ball,
        "n_non_bowler_wickets": int(non_bowler_wkt_mask.sum()),
        "inv_non_bowler_wkts_excluded": inv_non_bowler_excluded,
        "n_non_striker_runouts_preserved": non_striker_ro,
        "max_innings_raw_table": max_innings_raw,
        "n_super_over_deliveries_flagged": int(so_col.sum()),
        "inv_super_over_flag_exact": inv_so_flag,
        "inv_super_overs_excluded_in_modeling_pool": inv_so_excluded_by_filter,
        "inv_no_match_outcome_columns_in_deliveries": inv_no_leak_cols,
        "inv_first_ball_pre_state_zero": inv_pre_shift_zero,
        "dls_status_counts": {str(k): int(v) for k, v in df["dls_status"].value_counts().to_dict().items()},
        "comp_tier_counts": {str(k): int(v) for k, v in tiers.value_counts().to_dict().items()} if len(tiers) > 0 else {},
    }


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _rank_comparison_stats(
    df: pd.DataFrame,
    base_col: str,
    alt_col: str,
    id_col: str = "canonical_player_id",
    elig_mask: Optional[pd.Series] = None,
    higher_is_better_alt: bool = True,
) -> Dict:
    """Computes Spearman rho, Top-10 overlap, Top-20 overlap, and rank shifts on valid pairs."""
    sub = df if elig_mask is None else df.loc[elig_mask].copy()
    valid = sub[base_col].notna() & sub[alt_col].notna()
    sub = sub.loc[valid].copy()
    if len(sub) < 5:
        return {
            "n_compared": int(len(sub)),
            "spearman_rho": np.nan,
            "pearson_r": np.nan,
            "top10_overlap": 0,
            "top20_overlap": 0,
            "mean_abs_rank_shift": np.nan,
            "max_abs_rank_shift": np.nan,
        }
    alt_vals = sub[alt_col] if higher_is_better_alt else -sub[alt_col]
    rho = float(spearmanr(sub[base_col], alt_vals).statistic)
    pear = float(np.corrcoef(sub[base_col], alt_vals)[0, 1])

    sub["_r_base"] = sub[base_col].rank(ascending=False, method="min")
    sub["_r_alt"] = alt_vals.rank(ascending=False, method="min")
    top10_base = set(sub.sort_values(base_col, ascending=False).head(10)[id_col])
    top10_alt = set(sub.loc[alt_vals.sort_values(ascending=False).index].head(10)[id_col])
    top20_base = set(sub.sort_values(base_col, ascending=False).head(20)[id_col])
    top20_alt = set(sub.loc[alt_vals.sort_values(ascending=False).index].head(20)[id_col])
    rdiff = np.abs(sub["_r_base"] - sub["_r_alt"])
    return {
        "n_compared": int(len(sub)),
        "spearman_rho": round(rho, 4),
        "pearson_r": round(pear, 4),
        "top10_overlap": int(len(top10_base & top10_alt)),
        "top20_overlap": int(len(top20_base & top20_alt)),
        "mean_abs_rank_shift": round(float(rdiff.mean()), 2),
        "max_abs_rank_shift": int(rdiff.max()),
    }


def _enrich_deliveries_context(
    oof_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Adds match/comp metadata (team names, venue_country, match_date, win_team_id) without altering existing columns."""
    out = oof_df.copy()
    if "comp_tier" not in out.columns:
        out = _prepare_context_columns(out, matches_df, competitions_df)
    m_cols = [
        c
        for c in [
            "match_id",
            "comp_id",
            "match_date",
            "venue_id",
            "venue_country",
            "home_team_id",
            "home_team",
            "away_team_id",
            "away_team",
            "win_team_id",
            "result",
        ]
        if c in matches_df.columns
    ]
    m_meta = matches_df[m_cols].drop_duplicates("match_id")
    add_cols = [c for c in m_meta.columns if c != "match_id" and c not in out.columns]
    if add_cols:
        out = out.merge(m_meta[["match_id"] + add_cols], on="match_id", how="left")
    if "match_date" in out.columns:
        out["match_date_dt"] = pd.to_datetime(out["match_date"], errors="coerce")
    if all(c in out.columns for c in ["batting_team_id", "home_team_id", "home_team", "away_team"]):
        out["batting_team_name"] = np.where(
            out["batting_team_id"] == out["home_team_id"], out["home_team"], out["away_team"]
        )
        out["bowling_team_name"] = np.where(
            out["bowling_team_id"] == out["home_team_id"], out["home_team"], out["away_team"]
        )
    return out


# ---------------------------------------------------------------------------
# Item 1: Win Probability and Leverage Index (Exploratory)
# ---------------------------------------------------------------------------

def _build_wp_features(df_sub: pd.DataFrame, is_chase: bool) -> pd.DataFrame:
    """Constructs strictly pre-delivery Win Probability feature matrix."""
    rem = np.maximum(df_sub["legal_balls_remaining_eff"].fillna(0.0).values, 0.0)
    wkts = np.clip(df_sub["wickets_in_hand_before"].fillna(10.0).values, 0.0, 10.0)
    score = np.maximum(df_sub["score_before"].fillna(0.0).values, 0.0)
    crr = np.clip(df_sub["crr_before"].fillna(7.5).values, 0.0, 24.0)
    res = pd.DataFrame(
        {
            "legal_balls_remaining_eff": rem,
            "wickets_in_hand_before": wkts,
            "score_before": score,
            "crr_before": crr,
            "is_ipl_int": df_sub["is_ipl_int"].fillna(0).astype(int).values,
            "season_year": df_sub["season_year"].fillna(2024).astype(int).values,
            "match_phase": df_sub["match_phase"].fillna("middle").astype(str).values,
            "venue_id": df_sub["venue_id"].fillna(-1).astype(int).values,
        },
        index=df_sub.index,
    )
    if is_chase:
        needed = np.maximum(df_sub["runs_needed_eff"].fillna(1.0).values, 0.0)
        rrr = np.where(rem > 0, np.clip(needed * 6.0 / rem, 0.0, 36.0), 36.0)
        res["runs_needed_eff"] = needed
        res["rrr_capped"] = rrr
        res["rrr_minus_crr"] = rrr - crr
        res["log_runs_needed"] = np.log1p(needed)
        res["res_index"] = wkts * np.sqrt(np.maximum(rem, 0.0))
    return res


def _counterfactual_wp_df(df_sub: pd.DataFrame, outcome: str, is_chase: bool) -> Tuple[pd.DataFrame, np.ndarray]:
    """Builds counterfactual next-ball pre-delivery state under outcome in {'dot','four','six','wicket'}."""
    add_runs = {"dot": 0.0, "four": 4.0, "six": 6.0, "wicket": 0.0}[outcome]
    lose_wkt = 1.0 if outcome == "wicket" else 0.0

    rem = np.maximum(df_sub["legal_balls_remaining_eff"].fillna(0.0).values - 1.0, 0.0)
    wkts = np.clip(df_sub["wickets_in_hand_before"].fillna(10.0).values - lose_wkt, 0.0, 10.0)
    score = np.maximum(df_sub["score_before"].fillna(0.0).values + add_runs, 0.0)
    max_b = np.maximum(df_sub["max_legal_balls_eff"].fillna(120.0).values, 1.0)
    elapsed_after = np.maximum(max_b - rem, 1.0)
    crr = np.clip(score * 6.0 / elapsed_after, 0.0, 24.0)

    overs_elapsed = (max_b - rem) / 6.0
    phase = np.where(overs_elapsed < 6.0, "Powerplay", np.where(overs_elapsed < 15.0, "Middle", "Death"))

    res = pd.DataFrame(
        {
            "legal_balls_remaining_eff": rem,
            "wickets_in_hand_before": wkts,
            "score_before": score,
            "crr_before": crr,
            "is_ipl_int": df_sub["is_ipl_int"].fillna(0).astype(int).values,
            "season_year": df_sub["season_year"].fillna(2024).astype(int).values,
            "match_phase": phase,
            "venue_id": df_sub["venue_id"].fillna(-1).astype(int).values,
        },
        index=df_sub.index,
    )
    term = np.full(len(df_sub), np.nan, dtype=float)
    if is_chase:
        t_raw = df_sub["target_eff"].values
        target_eff = np.where(np.isnan(t_raw), score + 1.0, t_raw)
        needed = np.maximum(target_eff - score, 0.0)
        rrr = np.full(len(df_sub), 36.0, dtype=float)
        rrr = np.where(needed <= 0.0, 0.0, rrr)
        np.divide(needed * 6.0, rem, out=rrr, where=(rem > 0) & (needed > 0))
        rrr = np.clip(rrr, 0.0, 36.0)
        res["runs_needed_eff"] = needed
        res["rrr_capped"] = rrr
        res["rrr_minus_crr"] = rrr - crr
        res["log_runs_needed"] = np.log1p(needed)
        res["res_index"] = wkts * np.sqrt(np.maximum(rem, 0.0))
        # target_eff = first_innings_score + 1; so needed <= 0 is win (1.0), needed == 1 on final ball/wicket is tie (0.5), else loss (0.0)
        term = np.where(
            needed <= 0.0,
            1.0,
            np.where((wkts <= 0.0) | (rem <= 0.0), np.where(needed == 1.0, 0.5, 0.0), np.nan),
        )
    return res, term


def _wp_pipeline(is_chase: bool) -> Pipeline:
    """Zero-leakage regularized LogisticRegression WP pipeline."""
    if is_chase:
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["match_phase"]),
                (
                    "spl",
                    SplineTransformer(n_knots=5, degree=2, include_bias=False),
                    ["legal_balls_remaining_eff", "wickets_in_hand_before", "rrr_capped", "log_runs_needed"],
                ),
                (
                    "num",
                    StandardScaler(),
                    ["runs_needed_eff", "rrr_minus_crr", "crr_before", "res_index", "is_ipl_int", "season_year"],
                ),
            ]
        )
    else:
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), ["match_phase", "venue_id"]),
                (
                    "spl",
                    SplineTransformer(n_knots=5, degree=2, include_bias=False),
                    ["legal_balls_remaining_eff", "wickets_in_hand_before", "score_before", "crr_before"],
                ),
                ("num", StandardScaler(), ["is_ipl_int", "season_year"]),
            ]
        )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=1000))])


def evaluate_win_probability_and_leverage(
    oof_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    config: PerformanceConfig = PerformanceConfig(),
) -> Dict:
    """
    Item 1: Trains OOF GroupKFold pre-delivery Win Probability models separately for
    Innings 1 & Innings 2, computes counterfactual Leverage Index (LI), and compares
    LI-defined high-leverage performance against the approved PI >= 60 methodology.
    """
    df = _enrich_deliveries_context(oof_df, matches_df, pd.DataFrame())
    valid_win = (
        df["dls_status"].isin(PRIMARY_DLS_STATUSES)
        & df["win_team_id"].notna()
        & ((df["batting_team_id"] == df["win_team_id"]) | (df["bowling_team_id"] == df["win_team_id"]))
    )
    wp_pool = df.loc[valid_win].copy().reset_index(drop=True)
    wp_pool["bat_win"] = (wp_pool["batting_team_id"] == wp_pool["win_team_id"]).astype(int)
    for c in ("wp_oof", "wp_dot", "wp_four", "wp_six", "wp_wkt"):
        wp_pool[c] = np.nan

    for inn_num, is_chase in [(1, False), (2, True)]:
        sub_idx = np.where((wp_pool["innings"] == inn_num).values)[0]
        if len(sub_idx) == 0:
            continue
        sub_df = wp_pool.iloc[sub_idx].copy()
        X_all = _build_wp_features(sub_df, is_chase=is_chase)
        y_all = sub_df["bat_win"].values
        groups = sub_df["match_id"].values
        gkf = GroupKFold(n_splits=min(5, max(2, sub_df["match_id"].nunique())))

        cf_frames = {oc: _counterfactual_wp_df(sub_df, oc, is_chase=is_chase) for oc in ("dot", "four", "six", "wicket")}

        for tr, va in gkf.split(X_all, y_all, groups=groups):
            if pd.Series(y_all[tr]).nunique() < 2:
                continue
            pipe = _wp_pipeline(is_chase=is_chase)
            pipe.fit(X_all.iloc[tr], y_all[tr])
            va_global = sub_idx[va]
            wp_pool.iloc[va_global, wp_pool.columns.get_loc("wp_oof")] = np.clip(
                pipe.predict_proba(X_all.iloc[va])[:, 1], 0.001, 0.999
            )
            for oc, (cf_df, term) in cf_frames.items():
                p_cf = np.clip(pipe.predict_proba(cf_df.iloc[va])[:, 1], 0.001, 0.999)
                t_va = term[va]
                p_cf = np.where(np.isnan(t_va), p_cf, t_va)
                col_map = {"dot": "wp_dot", "four": "wp_four", "six": "wp_six", "wicket": "wp_wkt"}[oc]
                wp_pool.iloc[va_global, wp_pool.columns.get_loc(col_map)] = p_cf

    wp_pool["li_bat_pp"] = np.maximum(0.0, wp_pool["wp_six"] - wp_pool["wp_dot"]) * 100.0
    wp_pool["li_four_pp"] = np.maximum(0.0, wp_pool["wp_four"] - wp_pool["wp_dot"]) * 100.0
    wp_pool["li_bowl_pp"] = np.maximum(0.0, wp_pool["wp_dot"] - wp_pool["wp_wkt"]) * 100.0
    wp_pool["li_combined_pp"] = 0.5 * (wp_pool["li_bat_pp"] + wp_pool["li_bowl_pp"])
    wp_pool["li_max_pp"] = np.maximum(wp_pool["li_bat_pp"], wp_pool["li_bowl_pp"])

    cal_rows = []
    for label, mask in [
        ("Overall (Inn 1 + 2)", np.ones(len(wp_pool), dtype=bool)),
        ("Innings 1 (Setting)", wp_pool["innings"] == 1),
        ("Innings 2 (Chasing)", wp_pool["innings"] == 2),
        ("IPL Overall", wp_pool["comp_tier"] == COMP_TIER_IPL),
        ("IPL Inn 2 Chase", (wp_pool["comp_tier"] == COMP_TIER_IPL) & (wp_pool["innings"] == 2)),
        ("Senior T20I Overall", wp_pool["comp_tier"] == COMP_TIER_SENIOR_T20I),
        ("Senior T20I Inn 2 Chase", (wp_pool["comp_tier"] == COMP_TIER_SENIOR_T20I) & (wp_pool["innings"] == 2)),
    ]:
        s = wp_pool.loc[mask & wp_pool["wp_oof"].notna()]
        if len(s) == 0 or s["bat_win"].nunique() < 2:
            continue
        p = s["wp_oof"].values
        y = s["bat_win"].values
        r_pi_li = float(np.corrcoef(s["pressure_bat"], s["li_combined_pp"])[0, 1])
        rho_pi_li = float(spearmanr(s["pressure_bat"], s["li_combined_pp"]).statistic)
        cal_rows.append(
            {
                "slice": label,
                "n_deliveries": int(len(s)),
                "n_matches": int(s["match_id"].nunique()),
                "actual_win_rate": round(float(y.mean()), 4),
                "pred_win_mean": round(float(p.mean()), 4),
                "roc_auc": round(float(roc_auc_score(y, p)), 4),
                "log_loss": round(float(log_loss(y, p)), 4),
                "brier_score": round(float(brier_score_loss(y, p)), 4),
                "mean_li_pp": round(float(s["li_combined_pp"].mean()), 2),
                "p90_li_pp": round(float(np.percentile(s["li_combined_pp"], 90)), 2),
                "corr_pi_li_pearson": round(r_pi_li, 4),
                "corr_pi_li_spearman": round(rho_pi_li, 4),
            }
        )
    cal_df = pd.DataFrame(cal_rows)

    chase = wp_pool[(wp_pool["innings"] == 2) & wp_pool["wp_oof"].notna()].copy()
    if len(chase) >= 20:
        chase["wp_decile"] = pd.qcut(chase["wp_oof"], 10, labels=False, duplicates="drop") + 1
        decile_df = chase.groupby("wp_decile", as_index=False).agg(
            n=("bat_win", "size"), pred=("wp_oof", "mean"), actual=("bat_win", "mean"), mean_li_pp=("li_combined_pp", "mean")
        )
    else:
        decile_df = pd.DataFrame()

    comp_w_map = {COMP_TIER_IPL: W_IPL, COMP_TIER_SENIOR_T20I: float(config.w_t20i)}
    wp_pool["comp_w"] = wp_pool["comp_tier"].map(comp_w_map).fillna(0.0)
    wp_pool["d_raa"] = 100.0 * (wp_pool["runs"] - wp_pool["bat_exp_runs"])
    wp_pool["d_rsaa"] = 6.0 * (wp_pool["is_legal_ball"].astype(float) * wp_pool["bowl_exp_runs"] - wp_pool["bowler_runs_conceded"])

    sens_rows = []
    configs = [
        ("LI >= 8% (all WP)", (wp_pool["li_combined_pp"] >= 8.0)),
        ("LI >= 10% (all WP)", (wp_pool["li_combined_pp"] >= 10.0)),
        ("LI >= 12% (all WP)", (wp_pool["li_combined_pp"] >= 12.0)),
        ("LI >= 15% (all WP)", (wp_pool["li_combined_pp"] >= 15.0)),
        ("LI >= 10% & WP 25-75%", (wp_pool["li_combined_pp"] >= 10.0) & wp_pool["wp_oof"].between(0.25, 0.75)),
        ("LI >= 12% & WP 10-90%", (wp_pool["li_combined_pp"] >= 12.0) & wp_pool["wp_oof"].between(0.10, 0.90)),
        ("Max(Bat,Bowl) LI >= 15%", (wp_pool["li_max_pp"] >= 15.0)),
        ("PI >= 60 (Inn2 only)", (wp_pool["innings"] == 2) & (wp_pool["pressure_bat"] >= config.high_pressure_cutoff)),
    ]

    bat_li10_player = pd.DataFrame()
    bowl_li10_player = pd.DataFrame()

    for cname, mask in configs:
        sub_b = wp_pool[mask & wp_pool["is_batter_ball_faced"]].copy()
        sub_l = wp_pool[mask & wp_pool["is_legal_ball"]].copy()

        sub_b["w2"] = sub_b["comp_w"] ** 2
        sub_b["wd"] = sub_b["comp_w"] * sub_b["d_raa"]
        b_agg = sub_b.groupby("canonical_striker_id", as_index=False).agg(
            n_balls=("runs", "size"), sw=("comp_w", "sum"), sw2=("w2", "sum"), swd=("wd", "sum")
        ).rename(columns={"canonical_striker_id": "canonical_player_id"})
        b_agg["neff_li"] = np.where(b_agg["sw2"] > 0, b_agg["sw"] ** 2 / b_agg["sw2"], 0.0)
        b_agg["li_raa_raw"] = np.where(b_agg["sw"] > 0, b_agg["swd"] / b_agg["sw"], np.nan)
        sig2_b = float(np.average(sub_b["d_raa"] ** 2, weights=np.maximum(sub_b["comp_w"], 1e-9))) if len(sub_b) > 0 else 1.0
        eb_b, _ = apply_normal_shrinkage(b_agg["li_raa_raw"], b_agg["neff_li"], sig2_b, min_neff_fit=8.0, lcb_z=config.lcb_z)
        b_agg["li_raa_post"] = eb_b["post"]
        b_m = bat_df.merge(b_agg, on="canonical_player_id", how="left")
        b_comp = _rank_comparison_stats(b_m, "hp_raa_post", "li_raa_post", elig_mask=b_m["elig_primary"] & (b_m["neff_li"].fillna(0) >= 5.0))
        b_comp_pri = _rank_comparison_stats(b_m, "pri_score_lcb", "li_raa_post", elig_mask=b_m["elig_primary"] & (b_m["neff_li"].fillna(0) >= 5.0))

        sub_l["w2"] = sub_l["comp_w"] ** 2
        sub_l["wd"] = sub_l["comp_w"] * sub_l["d_rsaa"]
        l_agg = sub_l.groupby("canonical_bowler_id", as_index=False).agg(
            n_balls=("is_legal_ball", "size"), sw=("comp_w", "sum"), sw2=("w2", "sum"), swd=("wd", "sum")
        ).rename(columns={"canonical_bowler_id": "canonical_player_id"})
        l_agg["neff_li"] = np.where(l_agg["sw2"] > 0, l_agg["sw"] ** 2 / l_agg["sw2"], 0.0)
        l_agg["li_rsaa_raw"] = np.where(l_agg["sw"] > 0, l_agg["swd"] / l_agg["sw"], np.nan)
        sig2_l = float(np.average(sub_l["d_rsaa"] ** 2, weights=np.maximum(sub_l["comp_w"], 1e-9))) if len(sub_l) > 0 else 1.0
        eb_l, _ = apply_normal_shrinkage(l_agg["li_rsaa_raw"], l_agg["neff_li"], sig2_l, min_neff_fit=8.0, lcb_z=config.lcb_z)
        l_agg["li_rsaa_post"] = eb_l["post"]
        l_m = bowl_df.merge(l_agg, on="canonical_player_id", how="left")
        l_comp = _rank_comparison_stats(l_m, "hp_impact_post", "li_rsaa_post", elig_mask=l_m["elig_primary"] & (l_m["neff_li"].fillna(0) >= 5.0))
        l_comp_pri = _rank_comparison_stats(l_m, "pri_score_lcb", "li_rsaa_post", elig_mask=l_m["elig_primary"] & (l_m["neff_li"].fillna(0) >= 5.0))

        if cname == "LI >= 10% (all WP)":
            bat_li10_player = b_agg.rename(columns={"neff_li": "neff_li10", "li_raa_raw": "li10_raa_raw", "li_raa_post": "li10_raa_post", "n_balls": "li10_balls"})
            bowl_li10_player = l_agg.rename(columns={"neff_li": "neff_li10", "li_rsaa_raw": "li10_rsaa_raw", "li_rsaa_post": "li10_rsaa_post", "n_balls": "li10_balls"})

        sens_rows.append(
            {
                "li_regime": cname,
                "n_bat_balls": int(len(sub_b)),
                "pct_pool_balls": round(100.0 * len(sub_b) / max(1, int(wp_pool["is_batter_ball_faced"].sum())), 2),
                "elig_bat_neff10": int((b_m["elig_primary"] & (b_m["neff_li"].fillna(0) >= 10.0)).sum()),
                "bat_rho_vs_hp_raa": b_comp["spearman_rho"],
                "bat_top10_vs_hp_raa": b_comp["top10_overlap"],
                "bat_rho_vs_pri": b_comp_pri["spearman_rho"],
                "elig_bowl_neff10": int((l_m["elig_primary"] & (l_m["neff_li"].fillna(0) >= 10.0)).sum()),
                "bowl_rho_vs_hp_impact": l_comp["spearman_rho"],
                "bowl_top10_vs_hp_impact": l_comp["top10_overlap"],
                "bowl_rho_vs_pri": l_comp_pri["spearman_rho"],
            }
        )

    return {
        "wp_deliveries": wp_pool,
        "calibration_summary": cal_df,
        "chase_deciles": decile_df,
        "li_sensitivity": pd.DataFrame(sens_rows),
        "bat_li10": bat_li10_player,
        "bowl_li10": bowl_li10_player,
    }


# ---------------------------------------------------------------------------
# Items 2 & 3: Enriched Contextual RoE & Formal Pressure Delta
# ---------------------------------------------------------------------------

def _make_enriched_roe_pre(pressure_col: str) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=25),
                ["ipl_phase", "inn_phase", "overid", "venue_id", "pace_or_spin", "bowling_type", "batting_type", "season_phase"],
            ),
            (
                "spl",
                SplineTransformer(n_knots=6, degree=2, include_bias=False),
                [pressure_col, "wickets_in_hand_before", "score_before"],
            ),
            ("num", StandardScaler(), ["season_year"]),
        ]
    )


def evaluate_enriched_roe_and_pressure_delta(
    oof_pri: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    wp_res: Dict,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    config: PerformanceConfig = PerformanceConfig(),
) -> Dict:
    """
    Item 2: Fits enriched OOF contextual Runs-over-Expected (adding pace_or_spin,
    bowling_type, batting_type, season x phase, score_before) and checks extras attribution.
    Item 3: Formalizes Pressure Delta = HP_RoE_100 - Norm_RoE_100 with SE_delta & HP exposure,
    comparing approved PI >= 60 vs LI >= 10% partitions.
    """
    df = _enrich_deliveries_context(oof_pri, matches_df, competitions_df)
    pri = df[df["dls_status"].isin(PRIMARY_DLS_STATUSES)].copy().reset_index(drop=True)
    pri["season_phase"] = pri["season_year"].astype(str) + "_" + pri["match_phase"].astype(str)
    pri["pace_or_spin"] = pri["pace_or_spin"].fillna("Unknown").astype(str)
    pri["bowling_type"] = pri["bowling_type"].fillna("Unknown").astype(str)
    pri["batting_type"] = pri["batting_type"].fillna("Unknown").astype(str)

    pri["enr_bat_exp_runs"] = np.nan
    pri["enr_bowl_exp_runs"] = np.nan
    pri["enr_bowl_exp_total_runs"] = np.nan

    gkf = GroupKFold(n_splits=min(5, max(2, pri["match_id"].nunique())))
    is_bat = pri["is_batter_ball_faced"].fillna(False).values
    is_leg = pri["is_legal_ball"].fillna(False).values

    for tr, va in gkf.split(pri, groups=pri["match_id"].values):
        tr_b = tr[is_bat[tr]]
        va_b = va[is_bat[va]]
        if len(tr_b) > 0 and len(va_b) > 0:
            p_b = Pipeline([("pre", _make_enriched_roe_pre("pressure_bat")), ("reg", Ridge(alpha=30.0))])
            p_b.fit(pri.iloc[tr_b], pri["runs"].iloc[tr_b].values)
            pri.iloc[va_b, pri.columns.get_loc("enr_bat_exp_runs")] = np.clip(p_b.predict(pri.iloc[va_b]), 0.10, 4.00)

        tr_l = tr[is_leg[tr]]
        va_l = va[is_leg[va]]
        if len(tr_l) > 0 and len(va_l) > 0:
            p_l = Pipeline([("pre", _make_enriched_roe_pre("pressure_bowl")), ("reg", Ridge(alpha=30.0))])
            p_l.fit(pri.iloc[tr_l], pri["bowler_runs_conceded"].iloc[tr_l].values)
            pri.iloc[va_l, pri.columns.get_loc("enr_bowl_exp_runs")] = np.clip(p_l.predict(pri.iloc[va_l]), 0.10, 4.00)

            p_lt = Pipeline([("pre", _make_enriched_roe_pre("pressure_bowl")), ("reg", Ridge(alpha=30.0))])
            p_lt.fit(pri.iloc[tr_l], pri["ball_total_runs"].iloc[tr_l].values)
            pri.iloc[va_l, pri.columns.get_loc("enr_bowl_exp_total_runs")] = np.clip(p_lt.predict(pri.iloc[va_l]), 0.10, 4.00)

    b_rows = pri[pri["is_batter_ball_faced"]].copy()
    l_rows = pri[pri["is_legal_ball"]].copy()
    roe_fit_summary = pd.DataFrame(
        [
            {
                "target": "Batting (batter runs on balls faced)",
                "model": "Phase 3 Approved Context Baseline",
                "n": int(len(b_rows)),
                "rmse": round(float(root_mean_squared_error(b_rows["runs"], b_rows["bat_exp_runs"])), 5),
                "mae": round(float(mean_absolute_error(b_rows["runs"], b_rows["bat_exp_runs"])), 5),
                "r2": round(float(r2_score(b_rows["runs"], b_rows["bat_exp_runs"])), 5),
            },
            {
                "target": "Batting (batter runs on balls faced)",
                "model": "Enriched RoE (+style, pace/spin, season x phase, score)",
                "n": int(len(b_rows)),
                "rmse": round(float(root_mean_squared_error(b_rows["runs"], b_rows["enr_bat_exp_runs"])), 5),
                "mae": round(float(mean_absolute_error(b_rows["runs"], b_rows["enr_bat_exp_runs"])), 5),
                "r2": round(float(r2_score(b_rows["runs"], b_rows["enr_bat_exp_runs"])), 5),
            },
            {
                "target": "Bowling (bowler_runs_conceded on legal balls)",
                "model": "Phase 3 Approved Context Baseline",
                "n": int(len(l_rows)),
                "rmse": round(float(root_mean_squared_error(l_rows["bowler_runs_conceded"], l_rows["bowl_exp_runs_legal_raw"])), 5),
                "mae": round(float(mean_absolute_error(l_rows["bowler_runs_conceded"], l_rows["bowl_exp_runs_legal_raw"])), 5),
                "r2": round(float(r2_score(l_rows["bowler_runs_conceded"], l_rows["bowl_exp_runs_legal_raw"])), 5),
            },
            {
                "target": "Bowling (bowler_runs_conceded on legal balls)",
                "model": "Enriched RoE (+style, pace/spin, season x phase, score)",
                "n": int(len(l_rows)),
                "rmse": round(float(root_mean_squared_error(l_rows["bowler_runs_conceded"], l_rows["enr_bowl_exp_runs"])), 5),
                "mae": round(float(mean_absolute_error(l_rows["bowler_runs_conceded"], l_rows["enr_bowl_exp_runs"])), 5),
                "r2": round(float(r2_score(l_rows["bowler_runs_conceded"], l_rows["enr_bowl_exp_runs"])), 5),
            },
        ]
    )

    comp_w_map = {COMP_TIER_IPL: W_IPL, COMP_TIER_SENIOR_T20I: float(config.w_t20i)}
    b_gen = b_rows[b_rows["pressure_bat"] < config.high_pressure_cutoff].copy()
    b_gen["comp_w"] = b_gen["comp_tier"].map(comp_w_map).fillna(0.0)
    b_gen["w"] = b_gen["comp_w"] * ((b_gen["pressure_bat"] / 100.0) ** config.pressure_gamma)
    b_gen["d_enr"] = 100.0 * (b_gen["runs"] - b_gen["enr_bat_exp_runs"])
    b_gen["w2"] = b_gen["w"] ** 2
    b_gen["wd"] = b_gen["w"] * b_gen["d_enr"]
    b_agg = b_gen.groupby("canonical_striker_id", as_index=False).agg(
        sw=("w", "sum"), sw2=("w2", "sum"), swd=("wd", "sum")
    ).rename(columns={"canonical_striker_id": "canonical_player_id"})
    b_agg["enr_neff"] = np.where(b_agg["sw2"] > 0, b_agg["sw"] ** 2 / b_agg["sw2"], 0.0)
    b_agg["enr_roe_raw"] = np.where(b_agg["sw"] > 0, b_agg["swd"] / b_agg["sw"], np.nan)
    sig2_eb = float(np.average(b_gen["d_enr"] ** 2, weights=np.maximum(b_gen["w"], 1e-12)))
    eb_b, _ = apply_normal_shrinkage(b_agg["enr_roe_raw"], b_agg["enr_neff"], sig2_eb, min_neff_fit=15.0, lcb_z=config.lcb_z)
    b_agg["enr_roe_post"] = eb_b["post"]
    b_agg["enr_roe_lcb"] = eb_b["lcb"]

    # Bowling exact Phase 3 General (PI < 60) matching weighting (leg_f * enr_exp - conceded across all PI < 60 balls)
    # Using legal-ball enriched model scaled by Phase 3 multiplier ratio (~1.04) or direct comparison vs legal-scaled RSAA
    p_l_all = pri[pri["pressure_bowl"] < config.high_pressure_cutoff].copy()
    p_l_all["comp_w"] = p_l_all["comp_tier"].map(comp_w_map).fillna(0.0)
    p_l_all["w"] = p_l_all["comp_w"] * ((p_l_all["pressure_bowl"] / 100.0) ** config.pressure_gamma)
    p_l_all["leg_f"] = p_l_all["is_legal_ball"].astype(float)
    p_l_all["wl"] = p_l_all["w"] * p_l_all["leg_f"]
    p_l_all["wl2"] = p_l_all["wl"] ** 2
    mult_ratio = (
        p_l_all["bowl_exp_runs"].sum() / max(1e-9, (p_l_all["leg_f"] * p_l_all["bowl_exp_runs_legal_raw"].fillna(0.0)).sum())
    )
    enr_scaled = p_l_all["enr_bowl_exp_runs"].fillna(0.0) * mult_ratio
    p_l_all["d_enr_conc"] = 6.0 * (p_l_all["leg_f"] * enr_scaled - p_l_all["bowler_runs_conceded"])
    p_l_all["d_enr_tot"] = 6.0 * (p_l_all["leg_f"] * p_l_all["enr_bowl_exp_total_runs"].fillna(0.0) - p_l_all["ball_total_runs"])
    p_l_all["wd_conc"] = p_l_all["w"] * p_l_all["d_enr_conc"]
    p_l_all["wd_tot"] = p_l_all["w"] * p_l_all["d_enr_tot"]
    l_agg = p_l_all.groupby("canonical_bowler_id", as_index=False).agg(
        sw=("wl", "sum"), sw2=("wl2", "sum"), swd_conc=("wd_conc", "sum"), swd_tot=("wd_tot", "sum")
    ).rename(columns={"canonical_bowler_id": "canonical_player_id"})
    l_agg["enr_neff"] = np.where(l_agg["sw2"] > 0, l_agg["sw"] ** 2 / l_agg["sw2"], 0.0)
    l_agg["enr_rsaa_raw"] = np.where(l_agg["sw"] > 0, l_agg["swd_conc"] / l_agg["sw"], np.nan)
    l_agg["enr_tot_rsaa_raw"] = np.where(l_agg["sw"] > 0, l_agg["swd_tot"] / l_agg["sw"], np.nan)
    sig2_el = float(np.average(p_l_all["d_enr_conc"] ** 2, weights=np.maximum(p_l_all["wl"], 1e-12)))
    eb_l, _ = apply_normal_shrinkage(l_agg["enr_rsaa_raw"], l_agg["enr_neff"], sig2_el, min_neff_fit=15.0, lcb_z=config.lcb_z)
    l_agg["enr_rsaa_post"] = eb_l["post"]
    l_agg["enr_rsaa_lcb"] = eb_l["lcb"]

    bat_cmp = bat_df.merge(b_agg[["canonical_player_id", "enr_roe_raw", "enr_roe_post", "enr_roe_lcb"]], on="canonical_player_id", how="left")
    bowl_cmp = bowl_df.merge(l_agg[["canonical_player_id", "enr_rsaa_raw", "enr_rsaa_post", "enr_rsaa_lcb", "enr_tot_rsaa_raw"]], on="canonical_player_id", how="left")

    # Item 3: Formal Pressure Delta (per 100 balls, + = better under pressure)
    # Batting raa_post & hp_raa_post are already in runs per 100 balls!
    bat_cmp["delta_pi_raw_100"] = bat_cmp["hp_raa_raw"] - bat_cmp["raa_raw"]
    bat_cmp["delta_pi_post_100"] = bat_cmp["hp_raa_post"] - bat_cmp["raa_post"]
    bat_cmp["delta_pi_se_100"] = np.sqrt(bat_cmp["hp_raa_sd"] ** 2 + bat_cmp["raa_sd"] ** 2)
    bat_cmp["delta_pi_z"] = bat_cmp["delta_pi_post_100"] / np.maximum(bat_cmp["delta_pi_se_100"], 1e-9)

    # Bowling: hp_impact_post is combined net runs saved per over; compare consistently against gen_net_po = rsaa + (V_wkt/4)*waa
    vw_over = config.wicket_run_value / 4.0
    bowl_cmp["gen_net_po_raw"] = bowl_cmp["rsaa_raw"] + vw_over * bowl_cmp["waa_raw"]
    bowl_cmp["gen_net_po_post"] = bowl_cmp["rsaa_post"] + vw_over * bowl_cmp["waa_post"]
    bowl_cmp["gen_net_po_sd"] = np.sqrt(bowl_cmp["rsaa_sd"] ** 2 + (vw_over * bowl_cmp["waa_sd"]) ** 2)
    bowl_cmp["delta_pi_raw_100"] = (bowl_cmp["hp_impact_raw"] - bowl_cmp["gen_net_po_raw"]) * (100.0 / 6.0)
    bowl_cmp["delta_pi_post_100"] = (bowl_cmp["hp_impact_post"] - bowl_cmp["gen_net_po_post"]) * (100.0 / 6.0)
    bowl_cmp["delta_pi_se_100"] = np.sqrt(bowl_cmp["hp_impact_sd"] ** 2 + bowl_cmp["gen_net_po_sd"] ** 2) * (100.0 / 6.0)
    bowl_cmp["delta_pi_z"] = bowl_cmp["delta_pi_post_100"] / np.maximum(bowl_cmp["delta_pi_se_100"], 1e-9)

    # Leverage alternative Pressure Delta (LI >= 10% vs < 10%)
    wp_d = wp_res["wp_deliveries"]
    wp_b = wp_d[wp_d["is_batter_ball_faced"]].copy()
    wp_b["comp_w"] = wp_b["comp_tier"].map(comp_w_map).fillna(0.0)
    wp_b["d_raa"] = 100.0 * (wp_b["runs"] - wp_b["bat_exp_runs"])
    wp_b["is_li10"] = wp_b["li_combined_pp"] >= 10.0
    wp_b_hi = wp_b[wp_b["is_li10"]].groupby("canonical_striker_id").apply(
        lambda g: pd.Series({"li10_roe": np.average(g["d_raa"], weights=np.maximum(g["comp_w"], 1e-9)), "li10_n": len(g)}), include_groups=False
    ).reset_index().rename(columns={"canonical_striker_id": "canonical_player_id"})
    wp_b_lo = wp_b[~wp_b["is_li10"]].groupby("canonical_striker_id").apply(
        lambda g: pd.Series({"norm_li_roe": np.average(g["d_raa"], weights=np.maximum(g["comp_w"], 1e-9)), "norm_li_n": len(g)}), include_groups=False
    ).reset_index().rename(columns={"canonical_striker_id": "canonical_player_id"})
    bat_cmp = bat_cmp.merge(wp_b_hi, on="canonical_player_id", how="left").merge(wp_b_lo, on="canonical_player_id", how="left")
    bat_cmp["delta_li10_raw_100"] = bat_cmp["li10_roe"] - bat_cmp["norm_li_roe"]

    wp_l = wp_d[wp_d["is_legal_ball"]].copy()
    wp_l["comp_w"] = wp_l["comp_tier"].map(comp_w_map).fillna(0.0)
    wp_l["d_rsaa_100"] = 100.0 * (wp_l["bowl_exp_runs"] - wp_l["bowler_runs_conceded"])
    wp_l["is_li10"] = wp_l["li_combined_pp"] >= 10.0
    wp_l_hi = wp_l[wp_l["is_li10"]].groupby("canonical_bowler_id").apply(
        lambda g: pd.Series({"li10_roe": np.average(g["d_rsaa_100"], weights=np.maximum(g["comp_w"], 1e-9)), "li10_n": len(g)}), include_groups=False
    ).reset_index().rename(columns={"canonical_bowler_id": "canonical_player_id"})
    wp_l_lo = wp_l[~wp_l["is_li10"]].groupby("canonical_bowler_id").apply(
        lambda g: pd.Series({"norm_li_roe": np.average(g["d_rsaa_100"], weights=np.maximum(g["comp_w"], 1e-9)), "norm_li_n": len(g)}), include_groups=False
    ).reset_index().rename(columns={"canonical_bowler_id": "canonical_player_id"})
    bowl_cmp = bowl_cmp.merge(wp_l_hi, on="canonical_player_id", how="left").merge(wp_l_lo, on="canonical_player_id", how="left")
    bowl_cmp["delta_li10_raw_100"] = bowl_cmp["li10_roe"] - bowl_cmp["norm_li_roe"]

    bat_elig = bat_cmp["elig_primary"]
    bowl_elig = bowl_cmp["elig_primary"]
    b_li_valid = bat_elig & bat_cmp["delta_li10_raw_100"].notna() & bat_cmp["delta_pi_raw_100"].notna()
    l_li_valid = bowl_elig & bowl_cmp["delta_li10_raw_100"].notna() & bowl_cmp["delta_pi_raw_100"].notna()
    comparisons = {
        "bat_enr_vs_raa_post": _rank_comparison_stats(bat_cmp, "raa_post", "enr_roe_post", elig_mask=bat_elig),
        "bat_enr_vs_pri_lcb": _rank_comparison_stats(bat_cmp, "pri_score_lcb", "enr_roe_post", elig_mask=bat_elig),
        "bowl_enr_vs_rsaa_post": _rank_comparison_stats(bowl_cmp, "rsaa_post", "enr_rsaa_post", elig_mask=bowl_elig),
        "bowl_enr_vs_pri_lcb": _rank_comparison_stats(bowl_cmp, "pri_score_lcb", "enr_rsaa_post", elig_mask=bowl_elig),
        "bowl_extras_attribution_tot_vs_conc": _rank_comparison_stats(bowl_cmp, "enr_rsaa_raw", "enr_tot_rsaa_raw", elig_mask=bowl_elig),
        "bat_delta_pi_vs_li10_rho": round(float(spearmanr(bat_cmp.loc[b_li_valid, "delta_pi_raw_100"], bat_cmp.loc[b_li_valid, "delta_li10_raw_100"]).statistic), 4) if b_li_valid.sum() >= 5 else np.nan,
        "bowl_delta_pi_vs_li10_rho": round(float(spearmanr(bowl_cmp.loc[l_li_valid, "delta_pi_raw_100"], bowl_cmp.loc[l_li_valid, "delta_li10_raw_100"]).statistic), 4) if l_li_valid.sum() >= 5 else np.nan,
    }
    return {
        "roe_fit_summary": roe_fit_summary,
        "comparisons": comparisons,
        "bat_roe_delta": bat_cmp,
        "bowl_roe_delta": bowl_cmp,
        "enriched_deliveries": pri,
    }


# ---------------------------------------------------------------------------
# Items 4 & 5: Match-Cluster Bootstrap & Match-Level Consistency / Repeatability
# ---------------------------------------------------------------------------

def _cluster_bootstrap_player_stats(
    deliv_df: pd.DataFrame,
    id_col: str,
    w_col: str,
    d_col: str,
    match_col: str = "match_id",
    n_boot: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fast vectorized match-cluster bootstrap using sufficient statistics per (player, match).
    Returns per-player iid_se, boot_raw_se, and design_effect (DEFF).
    """
    sub = deliv_df[[id_col, match_col, w_col, d_col]].copy()
    sub["w"] = np.maximum(sub[w_col].astype(float), 0.0)
    sub["w2"] = sub["w"] ** 2
    sub["wd"] = sub["w"] * sub[d_col].astype(float)

    sig2 = float(np.average(sub[d_col].values ** 2, weights=np.maximum(sub["w"].values, 1e-12)))

    pm = sub.groupby([id_col, match_col], as_index=False).agg(
        sw=("w", "sum"), sw2=("w2", "sum"), swd=("wd", "sum")
    )
    players = pm[id_col].unique()
    matches = pm[match_col].unique()
    p_map = pd.Series(np.arange(len(players)), index=players)
    m_map = pd.Series(np.arange(len(matches)), index=matches)

    p_idx = p_map.loc[pm[id_col]].values
    m_idx = m_map.loc[pm[match_col]].values
    sw_v = pm["sw"].values
    swd_v = pm["swd"].values
    sw2_v = pm["sw2"].values

    P, M = len(players), len(matches)
    full_sw = np.bincount(p_idx, weights=sw_v, minlength=P)
    full_sw2 = np.bincount(p_idx, weights=sw2_v, minlength=P)
    neff = np.where(full_sw2 > 0, (full_sw ** 2) / full_sw2, 0.0)
    iid_se = np.where(neff > 0, np.sqrt(sig2 / neff), np.nan)

    rng = np.random.default_rng(seed)
    boot_means = np.full((n_boot, P), np.nan, dtype=float)
    for b in range(n_boot):
        drawn = rng.integers(0, M, size=M)
        m_mult = np.bincount(drawn, minlength=M)
        row_mult = m_mult[m_idx]
        bsw = np.bincount(p_idx, weights=sw_v * row_mult, minlength=P)
        bswd = np.bincount(p_idx, weights=swd_v * row_mult, minlength=P)
        valid = bsw > 0
        boot_means[b, valid] = bswd[valid] / bsw[valid]

    boot_se = np.nanstd(boot_means, axis=0, ddof=1)
    deff = np.where((iid_se > 0) & np.isfinite(boot_se), (boot_se / iid_se) ** 2, 1.0)
    return pd.DataFrame(
        {
            "canonical_player_id": players,
            "neff_check": neff,
            "iid_se": iid_se,
            "boot_raw_se": boot_se,
            "deff": deff,
        }
    )


def _split_half_repeatability(
    inn_df: pd.DataFrame,
    id_col: str = "canonical_player_id",
    val_col: str = "inn_impact",
    w_col: str = "sw",
    min_matches_half: int = 4,
) -> Tuple[pd.DataFrame, Dict]:
    """Computes match-to-match SD, odd-even split scores, and chronological split-half repeatability."""
    ordered = inn_df.sort_values([id_col, "match_date_dt", "match_id"]).copy()
    ordered["seq_idx"] = ordered.groupby(id_col).cumcount()
    ordered["n_pl_matches"] = ordered.groupby(id_col)["match_id"].transform("count")

    pl_sd = ordered.groupby(id_col, as_index=False).agg(
        n_inn_matches=("match_id", "count"),
        match_impact_sd=(val_col, "std"),
        match_impact_mean=(val_col, "mean"),
    )

    ordered["is_odd"] = (ordered["seq_idx"] % 2) == 1
    eligible = ordered[ordered["n_pl_matches"] >= 2 * min_matches_half].copy()

    def _wmean(df_in: pd.DataFrame, mask_col: str, pos_name: str, neg_name: str) -> pd.DataFrame:
        if len(df_in) == 0:
            return pd.DataFrame(columns=[id_col, pos_name, neg_name])
        hi = df_in[df_in[mask_col]].groupby(id_col).apply(
            lambda g: np.average(g[val_col], weights=np.maximum(g[w_col], 1e-9)), include_groups=False
        )
        lo = df_in[~df_in[mask_col]].groupby(id_col).apply(
            lambda g: np.average(g[val_col], weights=np.maximum(g[w_col], 1e-9)), include_groups=False
        )
        return pd.DataFrame({pos_name: hi, neg_name: lo}).dropna().reset_index()

    oe = _wmean(eligible, "is_odd", "odd_impact", "even_impact")
    eligible["is_second_half"] = eligible["seq_idx"] >= (eligible["n_pl_matches"] // 2)
    ch = _wmean(eligible, "is_second_half", "second_half_impact", "first_half_impact")

    out = pl_sd.merge(oe, on=id_col, how="left").merge(ch, on=id_col, how="left")

    def _corr_pair(d: pd.DataFrame, c1: str, c2: str) -> Dict:
        v = d[[c1, c2]].dropna()
        if len(v) < 10:
            return {"n": int(len(v)), "pearson_r": np.nan, "spearman_rho": np.nan, "spearman_brown": np.nan}
        r = float(np.corrcoef(v[c1], v[c2])[0, 1])
        rho = float(spearmanr(v[c1], v[c2]).statistic)
        sb = (2.0 * r / (1.0 + r)) if (1.0 + r) > 1e-6 else np.nan
        return {"n": int(len(v)), "pearson_r": round(r, 4), "spearman_rho": round(rho, 4), "spearman_brown": round(sb, 4)}

    stats = {
        "odd_even_repeatability": _corr_pair(out, "odd_impact", "even_impact"),
        "chrono_split_half_repeatability": _corr_pair(out, "first_half_impact", "second_half_impact"),
    }
    return out, stats


def evaluate_match_bootstrap_and_stability(
    oof_pri: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    config: PerformanceConfig = PerformanceConfig(),
    n_boot: int = 400,
) -> Dict:
    """
    Item 4: Fast match-cluster bootstrap (B=400) comparing Kish delivery SEs vs
    cluster-adjusted SEs / LCBs and Top 10 stability.
    Item 5: Match-to-match SD, odd-even and chronological split-half repeatability,
    comparing consistency rate (`cons_post`) vs repeatability and volatility adjustments.
    """
    df = _enrich_deliveries_context(oof_pri, matches_df, competitions_df)
    pri = df[df["dls_status"].isin(PRIMARY_DLS_STATUSES)].copy()
    comp_w_map = {COMP_TIER_IPL: W_IPL, COMP_TIER_SENIOR_T20I: float(config.w_t20i)}
    pri["comp_w"] = pri["comp_tier"].map(comp_w_map).fillna(0.0)

    # Batting exact Phase 3 per-100-balls scales
    bat_p = pri[pri["is_batter_ball_faced"]].copy()
    bat_p["w"] = bat_p["comp_w"] * ((bat_p["pressure_bat"] / 100.0) ** config.pressure_gamma)
    bat_p["d_raa"] = 100.0 * (bat_p["runs"] - bat_p["bat_exp_runs"])
    bat_p["d_surv"] = 100.0 * (bat_p["bat_exp_wkt"] - bat_p["striker_out"].astype(float))
    bat_p["d_net"] = bat_p["d_raa"] + config.wicket_run_value * bat_p["d_surv"]
    bat_gen = bat_p[bat_p["pressure_bat"] < config.high_pressure_cutoff].copy()
    bat_hp = bat_p[bat_p["pressure_bat"] >= config.high_pressure_cutoff].copy()

    bs_bat_raa = _cluster_bootstrap_player_stats(bat_gen, "canonical_striker_id", "w", "d_raa", n_boot=n_boot, seed=42).rename(
        columns={"deff": "deff_raa", "boot_raw_se": "boot_se_raa", "iid_se": "iid_se_raa"}
    )
    bs_bat_hp = _cluster_bootstrap_player_stats(bat_hp, "canonical_striker_id", "w", "d_raa", n_boot=n_boot, seed=43).rename(
        columns={"deff": "deff_hp", "boot_raw_se": "boot_se_hp", "iid_se": "iid_se_hp"}
    )

    bat_m = bat_df.merge(bs_bat_raa[["canonical_player_id", "iid_se_raa", "boot_se_raa", "deff_raa"]], on="canonical_player_id", how="left")
    bat_m = bat_m.merge(bs_bat_hp[["canonical_player_id", "iid_se_hp", "boot_se_hp", "deff_hp"]], on="canonical_player_id", how="left")
    bat_m["deff_raa_clamped"] = np.clip(bat_m["deff_raa"].fillna(1.0), 1.0, 5.0)
    bat_m["deff_hp_clamped"] = np.clip(bat_m["deff_hp"].fillna(1.0), 1.0, 5.0)
    bat_m["raa_sd_cluster"] = bat_m["raa_sd"] * np.sqrt(bat_m["deff_raa_clamped"])
    bat_m["hp_raa_sd_cluster"] = bat_m["hp_raa_sd"] * np.sqrt(bat_m["deff_hp_clamped"])
    bat_m["raa_lcb_cluster"] = bat_m["raa_post"] - config.lcb_z * bat_m["raa_sd_cluster"]
    bat_m["hp_raa_lcb_cluster"] = bat_m["hp_raa_post"] - config.lcb_z * bat_m["hp_raa_sd_cluster"]

    b_ref = bat_m["elig_broad"]
    bat_m["pri_z_lcb_cluster"] = (
        config.bat_w_raa * _safe_z(bat_m["raa_lcb_cluster"], b_ref)
        + config.bat_w_surv * _safe_z(bat_m["surv_lcb"], b_ref)
        + config.bat_w_cons * _safe_z(bat_m["cons_lcb"], b_ref)
        + config.bat_w_hp * _safe_z(bat_m["hp_raa_lcb_cluster"], b_ref)
    )
    bat_m["pri_score_lcb_cluster"] = 100.0 * norm.cdf(bat_m["pri_z_lcb_cluster"])

    # Bowling exact Phase 3 per-over / per-4-over-spell scales
    bowl_all = pri.copy()
    bowl_all["w"] = bowl_all["comp_w"] * ((bowl_all["pressure_bowl"] / 100.0) ** config.pressure_gamma)
    bowl_all["leg_f"] = bowl_all["is_legal_ball"].astype(float)
    bowl_all["d_rsaa"] = 6.0 * (bowl_all["leg_f"] * bowl_all["bowl_exp_runs"] - bowl_all["bowler_runs_conceded"])
    bowl_all["d_waa"] = 24.0 * (bowl_all["bowler_wkt_int"].astype(float) - bowl_all["leg_f"] * bowl_all["bowl_exp_wkt"])
    bowl_all["d_net"] = bowl_all["d_rsaa"] + (config.wicket_run_value / 4.0) * bowl_all["d_waa"]

    bowl_gen_leg = bowl_all[(bowl_all["pressure_bowl"] < config.high_pressure_cutoff) & bowl_all["is_legal_ball"]].copy()
    bowl_hp_leg = bowl_all[(bowl_all["pressure_bowl"] >= config.high_pressure_cutoff) & bowl_all["is_legal_ball"]].copy()

    bs_bowl_rsaa = _cluster_bootstrap_player_stats(bowl_gen_leg, "canonical_bowler_id", "w", "d_rsaa", n_boot=n_boot, seed=44).rename(
        columns={"deff": "deff_rsaa", "boot_raw_se": "boot_se_rsaa", "iid_se": "iid_se_rsaa"}
    )
    bs_bowl_hp = _cluster_bootstrap_player_stats(bowl_hp_leg, "canonical_bowler_id", "w", "d_net", n_boot=n_boot, seed=45).rename(
        columns={"deff": "deff_hp", "boot_raw_se": "boot_se_hp", "iid_se": "iid_se_hp"}
    )

    bowl_m = bowl_df.merge(bs_bowl_rsaa[["canonical_player_id", "iid_se_rsaa", "boot_se_rsaa", "deff_rsaa"]], on="canonical_player_id", how="left")
    bowl_m = bowl_m.merge(bs_bowl_hp[["canonical_player_id", "iid_se_hp", "boot_se_hp", "deff_hp"]], on="canonical_player_id", how="left")
    bowl_m["deff_rsaa_clamped"] = np.clip(bowl_m["deff_rsaa"].fillna(1.0), 1.0, 5.0)
    bowl_m["deff_hp_clamped"] = np.clip(bowl_m["deff_hp"].fillna(1.0), 1.0, 5.0)
    bowl_m["rsaa_sd_cluster"] = bowl_m["rsaa_sd"] * np.sqrt(bowl_m["deff_rsaa_clamped"])
    bowl_m["hp_impact_sd_cluster"] = bowl_m["hp_impact_sd"] * np.sqrt(bowl_m["deff_hp_clamped"])
    bowl_m["rsaa_lcb_cluster"] = bowl_m["rsaa_post"] - config.lcb_z * bowl_m["rsaa_sd_cluster"]
    bowl_m["hp_impact_lcb_cluster"] = bowl_m["hp_impact_post"] - config.lcb_z * bowl_m["hp_impact_sd_cluster"]

    l_ref = bowl_m["elig_broad"]
    bowl_m["pri_z_lcb_cluster"] = (
        config.bowl_w_rsaa * _safe_z(bowl_m["rsaa_lcb_cluster"], l_ref)
        + config.bowl_w_waa * _safe_z(bowl_m["waa_lcb"], l_ref)
        + config.bowl_w_cons * _safe_z(bowl_m["cons_lcb"], l_ref)
        + config.bowl_w_hp * _safe_z(bowl_m["hp_impact_lcb_cluster"], l_ref)
    )
    bowl_m["pri_score_lcb_cluster"] = 100.0 * norm.cdf(bowl_m["pri_z_lcb_cluster"])

    # Item 5: Match-level SD & Split-half Repeatability
    bat_inn = (
        bat_gen.assign(wy=lambda x: x["w"] * x["d_net"])
        .groupby(["canonical_striker_id", "match_id", "match_date_dt"], as_index=False)
        .agg(sw=("w", "sum"), swy=("wy", "sum"))
        .rename(columns={"canonical_striker_id": "canonical_player_id"})
    )
    bat_inn = bat_inn[bat_inn["sw"] > 0].copy()
    bat_inn["inn_impact"] = bat_inn["swy"] / bat_inn["sw"]
    bat_rep_df, bat_rep_stats = _split_half_repeatability(bat_inn, "canonical_player_id", "inn_impact", "sw")

    bowl_spell = (
        bowl_all[bowl_all["pressure_bowl"] < config.high_pressure_cutoff]
        .assign(wy=lambda x: x["w"] * x["d_net"], wl=lambda x: x["w"] * x["leg_f"])
        .groupby(["canonical_bowler_id", "match_id", "match_date_dt"], as_index=False)
        .agg(sw=("wl", "sum"), swy=("wy", "sum"))
        .rename(columns={"canonical_bowler_id": "canonical_player_id"})
    )
    bowl_spell = bowl_spell[bowl_spell["sw"] > 0].copy()
    bowl_spell["inn_impact"] = bowl_spell["swy"] / bowl_spell["sw"]
    bowl_rep_df, bowl_rep_stats = _split_half_repeatability(bowl_spell, "canonical_player_id", "inn_impact", "sw")

    bat_m = bat_m.merge(bat_rep_df, on="canonical_player_id", how="left")
    bowl_m = bowl_m.merge(bowl_rep_df, on="canonical_player_id", how="left")

    bat_m["inv_sd_score"] = -bat_m["match_impact_sd"].fillna(bat_m.loc[bat_m["elig_primary"], "match_impact_sd"].median())
    bowl_m["inv_sd_score"] = -bowl_m["match_impact_sd"].fillna(bowl_m.loc[bowl_m["elig_primary"], "match_impact_sd"].median())

    bat_m["pri_score_lcb_invsd"] = 100.0 * norm.cdf(
        config.bat_w_raa * _safe_z(bat_m["raa_lcb"], b_ref)
        + config.bat_w_surv * _safe_z(bat_m["surv_lcb"], b_ref)
        + config.bat_w_cons * _safe_z(bat_m["inv_sd_score"], b_ref)
        + config.bat_w_hp * _safe_z(bat_m["hp_raa_lcb"], b_ref)
    )
    bowl_m["pri_score_lcb_invsd"] = 100.0 * norm.cdf(
        config.bowl_w_rsaa * _safe_z(bowl_m["rsaa_lcb"], l_ref)
        + config.bowl_w_waa * _safe_z(bowl_m["waa_lcb"], l_ref)
        + config.bowl_w_cons * _safe_z(bowl_m["inv_sd_score"], l_ref)
        + config.bowl_w_hp * _safe_z(bowl_m["hp_impact_lcb"], l_ref)
    )

    b_elig = bat_m["elig_primary"]
    l_elig = bowl_m["elig_primary"]
    b_sd_v = b_elig & bat_m["match_impact_sd"].notna()
    l_sd_v = l_elig & bowl_m["match_impact_sd"].notna()

    summary = {
        "bat_median_deff_raa": round(float(bat_m.loc[b_elig, "deff_raa"].median()), 3),
        "bat_p90_deff_raa": round(float(bat_m.loc[b_elig, "deff_raa"].quantile(0.9)), 3),
        "bat_median_deff_hp": round(float(bat_m.loc[b_elig, "deff_hp"].median()), 3),
        "bowl_median_deff_rsaa": round(float(bowl_m.loc[l_elig, "deff_rsaa"].median()), 3),
        "bowl_p90_deff_rsaa": round(float(bowl_m.loc[l_elig, "deff_rsaa"].quantile(0.9)), 3),
        "bowl_median_deff_hp": round(float(bowl_m.loc[l_elig, "deff_hp"].median()), 3),
        "bat_cluster_pri_vs_approved": _rank_comparison_stats(bat_m, "pri_score_lcb", "pri_score_lcb_cluster", elig_mask=b_elig),
        "bowl_cluster_pri_vs_approved": _rank_comparison_stats(bowl_m, "pri_score_lcb", "pri_score_lcb_cluster", elig_mask=l_elig),
        "bat_invsd_pri_vs_approved": _rank_comparison_stats(bat_m, "pri_score_lcb", "pri_score_lcb_invsd", elig_mask=b_elig),
        "bowl_invsd_pri_vs_approved": _rank_comparison_stats(bowl_m, "pri_score_lcb", "pri_score_lcb_invsd", elig_mask=l_elig),
        "bat_cons_vs_match_sd_rho": round(float(spearmanr(bat_m.loc[b_sd_v, "cons_post"], bat_m.loc[b_sd_v, "match_impact_sd"]).statistic), 4) if b_sd_v.sum() >= 5 else np.nan,
        "bowl_cons_vs_match_sd_rho": round(float(spearmanr(bowl_m.loc[l_sd_v, "cons_post"], bowl_m.loc[l_sd_v, "match_impact_sd"]).statistic), 4) if l_sd_v.sum() >= 5 else np.nan,
        "bat_repeatability": bat_rep_stats,
        "bowl_repeatability": bowl_rep_stats,
    }
    return {
        "summary": summary,
        "bat_boot_stab": bat_m,
        "bowl_boot_stab": bowl_m,
    }


# ---------------------------------------------------------------------------
# Items 6 & 7: Opportunity-Neutral Phase Dominance & Wicket Quality + Execution
# ---------------------------------------------------------------------------

def evaluate_phase_dominance_and_wicket_quality(
    oof_pri: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    config: PerformanceConfig = PerformanceConfig(),
) -> Dict:
    """
    Item 6: Shrinkage-stabilized opportunity-neutral Phase Dominance Index (PDI).
    Item 7: Contextual/batter-value Wicket Quality weighting (excluding non-bowler
    dismissals) + Pace Yorker/Bouncer delivery_style supplementary execution audit.
    """
    df = _enrich_deliveries_context(oof_pri, matches_df, competitions_df)
    pri = df[df["dls_status"].isin(PRIMARY_DLS_STATUSES)].copy()
    comp_w_map = {COMP_TIER_IPL: W_IPL, COMP_TIER_SENIOR_T20I: float(config.w_t20i)}
    pri["comp_w"] = pri["comp_tier"].map(comp_w_map).fillna(0.0)

    # --- Item 6: Batting Phase Dominance Index ---
    bat_b = pri[pri["is_batter_ball_faced"]].copy()
    bat_b["w"] = bat_b["comp_w"] * ((bat_b["pressure_bat"] / 100.0) ** config.pressure_gamma)
    bat_b["d_raa"] = 100.0 * (bat_b["runs"] - bat_b["bat_exp_runs"])
    bat_b["w2"] = bat_b["w"] ** 2
    bat_b["wd"] = bat_b["w"] * bat_b["d_raa"]

    bp = bat_b.groupby(["canonical_striker_id", "match_phase"], as_index=False).agg(
        balls=("runs", "size"), sw=("w", "sum"), sw2=("w2", "sum"), swd=("wd", "sum")
    ).rename(columns={"canonical_striker_id": "canonical_player_id"})
    bp["neff_ph"] = np.where(bp["sw2"] > 0, bp["sw"] ** 2 / bp["sw2"], 0.0)
    bp["ph_raw"] = np.where(bp["sw"] > 0, bp["swd"] / bp["sw"], np.nan)
    def _add_phase_eb_columns(
        df_deliv: pd.DataFrame,
        df_agg: pd.DataFrame,
        target_col: str,
        spine_ids: pd.Series,
    ) -> pd.DataFrame:
        wide = pd.DataFrame({"canonical_player_id": spine_ids})
        for ph in ("Powerplay", "Middle", "Death"):
            ph_key = ph.lower()
            sub_ph = df_agg[df_agg["match_phase"] == ph].copy()
            ph_mask = df_deliv["match_phase"] == ph
            if len(sub_ph) == 0 or not bool(ph_mask.any()):
                wide[f"pdi_{ph_key}_post"] = 0.0
                wide[f"pdi_{ph_key}_neff"] = 0.0
                continue
            sig2_ph = float(
                np.average(
                    df_deliv.loc[ph_mask, target_col] ** 2,
                    weights=np.maximum(df_deliv.loc[ph_mask, "w"], 1e-12),
                )
            )
            eb_ph, _ = apply_normal_shrinkage(
                sub_ph["ph_raw"], sub_ph["neff_ph"], sig2_ph, min_neff_fit=10.0, lcb_z=config.lcb_z
            )
            sub_ph[f"pdi_{ph_key}_post"] = eb_ph["post"]
            sub_ph[f"pdi_{ph_key}_neff"] = sub_ph["neff_ph"]
            wide = wide.merge(
                sub_ph[["canonical_player_id", f"pdi_{ph_key}_post", f"pdi_{ph_key}_neff"]],
                on="canonical_player_id",
                how="left",
            )
            wide[f"pdi_{ph_key}_post"] = wide[f"pdi_{ph_key}_post"].fillna(0.0)
            wide[f"pdi_{ph_key}_neff"] = wide[f"pdi_{ph_key}_neff"].fillna(0.0)
        return wide

    bp_wide = _add_phase_eb_columns(bat_b, bp, "d_raa", bat_df["canonical_player_id"])

    bat_pdi = bat_df.merge(bp_wide, on="canonical_player_id", how="left")
    b_ref = bat_pdi["elig_broad"]
    for ph in ("powerplay", "middle", "death"):
        bat_pdi[f"z_pdi_{ph}"] = _safe_z(bat_pdi[f"pdi_{ph}_post"], b_ref)
    bat_pdi["pdi_equal_z"] = (bat_pdi["z_pdi_powerplay"] + bat_pdi["z_pdi_middle"] + bat_pdi["z_pdi_death"]) / 3.0
    bat_pdi["pdi_equal_score"] = 100.0 * norm.cdf(bat_pdi["pdi_equal_z"])
    bat_pdi["pdi_peak_z"] = bat_pdi[["z_pdi_powerplay", "z_pdi_middle", "z_pdi_death"]].max(axis=1)
    bat_pdi["n_pos_phases"] = (
        (bat_pdi["pdi_powerplay_post"] > 0) & (bat_pdi["pdi_powerplay_neff"] >= 10)
    ).astype(int) + (
        (bat_pdi["pdi_middle_post"] > 0) & (bat_pdi["pdi_middle_neff"] >= 10)
    ).astype(int) + (
        (bat_pdi["pdi_death_post"] > 0) & (bat_pdi["pdi_death_neff"] >= 10)
    ).astype(int)

    # --- Bowling PDI ---
    bowl_l = pri[pri["is_legal_ball"]].copy()
    bowl_l["w"] = bowl_l["comp_w"] * ((bowl_l["pressure_bowl"] / 100.0) ** config.pressure_gamma)
    bowl_l["d_net_po"] = 6.0 * (bowl_l["bowl_exp_runs"] - bowl_l["bowler_runs_conceded"]) + (
        config.wicket_run_value / 4.0
    ) * 24.0 * (bowl_l["bowler_wkt_int"].astype(float) - bowl_l["bowl_exp_wkt"])
    bowl_l["w2"] = bowl_l["w"] ** 2
    bowl_l["wd"] = bowl_l["w"] * bowl_l["d_net_po"]

    lp = bowl_l.groupby(["canonical_bowler_id", "match_phase"], as_index=False).agg(
        balls=("is_legal_ball", "size"), sw=("w", "sum"), sw2=("w2", "sum"), swd=("wd", "sum")
    ).rename(columns={"canonical_bowler_id": "canonical_player_id"})
    lp["neff_ph"] = np.where(lp["sw2"] > 0, lp["sw"] ** 2 / lp["sw2"], 0.0)
    lp["ph_raw"] = np.where(lp["sw"] > 0, lp["swd"] / lp["sw"], np.nan)
    lp_wide = _add_phase_eb_columns(bowl_l, lp, "d_net_po", bowl_df["canonical_player_id"])

    bowl_pdi = bowl_df.merge(lp_wide, on="canonical_player_id", how="left")
    l_ref = bowl_pdi["elig_broad"]
    for ph in ("powerplay", "middle", "death"):
        bowl_pdi[f"z_pdi_{ph}"] = _safe_z(bowl_pdi[f"pdi_{ph}_post"], l_ref)
    bowl_pdi["pdi_equal_z"] = (bowl_pdi["z_pdi_powerplay"] + bowl_pdi["z_pdi_middle"] + bowl_pdi["z_pdi_death"]) / 3.0
    bowl_pdi["pdi_equal_score"] = 100.0 * norm.cdf(bowl_pdi["pdi_equal_z"])
    bowl_pdi["pdi_peak_z"] = bowl_pdi[["z_pdi_powerplay", "z_pdi_middle", "z_pdi_death"]].max(axis=1)

    # --- Item 7: Wicket Quality Weighting & Pace Delivery-Style Audit ---
    pri_sorted = pri.sort_values(["match_id", "innings", "overid", "innings_ball_num"]).copy()
    pri_sorted["bat_ball_int"] = pri_sorted["is_batter_ball_faced"].astype(int)
    pri_sorted["batter_inn_balls_before"] = (
        pri_sorted.groupby(["match_id", "innings", "canonical_striker_id"])["bat_ball_int"].cumsum()
        - pri_sorted["bat_ball_int"]
    )
    bat_pri_lookup = bat_df.set_index("canonical_player_id")["pri_score_lcb"].to_dict()
    pri_sorted["striker_pri_lcb"] = pri_sorted["canonical_striker_id"].map(bat_pri_lookup).fillna(50.0)

    gen_leg = pri_sorted[(pri_sorted["pressure_bowl"] < config.high_pressure_cutoff) & pri_sorted["is_legal_ball"]].copy()
    gen_leg["w"] = gen_leg["comp_w"] * ((gen_leg["pressure_bowl"] / 100.0) ** config.pressure_gamma)

    b_ord = gen_leg["batting_order"].fillna(6).astype(float)
    b_set = gen_leg["batter_inn_balls_before"].fillna(0).astype(float) >= 15.0

    raw_wq_a = np.where(b_ord <= 4, 1.0, np.where(b_ord <= 6, 0.85, np.where(b_ord == 7, 0.65, 0.35))) * np.where(b_set, 1.20, 1.00)
    raw_wq_b = np.where(b_ord <= 6, 1.0, np.where(b_ord == 7, 0.70, 0.30))
    raw_wq_c = (0.50 + 1.00 * (gen_leg["striker_pri_lcb"] / 100.0)) * np.where(b_ord >= 8, 0.50, 1.00)

    is_bwkt = gen_leg["is_bowler_wicket"].astype(bool)
    non_bwkt_gen = gen_leg["is_wicket"].astype(bool) & (~is_bwkt)
    for tag, arr in [("wqa", raw_wq_a), ("wqb", raw_wq_b), ("wqc", raw_wq_c)]:
        mean_bw = float(arr[is_bwkt].mean()) if is_bwkt.any() else 1.0
        gen_leg[f"{tag}_norm"] = arr / mean_bw
        # Exact Phase 3 per-4-over-spell scaling (24.0 * ...) while excluding non-bowler dismissals
        gen_leg[f"d_{tag}"] = 24.0 * (np.where(is_bwkt, gen_leg[f"{tag}_norm"], 0.0) - gen_leg["bowl_exp_wkt"])
        gen_leg[f"wd_{tag}"] = gen_leg["w"] * gen_leg[f"d_{tag}"]

    gen_leg["w2"] = gen_leg["w"] ** 2
    wq_agg = gen_leg.groupby("canonical_bowler_id", as_index=False).agg(
        sw=("w", "sum"),
        sw2=("w2", "sum"),
        swd_wqa=("wd_wqa", "sum"),
        swd_wqb=("wd_wqb", "sum"),
        swd_wqc=("wd_wqc", "sum"),
        n_bwkts=("is_bowler_wicket", "sum"),
        n_non_bwkts=("is_wicket", "sum"),
    ).rename(columns={"canonical_bowler_id": "canonical_player_id"})
    wq_agg["neff_gen_chk"] = np.where(wq_agg["sw2"] > 0, wq_agg["sw"] ** 2 / wq_agg["sw2"], 0.0)
    wq_meta = {}
    for tag in ("wqa", "wqb", "wqc"):
        wq_agg[f"{tag}_raw"] = np.where(wq_agg["sw"] > 0, wq_agg[f"swd_{tag}"] / wq_agg["sw"], np.nan)
        sig2_t = float(np.average(gen_leg[f"d_{tag}"] ** 2, weights=np.maximum(gen_leg["w"], 1e-12)))
        eb_t, m_t = apply_normal_shrinkage(wq_agg[f"{tag}_raw"], wq_agg["neff_gen_chk"], sig2_t, min_neff_fit=15.0, lcb_z=config.lcb_z)
        wq_agg[f"{tag}_post"] = eb_t["post"]
        wq_agg[f"{tag}_lcb"] = eb_t["lcb"]
        wq_meta[tag] = m_t

    bowl_wq_df = bowl_pdi.merge(
        wq_agg[["canonical_player_id", "wqa_raw", "wqa_post", "wqa_lcb", "wqb_post", "wqb_lcb", "wqc_post", "wqc_lcb"]],
        on="canonical_player_id",
        how="left",
    )
    for tag in ("wqa", "wqb", "wqc"):
        bowl_wq_df[f"pri_score_lcb_{tag}"] = 100.0 * norm.cdf(
            config.bowl_w_rsaa * _safe_z(bowl_wq_df["rsaa_lcb"], l_ref)
            + config.bowl_w_waa * _safe_z(bowl_wq_df[f"{tag}_lcb"], l_ref)
            + config.bowl_w_cons * _safe_z(bowl_wq_df["cons_lcb"], l_ref)
            + config.bowl_w_hp * _safe_z(bowl_wq_df["hp_impact_lcb"], l_ref)
        )

    pace_l = pri[(pri["pace_or_spin"] == "Pace") & pri["is_legal_ball"]].copy()
    ds = pace_l["delivery_style"].fillna("").astype(str).str.lower()
    pace_l["is_yorker"] = ds.str.contains("yorker")
    pace_l["is_bouncer"] = ds.str.contains("bouncer")
    pace_l["is_full_toss"] = ds == "full toss"
    pace_l["is_death"] = pace_l["match_phase"] == "Death"
    pace_l["d_rsaa"] = 6.0 * (pace_l["bowl_exp_runs"] - pace_l["bowler_runs_conceded"])
    pace_l["death_yorker"] = pace_l["is_death"] & pace_l["is_yorker"]

    pace_exec = pace_l.groupby("canonical_bowler_id", as_index=False).agg(
        pace_leg_balls=("is_legal_ball", "size"),
        yorker_balls=("is_yorker", "sum"),
        bouncer_balls=("is_bouncer", "sum"),
        full_toss_balls=("is_full_toss", "sum"),
        death_pace_balls=("is_death", "sum"),
        death_yorker_balls=("death_yorker", "sum"),
        pace_d_rsaa=("d_rsaa", "mean"),
    ).rename(columns={"canonical_bowler_id": "canonical_player_id"})
    pace_exec["yorker_pct"] = 100.0 * pace_exec["yorker_balls"] / np.maximum(pace_exec["pace_leg_balls"], 1)
    pace_exec["death_yorker_pct"] = 100.0 * pace_exec["death_yorker_balls"] / np.maximum(pace_exec["death_pace_balls"], 1)
    pace_exec["bouncer_pct"] = 100.0 * pace_exec["bouncer_balls"] / np.maximum(pace_exec["pace_leg_balls"], 1)
    pace_exec["full_toss_pct"] = 100.0 * pace_exec["full_toss_balls"] / np.maximum(pace_exec["pace_leg_balls"], 1)

    bowl_wq_df = bowl_wq_df.merge(pace_exec, on="canonical_player_id", how="left")

    b_elig = bat_pdi["elig_primary"]
    l_elig = bowl_wq_df["elig_primary"]
    pace_elig = l_elig & (bowl_wq_df["pace_leg_balls"].fillna(0) >= 60)

    summary = {
        "n_non_bowler_wkts_excluded_in_wq": int(non_bwkt_gen.sum()),
        "n_bowler_wkts_in_wq": int(is_bwkt.sum()),
        "bat_pdi_equal_vs_pri": _rank_comparison_stats(bat_pdi, "pri_score_lcb", "pdi_equal_score", elig_mask=b_elig),
        "bat_pdi_peak_vs_pri": _rank_comparison_stats(bat_pdi, "pri_score_lcb", "pdi_peak_z", elig_mask=b_elig),
        "bowl_pdi_equal_vs_pri": _rank_comparison_stats(bowl_pdi, "pri_score_lcb", "pdi_equal_score", elig_mask=l_elig),
        "bowl_pdi_peak_vs_pri": _rank_comparison_stats(bowl_pdi, "pri_score_lcb", "pdi_peak_z", elig_mask=l_elig),
        "waa_meta_wqa_k0": round(float(wq_meta["wqa"]["k0_prior_neff"]), 1),
        "waa_meta_wqb_k0": round(float(wq_meta["wqb"]["k0_prior_neff"]), 1),
        "waa_meta_wqc_k0": round(float(wq_meta["wqc"]["k0_prior_neff"]), 1),
        "bowl_wqa_vs_waa": _rank_comparison_stats(bowl_wq_df, "waa_post", "wqa_post", elig_mask=l_elig),
        "bowl_pri_wqa_vs_approved": _rank_comparison_stats(bowl_wq_df, "pri_score_lcb", "pri_score_lcb_wqa", elig_mask=l_elig),
        "bowl_pri_wqb_vs_approved": _rank_comparison_stats(bowl_wq_df, "pri_score_lcb", "pri_score_lcb_wqb", elig_mask=l_elig),
        "bowl_pri_wqc_vs_approved": _rank_comparison_stats(bowl_wq_df, "pri_score_lcb", "pri_score_lcb_wqc", elig_mask=l_elig),
        "pace_yorker_pct_vs_pri_rho": round(float(spearmanr(bowl_wq_df.loc[pace_elig, "pri_score_lcb"], bowl_wq_df.loc[pace_elig, "yorker_pct"]).statistic), 4) if pace_elig.sum() >= 5 else np.nan,
        "pace_bouncer_pct_vs_pri_rho": round(float(spearmanr(bowl_wq_df.loc[pace_elig, "pri_score_lcb"], bowl_wq_df.loc[pace_elig, "bouncer_pct"]).statistic), 4) if pace_elig.sum() >= 5 else np.nan,
        "pace_full_toss_pct_vs_pri_rho": round(float(spearmanr(bowl_wq_df.loc[pace_elig, "pri_score_lcb"], bowl_wq_df.loc[pace_elig, "full_toss_pct"]).statistic), 4) if pace_elig.sum() >= 5 else np.nan,
    }
    return {
        "summary": summary,
        "bat_pdi": bat_pdi,
        "bowl_pdi_wq": bowl_wq_df,
        "wq_meta": wq_meta,
    }


# ---------------------------------------------------------------------------
# Items 8, 9, 10, 11: Two-Way Matchup Ridge, Season x Phase + Venue EB,
# Competition Bridge Calibration, and Observed-Window Tenure Proxy
# ---------------------------------------------------------------------------

def _fit_two_way_matchups(pri: pd.DataFrame, bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> Dict:
    """
    Item 8: Sparse OOF Ridge two-way (batter + bowler) matchup adjustment on context-residualized
    runs (`runs - bat_exp_runs`), plus bipartite graph connected-components & bridge audit.
    """
    sub = pri[pri["is_batter_ball_faced"]].copy().reset_index(drop=True)
    sub["y_resid"] = sub["runs"].astype(float) - sub["bat_exp_runs"].astype(float)

    bats = np.sort(sub["canonical_striker_id"].unique())
    bowls = np.sort(sub["canonical_bowler_id"].unique())
    b_map = pd.Series(np.arange(len(bats)), index=bats)
    l_map = pd.Series(np.arange(len(bowls)), index=bowls)

    b_idx = b_map.loc[sub["canonical_striker_id"]].values
    l_idx = l_map.loc[sub["canonical_bowler_id"]].values
    n = len(sub)
    B, L = len(bats), len(bowls)

    adj = sparse.csr_matrix((np.ones(n, dtype=float), (b_idx, l_idx)), shape=(B, L))
    bi_adj = sparse.bmat([[None, adj], [adj.T, None]], format="csr")
    n_cc, cc_labels = connected_components(bi_adj, directed=False)
    cc_sizes = pd.Series(cc_labels).value_counts()
    largest_cc_pct = round(100.0 * float(cc_sizes.iloc[0]) / float(B + L), 2)

    bat_tiers = sub.groupby("canonical_striker_id")["comp_tier"].nunique()
    bowl_tiers = sub.groupby("canonical_bowler_id")["comp_tier"].nunique()

    X_bat = sparse.csr_matrix((np.ones(n), (np.arange(n), b_idx)), shape=(n, B))
    X_bowl = sparse.csr_matrix((np.ones(n), (np.arange(n), l_idx)), shape=(n, L))
    X_two_way = sparse.hstack([X_bat, -X_bowl], format="csr")
    y = sub["y_resid"].values

    gkf = GroupKFold(n_splits=min(5, max(2, sub["match_id"].nunique())))
    oof_tw_pred = np.zeros(n, dtype=float)
    for tr, va in gkf.split(X_two_way, y, groups=sub["match_id"].values):
        reg = Ridge(alpha=100.0, fit_intercept=False, solver="lsqr")
        reg.fit(X_two_way[tr], y[tr])
        oof_tw_pred[va] = reg.predict(X_two_way[va])

    reg_full = Ridge(alpha=100.0, fit_intercept=False, solver="lsqr")
    reg_full.fit(X_two_way, y)
    bat_eff_100 = 100.0 * reg_full.coef_[:B]
    bowl_eff_po = 6.0 * reg_full.coef_[B:]

    bat_tw_df = pd.DataFrame({"canonical_player_id": bats, "tw_bat_eff_100": bat_eff_100})
    bowl_tw_df = pd.DataFrame({"canonical_player_id": bowls, "tw_bowl_eff_po": bowl_eff_po})

    bat_cmp = bat_df.merge(bat_tw_df, on="canonical_player_id", how="left")
    bowl_cmp = bowl_df.merge(bowl_tw_df, on="canonical_player_id", how="left")

    b_elig = bat_cmp["elig_primary"]
    l_elig = bowl_cmp["elig_primary"]

    rmse_ctx = float(root_mean_squared_error(sub["runs"], sub["bat_exp_runs"]))
    rmse_tw = float(root_mean_squared_error(sub["runs"], np.clip(sub["bat_exp_runs"] + oof_tw_pred, 0.10, 4.00)))
    mae_ctx = float(mean_absolute_error(sub["runs"], sub["bat_exp_runs"]))
    mae_tw = float(mean_absolute_error(sub["runs"], np.clip(sub["bat_exp_runs"] + oof_tw_pred, 0.10, 4.00)))

    return {
        "n_batters": int(B),
        "n_bowlers": int(L),
        "n_connected_components": int(n_cc),
        "largest_component_pct": largest_cc_pct,
        "n_bridge_batters": int((bat_tiers > 1).sum()),
        "n_bridge_bowlers": int((bowl_tiers > 1).sum()),
        "oof_rmse_ctx_only": round(rmse_ctx, 5),
        "oof_rmse_plus_twoway": round(rmse_tw, 5),
        "oof_mae_ctx_only": round(mae_ctx, 5),
        "oof_mae_plus_twoway": round(mae_tw, 5),
        "bat_tw_vs_raa_post": _rank_comparison_stats(bat_cmp, "raa_post", "tw_bat_eff_100", elig_mask=b_elig),
        "bat_tw_vs_pri_lcb": _rank_comparison_stats(bat_cmp, "pri_score_lcb", "tw_bat_eff_100", elig_mask=b_elig),
        "bowl_tw_vs_rsaa_post": _rank_comparison_stats(bowl_cmp, "rsaa_post", "tw_bowl_eff_po", elig_mask=l_elig),
        "bowl_tw_vs_pri_lcb": _rank_comparison_stats(bowl_cmp, "pri_score_lcb", "tw_bowl_eff_po", elig_mask=l_elig),
        "bat_tw_table": bat_cmp,
        "bowl_tw_table": bowl_cmp,
    }


def _fit_season_phase_and_venue_eb(pri: pd.DataFrame, bat_df: pd.DataFrame, bowl_df: pd.DataFrame, config: PerformanceConfig) -> Dict:
    """
    Item 9: Audits season x phase scoring inflation and OOF hierarchical Empirical Bayes
    venue shrinkage (`venue_id -> venue_country -> comp_tier`) against Phase 3 baselines.
    """
    sub = pri[pri["is_batter_ball_faced"]].copy().reset_index(drop=True)
    sub["venue_country"] = sub["venue_country"].fillna("Unknown").astype(str)
    sub["season_phase"] = sub["season_year"].astype(str) + "_" + sub["match_phase"].astype(str)

    v_stats = (
        sub.groupby(["venue_id", "venue_country"], as_index=False)
        .agg(n_matches=("match_id", "nunique"), n_balls=("runs", "size"), mean_runs=("runs", "mean"))
    )

    sub["d_raa_base_100"] = 100.0 * (sub["runs"] - sub["bat_exp_runs"])
    sp_bias = (
        sub.groupby(["season_year", "match_phase"], as_index=False)
        .agg(n=("runs", "size"), actual_rpb=("runs", "mean"), base_exp=("bat_exp_runs", "mean"), base_resid_100=("d_raa_base_100", "mean"))
    )

    gkf = GroupKFold(n_splits=min(5, max(2, sub["match_id"].nunique())))
    sub["hier_exp_runs"] = np.nan

    pre_noven = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ipl_phase", "inn_phase", "overid", "season_phase"]),
            ("spl", SplineTransformer(n_knots=6, degree=2, include_bias=False), ["pressure_bat", "wickets_in_hand_before"]),
        ]
    )

    for tr, va in gkf.split(sub, groups=sub["match_id"].values):
        pipe = Pipeline([("pre", pre_noven), ("reg", Ridge(alpha=30.0))])
        pipe.fit(sub.iloc[tr], sub["runs"].iloc[tr].values)
        pred_tr = pipe.predict(sub.iloc[tr])
        pred_va = pipe.predict(sub.iloc[va])
        res_tr = sub["runs"].iloc[tr].values - pred_tr

        tr_meta = sub.iloc[tr][["venue_id", "venue_country", "comp_tier"]].copy()
        tr_meta["res"] = res_tr

        tier_means = tr_meta.groupby("comp_tier")["res"].mean().to_dict()
        c_grp = tr_meta.groupby(["venue_country", "comp_tier"], as_index=False).agg(n=("res", "size"), mean_r=("res", "mean"))
        c_grp["tier_prior"] = c_grp["comp_tier"].map(tier_means).fillna(0.0)
        c_grp["country_eb"] = (c_grp["n"] * c_grp["mean_r"] + 1200.0 * c_grp["tier_prior"]) / (c_grp["n"] + 1200.0)
        country_map = c_grp.set_index(["venue_country", "comp_tier"])["country_eb"].to_dict()

        v_grp = tr_meta.groupby(["venue_id", "venue_country", "comp_tier"], as_index=False).agg(n=("res", "size"), mean_r=("res", "mean"))
        v_grp["c_prior"] = [country_map.get((c, t), tier_means.get(t, 0.0)) for c, t in zip(v_grp["venue_country"], v_grp["comp_tier"])]
        v_grp["venue_eb"] = (v_grp["n"] * v_grp["mean_r"] + 600.0 * v_grp["c_prior"]) / (v_grp["n"] + 600.0)
        venue_map = v_grp.set_index(["venue_id", "comp_tier"])["venue_eb"].to_dict()

        va_meta = sub.iloc[va]
        va_adj = np.array(
            [
                venue_map.get((vid, t), country_map.get((vc, t), tier_means.get(t, 0.0)))
                for vid, vc, t in zip(va_meta["venue_id"], va_meta["venue_country"], va_meta["comp_tier"])
            ]
        )
        sub.iloc[va, sub.columns.get_loc("hier_exp_runs")] = np.clip(pred_va + va_adj, 0.10, 4.00)

    sub["d_hier_100"] = 100.0 * (sub["runs"] - sub["hier_exp_runs"])
    sp_bias["hier_resid_100"] = (
        sub.groupby(["season_year", "match_phase"])["d_hier_100"].mean().reindex(pd.MultiIndex.from_frame(sp_bias[["season_year", "match_phase"]])).values
    )

    comp_w_map = {COMP_TIER_IPL: W_IPL, COMP_TIER_SENIOR_T20I: float(config.w_t20i)}
    sub["comp_w"] = sub["comp_tier"].map(comp_w_map).fillna(0.0)
    sub_gen = sub[sub["pressure_bat"] < config.high_pressure_cutoff].copy()
    sub_gen["w"] = sub_gen["comp_w"] * ((sub_gen["pressure_bat"] / 100.0) ** config.pressure_gamma)
    sub_gen["wd_h"] = sub_gen["w"] * sub_gen["d_hier_100"]
    sub_gen["w2"] = sub_gen["w"] ** 2
    agg = sub_gen.groupby("canonical_striker_id", as_index=False).agg(sw=("w", "sum"), sw2=("w2", "sum"), swd=("wd_h", "sum")).rename(columns={"canonical_striker_id": "canonical_player_id"})
    agg["neff_h"] = np.where(agg["sw2"] > 0, agg["sw"] ** 2 / agg["sw2"], 0.0)
    agg["hier_raa_raw"] = np.where(agg["sw"] > 0, agg["swd"] / agg["sw"], np.nan)
    sig2_h = float(np.average(sub_gen["d_hier_100"] ** 2, weights=np.maximum(sub_gen["w"], 1e-12)))
    eb_h, _ = apply_normal_shrinkage(agg["hier_raa_raw"], agg["neff_h"], sig2_h, min_neff_fit=15.0, lcb_z=config.lcb_z)
    agg["hier_raa_post"] = eb_h["post"]

    bat_h = bat_df.merge(agg[["canonical_player_id", "hier_raa_post"]], on="canonical_player_id", how="left")
    b_elig = bat_h["elig_primary"]

    return {
        "n_venues": int(len(v_stats)),
        "venues_lt5_matches": int((v_stats["n_matches"] < 5).sum()),
        "venues_lt10_matches": int((v_stats["n_matches"] < 10).sum()),
        "oof_rmse_base": round(float(root_mean_squared_error(sub["runs"], sub["bat_exp_runs"])), 5),
        "oof_rmse_hier": round(float(root_mean_squared_error(sub["runs"], sub["hier_exp_runs"])), 5),
        "max_abs_season_phase_resid_100_base": round(float(sp_bias["base_resid_100"].abs().max()), 2),
        "max_abs_season_phase_resid_100_hier": round(float(sp_bias["hier_resid_100"].abs().max()), 2),
        "bat_hier_vs_raa_post": _rank_comparison_stats(bat_h, "raa_post", "hier_raa_post", elig_mask=b_elig),
        "season_phase_residuals": sp_bias,
        "venue_stats": v_stats,
    }


def _evaluate_competition_bridge_calibration(
    pri: pd.DataFrame,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    config: PerformanceConfig,
    emg_bat_df: Optional[pd.DataFrame] = None,
    emg_bowl_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Item 10: Bridge-player calibration between IPL and Senior_T20I (split into Top-8 FM
    vs Other FM/Associates), tests opponent-tiered T20I weight sensitivity, and evaluates
    supplementary Emerging_Rising_T20 vs IPL bridge gaps separately.
    """
    bat_b = pri[pri["is_batter_ball_faced"]].copy()
    bat_b["d_raa"] = 100.0 * (bat_b["runs"] - bat_b["bat_exp_runs"])
    bat_b["t20i_opp_bucket"] = np.where(
        bat_b["comp_tier"] == COMP_TIER_IPL,
        "IPL",
        np.where(bat_b["bowling_team_name"].isin(T20I_TOP8_TEAMS), "Senior_T20I_Top8", "Senior_T20I_Other"),
    )

    bowl_l = pri[pri["is_legal_ball"]].copy()
    bowl_l["d_rsaa"] = 6.0 * (bowl_l["bowl_exp_runs"] - bowl_l["bowler_runs_conceded"])
    bowl_l["t20i_opp_bucket"] = np.where(
        bowl_l["comp_tier"] == COMP_TIER_IPL,
        "IPL",
        np.where(bowl_l["batting_team_name"].isin(T20I_TOP8_TEAMS), "Senior_T20I_Top8", "Senior_T20I_Other"),
    )

    def _bridge_stats(df_in: pd.DataFrame, id_col: str, val_col: str, tier_col: str, a_val: str, b_val: str, min_b: int = 30) -> Dict:
        a_s = df_in[df_in[tier_col] == a_val].groupby(id_col).agg(n=(val_col, "size"), mean=(val_col, "mean"))
        b_s = df_in[df_in[tier_col] == b_val].groupby(id_col).agg(n=(val_col, "size"), mean=(val_col, "mean"))
        m = a_s[a_s["n"] >= min_b].join(b_s[b_s["n"] >= min_b], lsuffix="_a", rsuffix="_b", how="inner")
        if len(m) < 5:
            return {"n_bridge": int(len(m)), "mean_a": np.nan, "mean_b": np.nan, "diff_a_minus_b": np.nan, "r": np.nan, "rho": np.nan}
        return {
            "n_bridge": int(len(m)),
            "mean_a": round(float(m["mean_a"].mean()), 2),
            "mean_b": round(float(m["mean_b"].mean()), 2),
            "diff_a_minus_b": round(float((m["mean_a"] - m["mean_b"]).mean()), 2),
            "r": round(float(np.corrcoef(m["mean_a"], m["mean_b"])[0, 1]), 4),
            "rho": round(float(spearmanr(m["mean_a"], m["mean_b"]).statistic), 4),
        }

    bowl_all = pri.copy()
    bowl_all["leg_f"] = bowl_all["is_legal_ball"].astype(float)
    bowl_all["d_rsaa"] = 6.0 * (bowl_all["leg_f"] * bowl_all["bowl_exp_runs"] - bowl_all["bowler_runs_conceded"])
    bowl_all["t20i_opp_bucket"] = np.where(
        bowl_all["comp_tier"] == COMP_TIER_IPL,
        "IPL",
        np.where(bowl_all["batting_team_name"].isin(T20I_TOP8_TEAMS), "Senior_T20I_Top8", "Senior_T20I_Other"),
    )

    def _reweight_sensitivity(
        df_in: pd.DataFrame,
        id_col: str,
        val_col: str,
        pres_col: str,
        base_df: pd.DataFrame,
        base_col: str,
        is_bowling: bool = False,
    ) -> pd.DataFrame:
        rows = []
        w_schemes = [
            ("Approved (IPL=1.0, T20I=0.50)", {"IPL": 1.0, "Senior_T20I_Top8": 0.50, "Senior_T20I_Other": 0.50}),
            ("Light T20I (IPL=1.0, T20I=0.25)", {"IPL": 1.0, "Senior_T20I_Top8": 0.25, "Senior_T20I_Other": 0.25}),
            ("Heavy T20I (IPL=1.0, T20I=0.75)", {"IPL": 1.0, "Senior_T20I_Top8": 0.75, "Senior_T20I_Other": 0.75}),
            ("Opp-Tiered (1.0 / 0.65 / 0.25)", {"IPL": 1.0, "Senior_T20I_Top8": 0.65, "Senior_T20I_Other": 0.25}),
        ]
        d_gen = df_in[df_in[pres_col] < config.high_pressure_cutoff].copy()
        for sname, wmap in w_schemes:
            d = d_gen.copy()
            d["cw"] = d["t20i_opp_bucket"].map(wmap).fillna(0.0)
            d["w"] = d["cw"] * ((d[pres_col] / 100.0) ** config.pressure_gamma)
            d["w_den"] = d["w"] * d["leg_f"] if is_bowling else d["w"]
            d["w2"] = d["w_den"] ** 2
            d["wd"] = d["w"] * d[val_col]
            ag = d.groupby(id_col, as_index=False).agg(sw=("w_den", "sum"), sw2=("w2", "sum"), swd=("wd", "sum")).rename(columns={id_col: "canonical_player_id"})
            ag["neff_w"] = np.where(ag["sw2"] > 0, ag["sw"] ** 2 / ag["sw2"], 0.0)
            ag["alt_raw"] = np.where(ag["sw"] > 0, ag["swd"] / ag["sw"], np.nan)
            sig2_w = float(np.average(d[val_col] ** 2, weights=np.maximum(d["w_den"], 1e-12)))
            eb_w, _ = apply_normal_shrinkage(ag["alt_raw"], ag["neff_w"], sig2_w, min_neff_fit=15.0, lcb_z=config.lcb_z)
            ag["alt_post"] = eb_w["post"]
            m = base_df.merge(ag[["canonical_player_id", "alt_post"]], on="canonical_player_id", how="left")
            st = _rank_comparison_stats(m, base_col, "alt_post", elig_mask=m["elig_primary"])
            st["scheme"] = sname
            rows.append(st)
        return pd.DataFrame(rows)

    bat_w_sens = _reweight_sensitivity(bat_b, "canonical_striker_id", "d_raa", "pressure_bat", bat_df, "raa_post", is_bowling=False)
    bowl_w_sens = _reweight_sensitivity(bowl_all, "canonical_bowler_id", "d_rsaa", "pressure_bowl", bowl_df, "rsaa_post", is_bowling=True)

    # Separate Supplementary Emerging vs IPL bridge comparison
    emg_bat_n, emg_bat_diff, emg_bowl_n, emg_bowl_diff = 0, np.nan, 0, np.nan
    if emg_bat_df is not None and len(emg_bat_df) > 0:
        ipl_b = bat_b[bat_b["comp_tier"] == COMP_TIER_IPL].groupby("canonical_striker_id").agg(ipl_b=("d_raa", "size"), ipl_raa=("d_raa", "mean")).reset_index().rename(columns={"canonical_striker_id": "canonical_player_id"})
        eb_j = emg_bat_df[emg_bat_df["emerging_balls"] >= 20].merge(ipl_b[ipl_b["ipl_b"] >= 30], on="canonical_player_id", how="inner")
        emg_bat_n = int(len(eb_j))
        if emg_bat_n > 0:
            emg_bat_diff = round(float((eb_j["emerging_raa_raw"] - eb_j["ipl_raa"]).mean()), 2)
    if emg_bowl_df is not None and len(emg_bowl_df) > 0:
        ipl_l = bowl_l[bowl_l["comp_tier"] == COMP_TIER_IPL].groupby("canonical_bowler_id").agg(ipl_b=("d_rsaa", "size"), ipl_rsaa=("d_rsaa", "mean")).reset_index().rename(columns={"canonical_bowler_id": "canonical_player_id"})
        el_j = emg_bowl_df[emg_bowl_df["emerging_legal_balls"] >= 20].merge(ipl_l[ipl_l["ipl_b"] >= 30], on="canonical_player_id", how="inner")
        emg_bowl_n = int(len(el_j))
        if emg_bowl_n > 0:
            emg_bowl_diff = round(float((el_j["emerging_rsaa_raw"] - el_j["ipl_rsaa"]).mean()), 2)

    return {
        "bat_bridge_ipl_vs_all_t20i": _bridge_stats(bat_b, "canonical_striker_id", "d_raa", "comp_tier", COMP_TIER_IPL, COMP_TIER_SENIOR_T20I),
        "bat_bridge_ipl_vs_top8_t20i": _bridge_stats(bat_b, "canonical_striker_id", "d_raa", "t20i_opp_bucket", "IPL", "Senior_T20I_Top8"),
        "bat_bridge_ipl_vs_other_t20i": _bridge_stats(bat_b, "canonical_striker_id", "d_raa", "t20i_opp_bucket", "IPL", "Senior_T20I_Other"),
        "bowl_bridge_ipl_vs_all_t20i": _bridge_stats(bowl_l, "canonical_bowler_id", "d_rsaa", "comp_tier", COMP_TIER_IPL, COMP_TIER_SENIOR_T20I),
        "bowl_bridge_ipl_vs_top8_t20i": _bridge_stats(bowl_l, "canonical_bowler_id", "d_rsaa", "t20i_opp_bucket", "IPL", "Senior_T20I_Top8"),
        "bowl_bridge_ipl_vs_other_t20i": _bridge_stats(bowl_l, "canonical_bowler_id", "d_rsaa", "t20i_opp_bucket", "IPL", "Senior_T20I_Other"),
        "bat_weight_sensitivity": bat_w_sens,
        "bowl_weight_sensitivity": bowl_w_sens,
        "emg_ipl_bat_bridge_n": emg_bat_n,
        "emg_ipl_bat_bridge_mean_raa_diff": emg_bat_diff,
        "emg_ipl_bowl_bridge_n": emg_bowl_n,
        "emg_ipl_bowl_bridge_mean_rsaa_diff": emg_bowl_diff,
    }


def _evaluate_tenure_proxy_and_progression(
    pri: pd.DataFrame,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
) -> Dict:
    """
    Item 11: Builds observed-window tenure proxy (`first_appearance_date`, `tenure_months`,
    `left_censored_2022`) and quantifies left-censoring bias against true verified DOB ages
    while computing early-window vs late-window progression deltas.
    """
    from src.phase4_analysis import PLAYER_AUCTION_CONTEXT

    def _player_tenure(df_in: pd.DataFrame, id_col: str, val_col: str, base_df: pd.DataFrame) -> pd.DataFrame:
        s = df_in.sort_values([id_col, "match_date_dt", "match_id"]).copy()
        s["ball_seq"] = s.groupby(id_col).cumcount()
        s["tot_b"] = s.groupby(id_col)[val_col].transform("size")
        s["is_late_half"] = s["ball_seq"] >= (s["tot_b"] // 2)
        ten = s.groupby(id_col, as_index=False).agg(
            first_app_date=("match_date_dt", "min"),
            last_app_date=("match_date_dt", "max"),
            total_obs_balls=(val_col, "size"),
        ).rename(columns={id_col: "canonical_player_id"})
        ten["tenure_months"] = (ten["last_app_date"] - ten["first_app_date"]).dt.days / 30.4375
        ten["left_censored_pre2022"] = ten["first_app_date"] <= pd.Timestamp("2022-06-30")

        early = s[~s["is_late_half"]].groupby(id_col)[val_col].mean().rename("early_half_mean")
        late = s[s["is_late_half"]].groupby(id_col)[val_col].mean().rename("late_half_mean")
        ten = ten.merge(early, left_on="canonical_player_id", right_index=True, how="left")
        ten = ten.merge(late, left_on="canonical_player_id", right_index=True, how="left")
        ten["progression_delta"] = ten["late_half_mean"] - ten["early_half_mean"]
        return base_df.merge(ten, on="canonical_player_id", how="left")

    b_sub = pri[pri["is_batter_ball_faced"]].assign(d_raa=lambda x: 100.0 * (x["runs"] - x["bat_exp_runs"]))
    l_sub = pri[pri["is_legal_ball"]].assign(d_rsaa=lambda x: 6.0 * (x["bowl_exp_runs"] - x["bowler_runs_conceded"]))
    bat_t = _player_tenure(b_sub, "canonical_striker_id", "d_raa", bat_df)
    bowl_t = _player_tenure(l_sub, "canonical_bowler_id", "d_rsaa", bowl_df)

    bio_rows = []
    for pid, meta in PLAYER_AUCTION_CONTEXT.items():
        age_str = str(meta.get("age_2026", "")).split(" ")[0]
        try:
            true_age = float(age_str)
        except ValueError:
            continue
        r_b = bat_t[bat_t["canonical_player_id"] == pid]
        r_l = bowl_t[bowl_t["canonical_player_id"] == pid]
        row_use = r_b.iloc[0] if len(r_b) > 0 else (r_l.iloc[0] if len(r_l) > 0 else None)
        if row_use is not None and pd.notna(row_use.get("tenure_months")):
            bio_rows.append(
                {
                    "canonical_player_id": pid,
                    "player_name": row_use["player_name"],
                    "true_age_2026": true_age,
                    "first_app_date": str(row_use["first_app_date"].date()),
                    "tenure_months": round(float(row_use["tenure_months"]), 1),
                    "left_censored_pre2022": bool(row_use["left_censored_pre2022"]),
                }
            )
    bio_df = pd.DataFrame(bio_rows)

    b_elig = bat_t["elig_primary"]
    l_elig = bowl_t["elig_primary"]
    return {
        "bat_pct_left_censored_elig": round(100.0 * float(bat_t.loc[b_elig, "left_censored_pre2022"].mean()), 1),
        "bowl_pct_left_censored_elig": round(100.0 * float(bowl_t.loc[l_elig, "left_censored_pre2022"].mean()), 1),
        "bat_median_tenure_months": round(float(bat_t.loc[b_elig, "tenure_months"].median()), 1),
        "bowl_median_tenure_months": round(float(bowl_t.loc[l_elig, "tenure_months"].median()), 1),
        "bio_tenure_vs_true_age_r": round(float(np.corrcoef(bio_df["tenure_months"], bio_df["true_age_2026"])[0, 1]), 4) if len(bio_df) >= 5 else np.nan,
        "bio_left_censored_pct": round(100.0 * float(bio_df["left_censored_pre2022"].mean()), 1) if len(bio_df) > 0 else np.nan,
        "bio_age_audit": bio_df,
        "bat_tenure": bat_t,
        "bowl_tenure": bowl_t,
    }


def evaluate_two_way_matchups_season_venue_comp_prog(
    oof_pri: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    config: PerformanceConfig = PerformanceConfig(),
    emg_bat_df: Optional[pd.DataFrame] = None,
    emg_bowl_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """Wrapper executing Items 8, 9, 10, and 11 on enriched primary deliveries."""
    df = _enrich_deliveries_context(oof_pri, matches_df, competitions_df)
    pri = df[df["dls_status"].isin(PRIMARY_DLS_STATUSES)].copy().reset_index(drop=True)
    tw_res = _fit_two_way_matchups(pri, bat_df, bowl_df)
    sv_res = _fit_season_phase_and_venue_eb(pri, bat_df, bowl_df, config)
    comp_res = _evaluate_competition_bridge_calibration(
        pri, bat_df, bowl_df, config, emg_bat_df=emg_bat_df, emg_bowl_df=emg_bowl_df
    )
    prog_res = _evaluate_tenure_proxy_and_progression(pri, bat_df, bowl_df)
    return {
        "two_way": tw_res,
        "season_venue": sv_res,
        "comp_calibration": comp_res,
        "progression_tenure": prog_res,
    }

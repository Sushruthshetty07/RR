"""
Phase 3 Player Performance Under Pressure Engine for Rajasthan Royals SupeRR Selector 2.0.

Builds separate, zero-leakage batting and bowling pressure evaluations combining:
1. Strict DLS confidence partitioning:
   - Primary pool: 'standard' and 'dls_pre_innings_confirmed' only.
   - Sensitivity pool: optionally includes non-ambiguous suspected DLS
     ('dls_innings_break_suspected', 'dls_post_stoppage_suspected').
   - Always excluded: ambiguous DLS deliveries (dls_ambiguous == True, 143 rows)
     and Super Overs (innings >= 3, 38 rows).
2. Out-of-fold (5-fold GroupKFold by match_id) contextual Expected Runs & Expected
   Wickets baselines conditioning strictly on pre-delivery match/pitch context.
3. Continuous pressure-weighted residuals (with IPL primary & T20I supporting weights),
   Kish effective sample sizes (N_eff), match-level consistency rates, and high-pressure
   (PI >= 60) verification.
4. Empirical Bayes hierarchical shrinkage (Marginal MLE Normal-Normal for rate/impact
   metrics; Beta-Binomial shrinkage for match-level positive-impact consistency) with
   posterior uncertainty & Lower Credible Bound (LCB) auction reliability scoring.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
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


PRIMARY_DLS_STATUSES = ("standard", "dls_pre_innings_confirmed")
SUSPECTED_DLS_STATUSES = ("dls_innings_break_suspected", "dls_post_stoppage_suspected")

COMP_TIER_IPL = "IPL"
COMP_TIER_SENIOR_T20I = "Senior_T20I"
COMP_TIER_EMERGING = "Emerging_Rising_T20"
PRIMARY_COMP_TIERS = (COMP_TIER_IPL, COMP_TIER_SENIOR_T20I)


@dataclass(frozen=True)
class PerformanceConfig:
    w_t20i: float = 0.50
    w_emerging: float = 0.00
    exclude_emerging_from_primary: bool = True
    pressure_gamma: float = 2.0
    high_pressure_cutoff: float = 60.0
    wicket_run_value: float = 7.5
    n_oof_folds: int = 5
    lcb_z: float = 0.50
    disjoint_hp_split: bool = True

    # Composite pillar weights (Batting)
    bat_w_raa: float = 0.45
    bat_w_surv: float = 0.25
    bat_w_cons: float = 0.15
    bat_w_hp: float = 0.15

    # Composite pillar weights (Bowling: reduced WAA from 0.30 -> 0.15 after k0=4,602 EB audit)
    bowl_w_rsaa: float = 0.55
    bowl_w_waa: float = 0.15
    bowl_w_cons: float = 0.15
    bowl_w_hp: float = 0.15

    # Primary auction eligibility thresholds
    min_neff_elig: float = 30.0
    min_ipl_balls_bat: int = 30
    min_ipl_balls_bowl: int = 30


def classify_comp_tier(
    df: pd.DataFrame,
    matches_df: Optional[pd.DataFrame] = None,
    competitions_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Classifies every delivery deterministically into one of three competition tiers:
      - 'IPL': Indian Premier League seasons (primary weight = 1.00)
      - 'Senior_T20I': Senior T20 Internationals (supporting weight = 0.50)
      - 'Emerging_Rising_T20': Emerging Teams Asia Cup / Asia Cup Rising Stars
        (supplementary only; excluded from primary IPL+T20I rankings)
    """
    if "comp_tier" in df.columns and df["comp_tier"].notna().all():
        return df["comp_tier"].astype(str)

    work = df[["match_id"]].copy() if "match_id" in df.columns else pd.DataFrame(index=df.index)
    for col in ("comp_id", "comp_name", "is_ipl"):
        if col in df.columns:
            work[col] = df[col]

    if "comp_id" not in work.columns and matches_df is not None and "comp_id" in matches_df.columns:
        work = work.merge(matches_df[["match_id", "comp_id"]].drop_duplicates("match_id"), on="match_id", how="left")

    if "comp_name" not in work.columns and competitions_df is not None and "comp_name" in competitions_df.columns and "comp_id" in work.columns:
        c_cols = ["comp_id", "comp_name"] + (["match_format"] if "match_format" in competitions_df.columns else [])
        work = work.merge(competitions_df[c_cols].drop_duplicates("comp_id"), on="comp_id", how="left")

    if "comp_name" in work.columns:
        cname = work["comp_name"]
        has_name = cname.notna()
        tier = np.where(
            cname.fillna("").str.contains("IPL", case=False, na=False),
            COMP_TIER_IPL,
            np.where(
                cname.fillna("").str.contains("Emerging|Rising", case=False, na=False),
                COMP_TIER_EMERGING,
                COMP_TIER_SENIOR_T20I,
            ),
        )
        # Fallback to is_ipl only for rows missing competition names (e.g. minimal unit tests)
        if not has_name.all() and "is_ipl" in work.columns:
            fb = np.where(work["is_ipl"].fillna(False), COMP_TIER_IPL, COMP_TIER_SENIOR_T20I)
            tier = np.where(has_name, tier, fb)
        return pd.Series(tier, index=df.index, name="comp_tier")

    is_ipl_series = df["is_ipl"].fillna(False) if "is_ipl" in df.columns else pd.Series(False, index=df.index)
    return pd.Series(
        np.where(is_ipl_series, COMP_TIER_IPL, COMP_TIER_SENIOR_T20I),
        index=df.index,
        name="comp_tier",
    )


def filter_dls_pool(
    df: pd.DataFrame,
    include_suspected_dls: bool = False,
    matches_df: Optional[pd.DataFrame] = None,
    competitions_df: Optional[pd.DataFrame] = None,
    exclude_emerging: bool = False,
) -> pd.DataFrame:
    """
    Applies the Phase 3 DLS & Super Over eligibility policy:
    - Always excludes Super Overs (innings >= 3) and ambiguous DLS (dls_ambiguous == True).
    - Primary mode keeps only 'standard' and 'dls_pre_innings_confirmed'.
    - Sensitivity mode additionally retains non-ambiguous suspected DLS deliveries.
    - Optionally excludes 'Emerging_Rising_T20' deliveries when exclude_emerging=True.
    """
    allowed = list(PRIMARY_DLS_STATUSES)
    if include_suspected_dls:
        allowed.extend(SUSPECTED_DLS_STATUSES)

    mask = (
        (~df["is_super_over"].fillna(False))
        & (~df["dls_ambiguous"].fillna(False))
        & (df["dls_status"].isin(allowed))
        & df["pressure_bat"].notna()
        & df["pressure_bowl"].notna()
    )
    out = df.loc[mask].copy()
    if exclude_emerging:
        out["comp_tier"] = classify_comp_tier(out, matches_df=matches_df, competitions_df=competitions_df)
        out = out.loc[out["comp_tier"] != COMP_TIER_EMERGING].copy()
    return out


def _prepare_context_columns(
    df: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    if "venue_id" not in out.columns or "comp_id" not in out.columns:
        cols_needed = ["match_id"] + [c for c in ("venue_id", "comp_id") if c not in out.columns]
        out = out.merge(matches_df[cols_needed], on="match_id", how="left")
    if "season" not in out.columns:
        out = out.merge(competitions_df[["comp_id", "season"]], on="comp_id", how="left")

    out["comp_tier"] = classify_comp_tier(out, matches_df=matches_df, competitions_df=competitions_df)
    out["venue_id"] = out["venue_id"].fillna(-1)
    season_str = out["season"].fillna("2024").astype(str).str.slice(0, 4)
    out["season_year"] = pd.to_numeric(season_str, errors="coerce").fillna(2024).astype(int)
    out["is_ipl_int"] = (out["comp_tier"] == COMP_TIER_IPL).astype(int)
    out["ipl_phase"] = out["is_ipl_int"].astype(str) + "_" + out["match_phase"].astype(str)
    out["inn_phase"] = out["innings"].astype(str) + "_" + out["match_phase"].astype(str)
    return out


def _make_context_preprocessor(pressure_col: str) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=30),
                ["ipl_phase", "inn_phase", "overid", "venue_id"],
            ),
            (
                "spl",
                SplineTransformer(n_knots=6, degree=2, include_bias=False),
                [pressure_col, "wickets_in_hand_before"],
            ),
            ("num", StandardScaler(), ["season_year"]),
        ]
    )


def _fit_predict_baseline_slice(
    pool: pd.DataFrame,
    tr_pri: np.ndarray,
    va_idx: np.ndarray,
    is_bat_ball: np.ndarray,
    is_leg_ball: np.ndarray,
    pre_bat: ColumnTransformer,
    pre_bowl: ColumnTransformer,
    target_df: Optional[pd.DataFrame] = None,
) -> None:
    """Fits batting & bowling contextual baseline models on `tr_pri` and writes predictions to `va_idx`."""
    dst = pool if target_df is None else target_df
    va_bat_mask = (dst["is_batter_ball_faced"].fillna(False).values[va_idx]) if target_df is not None else is_bat_ball[va_idx]
    va_leg_mask = (dst["is_legal_ball"].fillna(False).values[va_idx]) if target_df is not None else is_leg_ball[va_idx]

    tr_bat = tr_pri[is_bat_ball[tr_pri]]
    va_bat = va_idx[va_bat_mask]
    if len(tr_bat) > 0 and len(va_bat) > 0:
        bat_r_pipe = Pipeline([("pre", pre_bat), ("reg", Ridge(alpha=30.0))])
        bat_r_pipe.fit(pool.iloc[tr_bat], pool["runs"].iloc[tr_bat].values)
        dst.iloc[va_bat, dst.columns.get_loc("bat_exp_runs")] = np.clip(
            bat_r_pipe.predict(dst.iloc[va_bat]), 0.10, 4.00
        )

        if pool["striker_out"].iloc[tr_bat].nunique() > 1:
            bat_w_pipe = Pipeline(
                [("pre", pre_bat), ("clf", LogisticRegression(C=0.5, max_iter=1000))]
            )
            bat_w_pipe.fit(pool.iloc[tr_bat], pool["striker_out"].iloc[tr_bat].values)
            dst.iloc[va_bat, dst.columns.get_loc("bat_exp_wkt")] = np.clip(
                bat_w_pipe.predict_proba(dst.iloc[va_bat])[:, 1], 0.002, 0.50
            )
        else:
            dst.iloc[va_bat, dst.columns.get_loc("bat_exp_wkt")] = float(
                pool["striker_out"].iloc[tr_bat].mean()
            )

    tr_leg = tr_pri[is_leg_ball[tr_pri]]
    va_leg = va_idx[va_leg_mask]
    if len(tr_leg) > 0 and len(va_leg) > 0:
        bowl_r_pipe = Pipeline([("pre", pre_bowl), ("reg", Ridge(alpha=30.0))])
        bowl_r_pipe.fit(pool.iloc[tr_leg], pool["bowler_runs_conceded"].iloc[tr_leg].values)
        raw_er = np.clip(bowl_r_pipe.predict(dst.iloc[va_leg]), 0.10, 4.00)
        dst.iloc[va_leg, dst.columns.get_loc("bowl_exp_runs_legal_raw")] = raw_er

        if pool["bowler_wkt_int"].iloc[tr_leg].nunique() > 1:
            bowl_w_pipe = Pipeline(
                [("pre", pre_bowl), ("clf", LogisticRegression(C=0.5, max_iter=1000))]
            )
            bowl_w_pipe.fit(pool.iloc[tr_leg], pool["bowler_wkt_int"].iloc[tr_leg].values)
            raw_ew = np.clip(bowl_w_pipe.predict_proba(dst.iloc[va_leg])[:, 1], 0.002, 0.50)
        else:
            raw_ew = np.full(len(va_leg), float(pool["bowler_wkt_int"].iloc[tr_leg].mean()))
        dst.iloc[va_leg, dst.columns.get_loc("bowl_exp_wkt_legal_raw")] = raw_ew

        tr_df = pool.iloc[tr_pri]
        mult = tr_df.groupby(["is_ipl_int", "match_phase", "innings"]).agg(
            tot_r=("bowler_runs_conceded", "sum"),
            leg_r=("bowler_runs_conceded", lambda s: s[tr_df.loc[s.index, "is_legal_ball"]].sum()),
            tot_w=("bowler_wkt_int", "sum"),
            leg_w=("bowler_wkt_int", lambda s: s[tr_df.loc[s.index, "is_legal_ball"]].sum()),
        )
        mult["r_mult"] = mult["tot_r"] / np.maximum(1.0, mult["leg_r"])
        mult["w_mult"] = mult["tot_w"] / np.maximum(1.0, mult["leg_w"])

        va_sub = dst.iloc[va_leg][["is_ipl_int", "match_phase", "innings"]].join(
            mult[["r_mult", "w_mult"]], on=["is_ipl_int", "match_phase", "innings"]
        )
        r_mult = va_sub["r_mult"].fillna(1.04).values
        w_mult = va_sub["w_mult"].fillna(1.00).values
        dst.iloc[va_leg, dst.columns.get_loc("bowl_exp_runs")] = raw_er * r_mult
        dst.iloc[va_leg, dst.columns.get_loc("bowl_exp_wkt")] = raw_ew * w_mult


def _init_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["striker_out"] = (
        out["is_wicket"].fillna(False)
        & (out["canonical_dismissed_player_id"] == out["canonical_striker_id"])
    ).astype(int)
    out["bowler_wkt_int"] = out["is_bowler_wicket"].fillna(False).astype(int)
    out["bat_exp_runs"] = np.nan
    out["bat_exp_wkt"] = np.nan
    out["bowl_exp_runs_legal_raw"] = np.nan
    out["bowl_exp_wkt_legal_raw"] = np.nan
    out["bowl_exp_runs"] = 0.0
    out["bowl_exp_wkt"] = 0.0
    return out


def fit_predict_oof_baselines(
    df: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    include_suspected_dls: bool = False,
    n_splits: int = 5,
    exclude_emerging: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fits zero-leakage GroupKFold (by match_id) contextual Expected Runs and Expected
    Wickets baselines for both batting and bowling.

    Models are ALWAYS trained strictly on primary deliveries ('standard' +
    'dls_pre_innings_confirmed') within each training fold, so even when suspected
    DLS deliveries are included for sensitivity, they never influence baseline fits.
    When exclude_emerging=True (default), 'Emerging_Rising_T20' deliveries are
    excluded from the primary pool so they never contaminate IPL/Senior T20I baselines.
    """
    pool = filter_dls_pool(df, include_suspected_dls=include_suspected_dls)
    pool = _prepare_context_columns(pool, matches_df, competitions_df)
    if exclude_emerging:
        pool = pool.loc[pool["comp_tier"] != COMP_TIER_EMERGING].copy()
    pool = _init_expected_columns(pool.reset_index(drop=True))

    is_primary_fit = pool["dls_status"].isin(PRIMARY_DLS_STATUSES).values

    n_matches = pool["match_id"].nunique()
    splits = min(n_splits, max(2, n_matches))
    gkf = GroupKFold(n_splits=splits)

    pre_bat = _make_context_preprocessor("pressure_bat")
    pre_bowl = _make_context_preprocessor("pressure_bowl")

    groups = pool["match_id"].values
    is_bat_ball = pool["is_batter_ball_faced"].fillna(False).values
    is_leg_ball = pool["is_legal_ball"].fillna(False).values

    for tr_idx, va_idx in gkf.split(pool, groups=groups):
        tr_pri = tr_idx[is_primary_fit[tr_idx]]
        if len(tr_pri) == 0:
            tr_pri = tr_idx
        _fit_predict_baseline_slice(
            pool, tr_pri, va_idx, is_bat_ball, is_leg_ball, pre_bat, pre_bowl
        )

    # Collect OOF validation & calibration diagnostics (on primary rows)
    pri_rows = pool[pool["dls_status"].isin(PRIMARY_DLS_STATUSES)]
    pri_bat = pri_rows[pri_rows["is_batter_ball_faced"]]
    pri_leg = pri_rows[pri_rows["is_legal_ball"]]

    diagnostics = {
        "n_pool_deliveries": int(len(pool)),
        "n_primary_deliveries": int(len(pri_rows)),
        "n_bat_balls_faced": int(len(pri_bat)),
        "n_bowl_all_balls": int(len(pri_rows)),
        "n_bowl_legal_balls": int(len(pri_leg)),
        "bat_runs_actual_mean": float(pri_bat["runs"].mean()),
        "bat_runs_exp_mean": float(pri_bat["bat_exp_runs"].mean()),
        "bat_runs_rmse": float(root_mean_squared_error(pri_bat["runs"], pri_bat["bat_exp_runs"])),
        "bat_runs_mae": float(mean_absolute_error(pri_bat["runs"], pri_bat["bat_exp_runs"])),
        "bat_runs_r2": float(r2_score(pri_bat["runs"], pri_bat["bat_exp_runs"])),
        "bat_wkt_actual_mean": float(pri_bat["striker_out"].mean()),
        "bat_wkt_exp_mean": float(pri_bat["bat_exp_wkt"].mean()),
        "bat_wkt_logloss": float(log_loss(pri_bat["striker_out"], pri_bat["bat_exp_wkt"]))
        if pri_bat["striker_out"].nunique() > 1
        else np.nan,
        "bat_wkt_brier": float(brier_score_loss(pri_bat["striker_out"], pri_bat["bat_exp_wkt"])),
        "bat_wkt_auc": float(roc_auc_score(pri_bat["striker_out"], pri_bat["bat_exp_wkt"]))
        if pri_bat["striker_out"].nunique() > 1
        else np.nan,
        "bowl_runs_total_actual": float(pri_rows["bowler_runs_conceded"].sum()),
        "bowl_runs_total_exp": float(pri_rows["bowl_exp_runs"].sum()),
        "bowl_runs_legal_rmse": float(
            root_mean_squared_error(pri_leg["bowler_runs_conceded"], pri_leg["bowl_exp_runs_legal_raw"])
        ),
        "bowl_wkt_total_actual": float(pri_rows["bowler_wkt_int"].sum()),
        "bowl_wkt_total_exp": float(pri_rows["bowl_exp_wkt"].sum()),
        "bowl_wkt_legal_logloss": float(log_loss(pri_leg["bowler_wkt_int"], pri_leg["bowl_exp_wkt_legal_raw"]))
        if pri_leg["bowler_wkt_int"].nunique() > 1
        else np.nan,
        "bowl_wkt_legal_auc": float(roc_auc_score(pri_leg["bowler_wkt_int"], pri_leg["bowl_exp_wkt_legal_raw"]))
        if pri_leg["bowler_wkt_int"].nunique() > 1
        else np.nan,
    }
    return pool, diagnostics


def fit_empirical_bayes_normal(
    estimates: np.ndarray,
    se2: np.ndarray,
    min_fit_se2_quantile: float = 0.70,
) -> Dict[str, float]:
    """
    Fits Normal-Normal Empirical Bayes hierarchical prior parameters (mu0, tau2) via
    Marginal Maximum Likelihood on finite observations:
        y_j ~ N(mu0, tau2 + se2_j)
    Returns dict with mu0, tau2, and marginal MLE convergence info.
    """
    valid = np.isfinite(estimates) & np.isfinite(se2) & (se2 > 0)
    y = estimates[valid]
    s2 = se2[valid]

    if len(y) < 3:
        return {"mu0": 0.0, "tau2": float(np.var(y)) if len(y) > 1 else 1.0}

    # Restrict upper optimization bound reasonably via precision-weighted spread
    a = 1.0 / s2
    mu_init = float(np.sum(a * y) / np.sum(a))
    var_upper = max(1e-4, float(np.var(y) * 3.0))

    def neg_log_lik(tau2: float) -> float:
        v = tau2 + s2
        inv_v = 1.0 / v
        mu = np.sum(y * inv_v) / np.sum(inv_v)
        return 0.5 * float(np.sum(np.log(v) + ((y - mu) ** 2) * inv_v))

    res = minimize_scalar(neg_log_lik, bounds=(1e-8, var_upper), method="bounded")
    tau2_hat = max(1e-8, float(res.x))
    v_opt = tau2_hat + s2
    mu0_hat = float(np.sum(y / v_opt) / np.sum(1.0 / v_opt))
    return {"mu0": mu0_hat, "tau2": tau2_hat}


def apply_normal_shrinkage(
    raw_est: pd.Series,
    n_eff: pd.Series,
    sigma2_within: float,
    min_neff_fit: float = 10.0,
    lcb_z: float = 0.50,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Applies Normal-Normal Empirical Bayes shrinkage to raw player metric estimates:
      B_j = tau2 / (tau2 + SE_j^2),  SE_j^2 = sigma2_within / N_eff_j
      post_j = B_j * raw_j + (1 - B_j) * mu0
      sd_j = sqrt(B_j * SE_j^2)
      lcb_j = post_j - lcb_z * sd_j
    """
    valid_fit = (n_eff >= min_neff_fit) & raw_est.notna() & np.isfinite(raw_est)
    if valid_fit.sum() < 5:
        valid_fit = (n_eff > 0) & raw_est.notna() & np.isfinite(raw_est)

    se2_fit = sigma2_within / np.maximum(1e-6, n_eff[valid_fit].values)
    eb = fit_empirical_bayes_normal(raw_est[valid_fit].values, se2_fit)
    mu0, tau2 = eb["mu0"], eb["tau2"]
    k0 = sigma2_within / max(1e-9, tau2)

    se2_all = np.where(n_eff > 0, sigma2_within / np.maximum(1e-9, n_eff.values), np.inf)
    B = np.where((n_eff > 0) & raw_est.notna(), tau2 / (tau2 + se2_all), 0.0)
    raw_safe = raw_est.fillna(mu0).values
    post = B * raw_safe + (1.0 - B) * mu0
    post_sd = np.sqrt(np.maximum(0.0, tau2 * (1.0 - B)))
    lcb = post - lcb_z * post_sd

    res_df = pd.DataFrame(
        {
            "post": post,
            "sd": post_sd,
            "lcb": lcb,
            "B": B,
        },
        index=raw_est.index,
    )
    meta = {
        "mu0": float(mu0),
        "sigma_within": float(np.sqrt(sigma2_within)),
        "sigma2_within": float(sigma2_within),
        "tau_between": float(np.sqrt(tau2)),
        "tau2": float(tau2),
        "k0_prior_neff": float(k0),
        "n_players_fit": int(valid_fit.sum()),
    }
    return res_df, meta


def apply_beta_binomial_shrinkage(
    pos_rate: pd.Series,
    m_eff: pd.Series,
    min_meff_fit: float = 5.0,
    lcb_z: float = 0.50,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Beta-Binomial Empirical Bayes shrinkage for pressured appearance consistency rate:
      p_post = (M_eff * p_raw + M0 * mu0) / (M_eff + M0)
    Estimates M0 via precision-weighted method-of-moments across appearances.
    """
    fit_mask = (m_eff >= min_meff_fit) & pos_rate.notna()
    if fit_mask.sum() < 5:
        fit_mask = (m_eff > 0) & pos_rate.notna()

    p_f = pos_rate[fit_mask].values
    m_f = m_eff[fit_mask].values
    mu0 = float(np.average(p_f, weights=m_f)) if len(p_f) > 0 else 0.50
    mu0 = float(np.clip(mu0, 0.05, 0.95))

    # Observed weighted variance minus average sampling variance mu0*(1-mu0)/M_eff
    if len(p_f) > 2:
        obs_var = float(np.average((p_f - mu0) ** 2, weights=m_f))
        noise_var = float(np.average(mu0 * (1.0 - mu0) / m_f, weights=m_f))
        true_var = max(1e-4, obs_var - noise_var)
        # Var(Beta) = mu0*(1-mu0) / (M0 + 1) => M0 = max(2, mu0*(1-mu0)/true_var - 1)
        m0 = float(np.clip(mu0 * (1.0 - mu0) / true_var - 1.0, 2.0, 50.0))
    else:
        m0 = 10.0

    alpha0 = m0 * mu0
    beta0 = m0 * (1.0 - mu0)
    m_all = np.maximum(0.0, m_eff.values)
    p_safe = pos_rate.fillna(mu0).values
    alpha_post = alpha0 + m_all * p_safe
    beta_post = beta0 + m_all * (1.0 - p_safe)
    denom = alpha_post + beta_post

    p_post = alpha_post / denom
    p_sd = np.sqrt((alpha_post * beta_post) / ((denom ** 2) * (denom + 1.0)))
    p_lcb = np.clip(p_post - lcb_z * p_sd, 0.0, 1.0)
    B = np.where(m_all > 0, m_all / (m_all + m0), 0.0)

    res_df = pd.DataFrame(
        {"post": p_post, "sd": p_sd, "lcb": p_lcb, "B": B}, index=pos_rate.index
    )
    meta = {
        "mu0": float(mu0),
        "m0_prior_apps": float(m0),
        "alpha0": float(alpha0),
        "beta0": float(beta0),
        "n_players_fit": int(fit_mask.sum()),
    }
    return res_df, meta


def _safe_z(series: pd.Series, ref_mask: pd.Series) -> pd.Series:
    sub = series[ref_mask]
    mu = sub.mean() if len(sub) > 1 else series.mean()
    sd = sub.std(ddof=1) if len(sub) > 1 else series.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-9:
        sd = 1.0
    return (series - mu) / sd


def _prepare_tier_weighted_pool(
    df: pd.DataFrame,
    config: PerformanceConfig,
    comp_weight_override: Optional[float] = None,
) -> pd.DataFrame:
    """Attaches comp_tier, filters Emerging_Rising_T20 when excluded, and sets comp weights."""
    out = df.copy()
    out["comp_tier"] = classify_comp_tier(out)
    if config.exclude_emerging_from_primary and comp_weight_override is None:
        out = out[out["comp_tier"] != COMP_TIER_EMERGING].copy()
    if comp_weight_override is not None:
        out["comp_w"] = float(comp_weight_override)
    else:
        tier_weights = {
            COMP_TIER_IPL: 1.0,
            COMP_TIER_SENIOR_T20I: float(config.w_t20i),
            COMP_TIER_EMERGING: float(config.w_emerging),
        }
        out["comp_w"] = out["comp_tier"].map(tier_weights).fillna(float(config.w_t20i)).astype(float)
    out["is_ipl_tier"] = out["comp_tier"] == COMP_TIER_IPL
    out["is_t20i_tier"] = out["comp_tier"] == COMP_TIER_SENIOR_T20I
    return out


def _aggregate_batting_core(
    bat_df: pd.DataFrame,
    config: PerformanceConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Computes delivery-level batting residuals, player spine aggregates, and General-partition consistency."""
    bat = bat_df.copy()
    id_col = "canonical_striker_id" if "canonical_striker_id" in bat.columns else "striker_id"
    name_col = "canonical_striker_name" if "canonical_striker_name" in bat.columns else "striker_name"

    bat["press_w"] = (bat["pressure_bat"].values / 100.0) ** config.pressure_gamma
    bat["w"] = bat["comp_w"] * bat["press_w"]
    bat["w2"] = bat["w"] ** 2

    # Delivery-level residuals per 100 balls
    bat["d_raa"] = 100.0 * (bat["runs"] - bat["bat_exp_runs"])
    bat["d_surv"] = 100.0 * (bat["bat_exp_wkt"] - bat["striker_out"])
    bat["d_net"] = bat["d_raa"] + config.wicket_run_value * bat["d_surv"]

    bat["wy_raa"] = bat["w"] * bat["d_raa"]
    bat["wy_surv"] = bat["w"] * bat["d_surv"]
    bat["wy_net"] = bat["w"] * bat["d_net"]

    bat["is_hp"] = bat["pressure_bat"] >= config.high_pressure_cutoff
    bat["is_gen"] = (~bat["is_hp"]) if config.disjoint_hp_split else True

    bat["w_gen"] = np.where(bat["is_gen"], bat["w"], 0.0)
    bat["w_gen2"] = bat["w_gen"] ** 2
    bat["wy_gen_raa"] = np.where(bat["is_gen"], bat["wy_raa"], 0.0)
    bat["wy_gen_surv"] = np.where(bat["is_gen"], bat["wy_surv"], 0.0)

    bat["w_hp"] = np.where(bat["is_hp"], bat["w"], 0.0)
    bat["w_hp2"] = bat["w_hp"] ** 2
    bat["wy_hp_raa"] = np.where(bat["is_hp"], bat["wy_raa"], 0.0)

    bat["ipl_ball"] = bat["is_ipl_tier"].astype(int)
    bat["t20i_ball"] = bat["is_t20i_tier"].astype(int)
    bat["hp_ball"] = bat["is_hp"].astype(int)
    bat["ipl_hp_ball"] = (bat["is_ipl_tier"] & bat["is_hp"]).astype(int)

    # Aggregate player spine across all regular balls faced
    agg = bat.groupby(id_col, as_index=False).agg(
        player_name=(name_col, "first"),
        total_balls=("runs", "count"),
        ipl_balls=("ipl_ball", "sum"),
        t20i_balls=("t20i_ball", "sum"),
        hp60_balls=("hp_ball", "sum"),
        ipl_hp60_balls=("ipl_hp_ball", "sum"),
        n_matches=("match_id", "nunique"),
        mean_pi=("pressure_bat", "mean"),
        sw=("w", "sum"),
        sw2=("w2", "sum"),
        swy_net_all=("wy_net", "sum"),
        sw_gen=("w_gen", "sum"),
        sw_gen2=("w_gen2", "sum"),
        swy_raa=("wy_gen_raa", "sum"),
        swy_surv=("wy_gen_surv", "sum"),
        sw_hp=("w_hp", "sum"),
        sw_hp2=("w_hp2", "sum"),
        swy_hp_raa=("wy_hp_raa", "sum"),
    ).rename(columns={id_col: "canonical_player_id"})

    agg["neff"] = np.where(agg["sw2"] > 0, (agg["sw"] ** 2) / agg["sw2"], 0.0)
    agg["neff_gen"] = np.where(agg["sw_gen2"] > 0, (agg["sw_gen"] ** 2) / agg["sw_gen2"], 0.0)
    agg["neff_hp"] = np.where(agg["sw_hp2"] > 0, (agg["sw_hp"] ** 2) / agg["sw_hp2"], 0.0)

    agg["raa_raw"] = np.where(agg["sw_gen"] > 0, agg["swy_raa"] / agg["sw_gen"], np.nan)
    agg["surv_raw"] = np.where(agg["sw_gen"] > 0, agg["swy_surv"] / agg["sw_gen"], np.nan)
    agg["ev_net_all_raw"] = np.where(agg["sw"] > 0, agg["swy_net_all"] / agg["sw"], np.nan)
    agg["net_raw"] = agg["ev_net_all_raw"]
    agg["hp_raa_raw"] = np.where(agg["sw_hp"] > 0, agg["swy_hp_raa"] / agg["sw_hp"], np.nan)

    # Match-innings consistency strictly over General (PI < 60) deliveries when disjoint
    bat_gen = bat[bat["is_gen"]].copy()
    inn = (
        bat_gen.groupby([id_col, "match_id", "innings"], as_index=False)
        .agg(
            sw=("w", "sum"),
            swy_net=("wy_net", "sum"),
            max_pi=("pressure_bat", "max"),
            comp_w=("comp_w", "first"),
        )
    )
    inn = inn[inn["sw"] > 0].copy()
    inn["inn_impact"] = inn["swy_net"] / inn["sw"]
    inn["pos_inn"] = (inn["inn_impact"] > 0).astype(float)
    # Cap per-innings contribution weight so one long knock does not dominate consistency
    inn["u_m"] = (
        inn["comp_w"]
        * ((inn["max_pi"] / 100.0) ** (config.pressure_gamma / 2.0))
        * np.minimum(1.0, np.sqrt(inn["sw"]) / 1.5)
    )
    inn["u_m2"] = inn["u_m"] ** 2
    inn["u_pos"] = inn["u_m"] * inn["pos_inn"]

    inn_agg = inn.groupby(id_col, as_index=False).agg(
        su=("u_m", "sum"),
        su2=("u_m2", "sum"),
        supos=("u_pos", "sum"),
    ).rename(columns={id_col: "canonical_player_id"})
    inn_agg["meff_inn"] = np.where(inn_agg["su2"] > 0, (inn_agg["su"] ** 2) / inn_agg["su2"], 0.0)
    inn_agg["cons_raw"] = np.where(inn_agg["su"] > 0, inn_agg["supos"] / inn_agg["su"], np.nan)

    agg = agg.merge(inn_agg[["canonical_player_id", "meff_inn", "cons_raw"]], on="canonical_player_id", how="left")
    agg["meff_inn"] = agg["meff_inn"].fillna(0.0)
    return agg, bat, bat_gen


def evaluate_batting_pressure(
    oof_df: pd.DataFrame,
    config: PerformanceConfig = PerformanceConfig(),
    player_dir: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Computes continuous pressure-weighted, expected-adjusted, Empirical-Bayes-shrunk
    batting performance metrics for all canonical batters in the primary competition pool
    (IPL + Senior_T20I when exclude_emerging_from_primary=True).

    When config.disjoint_hp_split is True (default), deliveries are strictly
    partitioned with zero overlap:
      - General scoring (RAA), survival (Surv), and match consistency: PI < cutoff (60)
      - Dedicated high-pressure pillar (HP-RAA): PI >= cutoff (60)
      - Overall eligibility Kish N_eff and full-career EV benchmark span all balls.
    """
    bat_pool = _prepare_tier_weighted_pool(oof_df[oof_df["is_batter_ball_faced"]], config)
    agg, bat, bat_gen = _aggregate_batting_core(bat_pool, config)

    # Pooled within-player delivery noise variance inside each respective partition
    w_gen_arr = np.maximum(bat_gen["w"].values, 1e-12)
    sig2_raa = float(np.average(bat_gen["d_raa"].values ** 2, weights=w_gen_arr)) if len(bat_gen) > 0 else 1.0
    sig2_surv = float(np.average(bat_gen["d_surv"].values ** 2, weights=w_gen_arr)) if len(bat_gen) > 0 else 1.0

    w_all_arr = np.maximum(bat["w"].values, 1e-12)
    sig2_ev_all = float(np.average(bat["d_net"].values ** 2, weights=w_all_arr))

    hp_sub = bat[bat["is_hp"]]
    w_hp_arr = np.maximum(hp_sub["w"].values, 1e-12)
    sig2_hp = float(np.average(hp_sub["d_raa"].values ** 2, weights=w_hp_arr)) if len(hp_sub) > 0 else sig2_raa

    # Empirical Bayes Shrinkage (each partition uses its own disjoint Kish effective N)
    eb_raa, meta_raa = apply_normal_shrinkage(agg["raa_raw"], agg["neff_gen"], sig2_raa, min_neff_fit=15.0, lcb_z=config.lcb_z)
    eb_surv, meta_surv = apply_normal_shrinkage(agg["surv_raw"], agg["neff_gen"], sig2_surv, min_neff_fit=15.0, lcb_z=config.lcb_z)
    eb_hp, meta_hp = apply_normal_shrinkage(agg["hp_raa_raw"], agg["neff_hp"], sig2_hp, min_neff_fit=10.0, lcb_z=config.lcb_z)
    eb_cons, meta_cons = apply_beta_binomial_shrinkage(agg["cons_raw"], agg["meff_inn"], min_meff_fit=5.0, lcb_z=config.lcb_z)
    eb_ev_all, meta_ev_all = apply_normal_shrinkage(agg["ev_net_all_raw"], agg["neff"], sig2_ev_all, min_neff_fit=15.0, lcb_z=config.lcb_z)

    for prefix, eb_df in [
        ("raa", eb_raa),
        ("surv", eb_surv),
        ("hp_raa", eb_hp),
        ("cons", eb_cons),
        ("ev_all", eb_ev_all),
    ]:
        agg[f"{prefix}_post"] = eb_df["post"]
        agg[f"{prefix}_sd"] = eb_df["sd"]
        agg[f"{prefix}_lcb"] = eb_df["lcb"]
        agg[f"{prefix}_B"] = eb_df["B"]

    # Unified Expected-Value Run Equivalents (general-partition sum and direct all-ball EB)
    agg["ev_net_raw"] = agg["raa_raw"] + config.wicket_run_value * agg["surv_raw"]
    agg["ev_net_post"] = agg["raa_post"] + config.wicket_run_value * agg["surv_post"]
    agg["ev_net_lcb"] = agg["raa_lcb"] + config.wicket_run_value * agg["surv_lcb"]

    # Eligibility flags (total pressure N_eff + actual IPL balls faced)
    agg["is_ipl_exposed"] = agg["ipl_balls"] > 0
    agg["elig_primary"] = (agg["neff"] >= config.min_neff_elig) & (agg["ipl_balls"] >= config.min_ipl_balls_bat)
    agg["elig_broad"] = agg["neff"] >= config.min_neff_elig

    # Standardize posterior & LCB components against eligible broad pool
    ref_mask = agg["elig_broad"] if agg["elig_broad"].sum() >= 20 else pd.Series(True, index=agg.index)
    z_raa_p = _safe_z(agg["raa_post"], ref_mask)
    z_surv_p = _safe_z(agg["surv_post"], ref_mask)
    z_cons_p = _safe_z(agg["cons_post"], ref_mask)
    z_hp_p = _safe_z(agg["hp_raa_post"], ref_mask)

    z_raa_l = _safe_z(agg["raa_lcb"], ref_mask)
    z_surv_l = _safe_z(agg["surv_lcb"], ref_mask)
    z_cons_l = _safe_z(agg["cons_lcb"], ref_mask)
    z_hp_l = _safe_z(agg["hp_raa_lcb"], ref_mask)

    agg["pri_z_post"] = (
        config.bat_w_raa * z_raa_p
        + config.bat_w_surv * z_surv_p
        + config.bat_w_cons * z_cons_p
        + config.bat_w_hp * z_hp_p
    )
    agg["pri_z_lcb"] = (
        config.bat_w_raa * z_raa_l
        + config.bat_w_surv * z_surv_l
        + config.bat_w_cons * z_cons_l
        + config.bat_w_hp * z_hp_l
    )
    agg["pri_score_post"] = 100.0 * norm.cdf(agg["pri_z_post"])
    agg["pri_score_lcb"] = 100.0 * norm.cdf(agg["pri_z_lcb"])

    if player_dir is not None and "canonical_player_id" in player_dir.columns:
        cols = [c for c in ("canonical_player_id", "original_player_ids", "teams") if c in player_dir.columns]
        agg = agg.merge(player_dir[cols], on="canonical_player_id", how="left")

    eb_meta = {
        "raa": meta_raa,
        "surv": meta_surv,
        "hp_raa": meta_hp,
        "cons": meta_cons,
        "ev_all": meta_ev_all,
    }
    return agg, eb_meta


def _aggregate_bowling_core(
    bowl_df: pd.DataFrame,
    config: PerformanceConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Computes delivery-level bowling residuals, player spine aggregates, and General-partition spell consistency."""
    bowl = bowl_df.copy()
    id_col = "canonical_bowler_id" if "canonical_bowler_id" in bowl.columns else "bowler_id"
    name_col = "canonical_bowler_name" if "canonical_bowler_name" in bowl.columns else "bowler_name"

    bowl["press_w"] = (bowl["pressure_bowl"].values / 100.0) ** config.pressure_gamma
    bowl["w"] = bowl["comp_w"] * bowl["press_w"]
    bowl["leg_f"] = bowl["is_legal_ball"].astype(float)
    bowl["wl"] = bowl["w"] * bowl["leg_f"]
    bowl["wl2"] = bowl["wl"] ** 2

    # Residuals scaled to per-over (x6 runs saved) and per-4-over-spell (x24 wickets above exp)
    bowl["d_rsaa"] = 6.0 * (bowl["leg_f"] * bowl["bowl_exp_runs"] - bowl["bowler_runs_conceded"])
    bowl["d_waa"] = 24.0 * (bowl["bowler_wkt_int"].astype(float) - bowl["leg_f"] * bowl["bowl_exp_wkt"])
    # Combined per-over run-value saved equivalent (1 wicket = V_wkt runs => waa/4 = wkts/over)
    bowl["d_net_po"] = bowl["d_rsaa"] + (config.wicket_run_value / 4.0) * bowl["d_waa"]

    bowl["wy_rsaa"] = bowl["w"] * bowl["d_rsaa"]
    bowl["wy_waa"] = bowl["w"] * bowl["d_waa"]
    bowl["wy_net"] = bowl["w"] * bowl["d_net_po"]

    bowl["is_hp"] = bowl["pressure_bowl"] >= config.high_pressure_cutoff
    bowl["is_gen"] = (~bowl["is_hp"]) if config.disjoint_hp_split else True

    bowl["wl_gen"] = np.where(bowl["is_gen"], bowl["wl"], 0.0)
    bowl["wl_gen2"] = bowl["wl_gen"] ** 2
    bowl["wy_gen_rsaa"] = np.where(bowl["is_gen"], bowl["wy_rsaa"], 0.0)
    bowl["wy_gen_waa"] = np.where(bowl["is_gen"], bowl["wy_waa"], 0.0)

    bowl["wl_hp"] = np.where(bowl["is_hp"], bowl["wl"], 0.0)
    bowl["wl_hp2"] = bowl["wl_hp"] ** 2
    bowl["wy_hp_net"] = np.where(bowl["is_hp"], bowl["wy_net"], 0.0)

    bowl["ipl_leg"] = (bowl["is_ipl_tier"] & bowl["is_legal_ball"]).astype(int)
    bowl["t20i_leg"] = (bowl["is_t20i_tier"] & bowl["is_legal_ball"]).astype(int)
    bowl["leg_int"] = bowl["is_legal_ball"].astype(int)
    bowl["hp_leg"] = (bowl["is_hp"] & bowl["is_legal_ball"]).astype(int)
    bowl["ipl_hp_leg"] = (bowl["is_ipl_tier"] & bowl["is_hp"] & bowl["is_legal_ball"]).astype(int)

    agg = bowl.groupby(id_col, as_index=False).agg(
        player_name=(name_col, "first"),
        total_deliveries=("bowler_runs_conceded", "count"),
        total_legal_balls=("leg_int", "sum"),
        ipl_legal_balls=("ipl_leg", "sum"),
        t20i_legal_balls=("t20i_leg", "sum"),
        hp60_legal_balls=("hp_leg", "sum"),
        ipl_hp60_legal_balls=("ipl_hp_leg", "sum"),
        n_matches=("match_id", "nunique"),
        mean_pi=("pressure_bowl", "mean"),
        swl=("wl", "sum"),
        swl2=("wl2", "sum"),
        swy_net_all=("wy_net", "sum"),
        swl_gen=("wl_gen", "sum"),
        swl_gen2=("wl_gen2", "sum"),
        swy_rsaa=("wy_gen_rsaa", "sum"),
        swy_waa=("wy_gen_waa", "sum"),
        swl_hp=("wl_hp", "sum"),
        swl_hp2=("wl_hp2", "sum"),
        swy_hp_net=("wy_hp_net", "sum"),
    ).rename(columns={id_col: "canonical_player_id"})

    agg["neff"] = np.where(agg["swl2"] > 0, (agg["swl"] ** 2) / agg["swl2"], 0.0)
    agg["neff_gen"] = np.where(agg["swl_gen2"] > 0, (agg["swl_gen"] ** 2) / agg["swl_gen2"], 0.0)
    agg["neff_hp"] = np.where(agg["swl_hp2"] > 0, (agg["swl_hp"] ** 2) / agg["swl_hp2"], 0.0)

    agg["rsaa_raw"] = np.where(agg["swl_gen"] > 0, agg["swy_rsaa"] / agg["swl_gen"], np.nan)
    agg["waa_raw"] = np.where(agg["swl_gen"] > 0, agg["swy_waa"] / agg["swl_gen"], np.nan)
    agg["ev_net_all_raw"] = np.where(agg["swl"] > 0, agg["swy_net_all"] / agg["swl"], np.nan)
    agg["net_po_raw"] = agg["ev_net_all_raw"]
    agg["hp_impact_raw"] = np.where(agg["swl_hp"] > 0, agg["swy_hp_net"] / agg["swl_hp"], np.nan)

    # Match-spell consistency strictly over General (PI < 60) deliveries when disjoint
    bowl_gen = bowl[bowl["is_gen"]].copy()
    spell = (
        bowl_gen.groupby([id_col, "match_id", "innings"], as_index=False)
        .agg(
            swl=("wl", "sum"),
            swy_net=("wy_net", "sum"),
            max_pi=("pressure_bowl", "max"),
            comp_w=("comp_w", "first"),
        )
    )
    spell = spell[spell["swl"] > 0].copy()
    spell["spell_impact"] = spell["swy_net"] / spell["swl"]
    spell["pos_spell"] = (spell["spell_impact"] > 0).astype(float)
    spell["u_m"] = (
        spell["comp_w"]
        * ((spell["max_pi"] / 100.0) ** (config.pressure_gamma / 2.0))
        * np.minimum(1.0, np.sqrt(spell["swl"]) / 2.0)
    )
    spell["u_m2"] = spell["u_m"] ** 2
    spell["u_pos"] = spell["u_m"] * spell["pos_spell"]

    sp_agg = spell.groupby(id_col, as_index=False).agg(
        su=("u_m", "sum"),
        su2=("u_m2", "sum"),
        supos=("u_pos", "sum"),
    ).rename(columns={id_col: "canonical_player_id"})
    sp_agg["meff_spell"] = np.where(sp_agg["su2"] > 0, (sp_agg["su"] ** 2) / sp_agg["su2"], 0.0)
    sp_agg["cons_raw"] = np.where(sp_agg["su"] > 0, sp_agg["supos"] / sp_agg["su"], np.nan)

    agg = agg.merge(sp_agg[["canonical_player_id", "meff_spell", "cons_raw"]], on="canonical_player_id", how="left")
    agg["meff_spell"] = agg["meff_spell"].fillna(0.0)
    return agg, bowl, bowl_gen


def _merge_player_dir(agg: pd.DataFrame, player_dir: Optional[pd.DataFrame]) -> pd.DataFrame:
    if player_dir is not None and "canonical_player_id" in player_dir.columns:
        cols = [c for c in ("canonical_player_id", "original_player_ids", "teams") if c in player_dir.columns]
        return agg.merge(player_dir[cols], on="canonical_player_id", how="left")
    return agg


def evaluate_bowling_pressure(
    oof_df: pd.DataFrame,
    config: PerformanceConfig = PerformanceConfig(),
    player_dir: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Computes continuous pressure-weighted, expected-adjusted, Empirical-Bayes-shrunk
    bowling performance metrics (including wides/no-balls in conceded runs & wickets)
    for the primary competition pool (IPL + Senior_T20I when exclude_emerging_from_primary=True).

    When config.disjoint_hp_split is True (default), deliveries are strictly
    partitioned with zero overlap:
      - General runs saved (RSAA), wickets (WAA), and spell consistency: PI < cutoff (60)
      - Dedicated high-pressure pillar (HP-Impact): PI >= cutoff (60)
      - Overall eligibility Kish N_eff and full-career EV benchmark span all balls.
    """
    bowl_pool = _prepare_tier_weighted_pool(oof_df, config)
    agg, bowl, bowl_gen = _aggregate_bowling_core(bowl_pool, config)

    # Pooled within-bowler delivery noise variance inside each partition
    wl_gen_arr = np.maximum(bowl_gen["wl"].values, 1e-12)
    sig2_rsaa = float(np.average(bowl_gen["d_rsaa"].values ** 2, weights=wl_gen_arr)) if len(bowl_gen) > 0 else 1.0
    sig2_waa = float(np.average(bowl_gen["d_waa"].values ** 2, weights=wl_gen_arr)) if len(bowl_gen) > 0 else 1.0

    wl_all_arr = np.maximum(bowl["wl"].values, 1e-12)
    sig2_ev_all = float(np.average(bowl["d_net_po"].values ** 2, weights=wl_all_arr))

    hp_sub = bowl[bowl["is_hp"]]
    wl_hp_arr = np.maximum(hp_sub["wl"].values, 1e-12)
    sig2_hp = float(np.average(hp_sub["d_net_po"].values ** 2, weights=wl_hp_arr)) if len(hp_sub) > 0 else sig2_rsaa

    eb_rsaa, meta_rsaa = apply_normal_shrinkage(agg["rsaa_raw"], agg["neff_gen"], sig2_rsaa, min_neff_fit=15.0, lcb_z=config.lcb_z)
    eb_waa, meta_waa = apply_normal_shrinkage(agg["waa_raw"], agg["neff_gen"], sig2_waa, min_neff_fit=15.0, lcb_z=config.lcb_z)
    eb_hp, meta_hp = apply_normal_shrinkage(agg["hp_impact_raw"], agg["neff_hp"], sig2_hp, min_neff_fit=10.0, lcb_z=config.lcb_z)
    eb_cons, meta_cons = apply_beta_binomial_shrinkage(agg["cons_raw"], agg["meff_spell"], min_meff_fit=5.0, lcb_z=config.lcb_z)
    eb_ev_all, meta_ev_all = apply_normal_shrinkage(agg["ev_net_all_raw"], agg["neff"], sig2_ev_all, min_neff_fit=15.0, lcb_z=config.lcb_z)

    for prefix, eb_df in [
        ("rsaa", eb_rsaa),
        ("waa", eb_waa),
        ("hp_impact", eb_hp),
        ("cons", eb_cons),
        ("ev_all", eb_ev_all),
    ]:
        agg[f"{prefix}_post"] = eb_df["post"]
        agg[f"{prefix}_sd"] = eb_df["sd"]
        agg[f"{prefix}_lcb"] = eb_df["lcb"]
        agg[f"{prefix}_B"] = eb_df["B"]

    agg["ev_net_po_raw"] = agg["rsaa_raw"] + (config.wicket_run_value / 4.0) * agg["waa_raw"]
    agg["ev_net_po_post"] = agg["rsaa_post"] + (config.wicket_run_value / 4.0) * agg["waa_post"]
    agg["ev_net_po_lcb"] = agg["rsaa_lcb"] + (config.wicket_run_value / 4.0) * agg["waa_lcb"]

    agg["is_ipl_exposed"] = agg["ipl_legal_balls"] > 0
    agg["elig_primary"] = (agg["neff"] >= config.min_neff_elig) & (agg["ipl_legal_balls"] >= config.min_ipl_balls_bowl)
    agg["elig_broad"] = agg["neff"] >= config.min_neff_elig

    ref_mask = agg["elig_broad"] if agg["elig_broad"].sum() >= 20 else pd.Series(True, index=agg.index)
    z_rsaa_p = _safe_z(agg["rsaa_post"], ref_mask)
    z_waa_p = _safe_z(agg["waa_post"], ref_mask)
    z_cons_p = _safe_z(agg["cons_post"], ref_mask)
    z_hp_p = _safe_z(agg["hp_impact_post"], ref_mask)

    z_rsaa_l = _safe_z(agg["rsaa_lcb"], ref_mask)
    z_waa_l = _safe_z(agg["waa_lcb"], ref_mask)
    z_cons_l = _safe_z(agg["cons_lcb"], ref_mask)
    z_hp_l = _safe_z(agg["hp_impact_lcb"], ref_mask)

    agg["pri_z_post"] = (
        config.bowl_w_rsaa * z_rsaa_p
        + config.bowl_w_waa * z_waa_p
        + config.bowl_w_cons * z_cons_p
        + config.bowl_w_hp * z_hp_p
    )
    agg["pri_z_lcb"] = (
        config.bowl_w_rsaa * z_rsaa_l
        + config.bowl_w_waa * z_waa_l
        + config.bowl_w_cons * z_cons_l
        + config.bowl_w_hp * z_hp_l
    )
    agg["pri_score_post"] = 100.0 * norm.cdf(agg["pri_z_post"])
    agg["pri_score_lcb"] = 100.0 * norm.cdf(agg["pri_z_lcb"])

    agg = _merge_player_dir(agg, player_dir)

    eb_meta = {
        "rsaa": meta_rsaa,
        "waa": meta_waa,
        "hp_impact": meta_hp,
        "cons": meta_cons,
        "ev_all": meta_ev_all,
    }
    return agg, eb_meta


def evaluate_emerging_supplementary(
    df: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    config: PerformanceConfig = PerformanceConfig(),
    include_suspected_dls: bool = False,
    player_dir: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Evaluates Emerging/Rising T20 deliveries strictly as supplementary evidence.

    Contextual expected-runs/wickets baselines are trained ONLY on primary
    IPL + Senior_T20I deliveries (standard + confirmed DLS) and scored out-of-sample
    on Emerging_Rising_T20 deliveries so Emerging competitions never contaminate
    primary rankings or EB shrinkage priors.
    """
    full_pool = filter_dls_pool(df, include_suspected_dls=include_suspected_dls)
    full_pool = _prepare_context_columns(full_pool, matches_df, competitions_df)

    pri_pool = _init_expected_columns(
        full_pool[full_pool["comp_tier"].isin(PRIMARY_COMP_TIERS)].reset_index(drop=True)
    )
    emg_pool = _init_expected_columns(
        full_pool[full_pool["comp_tier"] == COMP_TIER_EMERGING].reset_index(drop=True)
    )

    if len(emg_pool) == 0 or len(pri_pool) == 0:
        return pd.DataFrame(), pd.DataFrame(), {
            "n_emerging_rows": int(len(emg_pool)),
            "n_emerging_matches": int(emg_pool["match_id"].nunique()) if len(emg_pool) > 0 else 0,
            "n_emerging_batters": 0,
            "n_emerging_bowlers": 0,
            "n_primary_fit_rows": 0,
        }

    tr_pri = np.flatnonzero(pri_pool["dls_status"].isin(PRIMARY_DLS_STATUSES).values)
    if len(tr_pri) == 0:
        tr_pri = np.arange(len(pri_pool))
    _fit_predict_baseline_slice(
        pri_pool,
        tr_pri,
        np.arange(len(emg_pool)),
        pri_pool["is_batter_ball_faced"].fillna(False).values,
        pri_pool["is_legal_ball"].fillna(False).values,
        _make_context_preprocessor("pressure_bat"),
        _make_context_preprocessor("pressure_bowl"),
        target_df=emg_pool,
    )

    bat_src = emg_pool[emg_pool["is_batter_ball_faced"]]
    if len(bat_src) > 0:
        bat_emg_in = _prepare_tier_weighted_pool(bat_src, config, comp_weight_override=1.0)
        bat_raw, _, _ = _aggregate_batting_core(bat_emg_in, config)
        bat_rename = {
            "total_balls": "emerging_balls",
            "hp60_balls": "emerging_hp60_balls",
            "n_matches": "emerging_matches",
            "mean_pi": "emerging_mean_pi",
            "neff": "emerging_neff",
            "neff_hp": "emerging_neff_hp",
            "meff_inn": "emerging_meff",
            "raa_raw": "emerging_raa_raw",
            "surv_raw": "emerging_surv_raw",
            "hp_raa_raw": "emerging_hp_raa_raw",
            "ev_net_all_raw": "emerging_ev_net_raw",
            "cons_raw": "emerging_cons_raw",
        }
        keep_bat = ["canonical_player_id", "player_name"] + list(bat_rename.keys())
        emg_bat = bat_raw[keep_bat].rename(columns=bat_rename)
        if "comp_name" in bat_emg_in.columns:
            id_col = "canonical_striker_id" if "canonical_striker_id" in bat_emg_in.columns else "striker_id"
            comp_map = (
                bat_emg_in.groupby(id_col)["comp_name"]
                .apply(lambda s: "; ".join(sorted(set(s.dropna().astype(str)))))
                .rename("emerging_competitions")
            )
            emg_bat = emg_bat.merge(comp_map, left_on="canonical_player_id", right_index=True, how="left")
        emg_bat = _merge_player_dir(emg_bat, player_dir)
    else:
        emg_bat = pd.DataFrame()

    bowl_emg_in = _prepare_tier_weighted_pool(emg_pool, config, comp_weight_override=1.0)
    bowl_raw, _, _ = _aggregate_bowling_core(bowl_emg_in, config)
    bowl_rename = {
        "total_deliveries": "emerging_deliveries",
        "total_legal_balls": "emerging_legal_balls",
        "hp60_legal_balls": "emerging_hp60_legal_balls",
        "n_matches": "emerging_matches",
        "mean_pi": "emerging_mean_pi",
        "neff": "emerging_neff",
        "neff_hp": "emerging_neff_hp",
        "meff_spell": "emerging_meff_spell",
        "rsaa_raw": "emerging_rsaa_raw",
        "waa_raw": "emerging_waa_raw",
        "hp_impact_raw": "emerging_hp_impact_raw",
        "ev_net_all_raw": "emerging_ev_net_po_raw",
        "cons_raw": "emerging_cons_raw",
    }
    keep_bowl = ["canonical_player_id", "player_name"] + list(bowl_rename.keys())
    emg_bowl = bowl_raw[keep_bowl].rename(columns=bowl_rename)
    if "comp_name" in bowl_emg_in.columns:
        id_col = "canonical_bowler_id" if "canonical_bowler_id" in bowl_emg_in.columns else "bowler_id"
        comp_map = (
            bowl_emg_in.groupby(id_col)["comp_name"]
            .apply(lambda s: "; ".join(sorted(set(s.dropna().astype(str)))))
            .rename("emerging_competitions")
        )
        emg_bowl = emg_bowl.merge(comp_map, left_on="canonical_player_id", right_index=True, how="left")
    emg_bowl = _merge_player_dir(emg_bowl, player_dir)

    diag = {
        "n_emerging_rows": int(len(emg_pool)),
        "n_emerging_matches": int(emg_pool["match_id"].nunique()),
        "n_emerging_batters": int(len(emg_bat)),
        "n_emerging_bowlers": int(len(emg_bowl)),
        "n_primary_fit_rows": int(len(tr_pri)),
    }
    return emg_bat, emg_bowl, diag


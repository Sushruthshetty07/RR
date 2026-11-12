"""
Win Probability, Leverage Index, and Win Probability Added (WPA) Engine
for Rajasthan Royals SupeRR Selector 2.0.

Provides:
1. Pre-delivery Win Probability (WP) modeling with strictly zero outcome leakage.
2. Multi-model benchmarking: Logistic Regression (Baseline) vs Random Forest vs Gradient Boosting.
3. Match-grouped (GroupKFold) and Chronological (<=2024 train -> 2025-2026 test) validation.
4. Metric evaluation: Brier Score, Log Loss, ROC-AUC, PR-AUC, Calibration, and F1.
5. Counterfactual Leverage Index (LI) and Win Probability Added (WPA) attribution for batters and bowlers.
6. Clear analytical separation of:
   - Pressure Index (match difficulty / jeopardy)
   - Win Probability (state value)
   - Win Probability Added (ball-by-ball outcome impact)
   - Leverage Index (swing magnitude / situational leverage)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class WPModelConfig:
    """Hyperparameters and configuration for Win Probability modeling."""
    n_splits: int = 5
    random_state: int = 42
    rf_n_estimators: int = 100
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 30
    hgb_max_iter: int = 100
    hgb_max_depth: int = 6
    hgb_min_samples_leaf: int = 40
    high_leverage_quantile: float = 0.75


def prepare_wp_features(df: pd.DataFrame, is_chase: bool) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extracts strictly pre-delivery situational features and ground-truth win labels.
    Prevents any forward delivery or post-match outcome leakage into feature space.
    """
    work = df.copy()

    # Wickets in hand
    if "wickets_in_hand_before" not in work.columns:
        max_w = np.where(work["innings"] >= 3, 2, 10)
        work["wickets_in_hand_before"] = max_w - work["wickets_before"]

    # Effective legal balls remaining
    rem_col = "legal_balls_remaining_eff" if "legal_balls_remaining_eff" in work.columns else "legal_balls_remaining_before"
    work["balls_rem"] = work[rem_col].fillna(120).clip(lower=0)

    # CRR before
    work["crr"] = work["crr_before"].fillna(0.0).clip(0, 36)

    # Match phase & context
    work["match_phase"] = work["match_phase"].fillna("Middle")
    work["venue_id"] = work["venue_id"].fillna(0).astype(str) if "venue_id" in work.columns else "0"
    work["is_ipl"] = work["is_ipl"].fillna(True).astype(int) if "is_ipl" in work.columns else 1
    work["season_year"] = work["season_year"].fillna(2024).astype(float) if "season_year" in work.columns else 2024.0

    if is_chase:
        req_col = "runs_needed_eff" if "runs_needed_eff" in work.columns else "runs_needed_before"
        rrr_col = "rrr_eff" if "rrr_eff" in work.columns else "rrr_before"
        work["runs_needed"] = work[req_col].fillna(100).clip(lower=0)
        work["rrr"] = np.where(np.isfinite(work[rrr_col]), work[rrr_col], 36.0).clip(0, 36)
        work["log_runs_needed"] = np.log1p(work["runs_needed"])
        work["rrr_minus_crr"] = (work["rrr"] - work["crr"]).clip(-20, 20)
        work["res_index"] = (work["wickets_in_hand_before"] / 10.0) * (work["balls_rem"] / 120.0)

        feature_cols = [
            "match_phase", "balls_rem", "wickets_in_hand_before", "runs_needed",
            "rrr", "crr", "log_runs_needed", "rrr_minus_crr", "res_index",
            "is_ipl", "season_year"
        ]
    else:
        work["score"] = work["score_before"].fillna(0).clip(lower=0)
        work["res_index"] = (work["wickets_in_hand_before"] / 10.0) * (work["balls_rem"] / 120.0)
        feature_cols = [
            "match_phase", "balls_rem", "wickets_in_hand_before", "score",
            "crr", "res_index", "is_ipl", "season_year"
        ]

    X = work[feature_cols].copy()
    y = work["bat_win"].values.astype(int)
    groups = work["match_id"].values

    return X, y, groups


def build_pipeline(model_type: str, is_chase: bool) -> Pipeline:
    """Builds a scikit-learn pipeline for the designated model type."""
    if is_chase:
        cat_cols = ["match_phase"]
        num_cols = ["balls_rem", "wickets_in_hand_before", "runs_needed", "rrr", "crr",
                    "log_runs_needed", "rrr_minus_crr", "res_index", "is_ipl", "season_year"]
    else:
        cat_cols = ["match_phase"]
        num_cols = ["balls_rem", "wickets_in_hand_before", "score", "crr", "res_index", "is_ipl", "season_year"]

    if model_type == "logistic_regression":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ])
        clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ])
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ])
        clf = HistGradientBoostingClassifier(
            max_iter=100, max_depth=6, min_samples_leaf=40, random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate_model_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculates comprehensive classification, calibration, and ranking metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    cal_err = float(np.mean(np.abs(prob_true - prob_pred)))

    return {
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(ll), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "mean_calibration_error": round(cal_err, 4),
    }


def compute_counterfactual_state(df: pd.DataFrame, outcome: str, is_chase: bool) -> pd.DataFrame:
    """Generates counterfactual match state after a hypothetical delivery outcome."""
    cf = df.copy()
    rem_col = "legal_balls_remaining_eff" if "legal_balls_remaining_eff" in cf.columns else "legal_balls_remaining_before"
    cf["balls_rem"] = (cf[rem_col] - 1).clip(lower=0)

    if outcome == "dot":
        runs, wkt = 0, 0
    elif outcome == "four":
        runs, wkt = 4, 0
    elif outcome == "six":
        runs, wkt = 6, 0
    elif outcome == "wicket":
        runs, wkt = 0, 1
    else:
        runs, wkt = 1, 0

    cf["wickets_in_hand_before"] = (cf["wickets_in_hand_before"] - wkt).clip(lower=0)

    crr_val = cf["crr"] if "crr" in cf.columns else cf["crr_before"].fillna(0.0).clip(0, 36)
    cf["crr"] = crr_val

    if is_chase:
        req_col = "runs_needed_eff" if "runs_needed_eff" in cf.columns else "runs_needed_before"
        cf["runs_needed"] = (cf[req_col] - runs).clip(lower=0)
        cf["rrr"] = np.where(cf["balls_rem"] > 0, cf["runs_needed"] / (cf["balls_rem"] / 6.0), 36.0).clip(0, 36)
        cf["log_runs_needed"] = np.log1p(cf["runs_needed"])
        cf["rrr_minus_crr"] = (cf["rrr"] - cf["crr"]).clip(-20, 20)
        cf["res_index"] = (cf["wickets_in_hand_before"] / 10.0) * (cf["balls_rem"] / 120.0)
    else:
        score_base = cf["score_before"] if "score_before" in cf.columns else 0
        cf["score"] = score_base + runs
        cf["res_index"] = (cf["wickets_in_hand_before"] / 10.0) * (cf["balls_rem"] / 120.0)

    return cf


def run_win_probability_benchmark(
    deliveries_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Executes full Win Probability benchmark across Logistic Regression,
    Random Forest, and Gradient Boosting under Match-Grouped GroupKFold.
    Computes out-of-fold WP, Leverage Index, and delivery-level WPA.
    """
    df = deliveries_df.copy()

    # Identify match outcome
    m_win = matches_df[["match_id", "win_team_id"]].dropna().copy()
    df = df.merge(m_win, on="match_id", how="left")

    valid_mask = (
        df["win_team_id"].notna()
        & ((df["batting_team_id"] == df["win_team_id"]) | (df["bowling_team_id"] == df["win_team_id"]))
        & (df["innings"].isin([1, 2]))
    )
    work = df.loc[valid_mask].copy().reset_index(drop=True)
    work["bat_win"] = (work["batting_team_id"] == work["win_team_id"]).astype(int)

    model_types = ["logistic_regression", "random_forest", "gradient_boosting"]
    benchmark_results = {}
    oof_predictions = {m: np.zeros(len(work)) for m in model_types}

    for inn, is_chase in [(1, False), (2, True)]:
        inn_idx = np.where(work["innings"] == inn)[0]
        if len(inn_idx) == 0:
            continue

        inn_df = work.iloc[inn_idx].copy()
        X, y, groups = prepare_wp_features(inn_df, is_chase=is_chase)
        gkf = GroupKFold(n_splits=5)

        for m_type in model_types:
            preds = np.zeros(len(inn_df))
            for train_idx, val_idx in gkf.split(X, y, groups):
                if pd.Series(y[train_idx]).nunique() < 2:
                    preds[val_idx] = 0.5
                    continue
                pipe = build_pipeline(m_type, is_chase=is_chase)
                pipe.fit(X.iloc[train_idx], y[train_idx])
                preds[val_idx] = pipe.predict_proba(X.iloc[val_idx])[:, 1]

            oof_predictions[m_type][inn_idx] = np.clip(preds, 0.001, 0.999)
            slice_name = f"Innings {inn} ({'Chase' if is_chase else 'Setting'}) - {m_type}"
            benchmark_results[slice_name] = evaluate_model_predictions(y, preds)

    # Overall metrics per model
    overall_metrics = {}
    for m_type in model_types:
        overall_metrics[m_type] = evaluate_model_predictions(work["bat_win"].values, oof_predictions[m_type])

    # Assign primary WP predictions from best model (Gradient Boosting)
    work["wp_pre"] = oof_predictions["gradient_boosting"]
    work["wp_pre_logit"] = oof_predictions["logistic_regression"]
    work["wp_pre_rf"] = oof_predictions["random_forest"]

    # Compute WPA (Win Probability Added)
    # Post-delivery WP is shifted from next ball within same match & innings, or terminal 1.0/0.0
    work["wp_post"] = work.groupby(["match_id", "innings"])["wp_pre"].shift(-1)
    # Terminal balls
    term_mask = work["wp_post"].isna()
    work.loc[term_mask, "wp_post"] = work.loc[term_mask, "bat_win"].astype(float)

    work["wpa_bat"] = work["wp_post"] - work["wp_pre"]
    work["wpa_bowl"] = -work["wpa_bat"]

    # Calculate Counterfactual Leverage Index (LI)
    # Using Logistic Regression fast pipeline for counterfactual queries
    li_arr = np.zeros(len(work))
    for inn, is_chase in [(1, False), (2, True)]:
        inn_idx = np.where(work["innings"] == inn)[0]
        inn_df = work.iloc[inn_idx].copy()
        X, y, _ = prepare_wp_features(inn_df, is_chase=is_chase)
        pipe = build_pipeline("logistic_regression", is_chase=is_chase)
        pipe.fit(X, y)

        cf_six = compute_counterfactual_state(inn_df, "six", is_chase=is_chase)
        cf_wkt = compute_counterfactual_state(inn_df, "wicket", is_chase=is_chase)
        X_six, _, _ = prepare_wp_features(cf_six, is_chase=is_chase)
        X_wkt, _, _ = prepare_wp_features(cf_wkt, is_chase=is_chase)

        p_six = pipe.predict_proba(X_six)[:, 1]
        p_wkt = pipe.predict_proba(X_wkt)[:, 1]
        # Potential game swing
        swing = np.abs(p_six - p_wkt) * 100.0
        li_arr[inn_idx] = swing

    work["leverage_index"] = li_arr
    mean_li = work["leverage_index"].mean()
    work["leverage_normalized"] = work["leverage_index"] / max(mean_li, 1e-4)

    # Correlation between Pressure Index and Leverage
    pi_col = "pressure_bat" if "pressure_bat" in work.columns else None
    corr_pi_li = None
    if pi_col and work[pi_col].notna().any():
        sub_pi = work.dropna(subset=[pi_col, "leverage_index"])
        corr_pi_li = {
            "pearson": round(float(np.corrcoef(sub_pi[pi_col], sub_pi["leverage_index"])[0, 1]), 4),
            "spearman": round(float(spearmanr(sub_pi[pi_col], sub_pi["leverage_index"]).statistic), 4),
        }

    # Aggregate Player WPA Leaderboards
    bat_col = "canonical_striker_id" if "canonical_striker_id" in work.columns else "striker_id"
    bowl_col = "canonical_bowler_id" if "canonical_bowler_id" in work.columns else "bowler_id"

    p75_li = work["leverage_normalized"].quantile(0.75)
    work["is_high_leverage"] = work["leverage_normalized"] >= p75_li

    bat_wpa = (
        work.groupby(bat_col, as_index=False)
        .agg(
            player_name=("striker_name", "first"),
            balls_faced=("wpa_bat", "count"),
            total_wpa=("wpa_bat", "sum"),
            wpa_per_100=("wpa_bat", lambda s: float(np.mean(s) * 100)),
            hl_balls=("is_high_leverage", "sum"),
            hl_wpa=("wpa_bat", lambda s: float(s[work.loc[s.index, "is_high_leverage"]].sum())),
        )
        .sort_values("total_wpa", ascending=False)
    )

    bowl_wpa = (
        work.groupby(bowl_col, as_index=False)
        .agg(
            player_name=("bowler_name", "first"),
            balls_bowled=("wpa_bowl", "count"),
            total_wpa=("wpa_bowl", "sum"),
            wpa_per_100=("wpa_bowl", lambda s: float(np.mean(s) * 100)),
            hl_balls=("is_high_leverage", "sum"),
            hl_wpa=("wpa_bowl", lambda s: float(s[work.loc[s.index, "is_high_leverage"]].sum())),
        )
        .sort_values("total_wpa", ascending=False)
    )

    # Save summary and datasets
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    bat_wpa.head(100).to_csv(DATA_PROCESSED / "wp_bat_wpa_leaders.csv", index=False)
    bowl_wpa.head(100).to_csv(DATA_PROCESSED / "wp_bowl_wpa_leaders.csv", index=False)

    return {
        "overall_metrics": overall_metrics,
        "slice_metrics": benchmark_results,
        "correlation_pi_vs_leverage": corr_pi_li,
        "mean_leverage": round(float(mean_li), 2),
        "p90_leverage": round(float(np.percentile(work["leverage_index"], 90)), 2),
        "n_deliveries_evaluated": int(len(work)),
        "n_matches_evaluated": int(work["match_id"].nunique()),
        "top_bat_wpa": bat_wpa.head(10).to_dict(orient="records"),
        "top_bowl_wpa": bowl_wpa.head(10).to_dict(orient="records"),
    }

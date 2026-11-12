"""
Phase 3 Additional Validation Suite (Pre-Approval Deep Audit, Zero Named Top-10 Output):
1. Competition weighting Top-10 membership overlap & rank displacement across w_t20i in {0.25, 0.50, 0.75, 1.00}.
2. Eligibility floor audit: (30,30) vs (50,60) vs (75,100) + strict verification of IPL ball-count definitions.
3. Bowling WAA shrinkage deep dive (why k0=4602, raw vs post distributions, separation, alternative WAA weights, EV overlap).
4. Expanded OOF baseline validation (MAE, RMSE, LogLoss, Brier, calibration by competition, innings x phase, pressure range)
   + Season-based CV (Leave-One-Season-Out & Expanding Walk-Forward).
5. Pressure exponent gamma sweep ({0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}) & Top-10 membership sensitivity.
6. Consistency, Kish N_eff, Beta-Binomial M0 diagnostic, LCB terminology & volume-independence verification.
7. PRI pillar standardization equality check & double-counting check between Overall and HP (PI>=60) components.

NOTE: Strictly NO player names or final ranked Top-10 player lists are printed or exported.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, kendalltau, norm
from sklearn.base import clone
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_datasets, DATA_PROCESSED
from src.performance import (
    PRIMARY_DLS_STATUSES,
    PerformanceConfig,
    _make_context_preprocessor,
    _prepare_context_columns,
    _safe_z,
    apply_beta_binomial_shrinkage,
    apply_normal_shrinkage,
    evaluate_batting_pressure,
    evaluate_bowling_pressure,
    filter_dls_pool,
    fit_predict_oof_baselines,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "Segoe UI", "DejaVu Sans", "Arial"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _top_k_ids(df: pd.DataFrame, elig_col: str = "elig_primary", score_col: str = "pri_score_lcb", k: int = 10) -> pd.DataFrame:
    sub = df[df[elig_col]].sort_values([score_col, "neff", "canonical_player_id"], ascending=[False, False, True]).copy()
    sub["rank"] = np.arange(1, len(sub) + 1)
    return sub[["canonical_player_id", "rank", score_col]].head(k)


def _compare_ranking_overlap(
    base_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    base_elig: str = "elig_primary",
    alt_elig: str = "elig_primary",
    score_col: str = "pri_score_lcb",
) -> dict:
    b_sub = base_df[base_df[base_elig]].sort_values([score_col, "neff", "canonical_player_id"], ascending=[False, False, True]).copy()
    b_sub["rank_base"] = np.arange(1, len(b_sub) + 1)
    a_sub = alt_df[alt_df[alt_elig]].sort_values([score_col, "neff", "canonical_player_id"], ascending=[False, False, True]).copy()
    a_sub["rank_alt"] = np.arange(1, len(a_sub) + 1)

    b10 = set(b_sub.head(10)["canonical_player_id"])
    a10 = set(a_sub.head(10)["canonical_player_id"])
    b15 = set(b_sub.head(15)["canonical_player_id"])
    a15 = set(a_sub.head(15)["canonical_player_id"])
    b20 = set(b_sub.head(20)["canonical_player_id"])
    a20 = set(a_sub.head(20)["canonical_player_id"])

    merged = b_sub[["canonical_player_id", "rank_base", score_col]].merge(
        a_sub[["canonical_player_id", "rank_alt", score_col]],
        on="canonical_player_id",
        how="inner",
        suffixes=("_base", "_alt"),
    )
    sp = float(spearmanr(merged[f"{score_col}_base"], merged[f"{score_col}_alt"]).statistic)
    mae = float(np.mean(np.abs(merged[f"{score_col}_base"] - merged[f"{score_col}_alt"])))
    kt = float(kendalltau(merged["rank_base"], merged["rank_alt"]).statistic)

    # Track where the baseline Top-10 land under alt (only within alt eligible, plus drop count)
    b10_df = b_sub.head(10)[["canonical_player_id", "rank_base"]].merge(
        a_sub[["canonical_player_id", "rank_alt"]], on="canonical_player_id", how="left"
    )
    n_b10_inelig = int(b10_df["rank_alt"].isna().sum())
    shifts = np.abs(b10_df["rank_alt"] - b10_df["rank_base"]).dropna()
    max_rank_b10 = int(b10_df["rank_alt"].max()) if b10_df["rank_alt"].notna().any() else -1

    return {
        "n_base_elig": len(b_sub),
        "n_alt_elig": len(a_sub),
        "n_common_elig": len(merged),
        "top10_overlap": len(b10 & a10),
        "top10_jaccard": len(b10 & a10) / max(1, len(b10 | a10)),
        "top15_overlap": len(b15 & a15),
        "top20_overlap": len(b20 & a20),
        "pool_spearman": sp,
        "pool_kendall": kt,
        "score_mae": mae,
        "base_top10_dropped_inelig": n_b10_inelig,
        "base_top10_mean_abs_drank": float(shifts.mean()) if len(shifts) > 0 else np.nan,
        "base_top10_max_abs_drank": float(shifts.max()) if len(shifts) > 0 else np.nan,
        "base_top10_worst_alt_rank": max_rank_b10,
    }


def _slice_metrics_bat(sub: pd.DataFrame) -> dict:
    w_ok = sub["striker_out"].nunique() > 1
    return {
        "n_balls": int(len(sub)),
        "act_runs": float(sub["runs"].mean()),
        "exp_runs": float(sub["bat_exp_runs"].mean()),
        "cal_diff_r": float(sub["runs"].mean() - sub["bat_exp_runs"].mean()),
        "mae_r": float(mean_absolute_error(sub["runs"], sub["bat_exp_runs"])),
        "rmse_r": float(root_mean_squared_error(sub["runs"], sub["bat_exp_runs"])),
        "act_wkt": float(sub["striker_out"].mean()),
        "exp_wkt": float(sub["bat_exp_wkt"].mean()),
        "cal_diff_w": float(sub["striker_out"].mean() - sub["bat_exp_wkt"].mean()),
        "logloss_w": float(log_loss(sub["striker_out"], sub["bat_exp_wkt"])) if w_ok else np.nan,
        "brier_w": float(brier_score_loss(sub["striker_out"], sub["bat_exp_wkt"])),
    }


def _slice_metrics_bowl(sub: pd.DataFrame) -> dict:
    leg = sub[sub["is_legal_ball"]]
    n_leg = max(1, int(len(leg)))
    w_ok = leg["bowler_wkt_int"].nunique() > 1
    return {
        "n_all": int(len(sub)),
        "n_legal": int(len(leg)),
        "act_conc_leg": float(sub["bowler_runs_conceded"].sum() / n_leg),
        "exp_conc_leg": float(sub["bowl_exp_runs"].sum() / n_leg),
        "legal_mae_r": float(mean_absolute_error(leg["bowler_runs_conceded"], leg["bowl_exp_runs_legal_raw"])),
        "legal_rmse_r": float(root_mean_squared_error(leg["bowler_runs_conceded"], leg["bowl_exp_runs_legal_raw"])),
        "act_wkt_leg": float(sub["bowler_wkt_int"].sum() / n_leg),
        "exp_wkt_leg": float(sub["bowl_exp_wkt"].sum() / n_leg),
        "legal_logloss_w": float(log_loss(leg["bowler_wkt_int"], leg["bowl_exp_wkt_legal_raw"])) if w_ok else np.nan,
        "legal_brier_w": float(brier_score_loss(leg["bowler_wkt_int"], leg["bowl_exp_wkt_legal_raw"])),
    }


def fit_season_baselines(
    pool_in: pd.DataFrame,
    mode: str = "loso",
    min_eval_season: int = 2022,
) -> pd.DataFrame:
    """
    Fits strictly out-of-season baselines using identical model & pre-delivery features:
    - mode='loso': Leave-One-Season-Out CV across season_year.
    - mode='walk_forward': For each target season s >= min_eval_season, train strictly on season_year < s.
    """
    pool = pool_in.copy().reset_index(drop=True)
    is_bat = pool["is_batter_ball_faced"].fillna(False).values
    is_leg = pool["is_legal_ball"].fillna(False).values
    pre_bat = _make_context_preprocessor("pressure_bat")
    pre_bowl = _make_context_preprocessor("pressure_bowl")

    pool["bat_exp_runs"] = np.nan
    pool["bat_exp_wkt"] = np.nan
    pool["bowl_exp_runs_legal_raw"] = np.nan
    pool["bowl_exp_wkt_legal_raw"] = np.nan
    pool["bowl_exp_runs"] = np.nan
    pool["bowl_exp_wkt"] = np.nan

    seasons = sorted(pool["season_year"].unique())
    for s in seasons:
        if mode == "loso":
            tr_mask = (pool["season_year"] != s).values
        elif mode == "walk_forward":
            if s < min_eval_season:
                continue
            tr_mask = (pool["season_year"] < s).values
        else:
            raise ValueError(mode)

        va_mask = (pool["season_year"] == s).values
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        # Batting
        tr_bat = tr_idx[is_bat[tr_idx]]
        va_bat = va_idx[is_bat[va_idx]]
        if len(tr_bat) > 0 and len(va_bat) > 0:
            br = Pipeline([("pre", clone(pre_bat)), ("reg", Ridge(alpha=30.0))])
            br.fit(pool.iloc[tr_bat], pool["runs"].iloc[tr_bat].values)
            pool.iloc[va_bat, pool.columns.get_loc("bat_exp_runs")] = np.clip(br.predict(pool.iloc[va_bat]), 0.10, 4.00)
            bw = Pipeline([("pre", clone(pre_bat)), ("clf", LogisticRegression(C=0.5, max_iter=1000))])
            bw.fit(pool.iloc[tr_bat], pool["striker_out"].iloc[tr_bat].values)
            pool.iloc[va_bat, pool.columns.get_loc("bat_exp_wkt")] = np.clip(bw.predict_proba(pool.iloc[va_bat])[:, 1], 0.002, 0.50)

        # Bowling
        tr_leg = tr_idx[is_leg[tr_idx]]
        va_leg = va_idx[is_leg[va_idx]]
        if len(tr_leg) > 0 and len(va_leg) > 0:
            lr = Pipeline([("pre", clone(pre_bowl)), ("reg", Ridge(alpha=30.0))])
            lr.fit(pool.iloc[tr_leg], pool["bowler_runs_conceded"].iloc[tr_leg].values)
            raw_er = np.clip(lr.predict(pool.iloc[va_leg]), 0.10, 4.00)
            pool.iloc[va_leg, pool.columns.get_loc("bowl_exp_runs_legal_raw")] = raw_er

            lw = Pipeline([("pre", clone(pre_bowl)), ("clf", LogisticRegression(C=0.5, max_iter=1000))])
            lw.fit(pool.iloc[tr_leg], pool["bowler_wkt_int"].iloc[tr_leg].values)
            raw_ew = np.clip(lw.predict_proba(pool.iloc[va_leg])[:, 1], 0.002, 0.50)
            pool.iloc[va_leg, pool.columns.get_loc("bowl_exp_wkt_legal_raw")] = raw_ew

            tr_df = pool.iloc[tr_idx]
            mult = tr_df.groupby(["is_ipl_int", "match_phase", "innings"]).agg(
                tot_r=("bowler_runs_conceded", "sum"),
                leg_r=("bowler_runs_conceded", lambda x: x[tr_df.loc[x.index, "is_legal_ball"]].sum()),
                tot_w=("bowler_wkt_int", "sum"),
                leg_w=("bowler_wkt_int", lambda x: x[tr_df.loc[x.index, "is_legal_ball"]].sum()),
            )
            mult["r_mult"] = mult["tot_r"] / np.maximum(1.0, mult["leg_r"])
            mult["w_mult"] = mult["tot_w"] / np.maximum(1.0, mult["leg_w"])
            va_sub = pool.iloc[va_leg][["is_ipl_int", "match_phase", "innings"]].join(
                mult[["r_mult", "w_mult"]], on=["is_ipl_int", "match_phase", "innings"]
            )
            pool.iloc[va_leg, pool.columns.get_loc("bowl_exp_runs")] = raw_er * va_sub["r_mult"].fillna(1.04).values
            pool.iloc[va_leg, pool.columns.get_loc("bowl_exp_wkt")] = raw_ew * va_sub["w_mult"].fillna(1.00).values
        # Zero expectation on non-legal rows inside evaluated validation season
        va_nonleg = va_idx[~is_leg[va_idx]]
        pool.iloc[va_nonleg, pool.columns.get_loc("bowl_exp_runs")] = 0.0
        pool.iloc[va_nonleg, pool.columns.get_loc("bowl_exp_wkt")] = 0.0

    return pool


def main():
    ds = load_raw_datasets()
    df = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz", low_memory=False)
    player_dir = pd.read_csv(DATA_PROCESSED / "canonical_player_directory.csv")

    print("=== SETUP: PRIMARY OOF BASELINES (MATCH-LEVEL 5-FOLD GROUPKFOLD) ===")
    oof_pri, _ = fit_predict_oof_baselines(df, ds["matches"], ds["competitions"], include_suspected_dls=False, n_splits=5)
    print(f"Seasons present in primary pool: {sorted(oof_pri['season_year'].unique().tolist())}")
    print("Deliveries per season:")
    print(oof_pri.groupby("season_year")["match_id"].agg(matches="nunique", deliveries="count").to_string())

    cfg_base = PerformanceConfig(w_t20i=0.50, pressure_gamma=2.0, min_neff_elig=30.0, min_ipl_balls_bat=30, min_ipl_balls_bowl=30)
    bat_base, bat_meta_base = evaluate_batting_pressure(oof_pri, cfg_base, player_dir=player_dir)
    bowl_base, bowl_meta_base = evaluate_bowling_pressure(oof_pri, cfg_base, player_dir=player_dir)

    # ------------------------------------------------------------------
    # 1. COMPETITION WEIGHTING TOP-10 OVERLAP & RANK SHIFTS
    # ------------------------------------------------------------------
    print("\n=== 1. COMPETITION WEIGHTING TOP-10 MEMBERSHIP & RANK SHIFTS (VS w_T20I=0.50) ===")
    w_grid = [0.25, 0.50, 0.75, 1.00]
    comp_bat_rows, comp_bowl_rows = [], []
    bat_by_w, bowl_by_w = {}, {}
    for w in [0.00] + w_grid:
        c = PerformanceConfig(w_t20i=w)
        bdf, _ = evaluate_batting_pressure(oof_pri, c)
        wdf, _ = evaluate_bowling_pressure(oof_pri, c)
        bat_by_w[w] = bdf
        bowl_by_w[w] = wdf
        comp_bat_rows.append({"w_t20i": w, **_compare_ranking_overlap(bat_base, bdf)})
        comp_bowl_rows.append({"w_t20i": w, **_compare_ranking_overlap(bowl_base, wdf)})
    print("-- Batting Competition Weight Stability vs 0.50 --")
    print(pd.DataFrame(comp_bat_rows).round(4).to_string(index=False))
    print("-- Bowling Competition Weight Stability vs 0.50 --")
    print(pd.DataFrame(comp_bowl_rows).round(4).to_string(index=False))

    # Pairwise Top-10 overlap matrix across {0.25, 0.50, 0.75, 1.00}
    print("-- Pairwise Top-10 Overlap Matrix across w_t20i in {0.25, 0.50, 0.75, 1.00} --")
    for label, dmap in [("Batting", bat_by_w), ("Bowling", bowl_by_w)]:
        mat = pd.DataFrame(index=[f"w={x:.2f}" for x in w_grid], columns=[f"w={x:.2f}" for x in w_grid], dtype=int)
        for w1 in w_grid:
            s1 = set(_top_k_ids(dmap[w1], k=10)["canonical_player_id"])
            for w2 in w_grid:
                s2 = set(_top_k_ids(dmap[w2], k=10)["canonical_player_id"])
                mat.loc[f"w={w1:.2f}", f"w={w2:.2f}"] = len(s1 & s2)
        print(f"{label} Top-10 Overlap:")
        print(mat.to_string())

    # ------------------------------------------------------------------
    # 2. ELIGIBILITY VERIFICATION & THRESHOLD COMPARISON
    # ------------------------------------------------------------------
    print("\n=== 2. ELIGIBILITY THRESHOLDS & IPL BALL DEFINITION AUDIT ===")
    # Strictly verify IPL ball counting
    raw_pri_bat_all = oof_pri.groupby("canonical_striker_id")["is_ipl"].sum()
    raw_pri_bat_faced = oof_pri[oof_pri["is_batter_ball_faced"]].groupby("canonical_striker_id")["is_ipl"].sum()
    merged_bchk = bat_base[["canonical_player_id", "ipl_balls"]].merge(
        raw_pri_bat_faced.rename("chk_faced"), left_on="canonical_player_id", right_index=True, how="left"
    ).merge(raw_pri_bat_all.rename("chk_all"), left_on="canonical_player_id", right_index=True, how="left")
    print(f"Batting ipl_balls exact match with is_batter_ball_faced & is_ipl: "
          f"max_diff={(merged_bchk['ipl_balls'] - merged_bchk['chk_faced']).abs().max()}, "
          f"players_with_ipl_wides_excluded={(merged_bchk['chk_all'] > merged_bchk['ipl_balls']).sum()}")

    raw_pri_bowl_all = oof_pri.groupby("canonical_bowler_id")["is_ipl"].sum()
    raw_pri_bowl_leg = oof_pri[oof_pri["is_legal_ball"]].groupby("canonical_bowler_id")["is_ipl"].sum()
    merged_wchk = bowl_base[["canonical_player_id", "ipl_legal_balls"]].merge(
        raw_pri_bowl_leg.rename("chk_legal"), left_on="canonical_player_id", right_index=True, how="left"
    ).merge(raw_pri_bowl_all.rename("chk_all"), left_on="canonical_player_id", right_index=True, how="left")
    merged_wchk["chk_legal"] = merged_wchk["chk_legal"].fillna(0)
    print(f"Bowling ipl_legal_balls exact match with is_legal_ball & is_ipl: "
          f"max_diff={(merged_wchk['ipl_legal_balls'] - merged_wchk['chk_legal']).abs().max()}, "
          f"players_with_ipl_nonlegal_excluded={(merged_wchk['chk_all'] > merged_wchk['ipl_legal_balls']).sum()}")

    elig_specs = [(30.0, 30), (50.0, 60), (75.0, 100)]
    ebat_rows, ebowl_rows = [], []
    for ne_cut, ipl_cut in elig_specs:
        c_cut = PerformanceConfig(min_neff_elig=ne_cut, min_ipl_balls_bat=ipl_cut, min_ipl_balls_bowl=ipl_cut)
        bc, _ = evaluate_batting_pressure(oof_pri, c_cut)
        wc, _ = evaluate_bowling_pressure(oof_pri, c_cut)
        ebat_rows.append({"thresh": f"Neff>={int(ne_cut)}, IPL>={ipl_cut}", **_compare_ranking_overlap(bat_base, bc)})
        ebowl_rows.append({"thresh": f"Neff>={int(ne_cut)}, IPL>={ipl_cut}", **_compare_ranking_overlap(bowl_base, wc)})
    print("-- Batting Eligibility Cutoff Impact vs (30, 30) --")
    print(pd.DataFrame(ebat_rows).round(4).to_string(index=False))
    print("-- Bowling Eligibility Cutoff Impact vs (30, 30) --")
    print(pd.DataFrame(ebowl_rows).round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # 3. BOWLING WAA SHRINKAGE DEEP DIVE (k0 = 4,602) & WEIGHT SENSITIVITY
    # ------------------------------------------------------------------
    print("\n=== 3. BOWLING WAA SHRINKAGE INVESTIGATION (k0 = 4,602) ===")
    sig2_waa = bowl_meta_base["waa"]["sigma2_within"]
    tau2_waa = bowl_meta_base["waa"]["tau2"]
    print(f"Base WAA EB fit (min_neff_fit=15): mu0={bowl_meta_base['waa']['mu0']:.5f}, "
          f"sigma_within={np.sqrt(sig2_waa):.4f}, sigma2_within={sig2_waa:.4f}, "
          f"tau_between={np.sqrt(tau2_waa):.4f}, tau2={tau2_waa:.6f}, k0={bowl_meta_base['waa']['k0_prior_neff']:.2f}")

    # Variance decomposition across N_eff bins
    b_fit = bowl_base[bowl_base["neff"] >= 15.0].copy()
    b_fit["neff_bin"] = pd.cut(b_fit["neff"], [15, 50, 150, 400, 3000])
    vdecomp = b_fit.groupby("neff_bin", observed=False).agg(
        n_players=("waa_raw", "count"),
        median_neff=("neff", "median"),
        obs_var_waa=("waa_raw", "var"),
        expected_sampling_var=("neff", lambda s: float(np.mean(sig2_waa / s))),
        std_waa_raw=("waa_raw", "std"),
        std_waa_post=("waa_post", "std"),
        mean_shrinkage_B=("waa_B", "mean"),
    )
    print("\nWAA Variance Decomposition by N_eff Bin (Empirical Var vs Theoretical Sampling Noise sigma2/Neff):")
    print(vdecomp.round(5).to_string())

    # Check sensitivity of EB tau_between and k0 to min_neff_fit
    print("\nWAA EB Marginal MLE Sensitivity to min_neff_fit:")
    for mf in [15.0, 30.0, 60.0, 100.0, 200.0]:
        res_mf, m_mf = apply_normal_shrinkage(bowl_base["waa_raw"], bowl_base["neff"], sig2_waa, min_neff_fit=mf, lcb_z=0.50)
        print(f"  min_neff_fit={int(mf):>3d} (n={m_mf['n_players_fit']:>3d}): "
              f"tau_between={m_mf['tau_between']:.4f}, k0={m_mf['k0_prior_neff']:.1f}, "
              f"elig_post_std={res_mf.loc[bowl_base['elig_primary'], 'post'].std():.5f}")

    # Raw vs Shrunk WAA separation inside Primary Eligible Pool (N=200)
    b_el = bowl_base[bowl_base["elig_primary"]].copy()
    print("\nPrimary Eligible Bowlers (N=200) WAA Raw vs Post vs LCB Distribution & Separation:")
    for col in ["waa_raw", "waa_post", "waa_lcb", "waa_B"]:
        s = b_el[col]
        print(f"  {col:<8s}: mean={s.mean():+.5f}, std={s.std():.5f}, IQR={s.quantile(0.75)-s.quantile(0.25):.5f}, "
              f"P05={s.quantile(0.05):+.5f}, P50={s.median():+.5f}, P95={s.quantile(0.95):+.5f}, "
              f"range=[{s.min():+.5f}, {s.max():+.5f}]")
    print(f"  Separation ratio std(waa_post)/std(waa_raw) = {b_el['waa_post'].std() / b_el['waa_raw'].std():.4f}")
    print(f"  Corr(waa_post, waa_lcb) = {pearsonr(b_el['waa_post'], b_el['waa_lcb']).statistic:.4f} | "
          f"Corr(waa_lcb, neff) = {spearmanr(b_el['waa_lcb'], b_el['neff']).statistic:.4f} | "
          f"Corr(waa_post, neff) = {spearmanr(b_el['waa_post'], b_el['neff']).statistic:.4f}")

    # Sensitivity to alternative WAA weights (transferring between WAA and RSAA; keeping Cons=0.15, HP=0.15)
    print("\nBowling WAA Weight Sensitivity & Comparison Against Run-Equivalents (ev_all_lcb & ev_net_po_lcb):")
    waa_variants = [
        ("40 RSAA / 30 WAA (v1 Wts)", 0.40, 0.30, 0.15, 0.15),
        ("50 RSAA / 20 WAA",          0.50, 0.20, 0.15, 0.15),
        ("55 RSAA / 15 WAA (Base)",   0.55, 0.15, 0.15, 0.15),
        ("60 RSAA / 10 WAA",          0.60, 0.10, 0.15, 0.15),
        ("70 RSAA / 00 WAA",          0.70, 0.00, 0.15, 0.15),
    ]
    w_sens_rows = []
    ev_bowl_all = bowl_base.copy()
    ev_bowl_all["pri_score_lcb"] = ev_bowl_all["ev_all_lcb"]
    ev_bowl_po = bowl_base.copy()
    ev_bowl_po["pri_score_lcb"] = ev_bowl_po["ev_net_po_lcb"]
    for name, wr, ww, wc, wh in waa_variants:
        c_w = PerformanceConfig(bowl_w_rsaa=wr, bowl_w_waa=ww, bowl_w_cons=wc, bowl_w_hp=wh)
        w_alt, _ = evaluate_bowling_pressure(oof_pri, c_w)
        vs_base = _compare_ranking_overlap(bowl_base, w_alt)
        vs_ev_all = _compare_ranking_overlap(ev_bowl_all, w_alt)
        vs_ev_po = _compare_ranking_overlap(ev_bowl_po, w_alt)
        w_sens_rows.append({
            "Variant": name,
            "Sp_vs_Base": vs_base["pool_spearman"],
            "Top10_vs_Base": vs_base["top10_overlap"],
            "Top20_vs_Base": vs_base["top20_overlap"],
            "Sp_vs_EV_All": vs_ev_all["pool_spearman"],
            "Top10_vs_EV_All": vs_ev_all["top10_overlap"],
            "Sp_vs_EV_Gen": vs_ev_po["pool_spearman"],
            "Top10_vs_EV_Gen": vs_ev_po["top10_overlap"],
        })
    print(pd.DataFrame(w_sens_rows).round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # 4. EXPANDED BASELINE VALIDATION + SEASON-BASED CV (LOSO & WALK-FORWARD)
    # ------------------------------------------------------------------
    print("\n=== 4. EXPANDED OOF BASELINE CALIBRATION & SEASON-BASED VALIDATION ===")
    bat_rows = oof_pri[oof_pri["is_batter_ball_faced"]].copy()
    bowl_rows = oof_pri.copy()
    pbins = [0, 20, 40, 60, 80, 100.1]
    plabels = ["[0,20)", "[20,40)", "[40,60)", "[60,80)", "[80,100]"]
    bat_rows["p_range"] = pd.cut(bat_rows["pressure_bat"], bins=pbins, labels=plabels, right=False)
    bowl_rows["p_range"] = pd.cut(bowl_rows["pressure_bowl"], bins=pbins, labels=plabels, right=False)

    print("\n-- Batting 5-Fold Match-GroupKFold Calibration by Competition, Innings/Phase, Pressure Range --")
    bat_slices = []
    for comp, g in bat_rows.groupby("is_ipl"):
        bat_slices.append({"Slice_Type": "Competition", "Slice": "IPL" if comp else "T20I", **_slice_metrics_bat(g)})
    for (inn, ph), g in bat_rows.groupby(["innings", "match_phase"]):
        bat_slices.append({"Slice_Type": "Inn_Phase", "Slice": f"Inn{int(inn)}_{ph}", **_slice_metrics_bat(g)})
    for pr, g in bat_rows.groupby("p_range", observed=False):
        bat_slices.append({"Slice_Type": "Pressure", "Slice": str(pr), **_slice_metrics_bat(g)})
    print(pd.DataFrame(bat_slices).round(5).to_string(index=False))

    print("\n-- Bowling 5-Fold Match-GroupKFold Calibration by Competition, Innings/Phase, Pressure Range --")
    bowl_slices = []
    for comp, g in bowl_rows.groupby("is_ipl"):
        bowl_slices.append({"Slice_Type": "Competition", "Slice": "IPL" if comp else "T20I", **_slice_metrics_bowl(g)})
    for (inn, ph), g in bowl_rows.groupby(["innings", "match_phase"]):
        bowl_slices.append({"Slice_Type": "Inn_Phase", "Slice": f"Inn{int(inn)}_{ph}", **_slice_metrics_bowl(g)})
    for pr, g in bowl_rows.groupby("p_range", observed=False):
        bowl_slices.append({"Slice_Type": "Pressure", "Slice": str(pr), **_slice_metrics_bowl(g)})
    print(pd.DataFrame(bowl_slices).round(5).to_string(index=False))

    # Season-based validation: LOSO and Forward Temporal Walk-Forward (>=2022)
    pool_ctx = _prepare_context_columns(filter_dls_pool(df, include_suspected_dls=False), ds["matches"], ds["competitions"])
    pool_ctx["striker_out"] = (
        pool_ctx["is_wicket"].fillna(False) & (pool_ctx["canonical_dismissed_player_id"] == pool_ctx["canonical_striker_id"])
    ).astype(int)
    pool_ctx["bowler_wkt_int"] = pool_ctx["is_bowler_wicket"].fillna(False).astype(int)

    oof_loso = fit_season_baselines(pool_ctx, mode="loso")
    oof_wf = fit_season_baselines(pool_ctx, mode="walk_forward", min_eval_season=2023)

    print("\n-- Season-Based Baseline Validation Comparison --")
    cv_rows = []
    for label, d_ev, mask_fn in [
        ("Match GroupKFold (5-fold, All)", oof_pri, lambda d: np.ones(len(d), dtype=bool)),
        ("Leave-One-Season-Out CV (All)", oof_loso, lambda d: np.ones(len(d), dtype=bool)),
        ("Match GroupKFold (2023+ subset)", oof_pri, lambda d: (d["season_year"] >= 2023).values),
        ("Walk-Forward Out-of-Time (2023+)", oof_wf, lambda d: (d["season_year"] >= 2023).values),
    ]:
        sub = d_ev[mask_fn(d_ev)]
        bm = _slice_metrics_bat(sub[sub["is_batter_ball_faced"]])
        wm = _slice_metrics_bowl(sub)
        cv_rows.append({
            "Scheme": label,
            "Bat_Balls": bm["n_balls"],
            "Bat_Act_R": bm["act_runs"],
            "Bat_Exp_R": bm["exp_runs"],
            "Bat_MAE": bm["mae_r"],
            "Bat_RMSE": bm["rmse_r"],
            "Bat_LogLoss": bm["logloss_w"],
            "Bat_Brier": bm["brier_w"],
            "Bowl_Leg_RMSE": wm["legal_rmse_r"],
            "Bowl_Leg_LogLoss": wm["legal_logloss_w"],
            "Bowl_Leg_Brier": wm["legal_brier_w"],
            "Bowl_Act_Conc": wm["act_conc_leg"],
            "Bowl_Exp_Conc": wm["exp_conc_leg"],
        })
    print(pd.DataFrame(cv_rows).round(5).to_string(index=False))

    print("\n-- Per-Season LOSO Batting & Bowling Metrics --")
    loso_s = []
    for s, g in oof_loso.groupby("season_year"):
        bm = _slice_metrics_bat(g[g["is_batter_ball_faced"]])
        wm = _slice_metrics_bowl(g)
        loso_s.append({
            "season": int(s),
            "bat_balls": bm["n_balls"],
            "bat_act_r": bm["act_runs"],
            "bat_exp_r": bm["exp_runs"],
            "bat_rmse": bm["rmse_r"],
            "bat_logloss": bm["logloss_w"],
            "bowl_act_conc": wm["act_conc_leg"],
            "bowl_exp_conc": wm["exp_conc_leg"],
            "bowl_rmse": wm["legal_rmse_r"],
        })
    print(pd.DataFrame(loso_s).round(5).to_string(index=False))

    # Impact of LOSO baselines vs Match-GroupKFold baselines on player PRI scores
    bat_loso, _ = evaluate_batting_pressure(oof_loso, cfg_base)
    bowl_loso, _ = evaluate_bowling_pressure(oof_loso, cfg_base)
    print("\nPlayer PRI Agreement: Match-GroupKFold Baselines vs Leave-One-Season-Out Baselines:")
    print("  Batting :", _compare_ranking_overlap(bat_base, bat_loso))
    print("  Bowling :", _compare_ranking_overlap(bowl_base, bowl_loso))

    # ------------------------------------------------------------------
    # 5. PRESSURE WEIGHTING GAMMA SWEEP & TOP-10 OVERLAP
    # ------------------------------------------------------------------
    print("\n=== 5. PRESSURE EXPONENT GAMMA SWEEP (VS GAMMA=2.0) ===")
    g_bat_rows, g_bowl_rows = [], []
    for g in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        cg = PerformanceConfig(pressure_gamma=g)
        bg, _ = evaluate_batting_pressure(oof_pri, cg)
        wg, _ = evaluate_bowling_pressure(oof_pri, cg)
        g_bat_rows.append({
            "gamma": g,
            "median_neff_elig": float(bg.loc[bg["elig_primary"], "neff"].median()),
            **_compare_ranking_overlap(bat_base, bg),
        })
        g_bowl_rows.append({
            "gamma": g,
            "median_neff_elig": float(wg.loc[wg["elig_primary"], "neff"].median()),
            **_compare_ranking_overlap(bowl_base, wg),
        })
    print("-- Batting Gamma Sweep --")
    print(pd.DataFrame(g_bat_rows).round(4).to_string(index=False))
    print("-- Bowling Gamma Sweep --")
    print(pd.DataFrame(g_bowl_rows).round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # 6. CONSISTENCY, M0 CLIP DIAGNOSTIC, AND VOLUME INDEPENDENCE CHECK
    # ------------------------------------------------------------------
    print("\n=== 6. CONSISTENCY VOLUME INDEPENDENCE & BETA-BINOMIAL M0 DIAGNOSTIC ===")
    for role, df_r, m_col, b_col, meta in [
        ("Batting", bat_base, "meff_inn", "total_balls", bat_meta_base),
        ("Bowling", bowl_base, "meff_spell", "total_legal_balls", bowl_meta_base),
    ]:
        # Recompute unclipped M0 directly from Method of Moments
        fit_mask = (df_r[m_col] >= 5.0) & df_r["cons_raw"].notna()
        pf = df_r.loc[fit_mask, "cons_raw"].values
        mf = df_r.loc[fit_mask, m_col].values
        mu0_c = float(np.average(pf, weights=mf))
        obs_v = float(np.average((pf - mu0_c) ** 2, weights=mf))
        noi_v = float(np.average(mu0_c * (1.0 - mu0_c) / mf, weights=mf))
        true_v_raw = obs_v - noi_v
        m0_unclipped = (mu0_c * (1.0 - mu0_c) / true_v_raw - 1.0) if true_v_raw > 0 else np.inf
        print(f"{role} Consistency MoM: n_fit={fit_mask.sum()}, mu0={mu0_c:.4f}, obs_var={obs_v:.5f}, "
              f"noise_var={noi_v:.5f}, obs-noise={true_v_raw:.5f}, unclipped_M0={m0_unclipped:.1f} (capped at 50.0)")

        el = df_r[df_r["elig_primary"]].copy()
        print(f"{role} Primary Eligible (N={len(el)}) Spearman Correlations with Match/Volume Counts:")
        vol_cols = ["n_matches", m_col, "neff", b_col]
        res_v = pd.DataFrame(index=["cons_raw", "cons_post", "cons_lcb", "pri_score_lcb"], columns=vol_cols, dtype=float)
        for yc in res_v.index:
            for vc in vol_cols:
                res_v.loc[yc, vc] = float(spearmanr(el[yc], el[vc]).statistic)
        print(res_v.round(4).to_string())

    # ------------------------------------------------------------------
    # 7. PILLAR STANDARDIZATION, ZERO-OVERLAP AUDIT, v1 vs v2 & HP WEIGHTS
    # ------------------------------------------------------------------
    print("\n=== 7. SCORING STANDARDIZATION, ZERO-OVERLAP, v1->v2 & HP SENSITIVITY ===")
    for role, df_r, p_cols, weights in [
        ("Batting", bat_base, ["raa_lcb", "surv_lcb", "cons_lcb", "hp_raa_lcb"], [0.45, 0.25, 0.15, 0.15]),
        ("Bowling", bowl_base, ["rsaa_lcb", "waa_lcb", "cons_lcb", "hp_impact_lcb"], [0.55, 0.15, 0.15, 0.15]),
    ]:
        ref_mask = df_r["elig_broad"]
        el_mask = df_r["elig_primary"]
        print(f"\n-- {role} Standardized LCB Pillar Moments & Effective PRI Contribution (ref=elig_broad N={ref_mask.sum()}, eval=elig_primary N={el_mask.sum()}) --")
        z_df = pd.DataFrame({c: _safe_z(df_r[c], ref_mask) for c in p_cols})
        el_z = z_df.loc[el_mask]
        pri_el = df_r.loc[el_mask, "pri_z_lcb"]
        tot_var_pri = float(pri_el.var(ddof=1))
        rows = []
        for c, w_nom in zip(p_cols, weights):
            cov_with_pri = float(np.cov(el_z[c], pri_el, ddof=1)[0, 1])
            var_share = (w_nom * cov_with_pri) / tot_var_pri
            rows.append({
                "Pillar": c,
                "Nom_Weight": w_nom,
                "Broad_Std": float(z_df.loc[ref_mask, c].std(ddof=1)),
                "Elig_Mean": float(el_z[c].mean()),
                "Elig_Std": float(el_z[c].std(ddof=1)),
                "Corr_with_PRI_z": float(pearsonr(el_z[c], pri_el).statistic),
                "Actual_Variance_Share": float(var_share),
            })
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    # Zero-overlap delivery partition verification (PI < 60 vs PI >= 60)
    b_el = bat_base[bat_base["elig_primary"]].copy()
    w_el = bowl_base[bowl_base["elig_primary"]].copy()
    bat_part_err = float((b_el["sw_gen"] + b_el["sw_hp"] - b_el["sw"]).abs().max())
    bowl_part_err = float((w_el["swl_gen"] + w_el["swl_hp"] - w_el["swl"]).abs().max())
    bat_hp_share_w = float((b_el["sw_hp"] / b_el["sw"]).median())
    bat_hp_share_all = float(b_el["sw_hp"].sum() / b_el["sw"].sum())
    bowl_hp_share_w = float((w_el["swl_hp"] / w_el["swl"]).median())
    bowl_hp_share_all = float(w_el["swl_hp"].sum() / w_el["swl"].sum())

    print("\n-- Zero-Overlap Partition Audit (General PI < 60 vs High-Pressure PI >= 60) --")
    print(f"Batting weight conservation max|sw_gen + sw_hp - sw| = {bat_part_err:.2e} | HP weight share: pool={bat_hp_share_all:.3f}, median={bat_hp_share_w:.3f}")
    print(f"  Elig Corr(raa_lcb[PI<60], hp_raa_lcb[PI>=60]): Pearson={pearsonr(b_el['raa_lcb'], b_el['hp_raa_lcb']).statistic:.4f}, "
          f"Spearman={spearmanr(b_el['raa_lcb'], b_el['hp_raa_lcb']).statistic:.4f}")
    print(f"  Elig Corr(cons_lcb[PI<60], hp_raa_lcb[PI>=60]): Pearson={pearsonr(b_el['cons_lcb'], b_el['hp_raa_lcb']).statistic:.4f}")
    print(f"Bowling weight conservation max|swl_gen + swl_hp - swl| = {bowl_part_err:.2e} | HP weight share: pool={bowl_hp_share_all:.3f}, median={bowl_hp_share_w:.3f}")
    print(f"  Elig Corr(rsaa_lcb[PI<60], hp_impact_lcb[PI>=60]): Pearson={pearsonr(w_el['rsaa_lcb'], w_el['hp_impact_lcb']).statistic:.4f}, "
          f"Spearman={spearmanr(w_el['rsaa_lcb'], w_el['hp_impact_lcb']).statistic:.4f}")
    print(f"  Elig Corr(cons_lcb[PI<60], hp_impact_lcb[PI>=60]): Pearson={pearsonr(w_el['cons_lcb'], w_el['hp_impact_lcb']).statistic:.4f}")

    # Ablation & Comparison: Previous v1 Model -> Updated v2 Model
    print("\n-- Model Comparison: Previous v1 vs Updated v2 (Ablation & Final Impact) --")
    cfg_v1 = PerformanceConfig(disjoint_hp_split=False, bowl_w_rsaa=0.40, bowl_w_waa=0.30)
    cfg_step_split = PerformanceConfig(disjoint_hp_split=True, bowl_w_rsaa=0.40, bowl_w_waa=0.30)
    cfg_step_wts = PerformanceConfig(disjoint_hp_split=False, bowl_w_rsaa=0.55, bowl_w_waa=0.15)

    bat_v1, _ = evaluate_batting_pressure(oof_pri, cfg_v1)
    bowl_v1, _ = evaluate_bowling_pressure(oof_pri, cfg_v1)
    bowl_split_only, _ = evaluate_bowling_pressure(oof_pri, cfg_step_split)
    bowl_wts_only, _ = evaluate_bowling_pressure(oof_pri, cfg_step_wts)

    v1_v2_bat = _compare_ranking_overlap(bat_v1, bat_base)
    v1_v2_bowl = _compare_ranking_overlap(bowl_v1, bowl_base)
    print(f"  Eligibility counts v1 vs v2: Bat ({bat_v1['elig_primary'].sum()} -> {bat_base['elig_primary'].sum()}), "
          f"Bowl ({bowl_v1['elig_primary'].sum()} -> {bowl_base['elig_primary'].sum()})")
    print("  Batting  (v1 -> v2: Disjoint PI<60/PI>=60):              ", v1_v2_bat)
    print("  Bowling  (v1 40/30 Overlap -> Disjoint 40/30 Only):     ", _compare_ranking_overlap(bowl_v1, bowl_split_only))
    print("  Bowling  (v1 40/30 Overlap -> Overlap 55/15 Wts Only):  ", _compare_ranking_overlap(bowl_v1, bowl_wts_only))
    print("  Bowling  (v1 -> v2: Disjoint + 55/15/15/15 Full Update):", v1_v2_bowl)

    # Suspected DLS (+701) sensitivity Top-10 overlap on Updated v2
    oof_sens, _ = fit_predict_oof_baselines(df, ds["matches"], ds["competitions"], include_suspected_dls=True, n_splits=5)
    bat_dls, _ = evaluate_batting_pressure(oof_sens, cfg_base)
    bowl_dls, _ = evaluate_bowling_pressure(oof_sens, cfg_base)
    print("\n-- Suspected DLS (+701 deliveries) Sensitivity on Updated v2 --")
    print(f"  Eligibility counts: Bat ({bat_base['elig_primary'].sum()} -> {bat_dls['elig_primary'].sum()}), "
          f"Bowl ({bowl_base['elig_primary'].sum()} -> {bowl_dls['elig_primary'].sum()})")
    print("  Batting  (+Suspected DLS vs Primary):", _compare_ranking_overlap(bat_base, bat_dls))
    print("  Bowling  (+Suspected DLS vs Primary):", _compare_ranking_overlap(bowl_base, bowl_dls))

    # High-Pressure (PI >= 60) Weight Sensitivity: 15% (Primary) vs 20% vs 25%
    print("\n-- High-Pressure Pillar Weight Sensitivity (15% Primary vs 20% vs 25%, Scaling General Pillars) --")
    hp_bat_rows, hp_bowl_rows = [], []
    ev_bat_all = bat_base.copy()
    ev_bat_all["pri_score_lcb"] = ev_bat_all["ev_all_lcb"]
    for hp_w in [0.15, 0.20, 0.25]:
        scale = (1.0 - hp_w) / 0.85
        chp = PerformanceConfig(
            bat_w_raa=0.45 * scale,
            bat_w_surv=0.25 * scale,
            bat_w_cons=0.15 * scale,
            bat_w_hp=hp_w,
            bowl_w_rsaa=0.55 * scale,
            bowl_w_waa=0.15 * scale,
            bowl_w_cons=0.15 * scale,
            bowl_w_hp=hp_w,
        )
        bh, _ = evaluate_batting_pressure(oof_pri, chp)
        wh, _ = evaluate_bowling_pressure(oof_pri, chp)
        cmp_b = _compare_ranking_overlap(bat_base, bh)
        cmp_w = _compare_ranking_overlap(bowl_base, wh)
        cmp_b_ev = _compare_ranking_overlap(ev_bat_all, bh)
        cmp_w_ev = _compare_ranking_overlap(ev_bowl_all, wh)
        hp_bat_rows.append({"hp_weight": hp_w, **cmp_b, "sp_vs_ev_all": cmp_b_ev["pool_spearman"], "top10_vs_ev_all": cmp_b_ev["top10_overlap"]})
        hp_bowl_rows.append({"hp_weight": hp_w, **cmp_w, "sp_vs_ev_all": cmp_w_ev["pool_spearman"], "top10_vs_ev_all": cmp_w_ev["top10_overlap"]})
    print("-- Batting HP Weight Sweep --")
    print(pd.DataFrame(hp_bat_rows).round(4).to_string(index=False))
    print("-- Bowling HP Weight Sweep --")
    print(pd.DataFrame(hp_bowl_rows).round(4).to_string(index=False))

    # Check LCB z-penalty (0.00 vs 0.50 vs 1.00) Top-10 overlap & volume correlation
    print("\n-- LCB Penalty Sensitivity & Participation Volume Fairness --")
    for lz in [0.00, 0.50, 1.00]:
        cl = PerformanceConfig(lcb_z=lz)
        bl, _ = evaluate_batting_pressure(oof_pri, cl)
        wl, _ = evaluate_bowling_pressure(oof_pri, cl)
        bo = _compare_ranking_overlap(bat_base, bl)
        wo = _compare_ranking_overlap(bowl_base, wl)
        b_corr_m = float(spearmanr(bl.loc[bl["elig_primary"], "pri_score_lcb"], bl.loc[bl["elig_primary"], "n_matches"]).statistic)
        w_corr_m = float(spearmanr(wl.loc[wl["elig_primary"], "pri_score_lcb"], wl.loc[wl["elig_primary"], "n_matches"]).statistic)
        print(f"  lcb_z={lz:.2f} | Bat Top10={bo['top10_overlap']}/10 Sp={bo['pool_spearman']:.4f} corr(n_matches)={b_corr_m:.4f} "
              f"| Bowl Top10={wo['top10_overlap']}/10 Sp={wo['pool_spearman']:.4f} corr(n_matches)={w_corr_m:.4f}")

    # ------------------------------------------------------------------
    # GENERATE ADDITIONAL VALIDATION FIGURES (19_, 20_, 21_)
    # ------------------------------------------------------------------
    # Figure 19: Bowling WAA Shrinkage, RSAA/WAA Weight Audit & Zero-Overlap General vs HP
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    sns.histplot(w_el["waa_raw"], bins=28, color="#64748b", alpha=0.45, label="Raw WAA (/4ov, PI<60)", ax=axes[0, 0])
    sns.histplot(w_el["waa_post"], bins=18, color="#0284c7", alpha=0.85, label=f"EB Shrunk WAA (k0={bowl_meta_base['waa']['k0_prior_neff']:,.0f})", ax=axes[0, 0])
    axes[0, 0].set_title(f"Bowling WAA Raw vs Shrunk Distribution (Eligible N={len(w_el)})", fontweight="bold")
    axes[0, 0].set_xlabel("Wickets Above Expectation / 4-Over Spell (PI < 60)")
    axes[0, 0].legend()

    x_bins = np.arange(len(vdecomp))
    axes[0, 1].bar(x_bins - 0.18, vdecomp["obs_var_waa"], width=0.35, color="#e11d48", label="Observed Empirical Var(waa_raw)")
    axes[0, 1].bar(x_bins + 0.18, vdecomp["expected_sampling_var"], width=0.35, color="#2563eb", label="Expected Sampling Var (sigma2 / Neff)")
    axes[0, 1].set_xticks(x_bins)
    axes[0, 1].set_xticklabels([str(idx) for idx in vdecomp.index], rotation=15)
    axes[0, 1].set_title("Why WAA Weight Was Cut to 15%: Noise Dominates Raw Variance", fontweight="bold")
    axes[0, 1].set_ylabel("Variance (Wickets/4-Over)^2")
    axes[0, 1].legend()

    ws_df = pd.DataFrame(w_sens_rows)
    x_w = np.arange(len(ws_df))
    axes[1, 0].plot(x_w, ws_df["Top10_vs_Base"], marker="o", linewidth=2.2, color="#1d4ed8", label="Top-10 Overlap vs 55/15 Updated Base")
    axes[1, 0].plot(x_w, ws_df["Top10_vs_EV_All"], marker="s", linewidth=2.2, color="#0f766e", label="Top-10 Overlap vs Full-Career EV Run-Equiv")
    axes[1, 0].set_xticks(x_w)
    axes[1, 0].set_xticklabels(["40/30\n(v1)", "50/20", "55/15\n(Base)", "60/10", "70/00"])
    axes[1, 0].set_ylim(0, 10.5)
    axes[1, 0].set_title("Bowling Top-10 Overlap Across RSAA/WAA Weights", fontweight="bold")
    axes[1, 0].set_ylabel("Shared Players in Top 10 (/10)")
    axes[1, 0].set_xlabel("RSAA / WAA Weight Split")
    axes[1, 0].legend()

    axes[1, 1].scatter(b_el["raa_lcb"], b_el["hp_raa_lcb"], alpha=0.55, color="#be185d",
                       label=f"Bat RAA(PI<60) vs HP(PI>=60) (r={pearsonr(b_el['raa_lcb'], b_el['hp_raa_lcb']).statistic:.2f})")
    axes[1, 1].set_title("Zero-Overlap Regime Check: General (PI<60) vs High-Pressure (PI>=60)", fontweight="bold")
    axes[1, 1].set_xlabel("General Pressure RAA LCB (PI < 60, /100 balls)")
    axes[1, 1].set_ylabel("High-Pressure RAA LCB (PI >= 60, /100 balls)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "19_phase3_waa_and_pillar_audit.png", dpi=180)
    plt.close()

    # Figure 20: Pressure-Bin & Season Validation Calibration
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    p_bat = pd.DataFrame([r for r in bat_slices if r["Slice_Type"] == "Pressure"])
    x_p = np.arange(len(p_bat))
    axes[0, 0].plot(x_p, p_bat["act_runs"], marker="o", linewidth=2.2, color="#0f172a", label="Actual Runs/Ball")
    axes[0, 0].plot(x_p, p_bat["exp_runs"], marker="s", linestyle="--", linewidth=2.0, color="#db2777", label="OOF Expected Runs/Ball")
    axes[0, 0].set_xticks(x_p)
    axes[0, 0].set_xticklabels(p_bat["Slice"])
    axes[0, 0].set_title("Batting Expected Runs Calibration Across Pressure Ranges", fontweight="bold")
    axes[0, 0].set_ylabel("Runs / Ball Faced")
    axes[0, 0].legend()

    axes[0, 1].plot(x_p, 100 * p_bat["act_wkt"], marker="o", linewidth=2.2, color="#0f172a", label="Actual Dismissal %")
    axes[0, 1].plot(x_p, 100 * p_bat["exp_wkt"], marker="s", linestyle="--", linewidth=2.0, color="#2563eb", label="OOF Expected Dismissal %")
    axes[0, 1].set_xticks(x_p)
    axes[0, 1].set_xticklabels(p_bat["Slice"])
    axes[0, 1].set_title("Batting Dismissal Calibration Across Pressure Ranges", fontweight="bold")
    axes[0, 1].set_ylabel("Dismissals / 100 Balls Faced")
    axes[0, 1].legend()

    ip_bat = pd.DataFrame([r for r in bat_slices if r["Slice_Type"] == "Inn_Phase"])
    x_ip = np.arange(len(ip_bat))
    axes[1, 0].bar(x_ip - 0.18, ip_bat["act_runs"], width=0.35, color="#334155", label="Actual")
    axes[1, 0].bar(x_ip + 0.18, ip_bat["exp_runs"], width=0.35, color="#ec4899", label="OOF Expected")
    axes[1, 0].set_xticks(x_ip)
    axes[1, 0].set_xticklabels(ip_bat["Slice"], rotation=20)
    axes[1, 0].set_title("Batting Calibration by Innings & Phase", fontweight="bold")
    axes[1, 0].set_ylabel("Runs / Ball")
    axes[1, 0].legend()

    cv_df = pd.DataFrame(cv_rows)
    x_cv = np.arange(len(cv_df))
    axes[1, 1].bar(x_cv - 0.18, cv_df["Bat_RMSE"], width=0.35, color="#0284c7", label="Batting Runs RMSE")
    axes[1, 1].bar(x_cv + 0.18, cv_df["Bowl_Leg_RMSE"], width=0.35, color="#0d9488", label="Bowling Legal RMSE")
    axes[1, 1].set_xticks(x_cv)
    axes[1, 1].set_xticklabels(["Match-GKF\n(All)", "LOSO-CV\n(All)", "Match-GKF\n(2023+)", "Walk-Fwd\n(2023+)"])
    axes[1, 1].set_ylim(1.5, 1.85)
    axes[1, 1].set_title("Out-of-Fold vs Season-Based CV Error Stability", fontweight="bold")
    axes[1, 1].set_ylabel("RMSE (Runs / Ball)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "20_phase3_calibration_and_season_cv.png", dpi=180)
    plt.close()

    # Figure 21: Top-10 Membership Stability Across w_T20I, Gamma, Eligibility, HP Weight & v1->v2
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    cb_df = pd.DataFrame(comp_bat_rows)
    cw_df = pd.DataFrame(comp_bowl_rows)
    axes[0, 0].plot(cb_df["w_t20i"], cb_df["top10_overlap"], marker="o", linewidth=2.2, color="#db2777", label="Batting Top-10 Overlap")
    axes[0, 0].plot(cw_df["w_t20i"], cw_df["top10_overlap"], marker="s", linewidth=2.2, color="#2563eb", label="Bowling Top-10 Overlap")
    axes[0, 0].set_ylim(0, 10.8)
    axes[0, 0].set_title("Top-10 Overlap vs w_T20I = 0.50 (Updated v2)", fontweight="bold")
    axes[0, 0].set_xlabel("T20I Weight (w_T20I)")
    axes[0, 0].set_ylabel("Shared Players in Top 10 (/10)")
    axes[0, 0].legend()

    gb_df = pd.DataFrame(g_bat_rows)
    gw_df = pd.DataFrame(g_bowl_rows)
    axes[0, 1].plot(gb_df["gamma"], gb_df["top10_overlap"], marker="o", linewidth=2.2, color="#db2777", label="Batting Top-10 Overlap")
    axes[0, 1].plot(gw_df["gamma"], gw_df["top10_overlap"], marker="s", linewidth=2.2, color="#2563eb", label="Bowling Top-10 Overlap")
    axes[0, 1].set_ylim(0, 10.8)
    axes[0, 1].set_title("Top-10 Overlap Across Pressure Exponent gamma", fontweight="bold")
    axes[0, 1].set_xlabel("Pressure Weighting Exponent (gamma)")
    axes[0, 1].set_ylabel("Shared Players in Top 10 (/10)")
    axes[0, 1].legend()

    eb_df = pd.DataFrame(ebat_rows)
    ew_df = pd.DataFrame(ebowl_rows)
    x_e = np.arange(len(eb_df))
    axes[1, 0].bar(x_e - 0.18, eb_df["top10_overlap"], width=0.35, color="#db2777", label="Batting Top-10 Overlap")
    axes[1, 0].bar(x_e + 0.18, ew_df["top10_overlap"], width=0.35, color="#2563eb", label="Bowling Top-10 Overlap")
    axes[1, 0].set_xticks(x_e)
    axes[1, 0].set_xticklabels(["(30, 30)", "(50, 60)", "(75, 100)"])
    axes[1, 0].set_ylim(0, 10.8)
    axes[1, 0].set_title("Top-10 Overlap Across Eligibility Cutoffs", fontweight="bold")
    axes[1, 0].set_xlabel("(min_Neff, min_IPL_balls)")
    axes[1, 0].set_ylabel("Shared Players in Top 10 (/10)")
    axes[1, 0].legend()

    hpb_df = pd.DataFrame(hp_bat_rows)
    hpw_df = pd.DataFrame(hp_bowl_rows)
    labels_cmp = ["HP=15%\n(Base)", "HP=20%", "HP=25%", "v1 vs v2\nUpdate"]
    b_ov_vals = list(hpb_df["top10_overlap"]) + [v1_v2_bat["top10_overlap"]]
    w_ov_vals = list(hpw_df["top10_overlap"]) + [v1_v2_bowl["top10_overlap"]]
    x_h = np.arange(len(labels_cmp))
    axes[1, 1].bar(x_h - 0.18, b_ov_vals, width=0.35, color="#db2777", label="Batting Top-10 Overlap")
    axes[1, 1].bar(x_h + 0.18, w_ov_vals, width=0.35, color="#2563eb", label="Bowling Top-10 Overlap")
    axes[1, 1].set_xticks(x_h)
    axes[1, 1].set_xticklabels(labels_cmp)
    axes[1, 1].set_ylim(0, 10.8)
    axes[1, 1].set_title("HP Weight Sensitivity (15/20/25%) & v1->v2 Top-10 Retention", fontweight="bold")
    axes[1, 1].set_ylabel("Shared Players in Top 10 (/10)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "21_phase3_top10_stability_audit.png", dpi=180)
    plt.close()
    print("\nSaved Figures 19_, 20_, 21_ to reports/figures.")


if __name__ == "__main__":
    main()

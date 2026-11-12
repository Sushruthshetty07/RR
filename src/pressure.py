"""
Continuous Pressure Index Engine (0-100) for Rajasthan Royals SupeRR Selector 2.0.

Implements four separate regular-innings pressure formulations plus separate
Super Over handling, grounded in pre-delivery situational state only:
  1. Batting during a chase (innings == 2)
  2. Batting while setting a target (innings == 1)
  3. Bowling while defending a target (innings == 2)
  4. Bowling during first innings (innings == 1)
  5. Super Overs (innings >= 3, odd setting / even chasing; kept separate)

Safeguards enforced:
- Pre-delivery causality only (never reads delivery outcomes).
- Leave-One-Match-Out (LOO) competition-specific par & expected wickets curves
  with recorded support counts and pooled/interpolated fallback below MIN_LOO_SUPPORT.
- Zero-leakage DLS handling: distinguishes pre-innings DLS revisions from
  timestamp-gap suspected mid-innings stoppages; pre-stoppage deliveries never
  use future post-rain revised_target values.
- Primary calculation uses knockout_multiplier = 1.00 (sensitivity tested at 1.05, 1.10).
- Super Over 85-100 base is retained separately and documented as an assumption.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PressureConfig:
    """Configuration parameters for the 0-100 Continuous Pressure Index."""

    # Minimum out-of-match records required for unsmoothed LOO baseline
    min_loo_support: int = 30
    # Inter-ball timestamp gap (minutes) indicating suspected mid-innings rain stoppage
    dls_gap_minutes: float = 12.0
    # Primary knockout multiplier (1.00 = no knockout boost; 1.05/1.10 tested in sensitivity)
    knockout_multiplier: float = 1.00

    # --- Regime 1: Batting Chase weights & parameters ---
    bat_chase_w_rrr: float = 0.40
    bat_chase_w_rate_diff: float = 0.20
    bat_chase_w_wkt: float = 0.25
    bat_chase_w_balls: float = 0.15
    bat_chase_rrr_mid: float = 9.5
    bat_chase_rrr_scale: float = 2.2
    bat_chase_diff_mid: float = 1.5
    bat_chase_diff_scale: float = 2.5
    bat_chase_wkt_power: float = 1.3
    bat_chase_ach_knee: float = 14.5
    bat_chase_ach_sigma: float = 4.5
    bat_chase_ach_floor: float = 0.15

    # --- Regime 2: Batting Setting weights & parameters ---
    bat_set_w_wkt: float = 0.45
    bat_set_w_deficit: float = 0.35
    bat_set_w_phase: float = 0.20
    bat_set_progress_balls: float = 24.0

    # --- Regime 3: Bowling Defending weights & parameters ---
    bowl_def_w_threat: float = 0.35
    bowl_def_w_wkt: float = 0.25
    bowl_def_w_late: float = 0.25
    bowl_def_w_cushion: float = 0.15
    bowl_def_rrr_peak: float = 9.8
    bowl_def_rrr_sigma: float = 3.3
    bowl_def_cushion_mid: float = 30.0
    bowl_def_cushion_scale: float = 12.0

    # --- Regime 4: Bowling Setting (1st Innings) weights & parameters ---
    bowl_set_w_dom: float = 0.45
    bowl_set_w_wkt: float = 0.30
    bowl_set_w_phase: float = 0.25
    bowl_set_progress_balls: float = 24.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid mapping (-inf, inf) -> (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def detect_dls_regimes(
    df: pd.DataFrame,
    matches_df: pd.DataFrame,
    gap_minutes: float = 12.0
) -> pd.DataFrame:
    """
    Classifies DLS matches and reconstructs pre-delivery effective target / ball
    limits without leaking future post-stoppage revised targets into pre-stoppage balls.

    Returns DataFrame columns appended:
    - dls_status: 'standard', 'dls_pre_innings_confirmed', 'dls_innings_break_suspected',
                  'dls_pre_stoppage_suspected', 'dls_post_stoppage_suspected',
                  or 'dls_ambiguous_timing'
    - dls_ambiguous: bool (True where pre-delivery target timing cannot be independently confirmed)
    - target_eff: effective pre-delivery target
    - max_legal_balls_eff: effective pre-delivery legal ball budget
    - runs_needed_eff: effective runs needed before delivery
    - legal_balls_remaining_eff: effective legal balls remaining before delivery
    - rrr_eff: effective required run rate before delivery
    """
    out = df.copy()
    out['dls_status'] = 'standard'
    out['dls_ambiguous'] = False
    out['target_eff'] = out['target']
    out['max_legal_balls_eff'] = out['max_legal_balls']
    out['legal_balls_remaining_eff'] = out['legal_balls_remaining_before']
    out['runs_needed_eff'] = out['runs_needed_before']
    out['rrr_eff'] = out['rrr_before']

    # Identify DLS matches in Innings 2
    rev_meta = matches_df[['match_id', 'revised_target', 'first_inn_revised_overs', 'second_inn_revised_overs']].copy()
    dls_ids = set(rev_meta[rev_meta['revised_target'].notna()]['match_id'])

    if not dls_ids:
        return out

    inn2_dls_mask = (out['innings'] == 2) & (out['match_id'].isin(dls_ids))
    if not inn2_dls_mask.any():
        return out

    # Inspect inter-ball timestamps within each DLS Innings 2
    dt_series = pd.to_datetime(out.loc[inn2_dls_mask, 'delivery_ist_time'], errors='coerce')
    out.loc[inn2_dls_mask, '_dt'] = dt_series

    for mid in sorted(dls_ids):
        mask = (out['match_id'] == mid) & (out['innings'] == 2)
        if not mask.any():
            continue

        idx = out.index[mask]
        rev_1 = out.loc[idx[0], 'first_inn_revised_overs']
        rev_2 = out.loc[idx[0], 'second_inn_revised_overs']
        inn1_runs = out.loc[idx[0], 'inn1_total_runs']

        dts = out.loc[idx, '_dt']
        if dts.isna().any():
            # Cannot verify stoppage timing when timestamps are missing
            out.loc[idx, 'dls_status'] = 'dls_ambiguous_timing'
            out.loc[idx, 'dls_ambiguous'] = True
            out.loc[idx, 'target_eff'] = np.nan
            out.loc[idx, 'runs_needed_eff'] = np.nan
            out.loc[idx, 'rrr_eff'] = np.nan
            continue

        gaps = dts.diff().dt.total_seconds().fillna(0.0) / 60.0
        stoppage_pos = np.where(gaps.values >= gap_minutes)[0]

        if len(stoppage_pos) > 0:
            # Timestamp-gap suspected mid-innings interruption!
            # Use the first major stoppage gap >= gap_minutes
            first_stop_i = int(stoppage_pos[0])
            pre_idx = idx[:first_stop_i]
            post_idx = idx[first_stop_i:]

            # Pre-stoppage deliveries MUST NOT leak future revised_target or rev_2
            pre_max_balls = int(rev_1) if pd.notnull(rev_1) else 120
            pre_target = (inn1_runs + 1.0) if pd.notnull(inn1_runs) else np.nan

            out.loc[pre_idx, 'dls_status'] = 'dls_pre_stoppage_suspected'
            out.loc[pre_idx, 'dls_ambiguous'] = True
            out.loc[pre_idx, 'target_eff'] = pre_target
            out.loc[pre_idx, 'max_legal_balls_eff'] = pre_max_balls
            out.loc[pre_idx, 'legal_balls_remaining_eff'] = (
                pre_max_balls - out.loc[pre_idx, 'legal_balls_before']
            ).clip(lower=0)
            out.loc[pre_idx, 'runs_needed_eff'] = (
                pre_target - out.loc[pre_idx, 'score_before']
            ).clip(lower=0)
            b_rem_pre = out.loc[pre_idx, 'legal_balls_remaining_eff']
            out.loc[pre_idx, 'rrr_eff'] = np.where(
                b_rem_pre > 0,
                out.loc[pre_idx, 'runs_needed_eff'] / (b_rem_pre / 6.0),
                np.where(out.loc[pre_idx, 'runs_needed_eff'] > 0, np.inf, np.nan)
            )

            # Post-stoppage deliveries use the revised target & revised ball limit
            out.loc[post_idx, 'dls_status'] = 'dls_post_stoppage_suspected'
            out.loc[post_idx, 'dls_ambiguous'] = False
        else:
            # Continuous play in Innings 2 (max gap < gap_minutes)
            if pd.notnull(rev_1) and pd.notnull(rev_2) and (rev_1 == rev_2):
                out.loc[idx, 'dls_status'] = 'dls_pre_innings_confirmed'
                out.loc[idx, 'dls_ambiguous'] = False
            else:
                out.loc[idx, 'dls_status'] = 'dls_innings_break_suspected'
                out.loc[idx, 'dls_ambiguous'] = False

    if '_dt' in out.columns:
        out.drop(columns=['_dt'], inplace=True)

    return out


def compute_loo_inn1_baselines(
    df: pd.DataFrame,
    min_support: int = 30
) -> pd.DataFrame:
    """
    Computes Leave-One-Match-Out (LOO) competition-specific ('IPL' vs 'T20I')
    empirical cumulative par score, score standard deviation, and expected wickets
    for Innings 1 deliveries.

    Prevents any match from contributing to its own baseline. Records exact
    contributing out-of-match counts (`loo_support_count`) and blends toward
    pooled curves / linear fallback whenever `loo_support_count < min_support`.
    """
    out = df.copy()

    # Initialize columns
    out['loo_exp_score'] = np.nan
    out['loo_std_score'] = np.nan
    out['loo_exp_wkts'] = np.nan
    out['loo_support_count'] = 0
    out['loo_low_support_flag'] = False

    inn1_std_mask = (out['innings'] == 1) & (out['max_legal_balls'] == 120)
    if not inn1_std_mask.any():
        return out

    # Take exactly 1 snapshot per (is_ipl, match_id, legal_balls_before)
    # Using first delivery at each legal ball index b in 0..119
    snap = (
        out.loc[inn1_std_mask]
        .groupby(['is_ipl', 'match_id', 'legal_balls_before'], as_index=False)
        .first()[['is_ipl', 'match_id', 'legal_balls_before', 'score_before', 'wickets_before']]
    )
    snap['score_sq'] = snap['score_before'] ** 2
    snap['wkts_sq'] = snap['wickets_before'] ** 2

    # Group totals by (is_ipl, legal_balls_before)
    comp_grp = snap.groupby(['is_ipl', 'legal_balls_before'], as_index=False).agg(
        c_sum_s=('score_before', 'sum'),
        c_sum_s2=('score_sq', 'sum'),
        c_sum_w=('wickets_before', 'sum'),
        c_n=('match_id', 'count'),
    )

    # Pooled totals by legal_balls_before (across IPL + T20I) for fallback smoothing
    pool_grp = snap.groupby('legal_balls_before', as_index=False).agg(
        p_sum_s=('score_before', 'sum'),
        p_sum_s2=('score_sq', 'sum'),
        p_sum_w=('wickets_before', 'sum'),
        p_n=('match_id', 'count'),
    )

    snap = snap.merge(comp_grp, on=['is_ipl', 'legal_balls_before'], how='left')
    snap = snap.merge(pool_grp, on='legal_balls_before', how='left')

    # LOO calculations for standard 20-over Innings 1 matches (exclude self)
    n_c_loo = (snap['c_n'] - 1).clip(lower=0)
    n_p_loo = (snap['p_n'] - 1).clip(lower=0)

    c_mean_s = np.where(n_c_loo > 0, (snap['c_sum_s'] - snap['score_before']) / n_c_loo, np.nan)
    c_mean_w = np.where(n_c_loo > 0, (snap['c_sum_w'] - snap['wickets_before']) / n_c_loo, np.nan)
    c_var_s = np.where(
        n_c_loo > 1,
        ((snap['c_sum_s2'] - snap['score_sq']) - n_c_loo * (c_mean_s ** 2)) / (n_c_loo - 1),
        np.nan
    )
    c_std_s = np.sqrt(np.maximum(0.0, c_var_s))

    p_mean_s = np.where(n_p_loo > 0, (snap['p_sum_s'] - snap['score_before']) / n_p_loo, np.nan)
    p_mean_w = np.where(n_p_loo > 0, (snap['p_sum_w'] - snap['wickets_before']) / n_p_loo, np.nan)
    p_var_s = np.where(
        n_p_loo > 1,
        ((snap['p_sum_s2'] - snap['score_sq']) - n_p_loo * (p_mean_s ** 2)) / (n_p_loo - 1),
        np.nan
    )
    p_std_s = np.sqrt(np.maximum(0.0, p_var_s))

    # Fallback / smoothing when n_c_loo < min_support:
    # Blend toward pooled LOO curve with weight alpha = n_c_loo / min_support
    alpha = np.clip(n_c_loo / float(min_support), 0.0, 1.0)
    blend_s = np.where(n_c_loo > 0, alpha * c_mean_s + (1.0 - alpha) * p_mean_s, p_mean_s)
    blend_w = np.where(n_c_loo > 0, alpha * c_mean_w + (1.0 - alpha) * p_mean_w, p_mean_w)
    blend_std = np.where(
        np.isfinite(c_std_s),
        alpha * c_std_s + (1.0 - alpha) * np.where(np.isfinite(p_std_s), p_std_s, 10.0),
        np.where(np.isfinite(p_std_s), p_std_s, 10.0)
    )

    # Final safety net if even pooled support < 2 (e.g., tiny synthetic test): linear RPO fallback
    b_vals = snap['legal_balls_before'].values.astype(float)
    par_rpo = np.where(snap['is_ipl'], 9.26, 8.45)
    fallback_s = par_rpo * (b_vals / 6.0)
    fallback_w = 5.0 * (b_vals / 120.0)
    fallback_std = 5.0 + 0.22 * b_vals

    snap['loo_exp_score'] = np.where(np.isfinite(blend_s) & (n_p_loo >= 2), blend_s, fallback_s)
    snap['loo_exp_wkts'] = np.where(np.isfinite(blend_w) & (n_p_loo >= 2), blend_w, fallback_w)
    snap['loo_std_score'] = np.maximum(5.0, np.where(np.isfinite(blend_std) & (n_p_loo >= 2), blend_std, fallback_std))
    snap['loo_support_count'] = n_c_loo
    snap['loo_low_support_flag'] = n_c_loo < min_support

    # Merge LOO values back onto standard 20-over Innings 1 deliveries
    loo_lookup = snap[['is_ipl', 'match_id', 'legal_balls_before',
                       'loo_exp_score', 'loo_std_score', 'loo_exp_wkts',
                       'loo_support_count', 'loo_low_support_flag']]

    merged = out.loc[inn1_std_mask, ['is_ipl', 'match_id', 'legal_balls_before']].merge(
        loo_lookup, on=['is_ipl', 'match_id', 'legal_balls_before'], how='left'
    )
    for col in ['loo_exp_score', 'loo_std_score', 'loo_exp_wkts', 'loo_support_count', 'loo_low_support_flag']:
        out.loc[inn1_std_mask, col] = merged[col].values

    # Handle shortened Innings 1 matches (max_legal_balls < 120):
    # Map progress fraction onto competition-level empirical curve (already excludes self since self isn't in 120-ball grp)
    short_inn1_mask = (out['innings'] == 1) & (out['max_legal_balls'] < 120)
    if short_inn1_mask.any():
        comp_curve = comp_grp.copy()
        comp_curve['c_mean_s'] = comp_curve['c_sum_s'] / comp_curve['c_n']
        comp_curve['c_mean_w'] = comp_curve['c_sum_w'] / comp_curve['c_n']
        comp_var = np.maximum(0.0, (comp_curve['c_sum_s2'] - comp_curve['c_n'] * (comp_curve['c_mean_s'] ** 2)) / np.maximum(1, comp_curve['c_n'] - 1))
        comp_curve['c_std_s'] = np.maximum(5.0, np.sqrt(comp_var))

        short_df = out.loc[short_inn1_mask, ['is_ipl', 'legal_balls_before', 'max_legal_balls']].copy()
        scale = short_df['max_legal_balls'] / 120.0
        short_df['b_equiv'] = np.clip(
            np.round(120.0 * short_df['legal_balls_before'] / short_df['max_legal_balls']).astype(int),
            0, 119
        )
        m_short = short_df.merge(
            comp_curve[['is_ipl', 'legal_balls_before', 'c_mean_s', 'c_std_s', 'c_mean_w', 'c_n']],
            left_on=['is_ipl', 'b_equiv'],
            right_on=['is_ipl', 'legal_balls_before'],
            how='left'
        )
        # Fallback if comp_curve empty (e.g. tiny synthetic test)
        rpo_fb = np.where(short_df['is_ipl'], 9.26, 8.45)
        b_eq = short_df['b_equiv'].values.astype(float)
        out.loc[short_inn1_mask, 'loo_exp_score'] = np.where(
            m_short['c_mean_s'].notna(), m_short['c_mean_s'].values * scale.values, rpo_fb * (short_df['legal_balls_before'].values / 6.0)
        )
        out.loc[short_inn1_mask, 'loo_std_score'] = np.maximum(
            5.0, np.where(m_short['c_std_s'].notna(), m_short['c_std_s'].values * scale.values, 5.0 + 0.22 * b_eq * scale.values)
        )
        out.loc[short_inn1_mask, 'loo_exp_wkts'] = np.where(
            m_short['c_mean_w'].notna(), m_short['c_mean_w'].values, 5.0 * (b_eq / 120.0)
        )
        supp = m_short['c_n'].fillna(0).astype(int).values
        out.loc[short_inn1_mask, 'loo_support_count'] = supp
        out.loc[short_inn1_mask, 'loo_low_support_flag'] = supp < min_support

    return out


def compute_pressure_scores(
    state_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    competitions_df: pd.DataFrame,
    config: PressureConfig = PressureConfig(),
) -> pd.DataFrame:
    """
    Computes continuous pre-delivery pressure indices (0-100) for all 4 regular-innings
    regimes plus separate Super Over pressure indices.

    Strictly uses pre-delivery situational columns only.
    """
    df = state_df.copy()

    # Attach tournament type (IPL vs T20I)
    comp_meta = matches_df[['match_id', 'comp_id']].merge(
        competitions_df[['comp_id', 'comp_name']], on='comp_id', how='left'
    )
    comp_meta['is_ipl'] = comp_meta['comp_name'].str.contains('IPL', case=False, na=False)
    if 'is_ipl' not in df.columns:
        df = df.merge(comp_meta[['match_id', 'is_ipl']], on='match_id', how='left')
    df['is_ipl'] = df['is_ipl'].fillna(False)

    # 1. Zero-leakage DLS handling
    df = detect_dls_regimes(df, matches_df, gap_minutes=config.dls_gap_minutes)

    # 2. Competition-specific Leave-One-Match-Out Innings 1 baselines
    df = compute_loo_inn1_baselines(df, min_support=config.min_loo_support)

    # Context multiplier (primary = 1.00; sensitivity tested at 1.05, 1.10)
    knock_flag = df['knock_out'].fillna(0).astype(int) if 'knock_out' in df.columns else 0
    m_ctx = np.where(knock_flag == 1, config.knockout_multiplier, 1.0)

    # Initialize output columns
    df['is_super_over'] = df['innings'] >= 3
    df['pressure_regime'] = 'unknown'
    df['pressure_bat'] = np.nan
    df['pressure_bowl'] = np.nan
    df['pressure_so_bat'] = np.nan
    df['pressure_so_bowl'] = np.nan

    # Diagnostic sub-components
    for col in [
        'comp_u_rrr', 'comp_d_rate', 'comp_r_wkt', 'comp_t_balls', 'comp_achievability',
        'comp_c_threat', 'comp_w_avail', 'comp_t_late', 'comp_c_cushion',
        'comp_p_deficit', 'comp_c_wkt', 'comp_b_dom', 'comp_w_threat', 'comp_phi_eff'
    ]:
        df[col] = np.nan

    # =========================================================================
    # REGIME 1 & 3: INNINGS 2 (Batting Chase & Bowling Defending)
    # Exclude ALL ambiguous DLS deliveries (dls_ambiguous == True) so
    # ambiguous target/overs never silently produce trusted pressure scores.
    # =========================================================================
    inn2_mask = (df['innings'] == 2) & (~df['dls_ambiguous'].fillna(False)) & df['rrr_eff'].notna()
    if inn2_mask.any():
        sub = df.loc[inn2_mask]
        rrr = sub['rrr_eff'].values
        crr = sub['crr_before'].values
        w_hand = sub['wickets_in_hand_before'].values.astype(float)
        b_rem = sub['legal_balls_remaining_eff'].values.astype(float)
        b_max = sub['max_legal_balls_eff'].values.astype(float)
        r_need = sub['runs_needed_eff'].values.astype(float)
        ctx = m_ctx[inn2_mask] if isinstance(m_ctx, np.ndarray) else m_ctx

        # --- Batting Chase ---
        u_rrr = _sigmoid((rrr - config.bat_chase_rrr_mid) / config.bat_chase_rrr_scale)
        d_rate = _sigmoid(((rrr - crr) - config.bat_chase_diff_mid) / config.bat_chase_diff_scale)
        r_wkt = np.clip((10.0 - w_hand) / 9.0, 0.0, 1.0) ** config.bat_chase_wkt_power
        t_balls = np.clip(1.0 - (b_rem / np.maximum(1.0, b_max)), 0.0, 1.0)

        a_base = 0.20 + 0.80 * np.exp(
            -(np.maximum(0.0, rrr - config.bat_chase_ach_knee) ** 2) / (2.0 * (config.bat_chase_ach_sigma ** 2))
        )
        # Additional extreme feasibility guard near end of chase (> 4.5 runs/ball when <= 18 balls left)
        rpb = r_need / np.maximum(1.0, b_rem)
        end_impossible = (b_rem <= 18.0) & (r_need > 4.5 * b_rem)
        excess_rpb = np.maximum(0.0, rpb - 3.5)
        a_final = np.where(
            end_impossible,
            a_base * np.exp(-(excess_rpb ** 2) / (2.0 * (1.2 ** 2))),
            a_base
        )
        ach = np.maximum(config.bat_chase_ach_floor, a_final)

        bat_chase_raw = (
            config.bat_chase_w_rrr * u_rrr
            + config.bat_chase_w_rate_diff * d_rate
            + config.bat_chase_w_wkt * r_wkt
            + config.bat_chase_w_balls * t_balls
        )
        pi_bat_chase = 100.0 * np.clip(ach * bat_chase_raw * ctx, 0.0, 1.0)

        # --- Bowling Defending ---
        c_threat = np.exp(-((rrr - config.bowl_def_rrr_peak) ** 2) / (2.0 * (config.bowl_def_rrr_sigma ** 2)))
        w_avail = np.clip(w_hand / 10.0, 0.0, 1.0) ** 0.9
        t_late = np.clip(1.0 - (b_rem / np.maximum(1.0, b_max)), 0.0, 1.0) ** 1.4
        c_cushion = _sigmoid((config.bowl_def_cushion_mid - r_need) / config.bowl_def_cushion_scale)
        l_alive = 0.25 + 0.75 * c_threat

        bowl_def_raw = (
            config.bowl_def_w_threat * c_threat
            + config.bowl_def_w_wkt * w_avail
            + config.bowl_def_w_late * t_late * l_alive
            + config.bowl_def_w_cushion * c_cushion * l_alive
        )
        pi_bowl_def = 100.0 * np.clip(bowl_def_raw * ctx, 0.0, 1.0)

        df.loc[inn2_mask, 'pressure_regime'] = 'bat_chase_bowl_defend'
        df.loc[inn2_mask, 'pressure_bat'] = pi_bat_chase
        df.loc[inn2_mask, 'pressure_bowl'] = pi_bowl_def
        df.loc[inn2_mask, 'comp_u_rrr'] = u_rrr
        df.loc[inn2_mask, 'comp_d_rate'] = d_rate
        df.loc[inn2_mask, 'comp_r_wkt'] = r_wkt
        df.loc[inn2_mask, 'comp_t_balls'] = t_balls
        df.loc[inn2_mask, 'comp_achievability'] = ach
        df.loc[inn2_mask, 'comp_c_threat'] = c_threat
        df.loc[inn2_mask, 'comp_w_avail'] = w_avail
        df.loc[inn2_mask, 'comp_t_late'] = t_late
        df.loc[inn2_mask, 'comp_c_cushion'] = c_cushion

    # =========================================================================
    # REGIME 2 & 4: INNINGS 1 (Batting Setting & Bowling Setting)
    # =========================================================================
    inn1_mask = df['innings'] == 1
    if inn1_mask.any():
        sub = df.loc[inn1_mask]
        b_bowled = sub['legal_balls_before'].values.astype(float)
        score = sub['score_before'].values.astype(float)
        w_before = sub['wickets_before'].values.astype(float)
        w_hand = sub['wickets_in_hand_before'].values.astype(float)
        exp_s = sub['loo_exp_score'].values.astype(float)
        std_s = np.maximum(5.0, sub['loo_std_score'].values.astype(float))
        exp_w = sub['loo_exp_wkts'].values.astype(float)
        overs = sub['overid'].values.astype(float)
        ctx = m_ctx[inn1_mask] if isinstance(m_ctx, np.ndarray) else m_ctx

        progress_bat = np.minimum(1.0, b_bowled / config.bat_set_progress_balls)
        progress_bowl = np.minimum(1.0, b_bowled / config.bowl_set_progress_balls)

        # --- Batting Setting ---
        z_def = (exp_s - score) / std_s
        p_deficit = progress_bat * _sigmoid(z_def)
        delta_w = np.maximum(0.0, w_before - exp_w)
        c_wkt = 0.55 * (1.0 - np.exp(-0.7 * delta_w)) + 0.45 * (np.clip(w_before / 10.0, 0.0, 1.0) ** 1.3)

        phi_bat = np.where(
            overs <= 6,
            0.15,
            np.where(overs <= 15, 0.25, 0.55 + 0.25 * np.clip((overs - 16.0) / 4.0, 0.0, 1.0))
        )
        cruising_relief = 1.0 - 0.35 * _sigmoid((-z_def) - 1.2) * (w_hand / 10.0)
        phi_bat_eff = phi_bat * cruising_relief

        bat_set_raw = (
            config.bat_set_w_wkt * c_wkt
            + config.bat_set_w_deficit * p_deficit
            + config.bat_set_w_phase * phi_bat_eff
        )
        pi_bat_set = 100.0 * np.clip(bat_set_raw * ctx, 0.0, 1.0)

        # --- Bowling Setting ---
        z_dom = (score - exp_s) / std_s
        b_dom = progress_bowl * _sigmoid(z_dom)
        delta_whand = exp_w - w_before
        w_threat = progress_bowl * _sigmoid(delta_whand / 1.2)
        w_avail_raw = np.clip(w_hand / 10.0, 0.0, 1.0) ** 1.2

        phi_bowl = np.where(
            overs <= 6,
            0.35,
            np.where(overs <= 15, 0.25, 0.65 + 0.25 * np.clip((overs - 16.0) / 4.0, 0.0, 1.0))
        )
        phi_bowl_eff = phi_bowl * w_avail_raw

        bowl_set_raw = (
            config.bowl_set_w_dom * b_dom
            + config.bowl_set_w_wkt * w_threat
            + config.bowl_set_w_phase * phi_bowl_eff
        )
        pi_bowl_set = 100.0 * np.clip(bowl_set_raw * ctx, 0.0, 1.0)

        df.loc[inn1_mask, 'pressure_regime'] = 'bat_set_bowl_set'
        df.loc[inn1_mask, 'pressure_bat'] = pi_bat_set
        df.loc[inn1_mask, 'pressure_bowl'] = pi_bowl_set
        df.loc[inn1_mask, 'comp_p_deficit'] = p_deficit
        df.loc[inn1_mask, 'comp_c_wkt'] = c_wkt
        df.loc[inn1_mask, 'comp_b_dom'] = b_dom
        df.loc[inn1_mask, 'comp_w_threat'] = w_threat
        df.loc[inn1_mask, 'comp_phi_eff'] = phi_bowl_eff

    # =========================================================================
    # REGIME 5: SUPER OVERS (Generalized innings >= 3; kept separate)
    # Note: 85-100 base scale is an analytical assumption documented separately.
    # =========================================================================
    so_odd_mask = (df['innings'] >= 3) & (df['innings'] % 2 == 1)
    if so_odd_mask.any():
        sub = df.loc[so_odd_mask]
        b_rem = sub['legal_balls_remaining_before'].values.astype(float)
        w_before = sub['wickets_before'].values.astype(float)
        ctx = m_ctx[so_odd_mask] if isinstance(m_ctx, np.ndarray) else m_ctx

        so_set_val = np.minimum(100.0, (85.0 + 6.0 * (1.0 - b_rem / 6.0) + 4.0 * (w_before / 2.0)) * ctx)
        df.loc[so_odd_mask, 'pressure_regime'] = 'super_over_set'
        df.loc[so_odd_mask, 'pressure_so_bat'] = so_set_val
        df.loc[so_odd_mask, 'pressure_so_bowl'] = so_set_val

    so_even_mask = (df['innings'] >= 4) & (df['innings'] % 2 == 0) & df['rrr_before'].notna()
    if so_even_mask.any():
        sub = df.loc[so_even_mask]
        rrr = sub['rrr_before'].values
        w_hand = sub['wickets_in_hand_before'].values.astype(float)
        ctx = m_ctx[so_even_mask] if isinstance(m_ctx, np.ndarray) else m_ctx

        a_so = 0.25 + 0.75 * np.exp(-(np.maximum(0.0, rrr - 18.0) ** 2) / (2.0 * (8.0 ** 2)))
        so_bat_val = np.minimum(100.0, (85.0 + 15.0 * np.clip(a_so * _sigmoid((rrr - 10.0) / 3.0), 0.0, 1.0)) * ctx)
        so_bowl_val = np.minimum(
            100.0,
            (85.0 + 15.0 * np.clip(np.exp(-((rrr - 10.0) ** 2) / (2.0 * (4.5 ** 2))) * (0.5 + 0.5 * w_hand / 2.0), 0.0, 1.0)) * ctx
        )
        df.loc[so_even_mask, 'pressure_regime'] = 'super_over_chase'
        df.loc[so_even_mask, 'pressure_so_bat'] = so_bat_val
        df.loc[so_even_mask, 'pressure_so_bowl'] = so_bowl_val

    return df


def compute_player_opportunities(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates regular-innings player pressure opportunity counts (`bat_opp`, `bowl_opp`),
    explicitly excluding Super Overs (`innings >= 3`) and ALL ambiguous DLS deliveries
    (`dls_ambiguous == True`), as well as any rows lacking valid regular pressure scores.
    """
    bat_id_col = 'canonical_striker_id' if 'canonical_striker_id' in df.columns else 'striker_id'
    bowl_id_col = 'canonical_bowler_id' if 'canonical_bowler_id' in df.columns else 'bowler_id'

    reg_valid = df[
        (~df['is_super_over'].fillna(False))
        & (~df['dls_ambiguous'].fillna(False))
        & df['pressure_bat'].notna()
        & df['pressure_bowl'].notna()
    ].copy()

    bat_df = reg_valid[reg_valid['is_batter_ball_faced']].copy()
    bat_df['_ge60'] = bat_df['pressure_bat'] >= 60.0
    bat_df['_ge70'] = bat_df['pressure_bat'] >= 70.0
    bat_df['_ipl'] = bat_df['is_ipl'].fillna(False)
    bat_df['_ipl_ge60'] = bat_df['_ge60'] & bat_df['_ipl']
    bat_df['_ipl_ge70'] = bat_df['_ge70'] & bat_df['_ipl']

    bat_opp = bat_df.groupby(bat_id_col, as_index=False).agg(
        total_balls=('pressure_bat', 'count'),
        ipl_balls=('_ipl', 'sum'),
        balls_ge60=('_ge60', 'sum'),
        balls_ge70=('_ge70', 'sum'),
        ipl_ge60=('_ipl_ge60', 'sum'),
        ipl_ge70=('_ipl_ge70', 'sum'),
    )

    bowl_df = reg_valid[reg_valid['is_legal_ball']].copy()
    bowl_df['_ge60'] = bowl_df['pressure_bowl'] >= 60.0
    bowl_df['_ge70'] = bowl_df['pressure_bowl'] >= 70.0
    bowl_df['_ipl'] = bowl_df['is_ipl'].fillna(False)
    bowl_df['_ipl_ge60'] = bowl_df['_ge60'] & bowl_df['_ipl']
    bowl_df['_ipl_ge70'] = bowl_df['_ge70'] & bowl_df['_ipl']

    bowl_opp = bowl_df.groupby(bowl_id_col, as_index=False).agg(
        total_balls=('pressure_bowl', 'count'),
        ipl_balls=('_ipl', 'sum'),
        balls_ge60=('_ge60', 'sum'),
        balls_ge70=('_ge70', 'sum'),
        ipl_ge60=('_ipl_ge60', 'sum'),
        ipl_ge70=('_ipl_ge70', 'sum'),
    )

    return bat_opp, bowl_opp


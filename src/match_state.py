"""
Match State Reconstruction Engine for Rajasthan Royals SupeRR Selector 2.0.

Reconstructs situational match context BEFORE every delivery:
- Cumulative score & wickets
- Wickets in hand (10 for regular innings, 2 for Super Overs)
- Legal balls bowled & remaining (accounting for wide, Overstep, and 'other' no-balls,
  as well as rain-shortened ball limits in matches.csv)
- Target & Required Run Rate (RRR) for chases (2nd innings & Super Over 2nd innings)
- Current Run Rate (CRR)
- Phase of match (Powerplay, Middle, Death, Super Over)
- Bowler wicket vs field wicket attribution, batter balls faced, bowler runs conceded
"""

import numpy as np
import pandas as pd

BOWLER_WICKET_TYPES = {
    'Bowled',
    'Caught',
    'LBW',
    'Stump',
    'Hit Wicket'
}

# Deliveries that do not count as a legal ball in the over (must be re-bowled):
# - 'wide': wide delivery
# - 'Overstep': front-foot no-ball
# - 'other': non-overstep no-ball (e.g., waist-high / fielding restriction)
NON_LEGAL_EXTRAS = {
    'wide',
    'Overstep',
    'other'
}

# Extras charged to the bowler's runs conceded (excluding byes, legbyes, penalty)
BOWLER_CHARGED_EXTRAS = {
    'wide',
    'Overstep',
    'other'
}


def reconstruct_match_situations(
    bbb_df: pd.DataFrame,
    matches_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Takes raw ball_by_ball DataFrame and matches DataFrame, and returns an enriched
    DataFrame with full pre-delivery situational match context.
    """
    df = bbb_df.copy()

    # Total runs on each delivery
    df['ball_total_runs'] = df['runs'] + df['extras1_runs'] + df['extras_2_runs']

    # Legal ball flag (wides, oversteps, and 'other' no-balls must be re-bowled)
    df['is_legal_ball'] = ~df['extras_type'].isin(NON_LEGAL_EXTRAS)

    # Batter ball faced flag (batters face legal balls + no-balls, not wides)
    df['is_batter_ball_faced'] = df['extras_type'] != 'wide'

    # Bowler runs conceded (batter runs + wides/no-ball penalties; excludes byes/legbyes/penalty)
    df['bowler_runs_conceded'] = df['runs'] + np.where(
        df['extras_type'].isin(BOWLER_CHARGED_EXTRAS),
        df['extras1_runs'],
        0
    )

    # Bowler credited wicket flag
    df['is_bowler_wicket'] = df['is_wicket'] & df['dismissal_type'].isin(BOWLER_WICKET_TYPES)

    # Sort strictly by match, innings, overid, over_ball_num, innings_ball_num to preserve causal timeline
    df.sort_values(
        by=['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num'],
        inplace=True
    )
    df.reset_index(drop=True, inplace=True)

    # Cumulative calculations within (match_id, innings)
    grouped = df.groupby(['match_id', 'innings'], group_keys=False)

    df['cum_runs_after'] = grouped['ball_total_runs'].cumsum()
    df['cum_wickets_after'] = grouped['is_wicket'].cumsum()
    df['cum_legal_balls_after'] = grouped['is_legal_ball'].cumsum()

    # Pre-delivery state (strictly BEFORE the current delivery is bowled)
    df['score_before'] = df['cum_runs_after'] - df['ball_total_runs']
    df['wickets_before'] = df['cum_wickets_after'] - df['is_wicket'].astype(int)
    df['legal_balls_before'] = df['cum_legal_balls_after'] - df['is_legal_ball'].astype(int)

    # Wickets in hand BEFORE delivery (2 wickets max in Super Overs, 10 in regular innings)
    max_wickets = np.where(df['innings'] >= 3, 2, 10)
    df['wickets_in_hand_before'] = max_wickets - df['wickets_before']

    # Merge match metadata (note: first_inn_revised_overs and second_inn_revised_overs
    # are stored in BALLS in matches.csv, e.g. 90.0 = 15 overs, 66.0 = 11 overs)
    matches_meta = matches_df[[
        'match_id', 'revised_target', 'first_inn_revised_overs',
        'second_inn_revised_overs', 'knock_out'
    ]].copy()
    df = df.merge(matches_meta, on='match_id', how='left')

    # Vectorized max legal balls per innings
    inn = df['innings']
    rev_1 = df['first_inn_revised_overs']
    rev_2 = df['second_inn_revised_overs']

    df['max_legal_balls'] = np.select(
        [
            inn >= 3,
            (inn == 1) & rev_1.notna(),
            (inn == 2) & rev_2.notna(),
            (inn == 2) & rev_1.notna(),
        ],
        [
            6,
            rev_1.fillna(120).astype(int),
            rev_2.fillna(120).astype(int),
            rev_1.fillna(120).astype(int),
        ],
        default=120
    )

    df['legal_balls_remaining_before'] = (df['max_legal_balls'] - df['legal_balls_before']).clip(lower=0)

    # Current Run Rate (CRR) BEFORE delivery
    df['crr_before'] = np.where(
        df['legal_balls_before'] > 0,
        df['score_before'] / (df['legal_balls_before'] / 6.0),
        0.0
    )

    # Calculate 1st innings totals and generalized Super Over preceding-innings totals
    inn_totals = (
        df.groupby(['match_id', 'innings'])['ball_total_runs']
        .sum()
        .reset_index()
    )

    inn1_totals = (
        inn_totals[inn_totals['innings'] == 1][['match_id', 'ball_total_runs']]
        .rename(columns={'ball_total_runs': 'inn1_total_runs'})
    )
    df = df.merge(inn1_totals, on='match_id', how='left')

    # Generalized Super Over target lookup: even Super Over innings (4, 6, ...) target (innings - 1) + 1
    so_prev_totals = inn_totals[inn_totals['innings'] >= 3].copy()
    so_prev_totals['innings'] = so_prev_totals['innings'] + 1
    so_prev_totals = so_prev_totals.rename(columns={'ball_total_runs': 'so_prev_total_runs'})
    df = df.merge(so_prev_totals[['match_id', 'innings', 'so_prev_total_runs']], on=['match_id', 'innings'], how='left')

    # Vectorized target calculation
    is_even_so = (df['innings'] >= 4) & (df['innings'] % 2 == 0)
    df['target'] = np.select(
        [
            (df['innings'] == 2) & df['revised_target'].notna(),
            (df['innings'] == 2) & df['inn1_total_runs'].notna(),
            is_even_so & df['so_prev_total_runs'].notna(),
        ],
        [
            df['revised_target'],
            df['inn1_total_runs'] + 1,
            df['so_prev_total_runs'] + 1,
        ],
        default=np.nan
    )

    # Chase situational metrics BEFORE delivery (for innings 2 and all even Super Over chases)
    is_chase = (df['innings'] == 2) | is_even_so
    df['runs_needed_before'] = np.where(
        is_chase,
        (df['target'] - df['score_before']).clip(lower=0),
        np.nan
    )

    df['rrr_before'] = np.where(
        is_chase & (df['legal_balls_remaining_before'] > 0),
        df['runs_needed_before'] / (df['legal_balls_remaining_before'] / 6.0),
        np.where(is_chase & (df['runs_needed_before'] > 0), np.inf, np.nan)
    )

    # Match Phase classification (overid is 1-indexed: 1..20)
    # Regular innings: Powerplay (1-6), Middle (7-15), Death (16-20); Super Overs: 'Super Over'
    df['match_phase'] = np.select(
        [
            df['innings'] >= 3,
            df['overid'] <= 6,
            df['overid'] <= 15,
        ],
        [
            'Super Over',
            'Powerplay',
            'Middle',
        ],
        default='Death'
    )

    return df

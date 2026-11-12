"""
Canonical Player Identity Resolution Module for Rajasthan Royals SupeRR Selector 2.0.

Preserves all original player IDs while adding verified canonical_player_id columns.
Never merges players based solely on matching names.

Audit-verified canonical mapping:
- 39444 -> 8638 (Basil Hameed): Confirmed same UAE RHB/Right-arm offbreak player;
  playing_eleven.csv records 8638 in Emerging Asia Cup 2024 matches 37087, 37092, 37095
  while ball_by_ball.csv records 39444.

Explicitly kept SEPARATE:
- 36153 (Syed Haider) vs 8729 (Haider Shah): Distinct names, not merged.
- 7446 vs 6922 (Abdul Samad): Distinct players (India/IPL vs Pakistan).
- 7338 vs 28756 (Haider Ali): Distinct players (Pakistan RHB vs UAE LHB bowler).
- 4722 vs 20172 (Rashid Khan): Distinct players (Afghanistan/GT vs Nepal).
- 39479 (Bashir Ahmad), 3083 (Mohammad Ali), 28978 (Muhammad Farooq):
  Secondary IDs have 0 deliveries and 0 playing XI rows; kept separate.
"""

from typing import Dict
import pandas as pd

# Verified ID alias -> canonical ID mapping (evidence-backed only)
CANONICAL_ID_MAP: Dict[int, int] = {
    39444: 8638,  # Basil Hameed (UAE)
}


def resolve_canonical_id(series: pd.Series) -> pd.Series:
    """
    Maps a Series of player IDs to canonical IDs while preserving NaNs and dtype.
    Original input Series is never modified in place.
    """
    if series.isna().any():
        # Float column (e.g., dismissed_player_id)
        return series.replace(CANONICAL_ID_MAP)
    return series.map(lambda x: CANONICAL_ID_MAP.get(int(x), int(x)))


def apply_canonical_identity(bbb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserves original striker_id, non_striker_id, bowler_id, and dismissed_player_id,
    and appends canonical_*_id columns (plus canonical_striker_name, canonical_bowler_name).
    """
    df = bbb_df.copy()

    df['canonical_striker_id'] = resolve_canonical_id(df['striker_id'])
    df['canonical_non_striker_id'] = resolve_canonical_id(df['non_striker_id'])
    df['canonical_bowler_id'] = resolve_canonical_id(df['bowler_id'])
    df['canonical_dismissed_player_id'] = resolve_canonical_id(df['dismissed_player_id'])

    # Map canonical names directly from canonical IDs (consistent across aliases)
    id_to_name = (
        pd.concat([
            df[['canonical_striker_id', 'striker_name']].rename(
                columns={'canonical_striker_id': 'cid', 'striker_name': 'cname'}
            ),
            df[['canonical_bowler_id', 'bowler_name']].rename(
                columns={'canonical_bowler_id': 'cid', 'bowler_name': 'cname'}
            ),
            df[['canonical_non_striker_id', 'non_striker_name']].rename(
                columns={'canonical_non_striker_id': 'cid', 'non_striker_name': 'cname'}
            ),
        ])
        .drop_duplicates(subset=['cid'])
        .set_index('cid')['cname']
        .to_dict()
    )

    df['canonical_striker_name'] = df['canonical_striker_id'].map(id_to_name)
    df['canonical_bowler_name'] = df['canonical_bowler_id'].map(id_to_name)

    return df


def build_canonical_player_directory(
    bbb_df: pd.DataFrame,
    mapping_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds a canonical player reference directory linking canonical_player_id
    to original player_ids, player_name, and teams represented.
    """
    m = mapping_df.copy()
    m['canonical_player_id'] = resolve_canonical_id(m['player_id'])

    directory = (
        m.groupby('canonical_player_id')
        .agg(
            canonical_name=('player_name', 'first'),
            original_player_ids=('player_id', lambda s: sorted(set(int(x) for x in s))),
            teams=('team_name', lambda s: sorted(set(s))),
            n_competitions=('comp_id', 'nunique'),
        )
        .reset_index()
    )

    directory['is_merged_alias'] = directory['original_player_ids'].apply(lambda x: len(x) > 1)
    return directory

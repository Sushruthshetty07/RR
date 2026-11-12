"""
Data loader and schema verification module for Rajasthan Royals SupeRR Selector 2.0.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_raw_datasets(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all raw CSV datasets without modifying original records.
    Returns a dictionary of DataFrames keyed by dataset name.
    """
    base_dir = data_dir or DATA_RAW
    datasets = {
        "competitions": pd.read_csv(base_dir / "competitions.csv"),
        "teams": pd.read_csv(base_dir / "teams.csv"),
        "matches": pd.read_csv(base_dir / "matches.csv"),
        "playing_eleven": pd.read_csv(base_dir / "playing_eleven.csv"),
        "player_mapping": pd.read_csv(base_dir / "competition_team_player_mapping.csv"),
        "ball_by_ball": pd.read_csv(base_dir / "ball_by_ball.csv", low_memory=False),
    }
    return datasets


def get_dataset_summary(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generates summary table with shapes, memory usage, and key info.
    """
    records = []
    for name, df in datasets.items():
        records.append({
            "Dataset": name,
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Memory (MB)": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "Null Values": df.isnull().sum().sum(),
        })
    return pd.DataFrame(records)

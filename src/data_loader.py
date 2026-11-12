"""
Data loader and schema verification module for Rajasthan Royals SupeRR Selector 2.0.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_raw_datasets(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all raw CSV datasets without modifying original records.
    Returns a dictionary of DataFrames keyed by dataset name.
    Raises FileNotFoundError with a clear message if any required file is missing.
    """
    base_dir = data_dir or DATA_RAW
    required_files = {
        "competitions": "competitions.csv",
        "teams": "teams.csv",
        "matches": "matches.csv",
        "playing_eleven": "playing_eleven.csv",
        "player_mapping": "competition_team_player_mapping.csv",
        "ball_by_ball": "ball_by_ball.csv",
    }

    datasets = {}
    missing = []
    for key, filename in required_files.items():
        filepath = base_dir / filename
        if not filepath.exists():
            missing.append(f"{filename} (expected at {filepath})")
        else:
            datasets[key] = pd.read_csv(filepath, low_memory=False if key == "ball_by_ball" else True)

    if missing:
        raise FileNotFoundError(
            f"Missing required raw competition dataset(s):\n  - " + "\n  - ".join(missing)
            + "\nPlease ensure all official competition CSV files are placed in data/raw/."
        )

    return datasets


def validate_raw_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Validates schemas, critical foreign keys, and data quality across raw datasets.
    Raises ValueError if required schemas are violated.
    """
    required_schemas = {
        "competitions": ["comp_id", "comp_name"],
        "teams": ["team_id", "team_name"],
        "matches": ["match_id", "comp_id", "match_date"],
        "playing_eleven": ["match_id", "team_id", "player_id"],
        "player_mapping": ["comp_id", "team_id", "player_id"],
        "ball_by_ball": ["match_id", "innings", "overid", "over_ball_num", "striker_id", "bowler_id", "runs"],
    }

    validation_results = {
        "passed": True,
        "dataset_checks": {},
        "integrity_checks": {},
    }

    for name, cols in required_schemas.items():
        if name not in datasets:
            validation_results["passed"] = False
            raise ValueError(f"Dataset '{name}' is missing from loaded datasets.")
        df = datasets[name]
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            validation_results["passed"] = False
            raise ValueError(f"Dataset '{name}' is missing required columns: {missing_cols}")
        if len(df) == 0:
            validation_results["passed"] = False
            raise ValueError(f"Dataset '{name}' is empty (0 rows).")

        validation_results["dataset_checks"][name] = {
            "row_count": len(df),
            "col_count": len(df.columns),
            "missing_required_cols": len(missing_cols),
        }

    # Integrity check: ball_by_ball matches match matches.csv
    bbb_matches = set(datasets["ball_by_ball"]["match_id"].unique())
    matches_ids = set(datasets["matches"]["match_id"].unique())
    unmatched_bbb = bbb_matches - matches_ids
    validation_results["integrity_checks"]["bbb_matches_in_matches_table"] = len(unmatched_bbb) == 0
    if unmatched_bbb:
        validation_results["passed"] = False
        raise ValueError(f"Found {len(unmatched_bbb)} match_ids in ball_by_ball that do not exist in matches.csv.")

    return validation_results


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

"""
Unified End-to-End Analytical Pipeline Runner for Rajasthan Royals SupeRR Selector 2.0.

Executes the complete reproducible data science pipeline in a single command:
1. Data Ingestion & Schema Verification
2. Ball-by-Ball Match State Reconstruction
3. Continuous Pressure Index Computation (4 Regimes + LOO Baselines)
4. Empirical Bayes Performance Modeling & Kish Effective Sample Sizes
5. Multi-Model Win Probability & Leverage Attribution (Logit, RF, GBDT)
6. Multi-Scenario Weight Sensitivity & Pareto Robustness Analysis
7. Final Validation & Dashboard Summary Generation

Usage:
  python scripts/run_full_pipeline.py [--all] [--step <step_name>]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader import load_raw_datasets, get_dataset_summary
from src.match_state import reconstruct_match_situations
from src.player_identity import apply_canonical_identity, build_canonical_player_directory
from src.pressure import compute_pressure_scores
from src.performance import (
    PerformanceConfig,
    evaluate_batting_pressure,
    evaluate_bowling_pressure,
    fit_predict_oof_baselines,
)
from src.win_probability import run_win_probability_benchmark
from src.weight_sensitivity import run_weight_sensitivity_analysis
from src.player_evaluation import evaluate_separated_dimensions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PipelineRunner")

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def step_data_verification():
    logger.info("=== STEP 1: Data Ingestion & Schema Verification ===")
    datasets = load_raw_datasets(DATA_RAW)
    summary = get_dataset_summary(datasets)
    logger.info(f"Loaded {len(datasets)} datasets successfully.")
    for _, row in summary.iterrows():
        logger.info(f"  {row['Dataset']}: {row['Rows']:,} rows, {row['Columns']} cols, {row['Memory (MB)']} MB")
    return datasets


def step_win_probability_and_leverage(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame):
    logger.info("=== STEP 2: Win Probability & Leverage Modeling ===")
    t0 = time.time()
    results = run_win_probability_benchmark(deliveries_df, matches_df)
    logger.info(f"WP benchmark completed in {time.time() - t0:.1f}s.")
    logger.info("  Model Comparison (ROC-AUC / Brier Score):")
    for m_name, m_metrics in results.get("overall_metrics", {}).items():
        logger.info(f"    - {m_name}: ROC-AUC={m_metrics['roc_auc']:.4f}, Brier={m_metrics['brier_score']:.4f}, CalErr={m_metrics['mean_calibration_error']:.4f}")

    logger.info(f"  Correlation PI vs Leverage: {results.get('correlation_pi_vs_leverage')}")
    out_path = DATA_PROCESSED / "wp_benchmark_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")
    return results


def step_weight_sensitivity(bat_df: pd.DataFrame, bowl_df: pd.DataFrame):
    logger.info("=== STEP 3: Weight Sensitivity & Pareto Robustness ===")
    t0 = time.time()
    results = run_weight_sensitivity_analysis(bat_df, bowl_df)
    logger.info(f"Weight sensitivity analysis completed in {time.time() - t0:.1f}s.")
    logger.info("  Batting Scenario Correlations with Baseline:")
    for sc, rho in results["batting"]["correlation_matrix"]["Baseline"].items():
        logger.info(f"    - {sc}: Spearman rho = {rho:.4f}")

    out_path = DATA_PROCESSED / "weight_sensitivity_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")
    return results


def step_player_evaluation_dimensions(bat_df: pd.DataFrame, bowl_df: pd.DataFrame):
    logger.info("=== STEP 4: Three Separated Evaluation Dimensions ===")
    t0 = time.time()
    results = evaluate_separated_dimensions(bat_df, bowl_df)
    logger.info(f"Separated evaluation dimensions generated in {time.time() - t0:.1f}s.")
    out_path = DATA_PROCESSED / "player_evaluation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")
    return results


def run_full_pipeline():
    logger.info("=================================================================")
    logger.info("  RAJASTHAN ROYALS SUPERR SELECTOR 2.0 — FULL REPRODUCIBLE PIPELINE")
    logger.info("=================================================================")
    start_time = time.time()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Verification
    datasets = step_data_verification()

    # Load preprocessed delivery records if available, or reconstruct
    deliv_path = DATA_PROCESSED / "deliveries_with_pressure.csv.gz"
    if deliv_path.exists():
        logger.info(f"Loading existing processed deliveries from {deliv_path}...")
        deliv = pd.read_csv(deliv_path)
    else:
        logger.info("Reconstructing match situations and pressure scores...")
        state_df = reconstruct_match_situations(datasets["ball_by_ball"], datasets["matches"])
        canon_df = apply_canonical_identity(state_df)
        deliv = compute_pressure_scores(canon_df, datasets["matches"], datasets["competitions"])
        deliv.to_csv(deliv_path, index=False, compression="gzip")
        logger.info(f"Reconstructed and saved {len(deliv):,} deliveries to {deliv_path}.")

    # Load batting and bowling preliminary data
    bat_path = DATA_PROCESSED / "final_bat_enriched_delta.csv"
    if not bat_path.exists():
        bat_path = DATA_PROCESSED / "phase3_batting_prelim.csv"
    bowl_path = DATA_PROCESSED / "final_bowl_enriched_delta.csv"
    if not bowl_path.exists():
        bowl_path = DATA_PROCESSED / "phase3_bowling_prelim.csv"

    bat_df = pd.read_csv(bat_path)
    bowl_df = pd.read_csv(bowl_path)

    # 2. Win Probability & Leverage
    wp_results = step_win_probability_and_leverage(deliv, datasets["matches"])

    # 3. Weight Sensitivity
    ws_results = step_weight_sensitivity(bat_df, bowl_df)

    # 4. Evaluation Dimensions
    pe_results = step_player_evaluation_dimensions(bat_df, bowl_df)

    total_time = time.time() - start_time
    logger.info("=================================================================")
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f} SECONDS")
    logger.info("Dashboard-ready files generated in data/processed/:")
    logger.info("  - wp_benchmark_summary.json")
    logger.info("  - weight_sensitivity_summary.json")
    logger.info("  - player_evaluation_summary.json")
    logger.info("  - wp_bat_wpa_leaders.csv")
    logger.info("  - wp_bowl_wpa_leaders.csv")
    logger.info("To launch the web dashboard: python scripts/run_dashboard.py")
    logger.info("=================================================================")


def main():
    parser = argparse.ArgumentParser(description="Run RR SupeRR Selector 2.0 Pipeline")
    parser.add_argument("--all", action="store_true", default=True, help="Run complete pipeline")
    parser.add_argument("--step", type=str, choices=["verify", "wp", "sensitivity", "eval"], help="Run specific step")
    args = parser.parse_args()

    if args.step:
        datasets = step_data_verification()
        if args.step == "verify":
            return
        matches = datasets["matches"]
        deliv = pd.read_csv(DATA_PROCESSED / "deliveries_with_pressure.csv.gz")
        bat_df = pd.read_csv(DATA_PROCESSED / "final_bat_enriched_delta.csv")
        bowl_df = pd.read_csv(DATA_PROCESSED / "final_bowl_enriched_delta.csv")
        if args.step == "wp":
            step_win_probability_and_leverage(deliv, matches)
        elif args.step == "sensitivity":
            step_weight_sensitivity(bat_df, bowl_df)
        elif args.step == "eval":
            step_player_evaluation_dimensions(bat_df, bowl_df)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()

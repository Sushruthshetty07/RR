"""
Unified End-to-End Analytical Pipeline Runner for Rajasthan Royals SupeRR Selector 2.0.

Executes the complete reproducible raw-to-final data science pipeline in a single command:
1. Data Ingestion & Schema Verification (data_loader.py)
2. Ball-by-Ball Pre-Delivery Match State Reconstruction (match_state.py)
3. Canonical Player Identity Resolution (player_identity.py)
4. Continuous 4-Regime Pressure Index Engine (pressure.py)
5. Zero-Leakage 5-Fold GroupKFold Expected Baselines (performance.py)
6. Empirical Bayes Shrinkage & Kish Effective Sample Sizes (performance.py)
7. Enriched Context & Pressure Delta Modeling (final_validation.py)
8. Multi-Model Win Probability & Counterfactual Leverage Attribution (win_probability.py)
9. Multi-Scenario Weight Sensitivity & Pareto Robustness Analysis (weight_sensitivity.py)
10. Official Competition Top 10 Deliverables & Dashboard Dataset Export (player_evaluation.py)

Usage:
  python scripts/run_full_pipeline.py [--force] [--step <step_name>]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader import load_raw_datasets, validate_raw_datasets, get_dataset_summary
from src.match_state import reconstruct_match_situations
from src.player_identity import apply_canonical_identity, build_canonical_player_directory
from src.pressure import compute_pressure_scores
from src.performance import (
    PerformanceConfig,
    evaluate_batting_pressure,
    evaluate_bowling_pressure,
    fit_predict_oof_baselines,
)
from src.final_validation import audit_enriched_roe_and_pressure_delta
from src.win_probability import run_win_probability_benchmark
from src.weight_sensitivity import run_weight_sensitivity_analysis
from src.player_evaluation import evaluate_separated_dimensions, export_official_rankings_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PipelineRunner")

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def step_1_data_ingestion_and_validation(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Step 1: Ingests raw datasets and verifies schemas & integrity."""
    logger.info("=== STEP 1/10: Raw Data Ingestion & Schema Verification ===")
    t0 = time.time()
    datasets = load_raw_datasets(data_dir or DATA_RAW)
    val_report = validate_raw_datasets(datasets)
    summary = get_dataset_summary(datasets)
    logger.info(f"Loaded {len(datasets)} raw datasets in {time.time() - t0:.1f}s. All schema checks passed.")
    for _, row in summary.iterrows():
        logger.info(f"  {row['Dataset']}: {row['Rows']:,} rows, {row['Columns']} cols, {row['Memory (MB)']} MB")
    return datasets


def step_2_and_3_match_state_and_identity(
    datasets: Dict[str, pd.DataFrame], force: bool = False
) -> pd.DataFrame:
    """Step 2 & 3: Causal match state reconstruction & canonical identity resolution."""
    deliv_path = DATA_PROCESSED / "deliveries_with_pressure.csv.gz"
    canon_dir_path = DATA_PROCESSED / "canonical_player_directory.csv"

    # Always ensure canonical player directory exists
    if not canon_dir_path.exists() or force:
        logger.info("Building canonical player directory...")
        player_dir = build_canonical_player_directory(
            datasets["ball_by_ball"],
            datasets["player_mapping"],
        )
        player_dir.to_csv(canon_dir_path, index=False)
        logger.info(f"Saved canonical player directory ({len(player_dir)} players) to {canon_dir_path}.")
    else:
        player_dir = pd.read_csv(canon_dir_path)

    if deliv_path.exists() and not force:
        logger.info(f"Loading existing processed deliveries from {deliv_path}...")
        return pd.read_csv(deliv_path, low_memory=False)

    logger.info("=== STEP 2/10: Causal Pre-Delivery Match State Reconstruction ===")
    t0 = time.time()
    state_df = reconstruct_match_situations(datasets["ball_by_ball"], datasets["matches"])
    logger.info(f"Reconstructed {len(state_df):,} delivery states in {time.time() - t0:.1f}s.")

    logger.info("=== STEP 3/10: Canonical Identity Mapping ===")
    canon_df = apply_canonical_identity(state_df)

    logger.info("=== STEP 4/10: 4-Regime Pressure Index Engine ===")
    t1 = time.time()
    deliv = compute_pressure_scores(canon_df, datasets["matches"], datasets["competitions"])
    deliv.to_csv(deliv_path, index=False, compression="gzip")
    logger.info(f"Computed pressure features and saved to {deliv_path} in {time.time() - t1:.1f}s.")
    return deliv


def step_5_and_6_baselines_and_performance(
    deliv: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
    cfg: PerformanceConfig,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Steps 5 & 6: 5-Fold GroupKFold OOF baselines, EB shrinkage, and player aggregation."""
    bat_prelim_path = DATA_PROCESSED / "phase3_batting_prelim.csv"
    bowl_prelim_path = DATA_PROCESSED / "phase3_bowling_prelim.csv"
    bat_enr_path = DATA_PROCESSED / "final_bat_enriched_delta.csv"
    bowl_enr_path = DATA_PROCESSED / "final_bowl_enriched_delta.csv"

    if bat_enr_path.exists() and bowl_enr_path.exists() and not force:
        logger.info("Loading existing enriched performance tables...")
        bat_df = pd.read_csv(bat_enr_path)
        bowl_df = pd.read_csv(bowl_enr_path)
        return bat_df, bowl_df

    logger.info("=== STEP 5/10: Fitting Zero-Leakage OOF Baselines (5-Fold GroupKFold by Match) ===")
    t0 = time.time()
    oof_pri, diag = fit_predict_oof_baselines(
        deliv,
        datasets["matches"],
        datasets["competitions"],
        include_suspected_dls=False,
        n_splits=cfg.n_oof_folds,
        exclude_emerging=True,
    )
    logger.info(f"OOF baselines fitted across {diag['n_pool_deliveries']:,} deliveries in {time.time() - t0:.1f}s.")

    logger.info("=== STEP 6/10: Empirical Bayes Performance Modeling & Kish Neff ===")
    t1 = time.time()
    player_dir_path = DATA_PROCESSED / "canonical_player_directory.csv"
    player_dir = pd.read_csv(player_dir_path) if player_dir_path.exists() else None

    bat_df, bat_meta = evaluate_batting_pressure(oof_pri, cfg, player_dir=player_dir)
    bowl_df, bowl_meta = evaluate_bowling_pressure(oof_pri, cfg, player_dir=player_dir)

    bat_df.sort_values("canonical_player_id").to_csv(bat_prelim_path, index=False)
    bowl_df.sort_values("canonical_player_id").to_csv(bowl_prelim_path, index=False)
    logger.info(f"Evaluated {len(bat_df)} batters and {len(bowl_df)} bowlers in {time.time() - t1:.1f}s.")

    logger.info("=== STEP 7/10: Enriched RoE & Pressure Delta Context Modeling ===")
    t2 = time.time()
    roe_res = audit_enriched_roe_and_pressure_delta(
        oof_pri, datasets["matches"], datasets["competitions"], bat_df, bowl_df, cfg
    )
    roe_res["bat_enriched_delta"].to_csv(bat_enr_path, index=False)
    roe_res["bowl_enriched_delta"].to_csv(bowl_enr_path, index=False)
    logger.info(f"Generated Enriched RoE and saved to {bat_enr_path} in {time.time() - t2:.1f}s.")

    return roe_res["bat_enriched_delta"], roe_res["bowl_enriched_delta"]


def step_8_win_probability_and_leverage(deliv: pd.DataFrame, matches: pd.DataFrame) -> Dict[str, Any]:
    """Step 8: Win Probability benchmark & counterfactual Leverage/WPA attribution."""
    logger.info("=== STEP 8/10: Win Probability Modeling & Leverage Attribution ===")
    t0 = time.time()
    wp_results = run_win_probability_benchmark(deliv, matches)
    out_path = DATA_PROCESSED / "wp_benchmark_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wp_results, f, indent=2)
    logger.info(f"WP benchmark completed and saved to {out_path} in {time.time() - t0:.1f}s.")
    return wp_results


def step_9_weight_sensitivity(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> Dict[str, Any]:
    """Step 9: 4-Scenario weight sensitivity and Pareto frontier analysis."""
    logger.info("=== STEP 9/10: Weight Sensitivity & Pareto Robustness Analysis ===")
    t0 = time.time()
    ws_results = run_weight_sensitivity_analysis(bat_df, bowl_df)
    out_path = DATA_PROCESSED / "weight_sensitivity_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ws_results, f, indent=2)
    logger.info(f"Weight sensitivity completed and saved to {out_path} in {time.time() - t0:.1f}s.")
    return ws_results


def step_10_player_evaluation_and_official_deliverables(
    bat_df: pd.DataFrame, bowl_df: pd.DataFrame
) -> Dict[str, Any]:
    """Step 10: Player evaluation dimensions and official Top 10 CSV exports."""
    logger.info("=== STEP 10/10: Player Evaluation Dimensions & Official Deliverables Export ===")
    t0 = time.time()
    pe_results = evaluate_separated_dimensions(bat_df, bowl_df)

    pe_path = DATA_PROCESSED / "player_evaluation_summary.json"
    with open(pe_path, "w", encoding="utf-8") as f:
        json.dump(pe_results, f, indent=2)

    # Export official CSV deliverables
    exported_csvs = export_official_rankings_csv(pe_results, DATA_PROCESSED)
    logger.info(f"Player evaluation completed and saved to {pe_path} in {time.time() - t0:.1f}s.")
    for name, path in exported_csvs.items():
        logger.info(f"  Exported deliverable [{name}]: {path}")

    return pe_results


def step_11_cricket_intelligence() -> Dict[str, Any]:
    """Step 11: Strategic cricket intelligence, matchup splits, and behavioral composure proxies."""
    logger.info("=== STEP 11/11: Strategic Cricket Intelligence & Composure Proxies ===")
    t0 = time.time()
    from src.cricket_intelligence import generate_cricket_intelligence_summary
    ci_results = generate_cricket_intelligence_summary()
    out_path = DATA_PROCESSED / "cricket_intelligence_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ci_results, f, indent=2)
    logger.info(f"Cricket intelligence completed and saved to {out_path} in {time.time() - t0:.1f}s.")
    return ci_results


def run_full_pipeline(force: bool = False) -> Dict[str, Any]:
    """Runs the complete, reproducible end-to-end pipeline from raw data to final deliverables."""
    logger.info("=================================================================")
    logger.info("  RAJASTHAN ROYALS SUPERR SELECTOR 2.0 — FULL REPRODUCIBLE PIPELINE")
    logger.info("=================================================================")
    start_time = time.time()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cfg = PerformanceConfig(w_t20i=0.50)

    # 1. Ingestion & Validation
    datasets = step_1_data_ingestion_and_validation()

    # 2-4. Match State, Canonical Identity & Pressure Scores
    deliv = step_2_and_3_match_state_and_identity(datasets, force=force)

    # 5-7. Baselines, Performance Evaluation & Enriched RoE
    bat_df, bowl_df = step_5_and_6_baselines_and_performance(deliv, datasets, cfg, force=force)

    # 8. Win Probability & Leverage Attribution
    wp_results = step_8_win_probability_and_leverage(deliv, datasets["matches"])

    # 9. Weight Sensitivity & Pareto Robustness
    ws_results = step_9_weight_sensitivity(bat_df, bowl_df)

    # 10. Separated Dimensions & Official Deliverables Export
    pe_results = step_10_player_evaluation_and_official_deliverables(bat_df, bowl_df)

    # 11. Strategic Cricket Intelligence & Composure Proxies
    ci_results = step_11_cricket_intelligence()

    total_time = time.time() - start_time
    logger.info("=================================================================")
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f} SECONDS")
    logger.info("Official Competition Deliverables Generated in data/processed/:")
    logger.info("  - official_top10_high_pressure_batters.csv (PRIMARY DELIVERABLE)")
    logger.info("  - official_top10_high_pressure_bowlers.csv (PRIMARY DELIVERABLE)")
    logger.info("  - official_top10_overall_reliability_batters.csv (SECONDARY)")
    logger.info("  - official_top10_overall_reliability_bowlers.csv (SECONDARY)")
    logger.info("  - player_evaluation_summary.json")
    logger.info("  - wp_benchmark_summary.json")
    logger.info("  - weight_sensitivity_summary.json")
    logger.info("  - cricket_intelligence_summary.json")
    logger.info("=================================================================")

    # Write pipeline execution summary
    exec_summary = {
        "status": "success",
        "total_runtime_seconds": round(total_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_deliveries": len(datasets["ball_by_ball"]),
        "processed_deliveries": len(deliv),
        "primary_eligible_batters": int(bat_df["elig_primary"].sum()),
        "primary_eligible_bowlers": int(bowl_df["elig_primary"].sum()),
        "top10_high_pressure_batters": [r["player_name"] for r in pe_results["batting"]["top10_high_pressure_robust"]],
        "top10_high_pressure_bowlers": [r["player_name"] for r in pe_results["bowling"]["top10_high_pressure_robust"]],
        "top10_overall_reliability_batters": [r["player_name"] for r in pe_results["batting"]["top10_overall_reliability"]],
        "top10_overall_reliability_bowlers": [r["player_name"] for r in pe_results["bowling"]["top10_overall_reliability"]],
    }
    with open(DATA_PROCESSED / "pipeline_execution_summary.json", "w", encoding="utf-8") as f:
        json.dump(exec_summary, f, indent=2)

    return exec_summary


def main():
    parser = argparse.ArgumentParser(description="Run RR SupeRR Selector 2.0 Pipeline")
    parser.add_argument("--all", action="store_true", default=True, help="Run complete pipeline")
    parser.add_argument("--force", action="store_true", default=False, help="Force recomputation of all steps")
    parser.add_argument(
        "--step",
        type=str,
        choices=["verify", "match_state", "performance", "wp", "sensitivity", "eval"],
        help="Run specific step",
    )
    args = parser.parse_args()

    if args.step:
        datasets = step_1_data_ingestion_and_validation()
        if args.step == "verify":
            return
        deliv = step_2_and_3_match_state_and_identity(datasets, force=args.force)
        cfg = PerformanceConfig(w_t20i=0.50)
        bat_df, bowl_df = step_5_and_6_baselines_and_performance(deliv, datasets, cfg, force=args.force)
        if args.step == "performance":
            return
        elif args.step == "wp":
            step_8_win_probability_and_leverage(deliv, datasets["matches"])
        elif args.step == "sensitivity":
            step_9_weight_sensitivity(bat_df, bowl_df)
        elif args.step == "eval":
            step_10_player_evaluation_and_official_deliverables(bat_df, bowl_df)
    else:
        run_full_pipeline(force=args.force)


if __name__ == "__main__":
    main()

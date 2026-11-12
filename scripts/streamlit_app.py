"""
Rajasthan Royals SupeRR Selector 2.0 — Streamlit Companion Dashboard.

Provides an interactive Streamlit UI matching the 7 core analytical pages:
1. Overview
2. Batter Scouting
3. Bowler Scouting
4. Player Comparison
5. Pressure and Leverage Analysis
6. RR Auction Intelligence
7. Model Evaluation

Usage:
  streamlit run scripts/streamlit_app.py
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

# Streamlit is imported conditionally so running without streamlit installed does not crash the CLI
try:
    import streamlit as st
except ImportError:
    st = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"


def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    if st is None:
        print("Streamlit is not installed in the current environment. To use this interface, run: pip install streamlit")
        return

    st.set_page_config(
        page_title="RR SupeRR Selector 2.0",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏏 Rajasthan Royals SupeRR Selector 2.0")
    st.caption("IPL Player Reliability, High-Pressure Clutch Evaluation & Auction Intelligence | 200,999 Balls | 867 Matches")

    # Load core datasets
    bat_df = pd.read_csv(DATA_PROCESSED / "final_bat_enriched_delta.csv") if (DATA_PROCESSED / "final_bat_enriched_delta.csv").exists() else pd.DataFrame()
    bowl_df = pd.read_csv(DATA_PROCESSED / "final_bowl_enriched_delta.csv") if (DATA_PROCESSED / "final_bowl_enriched_delta.csv").exists() else pd.DataFrame()
    wp_summary = load_json(DATA_PROCESSED / "wp_benchmark_summary.json")
    ws_summary = load_json(DATA_PROCESSED / "weight_sensitivity_summary.json")
    pe_summary = load_json(DATA_PROCESSED / "player_evaluation_summary.json")

    # Sidebar Navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Batter Scouting",
            "3. Bowler Scouting",
            "4. Player Comparison",
            "5. Pressure & Leverage",
            "6. RR Auction Intelligence",
            "7. Model Evaluation",
        ],
    )

    if page == "1. Overview":
        st.subheader("1. Executive Summary & Analytical Architecture")
        st.info(
            "Across **200,999 deliveries** (867 matches), we evaluate T20 players across three strictly separated dimensions: "
            "**(A) Overall Reliability**, **(B) High-Pressure Performance**, and **(C) Rajasthan Royals Tactical Fit**. "
            "We eliminate the conflation between explosive Powerplay pace-setting and late-innings clutch chasing, enforcing empirical Bayes shrinkage and Kish effective sample size gates."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Primary Deliveries", "193,144", "IPL (1.0) + T20I (0.5)")
        col2.metric("Eligible Universe", f"{len(bat_df[bat_df['elig_primary']])} Bat / {len(bowl_df[bowl_df['elig_primary']])} Bowl", "Neff >= 30")
        col3.metric("RR 2026 Baseline", "8-6 League (3rd)", "Guwahati 3/3 | Jaipur 1/4")
        col4.metric("WP vs PI Corr", "r = -0.20", "Pressure != Leverage")

        st.markdown("### Three Separated Evaluation Dimensions")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Dimension A: Overall Reliability")
            st.write("Evaluates volume & consistency across all phases using Kish effective sample size (Neff >= 30) and Bayesian LCB composite score (PRI_lcb).")
        with c2:
            st.markdown("#### Dimension B: High-Pressure Performance")
            st.write("Evaluates performance strictly on deliveries with PI >= 60. Imposes a strict evidence threshold (Neff_HP >= 30 for robust ranking). Displays 'Insufficient Evidence' for Neff_HP < 15.")
        with c3:
            st.markdown("#### Dimension C: RR Tactical Fit")
            st.write("Evaluates player compatibility against RR's specific tactical voids (Death Pace, Attacking Spin, RHB Finisher) and BCCI auction eligibility buckets.")

    elif page == "2. Batter Scouting":
        st.subheader("2. Batter Scouting: Overall Reliability vs High-Pressure Clutch")

        bat_elig = bat_df[bat_df["elig_primary"]].copy()

        view_mode = st.radio("Select View", ["Separated Leaderboards (Overall vs High Pressure)", "Complete Eligible Universe"], horizontal=True)

        if view_mode == "Separated Leaderboards (Overall vs High Pressure)":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### PRIMARY: Top 10 High-Pressure Batters (Neff_HP >= 30, Sorted by LCB)")
                robust_hp = bat_elig[bat_elig["neff_hp"] >= 30].sort_values("hp_raa_lcb", ascending=False).head(10)
                st.dataframe(robust_hp[["player_name", "hp_raa_lcb", "hp_raa_post", "neff_hp", "pri_score_lcb"]].reset_index(drop=True))

            with c2:
                st.markdown("#### SECONDARY: Top 10 Overall Reliable Batters (PRI_lcb)")
                top_ovr = bat_elig.sort_values("pri_score_lcb", ascending=False).head(10)
                st.dataframe(top_ovr[["player_name", "pri_score_lcb", "neff", "neff_hp"]].reset_index(drop=True))

            st.warning(
                "**Powerplay Pace-Setters (Zero HP Exposure):** Players like Vaibhav Sooryavanshi (#2 Overall) and Finn Allen (#5 Overall) "
                "earned elite PRI_lcb rankings from overs 1–6 destruction, but faced 0 high-pressure balls (Neff_HP = 0.0). They should never be misidentified as death closers."
            )
        else:
            search = st.text_input("Search Batter", "")
            if search:
                bat_elig = bat_elig[bat_elig["player_name"].str.contains(search, case=False, na=False)]
            st.dataframe(bat_elig[["player_name", "hp_raa_lcb", "hp_raa_post", "neff_hp", "pri_score_lcb", "neff"]].reset_index(drop=True))

    elif page == "3. Bowler Scouting":
        st.subheader("3. Bowler Scouting: High-Pressure Clutch vs Overall Reliability")
        bowl_elig = bowl_df[bowl_df["elig_primary"]].copy()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### PRIMARY: Top 10 High-Pressure Bowlers (Neff_HP >= 30, Sorted by LCB)")
            robust_hp_b = bowl_elig[bowl_elig["neff_hp"] >= 30].sort_values("hp_impact_lcb", ascending=False).head(10)
            st.dataframe(robust_hp_b[["player_name", "hp_impact_lcb", "hp_impact_post", "neff_hp", "pri_score_lcb"]].reset_index(drop=True))

        with c2:
            st.markdown("#### SECONDARY: Top 10 Overall Reliable Bowlers (PRI_lcb)")
            top_ovr_b = bowl_elig.sort_values("pri_score_lcb", ascending=False).head(10)
            st.dataframe(top_ovr_b[["player_name", "pri_score_lcb", "neff", "neff_hp"]].reset_index(drop=True))

    elif page == "4. Player Comparison":
        st.subheader("4. Side-by-Side Multi-Player Comparison")
        cids_input = st.text_input("Enter Canonical Player IDs (comma-separated)", "1699, 6396, 2933, 7920")
        st.caption("E.g., 1699 (Suryakumar), 6396 (Tim David), 2933 (Bumrah), 7920 (Ellis)")

    elif page == "5. Pressure & Leverage":
        st.subheader("5. Pressure and Leverage Analysis")
        st.info(
            "Pressure measures situational difficulty (e.g. high RRR), while Leverage measures win probability swing magnitude. "
            "Their correlation is slightly negative (r = -0.20) because hopeless blowouts feature high pressure but near-zero leverage."
        )
        if wp_summary:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Top 10 Batters by Win Probability Added (WPA)")
                st.dataframe(pd.DataFrame(wp_summary.get("top_bat_wpa", []))[["player_name", "total_wpa", "wpa_per_100", "hl_balls"]])
            with c2:
                st.markdown("#### Top 10 Bowlers by Win Probability Added (WPA)")
                st.dataframe(pd.DataFrame(wp_summary.get("top_bowl_wpa", []))[["player_name", "total_wpa", "wpa_per_100", "hl_balls"]])

    elif page == "6. RR Auction Intelligence":
        st.subheader("6. Rajasthan Royals Auction Intelligence & Target Shortlist")
        st.markdown(
            "**RR 2026 Audit:** RR finished 8-6 in the league and 3rd in the playoffs. RR was undefeated in Guwahati (3/3), "
            "but leaked 10.13 RPO in Jaipur (1/4 wins), exposing severe death-pace vulnerability."
        )

        st.markdown("### Top Tactical Voids & Recommended Targets")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.error("**Gap 1: Death Pace Specialist**\n\n*Targets:* Nathan Ellis (Trade), Josh Hazlewood (Trade), Sakib Hussain (Sleeper)")
        with col2:
            st.warning("**Gap 2: Attacking / Containment Spin**\n\n*Targets:* Wanindu Hasaranga (Pending), Adil Rashid (Auction), Mitchell Santner (Trade)")
        with col3:
            st.info("**Gap 3: Middle-Order Closer / Finisher**\n\n*Targets:* Tim David (Trade), Shashank Singh (Trade), Rajat Patidar (Trade)")

    elif page == "7. Model Evaluation":
        st.subheader("7. Machine Learning Pipeline & Statistical Validation")

        if wp_summary and "overall_metrics" in wp_summary:
            st.markdown("#### Win Probability Model Benchmark (5-Fold GroupKFold by Match)")
            st.dataframe(pd.DataFrame(wp_summary["overall_metrics"]).T)

        if ws_summary and "batting" in ws_summary:
            st.markdown("#### Weight Sensitivity: Pairwise Spearman Rank Correlations")
            st.dataframe(pd.DataFrame(ws_summary["batting"]["correlation_matrix"]))


if __name__ == "__main__":
    main()

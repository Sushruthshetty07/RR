"""
Phase 2 Execution Runner:
- Applies canonical player identity mapping (preserving original IDs)
- Reconstructs pre-delivery match states & computes 0-100 continuous Pressure Index
- Audits DLS status counts, LOO support counts, and Super Over counts
- Analyzes pressure distributions across all 4 regular regimes
- Analyzes player pressure opportunity distributions (IPL vs T20I)
- Performs parameter, knockout-multiplier, and competition-weight sensitivity analyses
- Saves Phase 2 visualizations (07_ to 12_) and processed artifacts
(Does NOT evaluate player performance or produce Top 10 rankings.)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_datasets, DATA_PROCESSED
from src.match_state import reconstruct_match_situations
from src.player_identity import apply_canonical_identity, build_canonical_player_directory
from src.pressure import compute_pressure_scores, compute_player_opportunities, PressureConfig

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    print("=== 1. LOADING RAW DATA & RECONSTRUCTING STATE ===")
    ds = load_raw_datasets()
    bbb_id = apply_canonical_identity(ds['ball_by_ball'])
    state_df = reconstruct_match_situations(bbb_id, ds['matches'])

    # Save canonical player directory
    player_dir = build_canonical_player_directory(bbb_id, ds['player_mapping'])
    player_dir.to_csv(DATA_PROCESSED / "canonical_player_directory.csv", index=False)
    print(f"Canonical player directory saved: {len(player_dir)} canonical players "
          f"(merged aliases: {player_dir['is_merged_alias'].sum()})")

    print("\n=== 2. COMPUTING PRIMARY PRESSURE SCORES (knockout_multiplier=1.00) ===")
    cfg_primary = PressureConfig(knockout_multiplier=1.00)
    df = compute_pressure_scores(state_df, ds['matches'], ds['competitions'], cfg_primary)

    # Save processed pressure dataset (without mutating data/raw/)
    out_path = DATA_PROCESSED / "deliveries_with_pressure.csv.gz"
    df.to_csv(out_path, index=False, compression='gzip')
    print(f"Saved processed dataset to {out_path} ({df.shape[0]} rows x {df.shape[1]} cols)")

    # --- AUDIT COUNTS: DLS, LOO SUPPORT, SUPER OVER, MISSING/AMBIGUOUS ---
    print("\n=== 3. MISSING / AMBIGUOUS DATA & SUB-REGIME COUNTS ===")
    print("DLS status counts (all deliveries):")
    print(df['dls_status'].value_counts().to_string())
    print(f"Total dls_ambiguous == True deliveries: {df['dls_ambiguous'].sum()} "
          f"(matches: {df[df['dls_ambiguous']]['match_id'].nunique()})")
    print(f"Super Over deliveries (innings >= 3): {df['is_super_over'].sum()} "
          f"(matches: {df[df['is_super_over']]['match_id'].nunique()})")

    inn1 = df[df['innings'] == 1]
    print(f"Innings 1 low LOO support (<{cfg_primary.min_loo_support} matches) deliveries: "
          f"{inn1['loo_low_support_flag'].sum()} / {len(inn1)} "
          f"(min support in std 20-ov={inn1[inn1['max_legal_balls']==120]['loo_support_count'].min()}, "
          f"min support overall={inn1['loo_support_count'].min()})")

    reg_trusted = df[(~df['is_super_over']) & (~df['dls_ambiguous'])].copy()
    print(f"Trusted regular deliveries missing pressure_bat: {reg_trusted['pressure_bat'].isna().sum()}")
    print(f"Trusted regular deliveries missing pressure_bowl: {reg_trusted['pressure_bowl'].isna().sum()}")
    print(f"Ambiguous DLS deliveries excluded (NaN pressure_bat/bowl): "
          f"{df[df['dls_ambiguous']]['pressure_bat'].isna().sum()} / {df['dls_ambiguous'].sum()}")

    # --- PRESSURE SCORE DISTRIBUTIONS FOR ALL 4 REGIMES ---
    print("\n=== 4. PRESSURE SCORE DISTRIBUTIONS ACROSS ALL 4 REGULAR REGIMES ===")
    regimes = [
        ("Batting: Chase (Inn 2)", df[(df['innings'] == 2) & df['pressure_bat'].notna()]['pressure_bat']),
        ("Batting: Setting (Inn 1)", df[(df['innings'] == 1) & df['pressure_bat'].notna()]['pressure_bat']),
        ("Bowling: Defending (Inn 2)", df[(df['innings'] == 2) & df['pressure_bowl'].notna()]['pressure_bowl']),
        ("Bowling: Setting (Inn 1)", df[(df['innings'] == 1) & df['pressure_bowl'].notna()]['pressure_bowl']),
    ]
    dist_rows = []
    for label, s in regimes:
        dist_rows.append({
            'Regime': label,
            'Deliveries': len(s),
            'Mean': s.mean(),
            'Std': s.std(),
            'Min': s.min(),
            'P10': s.quantile(0.10),
            'P25': s.quantile(0.25),
            'Median': s.median(),
            'P75': s.quantile(0.75),
            'P90': s.quantile(0.90),
            'P95': s.quantile(0.95),
            'Max': s.max(),
            'Pct>=60': (s >= 60).mean() * 100,
            'Pct>=70': (s >= 70).mean() * 100,
        })
    dist_table = pd.DataFrame(dist_rows)
    print(dist_table.round(2).to_string(index=False))

    so_bat = df[df['is_super_over']]['pressure_so_bat']
    so_bowl = df[df['is_super_over']]['pressure_so_bowl']
    print(f"\nSuper Over (separate, n={len(so_bat)}): Bat mean={so_bat.mean():.2f} [{so_bat.min():.1f}, {so_bat.max():.1f}] | "
          f"Bowl mean={so_bowl.mean():.2f} [{so_bowl.min():.1f}, {so_bowl.max():.1f}]")

    # --- PLAYER OPPORTUNITY DISTRIBUTIONS (IPL vs ALL) ---
    print("\n=== 5. PLAYER PRESSURE OPPORTUNITY DISTRIBUTIONS (NO PERFORMANCE SCORING) ===")
    bat_opp, bowl_opp = compute_player_opportunities(df)

    print(f"Batter Opportunity Summary (N={len(bat_opp)} canonical batters):")
    print(bat_opp[['total_balls', 'ipl_balls', 'balls_ge60', 'balls_ge70', 'ipl_ge60', 'ipl_ge70']].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95]
    ).round(1).to_string())

    print(f"\nBowler Opportunity Summary (N={len(bowl_opp)} canonical bowlers):")
    print(bowl_opp[['total_balls', 'ipl_balls', 'balls_ge60', 'balls_ge70', 'ipl_ge60', 'ipl_ge70']].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95]
    ).round(1).to_string())

    print("\nEligible Player Counts at Candidate High-Pressure (PI >= 60) Cutoffs:")
    for cutoff in [30, 50, 75, 100, 150, 200]:
        b_all = (bat_opp['balls_ge60'] >= cutoff).sum()
        b_ipl = (bat_opp['ipl_ge60'] >= cutoff).sum()
        w_all = (bowl_opp['balls_ge60'] >= cutoff).sum()
        w_ipl = (bowl_opp['ipl_ge60'] >= cutoff).sum()
        print(f"  >= {cutoff:3d} balls (PI>=60): Batters (All={b_all:3d}, IPL={b_ipl:3d}) | Bowlers (All={w_all:3d}, IPL={w_ipl:3d})")

    # --- SENSITIVITY ANALYSIS ---
    print("\n=== 6. SENSITIVITY ANALYSIS (WEIGHTS, KNOCKOUT MULTIPLIER, COMPETITION WEIGHTING) ===")
    sens_configs = [
        ("Knockout = 1.05", PressureConfig(knockout_multiplier=1.05)),
        ("Knockout = 1.10", PressureConfig(knockout_multiplier=1.10)),
        ("BatChase RRR-heavy (0.50/0.15/0.20/0.15)", PressureConfig(bat_chase_w_rrr=0.50, bat_chase_w_rate_diff=0.15, bat_chase_w_wkt=0.20)),
        ("BatChase Wkt-heavy (0.30/0.15/0.40/0.15)", PressureConfig(bat_chase_w_rrr=0.30, bat_chase_w_rate_diff=0.15, bat_chase_w_wkt=0.40)),
        ("BatSet Wkt-heavy (0.55/0.30/0.15)", PressureConfig(bat_set_w_wkt=0.55, bat_set_w_deficit=0.30, bat_set_w_phase=0.15)),
        ("BowlDef Threat-heavy (0.45/0.20/0.20/0.15)", PressureConfig(bowl_def_w_threat=0.45, bowl_def_w_wkt=0.20, bowl_def_w_late=0.20)),
        ("BowlDef Late-heavy (0.25/0.20/0.35/0.20)", PressureConfig(bowl_def_w_threat=0.25, bowl_def_w_wkt=0.20, bowl_def_w_late=0.35, bowl_def_w_cushion=0.20)),
        ("BowlSet Dom-heavy (0.55/0.25/0.20)", PressureConfig(bowl_set_w_dom=0.55, bowl_set_w_wkt=0.25, bowl_set_w_phase=0.20)),
    ]

    valid_idx = (~df['is_super_over']) & df['pressure_bat'].notna() & df['pressure_bowl'].notna()
    base_bat = df.loc[valid_idx, 'pressure_bat'].values
    base_bowl = df.loc[valid_idx, 'pressure_bowl'].values

    sens_results = []
    for name, cfg_alt in sens_configs:
        df_alt = compute_pressure_scores(state_df, ds['matches'], ds['competitions'], cfg_alt)
        alt_bat = df_alt.loc[valid_idx, 'pressure_bat'].values
        alt_bowl = df_alt.loc[valid_idx, 'pressure_bowl'].values

        sp_bat = spearmanr(base_bat, alt_bat).statistic
        sp_bowl = spearmanr(base_bowl, alt_bowl).statistic
        mae_bat = np.mean(np.abs(base_bat - alt_bat))
        mae_bowl = np.mean(np.abs(base_bowl - alt_bowl))
        # Agreement on PI >= 60 classification
        ov_bat = ((base_bat >= 60) == (alt_bat >= 60)).mean() * 100
        ov_bowl = ((base_bowl >= 60) == (alt_bowl >= 60)).mean() * 100
        sens_results.append({
            'Variant': name,
            'Spearman_Bat': sp_bat,
            'MAE_Bat': mae_bat,
            'ClassAgree60_Bat%': ov_bat,
            'Spearman_Bowl': sp_bowl,
            'MAE_Bowl': mae_bowl,
            'ClassAgree60_Bowl%': ov_bowl,
        })
    sens_df = pd.DataFrame(sens_results)
    print(sens_df.round(4).to_string(index=False))

    # Competition weighting impact on player effective high-pressure exposure (PI >= 60)
    print("\nCompetition Weighting Sensitivity (Effective High-Pressure Balls PI>=60 across T20I weights):")
    # Compare effective exposure ranking vs IPL-only and equal-weight
    bat_exp = bat_opp[bat_opp['balls_ge60'] >= 50].copy()
    bowl_exp = bowl_opp[bowl_opp['balls_ge60'] >= 50].copy()
    t20i_bat = bat_exp['balls_ge60'] - bat_exp['ipl_ge60']
    t20i_bowl = bowl_exp['balls_ge60'] - bowl_exp['ipl_ge60']

    comp_sens = []
    for w_t20i in [1.00, 0.75, 0.50, 0.25, 0.00]:
        eff_b = bat_exp['ipl_ge60'] + w_t20i * t20i_bat
        eff_w = bowl_exp['ipl_ge60'] + w_t20i * t20i_bowl
        comp_sens.append({
            'T20I_Weight': w_t20i,
            'Bat_Corr_vs_Equal': spearmanr(bat_exp['balls_ge60'], eff_b).statistic,
            'Bat_Corr_vs_IPLOnly': spearmanr(bat_exp['ipl_ge60'], eff_b).statistic,
            'Bowl_Corr_vs_Equal': spearmanr(bowl_exp['balls_ge60'], eff_w).statistic,
            'Bowl_Corr_vs_IPLOnly': spearmanr(bowl_exp['ipl_ge60'], eff_w).statistic,
        })
    print(pd.DataFrame(comp_sens).round(4).to_string(index=False))

    # --- GENERATE VISUALIZATIONS (07_ to 12_) ---
    print("\n=== 7. GENERATING PHASE 2 VISUALIZATIONS ===")
    generate_plots(df, ds['matches'], regimes, bat_opp, bowl_opp, sens_df)
    print("All Phase 2 analysis and visualizations complete!")


def generate_plots(df, matches_df, regimes, bat_opp, bowl_opp, sens_df):
    # Figure 07: Pressure score distributions for all 4 regimes
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ['#E91E63', '#FF9800', '#1E88E5', '#00897B']
    for ax, (label, s), col in zip(axes.flat, regimes, colors):
        sns.histplot(s, bins=40, kde=True, ax=ax, color=col, alpha=0.65, edgecolor='white')
        p50, p75, p90 = s.median(), s.quantile(0.75), s.quantile(0.90)
        ax.axvline(p50, color='black', linestyle='--', linewidth=1.2, label=f'Median={p50:.1f}')
        ax.axvline(p75, color='navy', linestyle=':', linewidth=1.3, label=f'P75={p75:.1f}')
        ax.axvline(60.0, color='red', linestyle='-.', linewidth=1.3, label=f'PI=60 ({(s>=60).mean()*100:.1f}%)')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Continuous Pressure Index (0-100)')
        ax.set_xlim(0, 100)
        ax.legend(fontsize=9)
    plt.suptitle('Phase 2: Continuous Pressure Index Distributions Across All 4 Regimes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_pressure_distributions_by_regime.png", dpi=200)
    plt.close()

    # Figure 08: LOO Par Score, Expected Wickets & Support Counts
    inn1 = df[(df['innings'] == 1) & (df['max_legal_balls'] == 120)]
    curve = inn1.groupby(['is_ipl', 'legal_balls_before']).agg(
        exp_s=('loo_exp_score', 'mean'),
        std_s=('loo_std_score', 'mean'),
        exp_w=('loo_exp_wkts', 'mean'),
        supp=('loo_support_count', 'min')
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for is_ipl, lbl, col in [(True, 'IPL', '#FF4081'), (False, 'T20I', '#1976D2')]:
        sub = curve[curve['is_ipl'] == is_ipl]
        axes[0].plot(sub['legal_balls_before'], sub['exp_s'], label=lbl, color=col, linewidth=2)
        axes[0].fill_between(sub['legal_balls_before'], sub['exp_s'] - sub['std_s'], sub['exp_s'] + sub['std_s'], color=col, alpha=0.15)
        axes[1].plot(sub['legal_balls_before'], sub['exp_w'], label=lbl, color=col, linewidth=2)
        axes[2].plot(sub['legal_balls_before'], sub['supp'], label=lbl, color=col, linewidth=2)

    axes[0].set_title('LOO Expected Score +/- 1 SD (Innings 1)', fontweight='bold')
    axes[0].set_xlabel('Legal Balls Bowled')
    axes[0].set_ylabel('Cumulative Runs')
    axes[0].legend()

    axes[1].set_title('LOO Expected Wickets Fallen (Innings 1)', fontweight='bold')
    axes[1].set_xlabel('Legal Balls Bowled')
    axes[1].set_ylabel('Wickets Lost')
    axes[1].legend()

    axes[2].axhline(30, color='red', linestyle='--', label='Min Support Fallback (30)')
    axes[2].set_title('LOO Sample Support Count (N - 1)', fontweight='bold')
    axes[2].set_xlabel('Legal Balls Bowled')
    axes[2].set_ylabel('Out-of-Match Contributing Count')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_loo_par_and_support_curves.png", dpi=200)
    plt.close()

    # Figure 09: Empirical Leverage vs Bowler Defending Curve + 2D Pressure Heatmaps
    inn2 = (
        df[(df['innings'] == 2) & df['pressure_bat'].notna()]
        .merge(matches_df[['match_id', 'win_team_id']], on='match_id', how='left')
    )
    inn2 = inn2[inn2['win_team_id'].notna()].copy()
    inn2['chase_won'] = (inn2['batting_team_id'] == inn2['win_team_id']).astype(float)
    inn2['rrr_bin'] = pd.cut(inn2['rrr_eff'], bins=[0, 4, 6, 7.5, 9, 10.5, 12, 14, 16, 20, 30])
    lev = inn2.groupby('rrr_bin', observed=False).agg(
        win_p=('chase_won', 'mean'),
        c_threat=('comp_c_threat', 'mean')
    ).reset_index()
    lev['emp_lev'] = 4 * lev['win_p'] * (1 - lev['win_p'])
    lev['mid'] = [2, 5, 6.75, 8.25, 9.75, 11.25, 13, 15, 18, 25]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(lev['mid'], lev['emp_lev'], marker='o', label='Empirical Leverage 4p(1-p)', color='#1565C0', linewidth=2)
    axes[0].plot(lev['mid'], lev['c_threat'], marker='s', linestyle='--', label='Model C_threat Bell Curve', color='#D81B60', linewidth=2)
    axes[0].set_title('Empirical Chase Leverage vs Bowler Threat Curve', fontweight='bold')
    axes[0].set_xlabel('Required Run Rate (RRR)')
    axes[0].set_ylabel('Normalized Tension (0-1)')
    axes[0].legend()

    inn2_sub = inn2[inn2['rrr_eff'] <= 30].copy()
    inn2_sub['rrr_b'] = pd.cut(inn2_sub['rrr_eff'], bins=[0, 6, 8, 10, 12, 15, 20, 30],
                               labels=['0-6', '6-8', '8-10', '10-12', '12-15', '15-20', '20-30'])
    inn2_sub['w_b'] = pd.cut(inn2_sub['wickets_in_hand_before'], bins=[0, 2, 4, 6, 8, 10],
                             labels=['1-2', '3-4', '5-6', '7-8', '9-10'])

    hm_bat = inn2_sub.pivot_table(index='w_b', columns='rrr_b', values='pressure_bat', aggfunc='mean', observed=False).iloc[::-1]
    hm_bowl = inn2_sub.pivot_table(index='w_b', columns='rrr_b', values='pressure_bowl', aggfunc='mean', observed=False).iloc[::-1]

    sns.heatmap(hm_bat, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1], vmin=0, vmax=90)
    axes[1].set_title('Mean Batting Chase Pressure (Garbage Time Decays >15)', fontweight='bold')
    axes[1].set_xlabel('RRR')
    axes[1].set_ylabel('Wickets in Hand')

    sns.heatmap(hm_bowl, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[2], vmin=0, vmax=90)
    axes[2].set_title('Mean Bowling Defending Pressure (Peaks in Knife-Edge)', fontweight='bold')
    axes[2].set_xlabel('RRR')
    axes[2].set_ylabel('Wickets in Hand')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_pressure_landscape_and_leverage.png", dpi=200)
    plt.close()

    # Figure 10: Representative Match Pressure Trajectories (Match 21586 IPL Final + Match 13512 1-run Thriller)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (mid, title) in zip(axes, [
        (21586, "Match 21586: IPL 2023 Final (DLS Stoppage @ Over 1.3)"),
        (13512, "Match 13512: ENG vs WI 2nd Innings (1-Run Thriller)")
    ]):
        sub = df[(df['match_id'] == mid) & (df['innings'] == 2)].sort_values(['overid', 'over_ball_num']).copy()
        sub['ball_seq'] = np.arange(1, len(sub) + 1)
        ax.plot(sub['ball_seq'], sub['pressure_bat'], label='Batting Pressure', color='#E91E63', linewidth=1.8)
        ax.plot(sub['ball_seq'], sub['pressure_bowl'], label='Bowling Pressure', color='#1976D2', linewidth=1.8)
        if (sub['dls_status'] == 'dls_pre_stoppage_suspected').any():
            stop_x = sub[sub['dls_status'] == 'dls_post_stoppage_suspected']['ball_seq'].min()
            ax.axvline(stop_x, color='black', linestyle='--', label='Rain Stoppage (Target 215 -> 171)')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Delivery Sequence (Innings 2)')
        ax.set_ylabel('Pressure Index (0-100)')
        ax.set_ylim(0, 102)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_match_pressure_trajectories.png", dpi=200)
    plt.close()

    # Figure 11: Opportunity Distributions (Batters & Bowlers, PI >= 60)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    b_plot = bat_opp[bat_opp['balls_ge60'] >= 20]
    w_plot = bowl_opp[bowl_opp['balls_ge60'] >= 20]
    sns.histplot(b_plot['balls_ge60'], bins=30, ax=axes[0], color='#FF4081', alpha=0.6, label='All Matches (IPL+T20I)')
    sns.histplot(b_plot[b_plot['ipl_ge60'] >= 20]['ipl_ge60'], bins=30, ax=axes[0], color='#880E4F', alpha=0.6, label='IPL Only')
    axes[0].axvline(50, color='black', linestyle='--', label='50 balls')
    axes[0].axvline(100, color='navy', linestyle=':', label='100 balls')
    axes[0].set_title('Batter High-Pressure Opportunity Dist (PI >= 60, >=20 balls)', fontweight='bold')
    axes[0].set_xlabel('Balls Faced at PI >= 60')
    axes[0].legend()

    sns.histplot(w_plot['balls_ge60'], bins=30, ax=axes[1], color='#1E88E5', alpha=0.6, label='All Matches (IPL+T20I)')
    sns.histplot(w_plot[w_plot['ipl_ge60'] >= 20]['ipl_ge60'], bins=30, ax=axes[1], color='#0D47A1', alpha=0.6, label='IPL Only')
    axes[1].axvline(60, color='black', linestyle='--', label='60 balls')
    axes[1].axvline(120, color='navy', linestyle=':', label='120 balls')
    axes[1].set_title('Bowler High-Pressure Opportunity Dist (PI >= 60, >=20 balls)', fontweight='bold')
    axes[1].set_xlabel('Legal Balls Bowled at PI >= 60')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_opportunity_distributions.png", dpi=200)
    plt.close()

    # Figure 12: Sensitivity Analysis Bar Chart
    fig, ax = plt.subplots(figsize=(12, 5))
    y = np.arange(len(sens_df))
    h = 0.35
    ax.barh(y - h/2, sens_df['Spearman_Bat'], height=h, label='Batting Pressure Spearman r', color='#E91E63', alpha=0.85)
    ax.barh(y + h/2, sens_df['Spearman_Bowl'], height=h, label='Bowling Pressure Spearman r', color='#1E88E5', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(sens_df['Variant'])
    ax.set_xlim(0.90, 1.005)
    ax.axvline(0.95, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Spearman Rank Correlation vs Primary Model')
    ax.set_title('Sensitivity Analysis: Rank Correlation Across Parameter & Knockout Variants', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_sensitivity_analysis.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

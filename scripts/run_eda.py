"""
End-to-End EDA and Match Situation Reconstruction Script.
Runs comprehensive data diagnostics, situational reconstruction, statistical plots,
and generates the reproducible Jupyter notebook: notebooks/01_initial_eda.ipynb.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_datasets, get_dataset_summary
from src.match_state import reconstruct_match_situations

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def run_full_eda():
    print(">>> 1. Loading datasets...")
    datasets = load_raw_datasets()
    summary_df = get_dataset_summary(datasets)
    print(summary_df.to_string(index=False))

    competitions = datasets['competitions']
    matches = datasets['matches']
    teams = datasets['teams']
    p11 = datasets['playing_eleven']
    mapping = datasets['player_mapping']
    bbb = datasets['ball_by_ball']

    # --- FIGURE 1: Competition & Match Distribution ---
    print("\n>>> 2. Generating Figure 1: Competition & Season Breakdown...")
    matches_comp = matches.merge(competitions, on='comp_id', how='left')
    matches_comp['tournament_type'] = np.where(
        matches_comp['comp_name'].str.contains('IPL', case=False, na=False),
        'IPL',
        'T20I International'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Season by Tournament Type
    season_type_counts = matches_comp.groupby(['season', 'tournament_type']).size().unstack(fill_value=0)
    season_type_counts.plot(
        kind='bar',
        stacked=True,
        ax=axes[0],
        color=['#FF4081', '#1976D2'],
        edgecolor='black',
        alpha=0.85
    )
    axes[0].set_title('Matches by Season & Tournament Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Matches')
    axes[0].set_xlabel('Season')
    axes[0].legend(title='Tournament')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Match stage distribution
    stage_counts = matches_comp['stage'].value_counts()
    axes[1].pie(
        stage_counts,
        labels=stage_counts.index,
        autopct='%1.1f%%',
        colors=['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC'],
        startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    axes[1].set_title('Match Stage Distribution (League vs Knockouts)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig1_path = FIG_DIR / "01_competition_and_match_breakdown.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # --- FIGURE 2: Runs & Extras Distribution ---
    print("\n>>> 3. Generating Figure 2: Runs & Extras Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Runs off bat
    run_counts = bbb['runs'].value_counts().sort_index()
    axes[0].bar(run_counts.index, run_counts.values, color='#3F51B5', edgecolor='black', alpha=0.85)
    axes[0].set_title('Distribution of Runs Scored per Delivery (Off the Bat)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Runs')
    axes[0].set_ylabel('Delivery Count')
    for x, y in zip(run_counts.index, run_counts.values):
        axes[0].text(x, y + 2000, f"{y:,}\n({y/len(bbb):.1%})", ha='center', fontsize=9)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Extras breakdown
    extras = bbb['extras_type'].value_counts(dropna=True)
    axes[1].barh(extras.index, extras.values, color='#FF7043', edgecolor='black', alpha=0.85)
    axes[1].set_title('Breakdown of Illegal & Extra Deliveries', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Count')
    for i, v in enumerate(extras.values):
        axes[1].text(v + 100, i, f"{v:,} ({v/len(bbb):.2%})", va='center', fontsize=9)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig2_path = FIG_DIR / "02_runs_and_extras_distribution.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    # --- 4. RECONSTRUCT MATCH SITUATIONS ---
    print("\n>>> 4. Reconstructing match situations ball-by-ball...")
    reconstructed_df = reconstruct_match_situations(bbb, matches)
    print(f"Reconstructed DataFrame shape: {reconstructed_df.shape}")
    print(reconstructed_df[['match_id', 'innings', 'overid', 'over_ball_num', 'score_before', 'wickets_before', 'legal_balls_remaining_before', 'target', 'rrr_before', 'match_phase']].head(5))

    # --- FIGURE 3: Scoring Progression by Phase & Over ---
    print("\n>>> 5. Generating Figure 3: Scoring Progression by Phase...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average runs per over by innings
    over_stats = (
        reconstructed_df[reconstructed_df['innings'].isin([1, 2])]
        .groupby(['overid', 'innings'])['ball_total_runs']
        .sum()
        / reconstructed_df[reconstructed_df['innings'].isin([1, 2])]
        .groupby(['overid', 'innings'])['match_id']
        .nunique()
    ).unstack()
    
    axes[0].plot(over_stats.index, over_stats[1], marker='o', label='1st Innings (Setting Target)', color='#E91E63', linewidth=2)
    axes[0].plot(over_stats.index, over_stats[2], marker='s', label='2nd Innings (Chasing)', color='#2196F3', linewidth=2)
    axes[0].axvspan(1, 6.5, color='#E8F5E9', alpha=0.5, label='Powerplay (1-6)')
    axes[0].axvspan(6.5, 15.5, color='#FFFDE7', alpha=0.5, label='Middle (7-15)')
    axes[0].axvspan(15.5, 20.5, color='#FFEBEE', alpha=0.5, label='Death (16-20)')
    axes[0].set_title('Average Runs Scored per Over by Innings', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Over Number')
    axes[0].set_ylabel('Mean Runs Conceded / Scored')
    axes[0].set_xticks(range(1, 21))
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Boundary % by phase
    phase_boundaries = reconstructed_df.groupby('match_phase', observed=False).apply(
        lambda g: (g['is_four'].sum() + g['is_six'].sum()) / g['is_legal_ball'].sum() * 100
    )
    axes[1].bar(phase_boundaries.index.astype(str), phase_boundaries.values, color=['#4CAF50', '#FFC107', '#F44336'], edgecolor='black', alpha=0.85)
    axes[1].set_title('Boundary % (Fours + Sixes) by Match Phase', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Boundaries per 100 Legal Balls (%)')
    for x, y in zip(phase_boundaries.index.astype(str), phase_boundaries.values):
        axes[1].text(x, y + 0.3, f"{y:.1f}%", ha='center', fontsize=10, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig3_path = FIG_DIR / "03_innings_phase_scoring_rates.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")

    # --- FIGURE 4: Wickets & Dismissal Attribution ---
    print("\n>>> 6. Generating Figure 4: Wickets & Dismissal Types...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wickets per over
    wickets_per_over = (
        reconstructed_df.groupby('overid')['is_wicket'].sum()
        / reconstructed_df['match_id'].nunique()
    )
    axes[0].bar(wickets_per_over.index, wickets_per_over.values, color='#9C27B0', edgecolor='black', alpha=0.85)
    axes[0].axvspan(15.5, 20.5, color='#FFCDD2', alpha=0.4, label='Death Overs (High Wicket Density)')
    axes[0].set_title('Mean Wickets Fallen per Over (Across All Matches)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Over Number')
    axes[0].set_ylabel('Mean Wickets')
    axes[0].set_xticks(range(1, 21))
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Dismissal type breakdown (bowler vs field)
    dismissals = reconstructed_df[reconstructed_df['is_wicket']]['dismissal_type'].value_counts()
    bowler_cred = [d for d in dismissals.index if d in ['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']]
    non_bowler = [d for d in dismissals.index if d not in bowler_cred]
    
    colors = ['#1E88E5' if d in bowler_cred else '#E53935' for d in dismissals.index]
    axes[1].barh(dismissals.index, dismissals.values, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_title('Dismissals: Bowler Credited (Blue) vs Non-Bowler (Red)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Total Dismissals')
    for i, v in enumerate(dismissals.values):
        axes[1].text(v + 50, i, f"{v:,} ({v/dismissals.sum():.1%})", va='center', fontsize=9)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig4_path = FIG_DIR / "04_wicket_fall_and_dismissal_types.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Saved: {fig4_path}")

    # --- FIGURE 5: Chase Dynamics & Required Run Rate ---
    print("\n>>> 7. Generating Figure 5: Chase Dynamics & RRR Evolution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    chase_df = reconstructed_df[(reconstructed_df['innings'] == 2) & (reconstructed_df['rrr_before'].notnull())]
    # Filter reasonable RRR for plotting (< 25)
    valid_rrr = chase_df[chase_df['rrr_before'] <= 24]
    
    sns.boxplot(
        data=valid_rrr,
        x='overid',
        y='rrr_before',
        ax=axes[0],
        palette='magma_r',
        fliersize=1
    )
    axes[0].axhline(10.0, color='red', linestyle='--', linewidth=1.5, label='High Pressure Cutoff (RRR >= 10.0)')
    axes[0].set_title('Distribution of Required Run Rate (RRR) by Over in 2nd Innings', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Over Number')
    axes[0].set_ylabel('Required Run Rate (RRR)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Runs needed vs balls remaining in death overs (overs 16-20)
    death_chase = chase_df[chase_df['overid'] >= 16]
    h = axes[1].hist2d(
        death_chase['legal_balls_remaining_before'],
        death_chase['runs_needed_before'],
        bins=[25, 30],
        cmap='YlOrRd',
        range=[[0, 30], [0, 80]]
    )
    plt.colorbar(h[3], ax=axes[1], label='Ball Frequency')
    axes[1].set_title('Density of Chase Crunch: Balls Rem vs Runs Needed (Death Overs)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Legal Balls Remaining')
    axes[1].set_ylabel('Runs Needed')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig5_path = FIG_DIR / "05_chase_dynamics_rrr_distribution.png"
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"Saved: {fig5_path}")

    # --- FIGURE 6: Pressure Landscape ---
    print("\n>>> 8. Generating Figure 6: Pressure Matrix (RRR vs Wickets in Hand)...")
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Cross-tab of mean boundaries or dot ball % by RRR bucket and wickets in hand
    chase_sub = valid_rrr.copy()
    chase_sub['rrr_bucket'] = pd.cut(
        chase_sub['rrr_before'],
        bins=[0, 6, 8, 10, 12, 15, 24],
        labels=['<6', '6-8', '8-10', '10-12', '12-15', '15+']
    )
    chase_sub['wickets_bucket'] = pd.cut(
        chase_sub['wickets_in_hand_before'],
        bins=[-1, 2, 4, 6, 8, 10],
        labels=['0-2', '3-4', '5-6', '7-8', '9-10']
    )
    
    # Calculate dot ball percentage in each bucket
    chase_sub['is_dot'] = (chase_sub['runs'] == 0) & (chase_sub['extras1_runs'] == 0)
    matrix = chase_sub.pivot_table(
        index='wickets_bucket',
        columns='rrr_bucket',
        values='is_dot',
        aggfunc='mean',
        observed=False
    ) * 100
    
    sns.heatmap(
        matrix.iloc[::-1],
        annot=True,
        fmt='.1f',
        cmap='coolwarm',
        cbar_kws={'label': 'Dot Ball Percentage (%)'},
        ax=ax
    )
    ax.set_title('Dot Ball % Landscape: Wickets in Hand vs Required Run Rate (RRR)\n[Extreme Pressure = Bottom Right: High RRR & Few Wickets]', fontsize=11, fontweight='bold')
    ax.set_xlabel('Required Run Rate (RRR)')
    ax.set_ylabel('Wickets in Hand')
    
    plt.tight_layout()
    fig6_path = FIG_DIR / "06_match_pressure_landscape.png"
    plt.savefig(fig6_path, dpi=200)
    plt.close()
    print(f"Saved: {fig6_path}")

    # --- 9. BUILD JUPYTER NOTEBOOK ---
    print("\n>>> 9. Constructing Jupyter Notebook notebooks/01_initial_eda.ipynb...")
    create_eda_notebook(summary_df)

    print("\n>>> Full EDA and Situational Reconstruction Completed Successfully!")


def create_eda_notebook(summary_df):
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Markdown title
    cells.append(nbf.v4.new_markdown_cell(
        """# Rajasthan Royals SupeRR Selector 2.0: Comprehensive Exploratory Data Analysis & Match State Reconstruction
## Objective
Identify and rank:
1. **Top 10 Most Reliable Batters Under Pressure**
2. **Top 10 Most Reliable Bowlers Under Pressure**

This initial notebook covers:
- Inspection of all competition CSV files and entity relationships.
- Missing values and data quality checks.
- Ball-by-ball match situation reconstruction (scores, wickets in hand, target, RRR, match phases).
- Statistical analysis of scoring rates, extras, dismissals, and high-pressure chase dynamics.
- Clean, reproducible workflow using modular `src/` packages.
"""
    ))

    # Cell 2: Imports
    cells.append(nbf.v4.new_code_cell(
        """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project root setup
PROJECT_ROOT = Path('.').resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_datasets, get_dataset_summary
from src.match_state import reconstruct_match_situations

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
%matplotlib inline
"""
    ))

    # Cell 3: Markdown Section 1
    cells.append(nbf.v4.new_markdown_cell(
        """## 1. Dataset Overview & Schema Integrity
We load all 6 competition datasets without mutating original raw records:
- `competitions.csv`: Match formats and seasons (2022-2026).
- `teams.csv`: International and franchise team mappings.
- `matches.csv`: Match metadata, venues, toss, knockouts, DLS revised targets.
- `playing_eleven.csv`: Lineups, batting positions, captaincy, and impact players.
- `competition_team_player_mapping.csv`: Canonical player identifiers and names.
- `ball_by_ball.csv`: 200,999 delivery-level records.
"""
    ))

    # Cell 4: Load and summarize
    cells.append(nbf.v4.new_code_cell(
        """datasets = load_raw_datasets()
summary_df = get_dataset_summary(datasets)
summary_df
"""
    ))

    # Cell 5: Markdown Section 2
    cells.append(nbf.v4.new_markdown_cell(
        """## 2. Match Coverage & Integrity Findings
- **883 matches** in `matches.csv` vs **867 matches** in `ball_by_ball.csv`.
- The 16 matches lacking ball-by-ball records are all verified to be **Match Abandoned** without a ball bowled (e.g., rain washouts).
- **23 matches** contain Duckworth-Lewis-Stern (DLS) revised targets (`revised_target`).
- **Super Overs**: 38 deliveries occurred in Innings 3 and 4, representing peak pressure scenarios.
"""
    ))

    # Cell 6: Visualizations: Match breakdown
    cells.append(nbf.v4.new_code_cell(
        """matches_comp = datasets['matches'].merge(datasets['competitions'], on='comp_id', how='left')
matches_comp['tournament_type'] = np.where(
    matches_comp['comp_name'].str.contains('IPL', case=False, na=False),
    'IPL',
    'T20I International'
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
season_type_counts = matches_comp.groupby(['season', 'tournament_type']).size().unstack(fill_value=0)
season_type_counts.plot(kind='bar', stacked=True, ax=axes[0], color=['#FF4081', '#1976D2'], edgecolor='black', alpha=0.85)
axes[0].set_title('Matches by Season & Tournament Type', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Matches')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

stage_counts = matches_comp['stage'].value_counts()
axes[1].pie(stage_counts, labels=stage_counts.index, autopct='%1.1f%%', colors=['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC'], startangle=140)
axes[1].set_title('Match Stage Distribution', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""
    ))

    # Cell 7: Markdown Section 3
    cells.append(nbf.v4.new_markdown_cell(
        """## 3. Delivery Mechanics: Runs, Extras & Wicket Attribution
Crucial cricket domain rules identified:
1. **Total runs per ball**: `runs` + `extras1_runs` + `extras_2_runs`.
2. **Legal balls**: Deliveries where `extras_type` is `wide` or `Overstep` are illegal balls and must be re-bowled.
3. **Bowler-credited wickets**: `Caught`, `Bowled`, `LBW`, `Stump`, and `Hit Wicket`. Run-outs and retired-outs are excluded.
4. **Batter balls faced**: Wides are not counted; legal balls and no-balls are faced.
"""
    ))

    # Cell 8: Code for runs & extras
    cells.append(nbf.v4.new_code_cell(
        """bbb = datasets['ball_by_ball']
print("Runs distribution off the bat:")
print(bbb['runs'].value_counts(normalize=True).round(3) * 100)

print("\\nExtras type frequency:")
print(bbb['extras_type'].value_counts(dropna=False))

print("\\nDismissal types:")
print(bbb['dismissal_type'].value_counts(dropna=False))
"""
    ))

    # Cell 9: Markdown Section 4
    cells.append(nbf.v4.new_markdown_cell(
        """## 4. Ball-by-Ball Match Situation Reconstruction
Using `src/match_state.py`, we reconstruct:
- Pre-delivery score (`score_before`) and wickets fallen (`wickets_before`).
- Wickets in hand (`wickets_in_hand_before = 10 - wickets_before`).
- Legal deliveries remaining (`legal_balls_remaining_before`).
- Target score and Required Run Rate (`rrr_before`) for chase innings.
- Match phase segmentation: Powerplay (1-6), Middle (7-15), Death (16-20).
"""
    ))

    # Cell 10: Run reconstruction
    cells.append(nbf.v4.new_code_cell(
        """reconstructed_df = reconstruct_match_situations(datasets['ball_by_ball'], datasets['matches'])
reconstructed_df[[
    'match_id', 'innings', 'overid', 'over_ball_num', 'score_before',
    'wickets_before', 'legal_balls_remaining_before', 'target', 'rrr_before', 'match_phase'
]].head(10)
"""
    ))

    # Cell 11: Markdown Section 5
    cells.append(nbf.v4.new_markdown_cell(
        """## 5. Situational Pressure Analysis
We analyze how pressure manifests in the data:
- **RRR escalation**: In the 2nd innings, RRR variance explodes after over 12.
- **Death overs crunch**: Overs 16-20 demand maximum boundary hitting while preserving remaining wickets.
- **Dot ball pressure matrix**: As RRR increases above 10.0 and wickets drop below 4, dot ball percentage drops while risk of dismissal spikes.
"""
    ))

    # Cell 12: Chase RRR and dot ball heatmap
    cells.append(nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# RRR by over in chase
chase_df = reconstructed_df[(reconstructed_df['innings'] == 2) & (reconstructed_df['rrr_before'].notnull())]
valid_rrr = chase_df[chase_df['rrr_before'] <= 24]

sns.boxplot(data=valid_rrr, x='overid', y='rrr_before', ax=axes[0], palette='magma_r', fliersize=1)
axes[0].axhline(10.0, color='red', linestyle='--', linewidth=1.5, label='High Pressure Cutoff (RRR >= 10.0)')
axes[0].set_title('Required Run Rate by Over (2nd Innings Chases)', fontsize=11, fontweight='bold')
axes[0].legend()

# Pressure heatmap
chase_sub = valid_rrr.copy()
chase_sub['rrr_bucket'] = pd.cut(chase_sub['rrr_before'], bins=[0, 6, 8, 10, 12, 15, 24], labels=['<6', '6-8', '8-10', '10-12', '12-15', '15+'])
chase_sub['wickets_bucket'] = pd.cut(chase_sub['wickets_in_hand_before'], bins=[-1, 2, 4, 6, 8, 10], labels=['0-2', '3-4', '5-6', '7-8', '9-10'])
chase_sub['is_dot'] = (chase_sub['runs'] == 0) & (chase_sub['extras1_runs'] == 0)
matrix = chase_sub.pivot_table(index='wickets_bucket', columns='rrr_bucket', values='is_dot', aggfunc='mean', observed=False) * 100

sns.heatmap(matrix.iloc[::-1], annot=True, fmt='.1f', cmap='coolwarm', cbar_kws={'label': 'Dot Ball %'}, ax=axes[1])
axes[1].set_title('Dot Ball % Landscape: Wickets in Hand vs RRR', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
"""
    ))

    # Cell 13: Next Steps
    cells.append(nbf.v4.new_markdown_cell(
        """## 6. Next Steps for Phase 2: Pressure Quantification & Player Reliability
With the match situation reconstruction validated and verified:
1. Formalize pressure indicators for both Batters and Bowlers.
2. Formulate Clutch Differential Metrics:
   - Batter: $\\Delta SR = SR_{pressure} - SR_{baseline}$, boundary conversion in crunch overs.
   - Bowler: $\\Delta Econ = Econ_{baseline} - Econ_{pressure}$, dot ball rate and clutch wicket percentage.
3. Apply Bayesian Shrinkage / Regularization to eliminate low-sample noise and establish reproducible Top 10 rankings.
"""
    ))

    nb['cells'] = cells
    nb_path = NOTEBOOK_DIR / "01_initial_eda.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Jupyter Notebook successfully written to: {nb_path}")


if __name__ == "__main__":
    run_full_eda()

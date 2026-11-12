"""Generate judge-facing audit figures (Figures 37 and 38) from Claude audit CSVs."""

from pathlib import Path
import os
import shutil
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
_artifact_env = os.environ.get("ARTIFACT_FIG_DIR")
ARTIFACT_FIG_DIR = Path(_artifact_env) if _artifact_env else FIG_DIR
if ARTIFACT_FIG_DIR != FIG_DIR:
    ARTIFACT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _short_name(name: str) -> str:
    replacements = {
        "Vaibhav Sooryavanshi": "V. Sooryavanshi",
        "Suryakumar Yadav": "S. Yadav",
        "Varun Chakravarthy": "V. Chakravarthy",
        "Bhuvneshwar Kumar": "B. Kumar",
        "Wanindu Hasaranga": "W. Hasaranga",
        "Ravichandran Ashwin": "R. Ashwin",
        "Jasprit Bumrah": "J. Bumrah",
        "Heinrich Klaasen": "H. Klaasen",
        "Nicholas Pooran": "N. Pooran",
        "Pathum Nissanka": "P. Nissanka",
        "Mitchell Santner": "M. Santner",
    }
    return replacements.get(name, name)


def make_figure_37_pressure_explainer() -> Path:
    """Figure 37: Why PI>=60 separates phase/order roles, HP weight/gate grid, and unified runs/over bowling HP view."""
    by_order = pd.read_csv(DATA_PROCESSED / "claude_hp_by_order.csv")
    by_phase = pd.read_csv(DATA_PROCESSED / "claude_hp_by_phase.csv")
    pi_prof = pd.read_csv(DATA_PROCESSED / "claude_bat_top15_pi_profiles.csv")
    hp_grid = pd.read_csv(DATA_PROCESSED / "claude_hp_weight_gate_grid.csv")
    bowl_delta = pd.read_csv(DATA_PROCESSED / "claude_bowl_top15_delta.csv")
    with open(DATA_PROCESSED / "claude_audit_summary.json", encoding="utf-8") as f:
        audit_json = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.4), dpi=200)
    fig.suptitle(
        "Pressure Mechanics Audit: Structural Openers Exposure, Sensitivity & Unified Bowling HP Scale",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    # Panel A: High-Pressure Share by Batting Order & Match Phase
    ax = axes[0, 0]
    order_seq = ["Opener (1-2)", "Top Order (3)", "Middle (4-5)", "Finisher (6+)"]
    phase_seq = ["Powerplay", "Middle", "Death"]
    bo = by_order.set_index("order_bucket").loc[order_seq].reset_index()
    bp = by_phase.set_index("match_phase").loc[phase_seq].reset_index()

    labels = list(bo["order_bucket"]) + [""] + list(bp["match_phase"])
    vals_ge60 = list(bo["pct_ge60"]) + [0.0] + list(bp["pct_ge60"])
    vals_40_59 = list(bo["pct_ge40"] - bo["pct_ge60"]) + [0.0] + list(bp["pct_ge40"] - bp["pct_ge60"])
    mean_pi = list(bo["mean_pi"]) + [np.nan] + list(bp["mean_pi"])

    x = np.arange(len(labels))
    bar_w = 0.62
    mask = [lbl != "" for lbl in labels]
    x_plot = x[mask]
    v60_plot = np.array(vals_ge60)[mask]
    v40_plot = np.array(vals_40_59)[mask]
    m_plot = np.array(mean_pi)[mask]

    ax.bar(x_plot, v60_plot, width=bar_w, color="#C0392B", alpha=0.90, label="High Pressure (PI >= 60)")
    ax.bar(x_plot, v40_plot, width=bar_w, bottom=v60_plot, color="#F39C12", alpha=0.78, label="Elevated (PI 40-59.9)")
    for xi, v60, v40, mpi in zip(x_plot, v60_plot, v40_plot, m_plot):
        ax.text(xi, v60 + v40 + 1.4, f"{v60:.1f}% HP\n(u={mpi:.1f})", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.axvline(3.98, color="#5D6D7E", linestyle="--", linewidth=1.2)
    ax.text(1.5, 89.5, "By Batting Order", ha="center", fontsize=9.2, fontweight="bold", color="#2C3E50")
    ax.text(6.0, 89.5, "By Match Phase", ha="center", fontsize=9.2, fontweight="bold", color="#2C3E50")
    ax.set_xticks(x_plot)
    ax.set_xticklabels([l for l in labels if l != ""], rotation=18, ha="right", fontsize=8.5)
    ax.set_ylim(0, 96)
    ax.set_ylabel("% of Legal / Batting Deliveries")
    ax.set_title("A. Why Openers & Powerplay Structurally See <2% PI >= 60 Balls", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.87), fontsize=7.9, framealpha=0.94)

    # Panel B: Top-15 Batters Powerplay Share vs Neff_HP
    ax = axes[0, 1]
    colors = [
        "#E74C3C" if r.neff_hp == 0 else ("#F39C12" if r.neff_hp < 45 else "#1F618D")
        for r in pi_prof.itertuples()
    ]
    sizes = np.clip(pi_prof["neff"] / 8.0, 45, 220)
    ax.scatter(pi_prof["pp_share_pct"], pi_prof["neff_hp"], s=sizes, c=colors, edgecolors="black", linewidth=0.8, alpha=0.88, zorder=4)
    ax.axhline(15.0, color="#7F8C8D", linestyle="--", linewidth=1.1)
    ax.text(2.0, 18.0, "Tested Gate Threshold (Neff_HP >= 15)", fontsize=7.5, color="#566573", fontweight="bold")
    ax.axvline(50.0, color="#95A5A6", linestyle=":", linewidth=1.1)

    # Exact coordinates in data space for crowded bottom-right & top-left clusters
    ann_specs = {
        "Vaibhav Sooryavanshi": ((53.0, 76.0), True),
        "Finn Allen": ((53.5, 98.0), True),
        "Priyansh Arya": ((57.0, 120.0), True),
        "Phil Salt": ((44.0, 78.0), True),
        "Pathum Nissanka": ((55.0, 52.0), True),
        "Sanju Samson": ((38.0, 86.0), True),
        "Shreyas Iyer": ((25.0, 62.0), True),
        "Rajat Patidar": ((16.5, 50.0), True),
        "Tim David": ((11.0, 272.0), False),
        "Andre Russell": ((4.5, 244.0), True),
        "Heinrich Klaasen": ((12.5, 200.0), True),
        "Suryakumar Yadav": ((29.0, 232.0), False),
        "Nicholas Pooran": ((24.5, 190.0), False),
        "Shashank Singh": ((9.5, 146.0), False),
        "Dinesh Karthik": ((7.5, 111.0), False),
    }
    for r in pi_prof.itertuples():
        xytext, use_arrow = ann_specs[r.player_name]
        arrow = dict(arrowstyle="-", color="#7F8C8D", lw=0.8) if use_arrow else None
        ax.annotate(
            f"#{int(r.raw_rank)} {_short_name(r.player_name)}",
            (r.pp_share_pct, r.neff_hp),
            xytext=xytext,
            textcoords="data",
            fontsize=7.6,
            fontweight="bold" if r.neff_hp == 0 or r.raw_rank == 1 else "normal",
            arrowprops=arrow,
            va="center",
        )
    ax.text(
        0.98, 0.97,
        "Bubble area ~ Neff\nRed: Neff_HP = 0 (PP-only)\nAmber: Low HP (<45)\nBlue: Battle-Tested HP",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
        bbox=dict(boxstyle="round", facecolor="#F8F9F9", alpha=0.94),
    )
    ax.set_xlabel("Powerplay Ball Share (% of Career Balls)")
    ax.set_ylabel("High-Pressure Effective Sample Size (Neff_HP)")
    ax.set_xlim(-4, 83)
    ax.set_ylim(-15, 295)
    ax.set_title("B. Top-15 Batters: Powerplay Role Concentration vs HP Evidence", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.25)

    # Panel C: HP Weight x Gate Sensitivity Grid
    ax = axes[1, 0]
    g0 = hp_grid[hp_grid["min_neff_hp_gate"] == 0.0].sort_values("w_hp")
    w_pct = (g0["w_hp"] * 100).astype(int)
    ax.plot(w_pct, g0["bat_rho_full_pool"], "o-", color="#1F618D", linewidth=2.2, label="Batting Full-Pool Spearman rho (Gate=0)")
    ax.plot(w_pct, g0["bowl_rho_full_pool"], "s-", color="#117A65", linewidth=2.2, label="Bowling Full-Pool Spearman rho (Gate=0)")
    ax.plot(w_pct, g0["bat_top10_overlap_vs_approved"] / 10.0, "^--", color="#5DADE2", linewidth=1.7, label="Batting Top-10 Overlap / 10 (Gate=0)")
    ax.plot(w_pct, g0["bowl_top10_overlap_vs_approved"] / 10.0, "d--", color="#48C9B0", linewidth=1.7, label="Bowling Top-10 Overlap / 10 (Gate=0)")
    gate_box = (
        f"Alternative PP Pressure (PI>=62, +16,969 balls):\n"
        f"  Bat rho = {audit_json['pp_alt_summary']['spearman_rho_vs_approved']:.4f}, Top-10 = {audit_json['pp_alt_summary']['top10_overlap_vs_approved']}/10\n"
        f"Minimum Neff_HP Gate Impact (w_hp=15%):\n"
        f"  Gate >= 0: 188 Bat (10/10) | 200 Bowl (10/10)\n"
        f"  Gate >=15: 152 Bat (-19%, 8/10) | 176 Bowl (10/10)\n"
        f"  Gate >=30: 124 Bat (-34%, 8/10) | 156 Bowl (10/10)"
    )
    ax.text(
        0.03, 0.25, gate_box, transform=ax.transAxes, fontsize=7.5, va="bottom",
        bbox=dict(boxstyle="round", facecolor="#F4F6F7", alpha=0.95),
    )
    ax.set_xticks([15, 25, 35, 40])
    ax.set_xticklabels(["15%\n(Approved)", "25%", "35%", "40%"])
    ax.set_ylim(0.68, 1.02)
    ax.set_xlabel("HP Component Weight (Remainder Split Proportionally)")
    ax.set_ylabel("Rank Correlation / Top-10 Overlap Ratio")
    ax.set_title("C. HP Weight (15%-40%) & Eligibility Gate Sensitivity", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=7.4, framealpha=0.94)

    # Panel D: Unified Runs/Over Bowling Comparison (Retiring Bowling Pressure Delta)
    ax = axes[1, 1]
    bd15 = bowl_delta.head(15).iloc[::-1].copy()
    y = np.arange(len(bd15))
    h = 0.38
    ax.barh(y + h / 2, bd15["gen_net_po_post"], height=h, color="#1F618D", alpha=0.88, label="General Net Saved (PI < 60, runs/over)")
    ax.barh(y - h / 2, bd15["hp_impact_post"], height=h, color="#D35400", alpha=0.88, label="High-Pressure Impact (PI >= 60, runs/over)")
    ax.axvline(0.0, color="#2C3E50", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{int(r.raw_rank)} {_short_name(r.player_name)}" for r in bd15.itertuples()], fontsize=8.0)
    ax.set_xlabel("Posterior Runs Saved / Impact (Positive = Better, Runs/Over)")
    ax.set_xlim(-1.10, 2.72)
    ax.set_title("D. Unified Runs/Over Scale: General vs HP Bowling Impact (Delta Retired)", fontsize=10.5, fontweight="bold")
    t_note = (
        "Why Bowl Pressure Delta Was Retired:\n"
        "  55.5% negative (HP overs >1.1 runs/ov harder)\n"
        "  Temporal r (2022-24 -> 25-26): -0.2197 (N=131)\n"
        "  vs Absolute HP-Impact r: +0.1220\n"
        "  vs Bat Delta temporal r: +0.2987 (N=97)"
    )
    ax.text(
        0.985, 0.03, t_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4,
        bbox=dict(boxstyle="round", facecolor="#FDEDEC", edgecolor="#C0392B", alpha=0.95),
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.255), fontsize=7.5, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / "37_claude_pressure_explainer_and_hp_exposure.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    ARTIFACT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, ARTIFACT_FIG_DIR / out_path.name)
    return out_path


def make_figure_38_rank_uncertainty() -> Path:
    """Figure 38: B=2,000 Full-Composite Bootstrap Rank Uncertainty Intervals (Top-15 Bat & Bowl)."""
    bat_b = pd.read_csv(DATA_PROCESSED / "claude_bat_bootstrap_ranks.csv").head(15).iloc[::-1].copy()
    bowl_b = pd.read_csv(DATA_PROCESSED / "claude_bowl_bootstrap_ranks.csv").head(15).iloc[::-1].copy()

    tier_colors = {
        "Tier 1A: Proven HP Core Lock": "#117A65",
        "Tier 1B: Low-HP Statistical Lock": "#D68910",
        "Tier 2: Top-10 Contender / Bubble": "#2471A3",
        "Tier 3: Watchlist Fringe": "#7F8C8D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 9.2), dpi=200)
    fig.suptitle(
        "Full-Composite Rank Uncertainty (B = 2,000 Match-Cluster Bootstrap): Point Rank vs [P05, P95] Interval & Top-10 Inclusion",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    panels = [
        (axes[0], bat_b, "A. Top-15 Batters (Unrestricted Ranks 1-15, Active #1-#13)", 162),
        (axes[1], bowl_b, "B. Top-15 Bowlers (Unrestricted Ranks 1-15, Active #1-#14)", 148),
    ]

    for ax, df, title, xlim_hi in panels:
        y = np.arange(len(df))
        ax.axvline(10.5, color="#C0392B", linestyle="--", linewidth=1.3, label="Top-10 Cutoff (Rank <= 10)")
        for yi, r in zip(y, df.itertuples()):
            col = "#566573" if r.is_ipl_retired else tier_colors.get(r.perf_tier, "#2471A3")
            ax.plot([r.rank_p05, r.rank_p95], [yi, yi], color=col, linewidth=3.2, alpha=0.55, zorder=2)
            ax.scatter([r.rank_p05, r.rank_p95], [yi, yi], color=col, marker="|", s=80, linewidth=2.0, zorder=3)
            ax.scatter(r.median_rank, yi, color=col, marker="o", s=58, edgecolors="black", linewidth=0.7, zorder=4)
            ax.scatter(r.raw_rank, yi, color="white", marker="D", s=34, edgecolors="black", linewidth=1.0, zorder=5)
            hp_tag = " | HP=0" if r.neff_hp == 0 else ""
            ax.text(
                r.rank_p95 + 1.8,
                yi,
                f"Inc={r.top10_inc_pct:.1f}% [{int(round(r.rank_p05))}-{int(round(r.rank_p95))}]{hp_tag}",
                va="center",
                fontsize=7.6,
                fontweight="bold" if r.top10_inc_pct >= 75 else "normal",
                color=col,
            )

        ylabels = []
        for r in df.itertuples():
            tag = " [RET]" if r.is_ipl_retired else (f" (Act #{int(r.active_rank)})" if pd.notna(r.active_rank) and int(r.active_rank) != int(r.raw_rank) else "")
            ylabels.append(f"#{int(r.raw_rank)} {_short_name(r.player_name)}{tag}")

        ax.set_yticks(y)
        ax.set_yticklabels(ylabels, fontsize=8.4)
        ax.set_xlim(-1, xlim_hi)
        ax.set_xlabel("Composite Rank Across 2,000 Match-Cluster Bootstrap Replicates (Lower = Better)")
        ax.set_title(title, fontsize=10.8, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)

    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor="black", markersize=7, label="Approved Point Rank"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2471A3", markeredgecolor="black", markersize=7, label="Bootstrap Median Rank"),
        Line2D([0], [0], color="#117A65", lw=3.2, label="Tier 1A: Proven HP Lock (>80% Top-10)"),
        Line2D([0], [0], color="#D68910", lw=3.2, label="Tier 1B: Low-HP Statistical Lock (PP)"),
        Line2D([0], [0], color="#2471A3", lw=3.2, label="Tier 2: Bubble Contender (24%-62%)"),
        Line2D([0], [0], color="#566573", lw=3.2, label="Tier 3 / IPL Retired (Russell / Karthik / Ashwin)"),
        Line2D([0], [0], color="#C0392B", lw=1.3, linestyle="--", label="Top-10 Threshold"),
    ]
    axes[1].legend(handles=handles, loc="upper right", fontsize=7.6, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path = FIG_DIR / "38_claude_rank_uncertainty_intervals.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    ARTIFACT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, ARTIFACT_FIG_DIR / out_path.name)
    return out_path


def main() -> None:
    p37 = make_figure_37_pressure_explainer()
    p38 = make_figure_38_rank_uncertainty()
    print(f"Saved: {p37}")
    print(f"Saved: {p38}")


if __name__ == "__main__":
    main()

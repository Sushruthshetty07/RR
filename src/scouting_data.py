"""
Read-only analytical + external IPL 2027 scouting data service for the
Rajasthan Royals SupeRR Selector web application.

Strictly preserves all approved Phase 3/4 statistical outputs in data/processed/
and layers external IPL 2027 transfer & availability registries from data/external/
at presentation/scouting time only.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

AVAILABILITY_STATUSES = [
    "Confirmed auction-available",
    "Released, auction registration pending/unverified",
    "Contracted to another franchise",
    "Potential trade candidate",
    "Existing Rajasthan Royals player",
    "Retired",
    "Unknown/unverified",
]

EXCLUDED_FROM_ACQUISITION = {
    "Retired",
    "Existing Rajasthan Royals player",
    "Contracted to another franchise",
}


def _clean_float(val: Any, digits: int = 2) -> Optional[float]:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, digits)
    except (TypeError, ValueError):
        return None


def _clean_int(val: Any) -> int:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return 0


def _parse_orig_ids(val: Any) -> List[int]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    out: List[int] = []
    for tok in str(val).replace(",", ";").split(";"):
        tok = tok.strip()
        if tok:
            try:
                out.append(int(float(tok)))
            except ValueError:
                pass
    return out


def _classify_confidence_tier(neff: float, neff_hp: float) -> str:
    if neff >= 400 and neff_hp >= 50:
        return "High Confidence (Full + HP)"
    if neff >= 400 and neff_hp < 15:
        return "High Volume / Low HP Sample"
    if neff >= 150:
        return "Moderate Confidence"
    return "Small Sample / High Uncertainty"


def _classify_pressure_tier(pri_score: float, neff_hp: float) -> str:
    if neff_hp <= 0:
        return "Unproven in HP (Neff_HP = 0)"
    if pri_score >= 90 and neff_hp >= 30:
        return "Elite Proven Under Pressure"
    if pri_score >= 80:
        return "Strong Pressure Performer"
    if pri_score >= 60:
        return "Above-Average / Depth Option"
    return "Below-Threshold / Depth"


def _classify_ipl_exp(ipl_balls: int) -> str:
    if ipl_balls >= 600:
        return "Established IPL Starter (600+)"
    if ipl_balls >= 180:
        return "Moderate IPL Exposure (180-599)"
    if ipl_balls >= 30:
        return "Limited IPL Exposure (30-179)"
    return "Minimal IPL (<30)"


def _classify_bat_role(med_order: Optional[float], wk_apps: int, is_ar: bool) -> str:
    if wk_apps >= 3:
        return "Wicketkeeper-Batter"
    if is_ar:
        return "All-Rounder"
    if med_order is None:
        return "Batter"
    if med_order <= 2.5:
        return "Opener"
    if med_order <= 4.5:
        return "Top / Middle Order (#3-#4)"
    return "Finisher / Lower Middle (#5+)"


def _classify_bowl_role(pace_or_spin: str, death_share: float, pp_share: float, is_ar: bool) -> str:
    base = "Spin" if str(pace_or_spin).lower().startswith("spin") else "Pace"
    if is_ar:
        return f"{base} All-Rounder"
    if death_share >= 32:
        return f"{base} (Death Specialist)"
    if pp_share >= 35:
        return f"{base} (Powerplay Specialist)"
    return f"{base} (Middle / Multi-Phase)"


def _hp_evidence_banner(neff_hp: float) -> Optional[str]:
    if neff_hp <= 0:
        return "High-pressure performance has NOT been demonstrated by the available sample (Neff_HP = 0)."
    if neff_hp < 15:
        return f"Limited high-pressure exposure (Neff_HP = {neff_hp:.1f} < 15); treat HP estimates with caution."
    return None


class ScoutingStore:
    """In-memory read-only store combining frozen Phase 3/4 outputs with IPL 2027 registries."""

    def __init__(self) -> None:
        self.transfers_data = self._load_json(DATA_EXTERNAL / "ipl_2027_transfers.json")
        self.availability_data = self._load_json(DATA_EXTERNAL / "ipl_2027_player_availability.json")
        self.rr_context = self._load_json(DATA_PROCESSED / "phase4_rr_2026_context.json")
        self.claude_summary = self._load_json(DATA_PROCESSED / "claude_audit_summary.json")

        self.rr_roster_df = pd.read_csv(DATA_PROCESSED / "phase4_rr_2026_roster.csv")
        self.rr_ids = set(self.rr_roster_df["canonical_player_id"].astype(int).tolist())

        self.bat_prelim = pd.read_csv(DATA_PROCESSED / "phase3_batting_prelim.csv")
        self.bowl_prelim = pd.read_csv(DATA_PROCESSED / "phase3_bowling_prelim.csv")
        self.bat_elig = (
            self.bat_prelim[self.bat_prelim["elig_primary"] == True]
            .sort_values("pri_score_lcb", ascending=False)
            .reset_index(drop=True)
        )
        self.bowl_elig = (
            self.bowl_prelim[self.bowl_prelim["elig_primary"] == True]
            .sort_values("pri_score_lcb", ascending=False)
            .reset_index(drop=True)
        )

        self.bat_elig_ids = set(self.bat_elig["canonical_player_id"].astype(int).tolist())
        self.bowl_elig_ids = set(self.bowl_elig["canonical_player_id"].astype(int).tolist())
        self.dual_elig_ids = self.bat_elig_ids & self.bowl_elig_ids

        self.bat_boot = pd.read_csv(DATA_PROCESSED / "claude_bat_bootstrap_ranks.csv").set_index("canonical_player_id")
        self.bowl_boot = pd.read_csv(DATA_PROCESSED / "claude_bowl_bootstrap_ranks.csv").set_index("canonical_player_id")
        self.bat_ctx = pd.read_csv(DATA_PROCESSED / "phase4_bat_context_metrics.csv").set_index("canonical_player_id")
        self.bowl_ctx = pd.read_csv(DATA_PROCESSED / "phase4_bowl_context_metrics.csv").set_index("canonical_player_id")
        self.bat_yoy = pd.read_csv(DATA_PROCESSED / "phase4_bat_yoy.csv")
        self.bowl_yoy = pd.read_csv(DATA_PROCESSED / "phase4_bowl_yoy.csv")
        self.bat_cards = pd.read_csv(DATA_PROCESSED / "phase4_bat_benchmark15_cards.csv").set_index("canonical_player_id")
        self.bowl_cards = pd.read_csv(DATA_PROCESSED / "phase4_bowl_benchmark15_cards.csv").set_index("canonical_player_id")
        self.archetype_leaders = pd.read_csv(DATA_PROCESSED / "final_7_archetype_leaders.csv")

        raw_bat_meta, raw_bowl_meta = self._extract_raw_player_metadata()
        self.bat_records = self._build_batter_records(raw_bat_meta)
        self.bowl_records = self._build_bowler_records(raw_bowl_meta)
        self.bat_by_id = {r["canonical_player_id"]: r for r in self.bat_records}
        self.bowl_by_id = {r["canonical_player_id"]: r for r in self.bowl_records}

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _extract_raw_player_metadata(self) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """Fast extraction of role, handedness, phase splits, and recency from data/raw/."""
        orig_to_canon_bat: Dict[int, int] = {}
        for _, row in self.bat_elig.iterrows():
            cid = int(row["canonical_player_id"])
            for oid in _parse_orig_ids(row.get("original_player_ids")):
                orig_to_canon_bat[oid] = cid
            orig_to_canon_bat.setdefault(cid, cid)

        orig_to_canon_bowl: Dict[int, int] = {}
        for _, row in self.bowl_elig.iterrows():
            cid = int(row["canonical_player_id"])
            for oid in _parse_orig_ids(row.get("original_player_ids")):
                orig_to_canon_bowl[oid] = cid
            orig_to_canon_bowl.setdefault(cid, cid)

        matches = pd.read_csv(DATA_RAW / "matches.csv", usecols=["match_id", "comp_id", "match_date"])
        comps = pd.read_csv(DATA_RAW / "competitions.csv", usecols=["comp_id", "comp_name", "season"])
        matches = matches.merge(comps, on="comp_id", how="left")
        matches["is_ipl"] = matches["comp_name"].astype(str).str.contains("Indian Premier League", case=False, na=False)
        matches["is_ipl_2026"] = matches["is_ipl"] & (matches["season"].astype(str) == "2026")

        bbb = pd.read_csv(
            DATA_RAW / "ball_by_ball.csv",
            usecols=[
                "match_id", "overid", "striker_id", "bowler_id",
                "batting_type", "bowling_type", "pace_or_spin", "bowl_speed",
                "runs", "is_wicket", "extras_type", "extras1_runs",
            ],
            low_memory=False,
        ).merge(matches[["match_id", "match_date", "is_ipl", "is_ipl_2026"]], on="match_id", how="left")

        bbb["phase"] = np.where(bbb["overid"] <= 6, "PP", np.where(bbb["overid"] <= 15, "Mid", "Death"))
        ex = bbb["extras_type"].astype(str).str.lower()
        bbb["is_wide"] = ex.str.contains("wide", na=False)
        bbb["is_noball"] = ex.str.contains("no", na=False)
        bbb["is_legal"] = (~bbb["is_wide"]) & (~bbb["is_noball"])
        bbb["is_bat_ball"] = ~bbb["is_wide"]

        pe = pd.read_csv(DATA_RAW / "playing_eleven.csv", usecols=["match_id", "player_id", "batting_order", "is_wk", "is_played"])
        pe = pe[pe["is_played"] == 1].copy()
        pe["canon_bat"] = pe["player_id"].map(orig_to_canon_bat)
        pe_bat = pe.dropna(subset=["canon_bat"]).copy()
        pe_bat["canon_bat"] = pe_bat["canon_bat"].astype(int)
        pe_agg = pe_bat.groupby("canon_bat").agg(
            med_bat_order=("batting_order", "median"),
            wk_apps=("is_wk", "sum"),
        )

        # Batting aggregation
        bbb_bat = bbb[bbb["striker_id"].isin(orig_to_canon_bat) & bbb["is_bat_ball"]].copy()
        bbb_bat["cid"] = bbb_bat["striker_id"].map(orig_to_canon_bat).astype(int)
        bat_meta: Dict[int, Dict[str, Any]] = {}
        for cid, g in bbb_bat.groupby("cid"):
            hand_mode = g["batting_type"].dropna().mode()
            hand = str(hand_mode.iloc[0]) if len(hand_mode) else "Unknown"
            if "left" in hand.lower():
                hand_short = "LHB"
            elif "right" in hand.lower():
                hand_short = "RHB"
            else:
                hand_short = hand
            tot_b = len(g)
            tot_r = int(g["runs"].fillna(0).sum())
            pp = g[g["phase"] == "PP"]
            mid = g[g["phase"] == "Mid"]
            dth = g[g["phase"] == "Death"]
            ipl_g = g[g["is_ipl"] == True]
            pe_row = pe_agg.loc[cid] if cid in pe_agg.index else None
            bat_meta[int(cid)] = {
                "batting_hand": hand_short,
                "med_bat_order": _clean_float(pe_row["med_bat_order"], 1) if pe_row is not None else None,
                "wk_apps": int(pe_row["wk_apps"]) if pe_row is not None else 0,
                "last_match_date": str(g["match_date"].max()),
                "last_ipl_date": str(ipl_g["match_date"].max()) if len(ipl_g) else None,
                "ipl_2026_balls": int((g["is_ipl_2026"] == True).sum()),
                "runs_scored": tot_r,
                "career_sr": round(100.0 * tot_r / tot_b, 1) if tot_b else None,
                "powerplay_balls": len(pp),
                "powerplay_share_pct": round(100.0 * len(pp) / tot_b, 1) if tot_b else 0.0,
                "powerplay_sr": round(100.0 * pp["runs"].sum() / len(pp), 1) if len(pp) else None,
                "middle_balls": len(mid),
                "middle_share_pct": round(100.0 * len(mid) / tot_b, 1) if tot_b else 0.0,
                "middle_sr": round(100.0 * mid["runs"].sum() / len(mid), 1) if len(mid) else None,
                "death_balls": len(dth),
                "death_share_pct": round(100.0 * len(dth) / tot_b, 1) if tot_b else 0.0,
                "death_sr": round(100.0 * dth["runs"].sum() / len(dth), 1) if len(dth) else None,
            }

        # Bowling aggregation
        bbb_bowl = bbb[bbb["bowler_id"].isin(orig_to_canon_bowl)].copy()
        bbb_bowl["cid"] = bbb_bowl["bowler_id"].map(orig_to_canon_bowl).astype(int)
        bbb_bowl["bowler_runs"] = bbb_bowl["runs"].fillna(0) + np.where(
            bbb_bowl["is_wide"] | bbb_bowl["is_noball"], bbb_bowl["extras1_runs"].fillna(0), 0
        )
        bowl_meta: Dict[int, Dict[str, Any]] = {}
        for cid, g in bbb_bowl.groupby("cid"):
            leg = g[g["is_legal"]]
            tot_l = len(leg)
            ps_mode = g["pace_or_spin"].dropna().mode()
            bt_mode = g["bowling_type"].dropna().mode()
            pace_spin = str(ps_mode.iloc[0]) if len(ps_mode) else "Unknown"
            subtype = str(bt_mode.iloc[0]) if len(bt_mode) else "Unknown"
            speeds = g["bowl_speed"].dropna()
            speeds = speeds[(speeds >= 60) & (speeds <= 165)]
            mean_spd = round(float(speeds.mean()), 1) if len(speeds) >= 10 else None
            wkts = int(g["is_wicket"].fillna(0).sum())
            runs_conc = float(g["bowler_runs"].sum())

            def _phase_bowl(sub: pd.DataFrame) -> Tuple[int, Optional[float]]:
                lb = int(sub["is_legal"].sum())
                ec = round(6.0 * sub["bowler_runs"].sum() / lb, 2) if lb > 0 else None
                return lb, ec

            pp_b, pp_ec = _phase_bowl(g[g["phase"] == "PP"])
            mid_b, mid_ec = _phase_bowl(g[g["phase"] == "Mid"])
            dth_b, dth_ec = _phase_bowl(g[g["phase"] == "Death"])
            ipl_g = g[g["is_ipl"] == True]
            bowl_meta[int(cid)] = {
                "pace_or_spin": pace_spin,
                "bowling_subtype": subtype,
                "mean_speed_kph": mean_spd,
                "last_match_date": str(g["match_date"].max()),
                "last_ipl_date": str(ipl_g["match_date"].max()) if len(ipl_g) else None,
                "ipl_2026_balls": int((leg["is_ipl_2026"] == True).sum()),
                "wkts_taken": wkts,
                "career_econ": round(6.0 * runs_conc / tot_l, 2) if tot_l else None,
                "powerplay_balls": pp_b,
                "powerplay_share_pct": round(100.0 * pp_b / tot_l, 1) if tot_l else 0.0,
                "powerplay_econ": pp_ec,
                "middle_balls": mid_b,
                "middle_share_pct": round(100.0 * mid_b / tot_l, 1) if tot_l else 0.0,
                "middle_econ": mid_ec,
                "death_balls": dth_b,
                "death_share_pct": round(100.0 * dth_b / tot_l, 1) if tot_l else 0.0,
                "death_econ": dth_ec,
            }

        return bat_meta, bowl_meta

    def _get_availability(self, cid: int, player_name: str, teams_str: str) -> Dict[str, Any]:
        reg = self.availability_data.get("players", {}).get(str(cid))
        if reg:
            return {
                "availability_status": reg["availability_status"],
                "ipl_2026_team": reg.get("ipl_2026_team", "Unknown"),
                "current_2027_team": reg.get("current_2027_team", "Unverified"),
                "trade_note": reg.get("trade_note"),
                "acquisition_pathway": reg.get("acquisition_pathway", ""),
                "last_verified_date": reg.get("last_verified_date", "2026-11-12"),
                "source_name": reg.get("source_name", "External Registry"),
                "source_url": reg.get("source_url", ""),
            }
        if cid in self.rr_ids:
            return {
                "availability_status": "Existing Rajasthan Royals player",
                "ipl_2026_team": "Rajasthan Royals",
                "current_2027_team": "Rajasthan Royals",
                "trade_note": None,
                "acquisition_pathway": "Excluded — Existing RR squad",
                "last_verified_date": "2026-11-12",
                "source_name": "Phase 4 RR 2026 Roster",
                "source_url": "https://www.iplt20.com/teams/rajasthan-royals",
            }
        return {
            "availability_status": "Unknown/unverified",
            "ipl_2026_team": "Unverified (check pre-Nov 17 lists)",
            "current_2027_team": "Unverified",
            "trade_note": None,
            "acquisition_pathway": "Await Nov 17 retention & Nov 23 auction registration",
            "last_verified_date": "2026-11-12",
            "source_name": "Pending official BCCI IPL 2027 list",
            "source_url": "https://www.iplt20.com/",
        }

    def _build_batter_records(self, raw_meta: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        n_total = len(self.bat_elig)
        for idx, row in self.bat_elig.iterrows():
            cid = int(row["canonical_player_id"])
            rank = idx + 1
            emp_pct = round(100.0 * (n_total - rank + 1) / n_total, 1)
            meta = raw_meta.get(cid, {})
            boot = self.bat_boot.loc[cid] if cid in self.bat_boot.index else None
            ctx = self.bat_ctx.loc[cid] if cid in self.bat_ctx.index else None
            card = self.bat_cards.loc[cid] if cid in self.bat_cards.index else None
            avail = self._get_availability(cid, str(row["player_name"]), str(row.get("teams", "")))
            is_ar = cid in self.dual_elig_ids
            neff = float(row["neff"])
            neff_hp = float(row["neff_hp"])
            pri_lcb = float(row["pri_score_lcb"])
            role_label = _classify_bat_role(meta.get("med_bat_order"), meta.get("wk_apps", 0), is_ar)
            ipl_balls = int(row["ipl_balls"])
            records.append({
                "canonical_player_id": cid,
                "player_name": str(row["player_name"]),
                "discipline": "Batting",
                "is_all_rounder": is_ar,
                "raw_pri_lcb_rank": rank,
                "emp_pct_elig_primary": emp_pct,
                "pri_score_lcb": round(pri_lcb, 2),
                "pri_score_post": round(float(row["pri_score_post"]), 2),
                "neff": round(neff, 1),
                "neff_gen": round(float(row["neff_gen"]), 1),
                "neff_hp": round(neff_hp, 1),
                "hp_banner": _hp_evidence_banner(neff_hp),
                "n_matches": int(row["n_matches"]),
                "ipl_balls": ipl_balls,
                "t20i_balls": int(row["t20i_balls"]),
                "hp60_balls": int(row["hp60_balls"]),
                "ipl_hp60_balls": int(row["ipl_hp60_balls"]),
                "ipl_2026_balls": meta.get("ipl_2026_balls", 0),
                "last_match_date": meta.get("last_match_date"),
                "last_ipl_date": meta.get("last_ipl_date"),
                # Components & CI
                "raa_post": _clean_float(row["raa_post"], 2),
                "raa_lcb": _clean_float(row["raa_lcb"], 2),
                "surv_post": _clean_float(row["surv_post"], 2),
                "surv_lcb": _clean_float(row["surv_lcb"], 2),
                "cons_post": _clean_float(row["cons_post"], 2),
                "hp_raa_post": _clean_float(row["hp_raa_post"], 2),
                "hp_raa_lcb": _clean_float(row["hp_raa_lcb"], 2),
                "ev_all_post": _clean_float(row["ev_all_post"], 2),
                "ev_all_lcb": _clean_float(row["ev_all_lcb"], 2),
                "pri_p05": _clean_float(boot["pri_p05"], 2) if boot is not None else None,
                "pri_p95": _clean_float(boot["pri_p95"], 2) if boot is not None else None,
                "rank_p05": _clean_int(boot["rank_p05"]) if boot is not None else None,
                "rank_p95": _clean_int(boot["rank_p95"]) if boot is not None else None,
                "top10_inc_pct": _clean_float(boot["top10_inc_pct"], 1) if boot is not None else None,
                # Role & Context
                "role_label": role_label,
                "batting_hand": meta.get("batting_hand", "Unknown"),
                "med_bat_order": meta.get("med_bat_order"),
                "wk_apps": meta.get("wk_apps", 0),
                "runs_scored": meta.get("runs_scored", 0),
                "career_sr": meta.get("career_sr"),
                "true_sr": _clean_float(ctx["true_sr"], 1) if ctx is not None else None,
                "ipl_true_sr": _clean_float(ctx["ipl_true_sr"], 1) if ctx is not None else None,
                "t20i_top8_true_sr": _clean_float(ctx["t20i_top8_true_sr"], 1) if ctx is not None else None,
                "powerplay_balls": meta.get("powerplay_balls", 0),
                "powerplay_share_pct": meta.get("powerplay_share_pct", 0.0),
                "powerplay_sr": meta.get("powerplay_sr"),
                "middle_balls": meta.get("middle_balls", 0),
                "middle_share_pct": meta.get("middle_share_pct", 0.0),
                "middle_sr": meta.get("middle_sr"),
                "death_balls": meta.get("death_balls", 0),
                "death_share_pct": meta.get("death_share_pct", 0.0),
                "death_sr": meta.get("death_sr"),
                # Classifications & Filters
                "confidence_tier": _classify_confidence_tier(neff, neff_hp),
                "pressure_tier": _classify_pressure_tier(pri_lcb, neff_hp),
                "ipl_exp_tier": _classify_ipl_exp(ipl_balls),
                "nationality": str(card["nationality"]) if card is not None and pd.notna(card.get("nationality")) else "Unverified",
                "cap_status": str(card["cap_status"]) if card is not None and pd.notna(card.get("cap_status")) else "Unverified",
                "age_2026": str(card["age_2026"]) if card is not None and pd.notna(card.get("age_2026")) else "Unverified",
                **avail,
            })
        return records

    def _build_bowler_records(self, raw_meta: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        n_total = len(self.bowl_elig)
        for idx, row in self.bowl_elig.iterrows():
            cid = int(row["canonical_player_id"])
            rank = idx + 1
            emp_pct = round(100.0 * (n_total - rank + 1) / n_total, 1)
            meta = raw_meta.get(cid, {})
            boot = self.bowl_boot.loc[cid] if cid in self.bowl_boot.index else None
            ctx = self.bowl_ctx.loc[cid] if cid in self.bowl_ctx.index else None
            card = self.bowl_cards.loc[cid] if cid in self.bowl_cards.index else None
            avail = self._get_availability(cid, str(row["player_name"]), str(row.get("teams", "")))
            is_ar = cid in self.dual_elig_ids
            neff = float(row["neff"])
            neff_hp = float(row["neff_hp"])
            pri_lcb = float(row["pri_score_lcb"])
            role_label = _classify_bowl_role(
                meta.get("pace_or_spin", "Pace"),
                meta.get("death_share_pct", 0.0),
                meta.get("powerplay_share_pct", 0.0),
                is_ar,
            )
            ipl_balls = int(row["ipl_legal_balls"])
            records.append({
                "canonical_player_id": cid,
                "player_name": str(row["player_name"]),
                "discipline": "Bowling",
                "is_all_rounder": is_ar,
                "raw_pri_lcb_rank": rank,
                "emp_pct_elig_primary": emp_pct,
                "pri_score_lcb": round(pri_lcb, 2),
                "pri_score_post": round(float(row["pri_score_post"]), 2),
                "neff": round(neff, 1),
                "neff_gen": round(float(row["neff_gen"]), 1),
                "neff_hp": round(neff_hp, 1),
                "hp_banner": _hp_evidence_banner(neff_hp),
                "n_matches": int(row["n_matches"]),
                "ipl_balls": ipl_balls,
                "t20i_balls": int(row["t20i_legal_balls"]),
                "hp60_balls": int(row["hp60_legal_balls"]),
                "ipl_hp60_balls": int(row["ipl_hp60_legal_balls"]),
                "ipl_2026_balls": meta.get("ipl_2026_balls", 0),
                "last_match_date": meta.get("last_match_date"),
                "last_ipl_date": meta.get("last_ipl_date"),
                # Unified runs/over components & CI
                "rsaa_post": _clean_float(row["rsaa_post"], 2),
                "rsaa_lcb": _clean_float(row["rsaa_lcb"], 2),
                "waa_post": _clean_float(row["waa_post"], 2),
                "waa_lcb": _clean_float(row["waa_lcb"], 2),
                "cons_post": _clean_float(row["cons_post"], 2),
                "hp_impact_post": _clean_float(row["hp_impact_post"], 2),
                "hp_impact_lcb": _clean_float(row["hp_impact_lcb"], 2),
                "ev_all_post": _clean_float(row["ev_all_post"], 2),
                "ev_all_lcb": _clean_float(row["ev_all_lcb"], 2),
                "pri_p05": _clean_float(boot["pri_p05"], 2) if boot is not None else None,
                "pri_p95": _clean_float(boot["pri_p95"], 2) if boot is not None else None,
                "rank_p05": _clean_int(boot["rank_p05"]) if boot is not None else None,
                "rank_p95": _clean_int(boot["rank_p95"]) if boot is not None else None,
                "top10_inc_pct": _clean_float(boot["top10_inc_pct"], 1) if boot is not None else None,
                # Role & Context
                "role_label": role_label,
                "pace_or_spin": meta.get("pace_or_spin", "Unknown"),
                "bowling_subtype": meta.get("bowling_subtype", "Unknown"),
                "mean_speed_kph": meta.get("mean_speed_kph"),
                "wkts_taken": meta.get("wkts_taken", 0),
                "career_econ": meta.get("career_econ"),
                "true_econ": _clean_float(ctx["true_econ"], 2) if ctx is not None else None,
                "ipl_true_econ": _clean_float(ctx["ipl_true_econ"], 2) if ctx is not None else None,
                "t20i_top8_true_econ": _clean_float(ctx["t20i_top8_true_econ"], 2) if ctx is not None else None,
                "powerplay_balls": meta.get("powerplay_balls", 0),
                "powerplay_share_pct": meta.get("powerplay_share_pct", 0.0),
                "powerplay_econ": meta.get("powerplay_econ"),
                "middle_balls": meta.get("middle_balls", 0),
                "middle_share_pct": meta.get("middle_share_pct", 0.0),
                "middle_econ": meta.get("middle_econ"),
                "death_balls": meta.get("death_balls", 0),
                "death_share_pct": meta.get("death_share_pct", 0.0),
                "death_econ": meta.get("death_econ"),
                # Classifications & Filters
                "confidence_tier": _classify_confidence_tier(neff, neff_hp),
                "pressure_tier": _classify_pressure_tier(pri_lcb, neff_hp),
                "ipl_exp_tier": _classify_ipl_exp(ipl_balls),
                "nationality": str(card["nationality"]) if card is not None and pd.notna(card.get("nationality")) else "Unverified",
                "cap_status": str(card["cap_status"]) if card is not None and pd.notna(card.get("cap_status")) else "Unverified",
                "age_2026": str(card["age_2026"]) if card is not None and pd.notna(card.get("age_2026")) else "Unverified",
                **avail,
            })
        return records

    @staticmethod
    def _apply_filters(
        records: List[Dict[str, Any]],
        search: str = "",
        role: str = "",
        availability: str = "",
        ipl_exp: str = "",
        pressure_tier: str = "",
        confidence_tier: str = "",
        all_rounder_only: bool = False,
    ) -> List[Dict[str, Any]]:
        out = records
        if search:
            q = search.strip().lower()
            out = [
                r for r in out
                if q in r["player_name"].lower()
                or q in str(r.get("current_2027_team", "")).lower()
                or q in str(r.get("role_label", "")).lower()
            ]
        if role:
            out = [r for r in out if role.lower() in r.get("role_label", "").lower()]
        if availability:
            out = [r for r in out if r.get("availability_status") == availability]
        if ipl_exp:
            out = [r for r in out if r.get("ipl_exp_tier") == ipl_exp]
        if pressure_tier:
            out = [r for r in out if r.get("pressure_tier") == pressure_tier]
        if confidence_tier:
            out = [r for r in out if r.get("confidence_tier") == confidence_tier]
        if all_rounder_only:
            out = [r for r in out if r.get("is_all_rounder")]
        return out

    def get_dashboard_a(self, discipline: str = "batting", **filters: Any) -> Dict[str, Any]:
        """Complete unrestricted analytical universe (188 batters, 200 bowlers)."""
        base = self.bat_records if discipline.lower().startswith("bat") else self.bowl_records
        filtered = self._apply_filters(base, **filters)
        return {
            "view": "Dashboard A: Performance Analytics (Complete Universe)",
            "discipline": "Batting" if discipline.lower().startswith("bat") else "Bowling",
            "total_eligible": len(base),
            "filtered_count": len(filtered),
            "records": filtered,
        }

    def get_dashboard_b(self, discipline: str = "all", **filters: Any) -> Dict[str, Any]:
        """
        Primary RR Auction Scouting view with strict bucket separation:
        - confirmed_auction (0 today before Nov 17/23)
        - pending_registration (released/uncontracted awaiting Nov 23)
        - trade_watchlist (separate watchlist, NEVER mixed into ordinary auction shortlist)
        - rr_squad (excluded from acquisition targets)
        - contracted_elsewhere (excluded from auction shortlist)
        - retired (excluded from acquisition targets)
        - unknown_unverified
        """
        if discipline.lower().startswith("bat"):
            pool = list(self.bat_records)
        elif discipline.lower().startswith("bowl"):
            pool = list(self.bowl_records)
        else:
            # Combined pool keeping highest PRI discipline entry if dual-eligible, or both tagged
            seen = set()
            pool = []
            for r in sorted(self.bat_records + self.bowl_records, key=lambda x: x["pri_score_lcb"], reverse=True):
                key = (r["canonical_player_id"], r["discipline"])
                if key not in seen:
                    seen.add(key)
                    pool.append(r)

        filtered = self._apply_filters(pool, **filters)
        buckets: Dict[str, List[Dict[str, Any]]] = {
            "confirmed_auction": [],
            "pending_registration": [],
            "trade_watchlist": [],
            "rr_squad": [],
            "contracted_elsewhere": [],
            "retired": [],
            "unknown_unverified": [],
        }
        for r in filtered:
            st = r["availability_status"]
            if st == "Confirmed auction-available":
                buckets["confirmed_auction"].append(r)
            elif st == "Released, auction registration pending/unverified":
                buckets["pending_registration"].append(r)
            elif st == "Potential trade candidate":
                buckets["trade_watchlist"].append(r)
            elif st == "Existing Rajasthan Royals player":
                buckets["rr_squad"].append(r)
            elif st == "Contracted to another franchise":
                buckets["contracted_elsewhere"].append(r)
            elif st == "Retired":
                buckets["retired"].append(r)
            else:
                buckets["unknown_unverified"].append(r)

        return {
            "view": "Dashboard B: RR Auction Scouting (PRIMARY)",
            "calendar": self.transfers_data["calendar"],
            "counts": {k: len(v) for k, v in buckets.items()},
            "buckets": buckets,
        }

    def get_rr_squad(self) -> Dict[str, Any]:
        """Verified 2026 RR 26-row roster + phase performance gaps for 2027 scouting."""
        rows = []
        for _, r in self.rr_roster_df.iterrows():
            cid = int(r["canonical_player_id"])
            bat = self.bat_by_id.get(cid)
            bowl = self.bowl_by_id.get(cid)
            rows.append({
                "canonical_player_id": cid,
                "canonical_name": str(r["canonical_name"]),
                "apps_2026": int(r["apps_2026"]),
                "wk_apps_2026": int(r["wk_apps_2026"]),
                "capt_apps_2026": int(r["capt_apps_2026"]),
                "med_bat_order": _clean_float(r["med_bat_order"], 1),
                "bat_pri_lcb": bat["pri_score_lcb"] if bat else None,
                "bat_rank": bat["raw_pri_lcb_rank"] if bat else None,
                "bat_neff_hp": bat["neff_hp"] if bat else None,
                "bowl_pri_lcb": bowl["pri_score_lcb"] if bowl else None,
                "bowl_rank": bowl["raw_pri_lcb_rank"] if bowl else None,
                "bowl_neff_hp": bowl["neff_hp"] if bowl else None,
                "availability_status": "Existing Rajasthan Royals player",
            })
        return {
            "roster": rows,
            "rr_2026_context": self.rr_context.get("rr_2026_context", {}),
            "tactical_gaps": [
                {
                    "gap": "Death-Overs Batting SR & Wicket Preservation",
                    "evidence": "RR 2026 Death SR 163.4 vs League 169.4; Death wicket% 8.8% vs League 7.4%.",
                    "target_profile": "High-pressure #5-#7 finishers (Neff_HP >= 30).",
                },
                {
                    "gap": "Middle-Overs Spin / Wrist-Spin HP Control",
                    "evidence": "RR 2026 Middle bowling econ 9.43 vs League 9.22; Hasaranga released pre-2026, Theekshana unsold.",
                    "target_profile": "Proven middle-over / HP spin controllers (Zampa, Rashid, Theekshana pending; Hasaranga, Santner, Brar trade watchlist).",
                },
                {
                    "gap": "Powerplay & Death Pace Depth",
                    "evidence": "RR 2026 Pace econ 10.21 vs League 9.97; Powerplay econ 10.15 vs League 9.95.",
                    "target_profile": "Multi-phase or death pace specialists (Mustafizur, Mark Wood, Little pending; Sakib Hussain, Ngidi, Chahar trade watchlist).",
                },
            ],
        }

    def get_trades_overview(self) -> Dict[str, Any]:
        return self.transfers_data

    def get_player_detail(self, cid: int) -> Optional[Dict[str, Any]]:
        bat = self.bat_by_id.get(cid)
        bowl = self.bowl_by_id.get(cid)
        if not bat and not bowl:
            return None
        primary = bat or bowl
        bat_yoy = (
            self.bat_yoy[self.bat_yoy["canonical_player_id"] == cid]
            .sort_values("season_yr")
            .to_dict(orient="records")
        )
        bowl_yoy = (
            self.bowl_yoy[self.bowl_yoy["canonical_player_id"] == cid]
            .sort_values("season_yr")
            .to_dict(orient="records")
        )
        return {
            "canonical_player_id": cid,
            "player_name": primary["player_name"],
            "availability_status": primary["availability_status"],
            "ipl_2026_team": primary["ipl_2026_team"],
            "current_2027_team": primary["current_2027_team"],
            "trade_note": primary.get("trade_note"),
            "acquisition_pathway": primary["acquisition_pathway"],
            "last_verified_date": primary["last_verified_date"],
            "source_name": primary["source_name"],
            "source_url": primary["source_url"],
            "nationality": primary["nationality"],
            "cap_status": primary["cap_status"],
            "age_2026": primary["age_2026"],
            "is_all_rounder": primary["is_all_rounder"],
            "batting": bat,
            "bowling": bowl,
            "batting_yoy": bat_yoy,
            "bowling_yoy": bowl_yoy,
        }

    def compare_players(self, cids: List[int]) -> List[Dict[str, Any]]:
        out = []
        for cid in cids[:4]:
            detail = self.get_player_detail(cid)
            if detail:
                out.append(detail)
        return out

    def get_overview(self) -> Dict[str, Any]:
        dash_b = self.get_dashboard_b()
        return {
            "as_of_date": self.transfers_data["metadata"]["as_of_date"],
            "calendar": self.transfers_data["calendar"],
            "universe_counts": {
                "eligible_batters": len(self.bat_records),
                "eligible_bowlers": len(self.bowl_records),
                "dual_eligible_all_rounders": len(self.dual_elig_ids),
                "primary_deliveries": self.claude_summary.get("accounting", {}).get("n_primary_reg_no_emg_rows", 193144),
            },
            "dashboard_b_counts": dash_b["counts"],
            "completed_trades_count": len(self.transfers_data["official_completed_trades"]),
            "rumours_count": len(self.transfers_data["reported_negotiations_not_completed"]),
            "top5_pending_pool": sorted(
                dash_b["buckets"]["pending_registration"],
                key=lambda x: x["pri_score_lcb"],
                reverse=True,
            )[:6],
            "top5_trade_watchlist": sorted(
                dash_b["buckets"]["trade_watchlist"],
                key=lambda x: x["pri_score_lcb"],
                reverse=True,
            )[:6],
            "top5_batters_universe": self.bat_records[:5],
            "top5_bowlers_universe": self.bowl_records[:5],
            "archetype_leaders": self.archetype_leaders.to_dict(orient="records"),
        }

    def get_methodology(self) -> Dict[str, Any]:
        return {
            "accounting": self.claude_summary.get("accounting", {}),
            "competition_weights": {
                "IPL": 1.00,
                "Senior_T20I": 0.50,
                "Emerging_Rising_T20": 0.00,
            },
            "eligibility_gates": {
                "min_neff": 30,
                "min_ipl_balls": 30,
                "eligible_batters": len(self.bat_records),
                "eligible_bowlers": len(self.bowl_records),
            },
            "pressure_framework": {
                "gamma": 2.0,
                "hp_threshold": "PI >= 60",
                "general_partition": "PI < 60",
                "pri_interpretation": "PRI_lcb is a bounded 0-100 empirical composite of Bayesian LCB components; it is NOT a calibrated probability.",
                "bowling_units": "Unified runs/over scale across RSAA, WAA, and HP-Impact (deprecated Pressure Delta removed from primary UI).",
                "hp_zero_rule": "Players with Neff_HP = 0 are explicitly flagged as unproven in high-pressure situations.",
            },
            "data_availability_governance": self.transfers_data["calendar"],
        }


@lru_cache(maxsize=1)
def get_store() -> ScoutingStore:
    return ScoutingStore()

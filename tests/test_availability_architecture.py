"""
Unit and regression tests for Layer 2 IPL 2027 Player Availability Architecture.
Validates:
1. Canonical 1,216 player universe integrity & exclusivity
2. Separate availability_status and auction_registration_status taxonomies
3. Hard-exclusion of CONFIRMED_OTHER_TEAM (e.g. Nathan Ellis) and RETIRED_UNAVAILABLE from RR targets
4. Duplicate name resolution via distinct canonical player IDs
5. Retention drop and final auction register transition rules
6. Golden CSV checksum integrity for Layer 1 deliverables
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import pytest
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.scouting_data import (
    get_store,
    _derive_acquisition_pathway,
)

EXTERNAL_DIR = BASE_DIR / "data" / "external"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
AVAILABILITY_FILE = EXTERNAL_DIR / "ipl_2027_player_availability.json"
CANONICAL_PLAYERS_FILE = PROCESSED_DIR / "canonical_player_directory.csv"

ALLOWED_AVAILABILITY_STATUSES = {
    "CONFIRMED_OTHER_TEAM",
    "RR_2026_SQUAD",
    "RR_CONFIRMED_RETAINED",
    "PENDING_RETENTION",
    "CONFIRMED_RELEASED",
    "RETIRED_UNAVAILABLE",
    "UNVERIFIED",
}

ALLOWED_REGISTRATION_STATUSES = {
    "NOT_APPLICABLE",
    "PENDING",
    "REGISTERED",
    "NOT_REGISTERED",
    "WITHDRAWN",
    "NOT_IN_FINAL_REGISTER",
}

EXPECTED_CSV_HASHES = {
    "official_top10_high_pressure_batters.csv": {
        "crlf_raw": "A15F8064BA7235907E91CE1A50F4B7C7",
        "canonical_lf": "576930664161ED9ABC01D8B75B1B3754",
    },
    "official_top10_high_pressure_bowlers.csv": {
        "crlf_raw": "FFC8B156D8F01D77157B397C0371F397",
        "canonical_lf": "108AD7AF9E0D617F0DB668FB1E76937D",
    },
    "official_top10_overall_reliability_batters.csv": {
        "crlf_raw": "22A073F5A42CB21C3530436824F5E96F",
        "canonical_lf": "6D61BEF2963AE1D9223AAD7809BA25CF",
    },
    "official_top10_overall_reliability_bowlers.csv": {
        "crlf_raw": "5EAB5D69240D0EA2981463557E9A3440",
        "canonical_lf": "AFA6579BBD4B6D6AAEFFA432C18863E7",
    },
}


def test_availability_registry_completeness():
    """Verify that all 1,216 canonical players have exact, valid availability records."""
    assert AVAILABILITY_FILE.exists(), f"Missing {AVAILABILITY_FILE}"
    with open(AVAILABILITY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["metadata"]["summary_counts"]
    players = data["players"]

    assert summary["total_canonical_players"] == 1216
    assert len(players) == 1216

    # Verify canonical players CSV matches
    df_can = pd.read_csv(CANONICAL_PLAYERS_FILE)
    canonical_ids = set(df_can["canonical_player_id"].unique())
    registry_ids = {int(k) for k in players.keys()}

    assert canonical_ids == registry_ids
    assert len(registry_ids) == 1216


def test_status_taxonomies_and_exclusivity():
    """Verify that statuses are mutually exclusive and conform to the approved taxonomy."""
    with open(AVAILABILITY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    status_counts = {}
    reg_counts = {}

    for p in data["players"].values():
        av_stat = p["availability_status"]
        reg_stat = p["auction_registration_status"]

        assert av_stat in ALLOWED_AVAILABILITY_STATUSES, f"Invalid av_status: {av_stat} for {p['player_name']}"
        assert reg_stat in ALLOWED_REGISTRATION_STATUSES, f"Invalid reg_stat: {reg_stat} for {p['player_name']}"

        status_counts[av_stat] = status_counts.get(av_stat, 0) + 1
        reg_counts[reg_stat] = reg_counts.get(reg_stat, 0) + 1

    # Exact expected counts for current baseline state
    assert status_counts["CONFIRMED_OTHER_TEAM"] == 10
    assert status_counts["RR_2026_SQUAD"] == 26
    assert status_counts.get("RR_CONFIRMED_RETAINED", 0) == 0
    assert status_counts["PENDING_RETENTION"] == 231
    assert status_counts.get("CONFIRMED_RELEASED", 0) == 0
    assert status_counts["RETIRED_UNAVAILABLE"] == 10
    assert status_counts["UNVERIFIED"] == 939
    assert sum(status_counts.values()) == 1216

    # Registration statuses: exactly 20 NOT_APPLICABLE (10 traded + 10 retired), 1196 PENDING
    assert reg_counts["NOT_APPLICABLE"] == 20
    assert reg_counts["PENDING"] == 1196
    assert sum(reg_counts.values()) == 1216


def test_hard_exclusion_guardrails_for_other_teams_and_retired():
    """
    Test that CONFIRMED_OTHER_TEAM and RETIRED_UNAVAILABLE players are strictly
    hard-excluded from Rajasthan Royals actionable auction shortlists.
    """
    store = get_store()
    intel = store.get_rr_auction_intelligence()
    target_names = set()
    for g in intel["identified_roster_gaps"]:
        for cand in g["target_candidates"]:
            target_names.add(cand["name"])

    # 1. Nathan Ellis traded to SRH must NOT be an RR target
    assert "Nathan Ellis" not in target_names

    # 2. Hardik Pandya, Arshdeep Singh, Rishabh Pant, Kuldeep Yadav must NOT be RR targets
    traded_players = [
        "Hardik Pandya", "Arshdeep Singh", "Rishabh Pant", "Kuldeep Yadav",
        "T. Natarajan", "Zeeshan Ansari", "Liam Livingstone", "Jack Edwards",
        "Khaleel Ahmed", "Nathan Ellis"
    ]
    for tp in traded_players:
        assert tp not in target_names, f"{tp} must be excluded from RR targets"

    # 3. Retired players must NOT be RR targets
    retired_players = [
        "Dinesh Karthik", "Andre Russell", "Ravichandran Ashwin", "Piyush Chawla",
        "Shikhar Dhawan", "Dwayne Bravo", "Ambati Rayudu", "Wriddhiman Saha",
        "Kedar Jadhav", "Robin Uthappa"
    ]
    for rp in retired_players:
        assert rp not in target_names, f"{rp} must be excluded from RR targets"

    # 4. Verify Dashboard B exclusion counts
    db = store.get_dashboard_b()
    assert db["counts"]["excluded_other_franchises"] == 10
    assert db["counts"]["excluded_retired"] == 10
    # Before Nov 17 drop, actionable auction targets must be 0
    assert db["counts"]["actionable_auction_targets"] == 0


def test_pending_retention_and_rr_2026_squad_statuses():
    """Verify specific key players have the exact required pre-retention statuses."""
    store = get_store()

    # Nathan Ellis: CONFIRMED_OTHER_TEAM (SRH)
    ellis = store.get_player_detail(7920)
    assert ellis is not None
    assert ellis["availability_status"] == "CONFIRMED_OTHER_TEAM"
    assert ellis["current_2027_team"] == "Sunrisers Hyderabad"
    assert "SRH" in ellis.get("trade_note", "") or "Sunrisers" in ellis.get("trade_note", "")
    assert "Excluded" in ellis["acquisition_pathway"]

    # Sanju Samson: PENDING_RETENTION (CSK)
    samson = store.get_player_detail(2258)
    assert samson is not None
    assert samson["availability_status"] == "PENDING_RETENTION"
    assert "Chennai Super Kings" in samson["current_2027_team"]
    assert "Watchlist" in samson["acquisition_pathway"] or "Pending" in samson["acquisition_pathway"]

    # Dhruv Jurel: RR_2026_SQUAD (RR)
    jurel = store.get_player_detail(7260)
    assert jurel is not None
    assert jurel["availability_status"] == "RR_2026_SQUAD"
    assert "Rajasthan Royals" in jurel["current_2027_team"]
    assert "Internal Squad" in jurel["acquisition_pathway"]

    # Yashasvi Jaiswal & Riyan Parag: RR_2026_SQUAD
    jaiswal = store.get_player_detail(6819)
    assert jaiswal is not None
    assert jaiswal["availability_status"] == "RR_2026_SQUAD"
    parag = store.get_player_detail(6423)
    assert parag is not None
    assert parag["availability_status"] == "RR_2026_SQUAD"

    # Josh Hazlewood, Mitchell Santner, Tim David, Sakib Hussain: PENDING_RETENTION
    hazlewood = store.get_player_detail(830)
    assert hazlewood["availability_status"] == "PENDING_RETENTION"
    santner = store.get_player_detail(4101)
    assert santner["availability_status"] == "PENDING_RETENTION"
    david = store.get_player_detail(6396)
    assert david["availability_status"] == "PENDING_RETENTION"
    sakib = store.get_player_detail(35318)
    assert sakib["availability_status"] == "PENDING_RETENTION"


def test_duplicate_name_disambiguation():
    """Verify that players sharing identical names are resolved to distinct canonical IDs."""
    with open(AVAILABILITY_FILE, "r", encoding="utf-8") as f:
        players = json.load(f)["players"]

    # Abdul Samad
    samad_ipl = players.get("7446")
    samad_pak = players.get("6922")
    assert samad_ipl is not None and samad_pak is not None
    assert samad_ipl["canonical_player_id"] != samad_pak["canonical_player_id"]
    assert samad_ipl["player_name"] == samad_pak["player_name"] == "Abdul Samad"
    assert samad_ipl["availability_status"] == "PENDING_RETENTION"
    assert samad_pak["availability_status"] == "UNVERIFIED"

    # Rashid Khan
    rashid_gt = players.get("4722")
    rashid_nep = players.get("20172")
    assert rashid_gt is not None and rashid_nep is not None
    assert rashid_gt["canonical_player_id"] != rashid_nep["canonical_player_id"]
    assert rashid_gt["player_name"] == rashid_nep["player_name"] == "Rashid Khan"
    assert rashid_gt["availability_status"] == "PENDING_RETENTION"
    assert rashid_nep["availability_status"] == "UNVERIFIED"

    # Haider Ali
    haider_pak = players.get("7338")
    haider_uae = players.get("28756")
    assert haider_pak is not None and haider_uae is not None
    assert haider_pak["canonical_player_id"] != haider_uae["canonical_player_id"]
    assert haider_pak["player_name"] == haider_uae["player_name"] == "Haider Ali"

    # Mohammad Ali
    mali_1 = players.get("3083")
    mali_2 = players.get("20229")
    assert mali_1 is not None and mali_2 is not None
    assert mali_1["canonical_player_id"] != mali_2["canonical_player_id"]
    assert mali_1["player_name"] == mali_2["player_name"] == "Mohammad Ali"

    # Muhammad Farooq
    farooq_1 = players.get("27373")
    farooq_2 = players.get("28978")
    assert farooq_1 is not None and farooq_2 is not None
    assert farooq_1["canonical_player_id"] != farooq_2["canonical_player_id"]
    assert farooq_1["player_name"] == farooq_2["player_name"] == "Muhammad Farooq"

    # Bashir Ahmad
    bashir_1 = players.get("29833")
    bashir_2 = players.get("39479")
    assert bashir_1 is not None and bashir_2 is not None
    assert bashir_1["canonical_player_id"] != bashir_2["canonical_player_id"]
    assert bashir_1["player_name"] == bashir_2["player_name"] == "Bashir Ahmad"


def test_acquisition_pathway_logic():
    """Verify that acquisition pathways are correctly derived from both status dimensions."""
    assert "Excluded" in _derive_acquisition_pathway("CONFIRMED_OTHER_TEAM", "NOT_APPLICABLE")
    assert "Contracted" in _derive_acquisition_pathway("CONFIRMED_OTHER_TEAM", "NOT_APPLICABLE")
    assert "Retired" in _derive_acquisition_pathway("RETIRED_UNAVAILABLE", "NOT_APPLICABLE")
    assert "Confirmed Retained" in _derive_acquisition_pathway("RR_CONFIRMED_RETAINED", "NOT_APPLICABLE")
    assert "Retention Pending" in _derive_acquisition_pathway("RR_2026_SQUAD", "PENDING")
    assert "Pending Franchise Release" in _derive_acquisition_pathway("PENDING_RETENTION", "PENDING")
    assert "Awaiting Final Auction Register" in _derive_acquisition_pathway("CONFIRMED_RELEASED", "REGISTERED")
    assert "Released" in _derive_acquisition_pathway("CONFIRMED_RELEASED", "PENDING")
    assert "Actionable Auction Target" in _derive_acquisition_pathway("CONFIRMED_AUCTION_REGISTERED", "REGISTERED")


def test_official_deliverable_csv_checksums():
    """
    Ensure the 4 official Top 10 deliverable CSVs have strictly unmodified checksums,
    preserving Layer 1 analytical integrity across both Windows (CRLF) and Linux (LF).
    """
    for filename, exp in EXPECTED_CSV_HASHES.items():
        file_path = PROCESSED_DIR / filename
        assert file_path.exists(), f"Deliverable CSV missing: {file_path}"
        raw_bytes = file_path.read_bytes()

        # 1. Strict raw byte verification against approved platform representations
        computed_raw_hash = hashlib.md5(raw_bytes).hexdigest().upper()
        valid_hashes = {exp["crlf_raw"], exp["canonical_lf"]}
        assert computed_raw_hash in valid_hashes, (
            f"Hash mismatch for {filename}: expected one of {valid_hashes}, got {computed_raw_hash}"
        )

        # 2. Canonical content verification (LF-normalized)
        norm_bytes = raw_bytes.replace(b"\r\n", b"\n")
        computed_norm_hash = hashlib.md5(norm_bytes).hexdigest().upper()
        assert computed_norm_hash == exp["canonical_lf"], (
            f"Canonical content hash mismatch for {filename}: expected {exp['canonical_lf']}, got {computed_norm_hash}"
        )

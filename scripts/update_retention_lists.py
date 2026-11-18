"""
BCCI IPL 2027 Regulatory Retention and Auction Register Ingestion Workflow
========================================================================
Designed for the Rajasthan Royals SupeRR Selector 2.0 Two-Layer Architecture.

Timeline Milestones:
- 17 Nov 2026: Official Franchise Retention / Release Announcements
- 23 Nov 2026: Official Player Auction Registration Cutoff
- 01 Dec 2026: Franchise Wishlist Submissions
- 03 Dec 2026: Official BCCI Final Auction Register Published
- 16 Dec 2026: IPL 2027 Mini-Auction (Goa)

Usage:
    python scripts/update_retention_lists.py --mode retention --input path/to/retention_data.json
    python scripts/update_retention_lists.py --mode auction_register --input path/to/final_register.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AVAIL_FILE = ROOT / "data" / "external" / "ipl_2027_player_availability.json"

VALID_AVAIL_STATUSES = {
    "CONFIRMED_OTHER_TEAM",
    "RR_2026_SQUAD",
    "RR_CONFIRMED_RETAINED",
    "PENDING_RETENTION",
    "CONFIRMED_RELEASED",
    "RETIRED_UNAVAILABLE",
    "UNVERIFIED",
}

VALID_AUCTION_STATUSES = {
    "NOT_APPLICABLE",
    "PENDING",
    "REGISTERED",
    "NOT_REGISTERED",
    "WITHDRAWN",
    "NOT_IN_FINAL_REGISTER",
}


def load_availability():
    if not AVAIL_FILE.exists():
        raise FileNotFoundError(f"Availability registry not found at {AVAIL_FILE}")
    with open(AVAIL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_availability(data):
    with open(AVAIL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def process_retention_drop(drop_file_path: Path):
    """
    Ingests official 17 Nov 2026 BCCI retention/release list.
    Expected format per record:
    {
        "canonical_player_id": 1699,
        "new_availability_status": "CONFIRMED_OTHER_TEAM" | "RR_CONFIRMED_RETAINED" | "CONFIRMED_RELEASED",
        "current_2027_team": "Mumbai Indians" | "Rajasthan Royals" | "Released",
        "source_url": "https://www.iplt20.com/news/...",
        "verification_date": "2026-11-17",
        "evidence_note": "Retained by MI for INR 18.00 Cr"
    }
    """
    with open(drop_file_path, "r", encoding="utf-8") as f:
        drop_data = json.load(f)

    reg_data = load_availability()
    players = reg_data["players"]
    updates_count = 0

    records = drop_data.get("declarations", drop_data)
    if not isinstance(records, list):
        raise ValueError("Retention file must contain a list of player declarations under key 'declarations' or at root.")

    for item in records:
        cid_str = str(item["canonical_player_id"])
        new_status = item["new_availability_status"]
        if new_status not in {"CONFIRMED_OTHER_TEAM", "RR_CONFIRMED_RETAINED", "CONFIRMED_RELEASED"}:
            raise ValueError(f"Invalid post-retention status '{new_status}' for player {cid_str}")
        if not item.get("source_url"):
            raise ValueError(f"Strict compliance: source_url required for player {cid_str}")

        if cid_str in players:
            p = players[cid_str]
            old_status = p["availability_status"]
            p["availability_status"] = new_status
            p["current_2027_team"] = item.get("current_2027_team", p["current_2027_team"])
            p["source_url"] = item["source_url"]
            p["source_name"] = item.get("source_name", "Official BCCI IPL 2027 Retention Announcement")
            p["last_verified_date"] = item.get("verification_date", "2026-11-17")
            p["verification_confidence"] = "Official BCCI / Franchise Declaration"
            p["notes"] = item.get("evidence_note", f"Transitioned from {old_status} to {new_status} on Nov 17")
            if new_status in {"CONFIRMED_OTHER_TEAM", "RR_CONFIRMED_RETAINED"}:
                p["auction_registration_status"] = "NOT_APPLICABLE"
            elif new_status == "CONFIRMED_RELEASED":
                p["auction_registration_status"] = "PENDING"
            updates_count += 1

    # Recalculate summary counts
    summary = {}
    for st in VALID_AVAIL_STATUSES:
        summary[st] = sum(1 for p in players.values() if p.get("availability_status") == st)
    reg_data["metadata"]["summary_counts"] = summary
    reg_data["metadata"]["last_updated"] = "2026-11-17"

    save_availability(reg_data)
    print(f"Successfully processed {updates_count} player retention declarations.")


def process_auction_register(register_file_path: Path):
    """
    Ingests official 3 Dec 2026 BCCI Final Auction Register.
    Expected format per record:
    {
        "canonical_player_id": 4847,
        "auction_set": "BA1",
        "base_price_lakh": 200,
        "source_url": "https://www.iplt20.com/auction/register-2027",
        "verification_date": "2026-12-03"
    }
    """
    with open(register_file_path, "r", encoding="utf-8") as f:
        reg_input = json.load(f)

    reg_data = load_availability()
    players = reg_data["players"]
    promoted_count = 0

    register_ids = set()
    records = reg_input.get("auction_players", reg_input)
    for item in records:
        cid_str = str(item["canonical_player_id"])
        register_ids.add(cid_str)
        if cid_str in players:
            p = players[cid_str]
            # Only promote released or unverified free agents
            if p["availability_status"] in {"CONFIRMED_RELEASED", "UNVERIFIED"}:
                p["auction_registration_status"] = "REGISTERED"
                p["source_url"] = item.get("source_url", p["source_url"])
                p["source_name"] = "Official BCCI Final Auction Register"
                p["last_verified_date"] = item.get("verification_date", "2026-12-03")
                p["verification_confidence"] = "Verified on Official BCCI Auction Register"
                p["notes"] = f"Verified on BCCI final auction list (Set {item.get('auction_set', 'N/A')}, Base ₹{item.get('base_price_lakh', 'N/A')}L)"
                promoted_count += 1

    # Mark released players who were not in the final register
    for cid_str, p in players.items():
        if p["availability_status"] == "CONFIRMED_RELEASED" and cid_str not in register_ids:
            p["auction_registration_status"] = "NOT_IN_FINAL_REGISTER"

    save_availability(reg_data)
    print(f"Successfully verified {promoted_count} players on official BCCI Final Auction Register.")


def main():
    parser = argparse.ArgumentParser(description="Ingest BCCI retention/release and auction register updates.")
    parser.add_argument("--mode", choices=["retention", "auction_register"], required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to JSON file containing official updates.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)

    if args.mode == "retention":
        process_retention_drop(input_path)
    elif args.mode == "auction_register":
        process_auction_register(input_path)


if __name__ == "__main__":
    main()

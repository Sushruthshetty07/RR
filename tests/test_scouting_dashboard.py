"""
Unit and API integration tests for the RR SupeRR Selector IPL 2027
Scouting & Performance Analytics Dashboard.
"""

import asyncio
import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scouting_data import EXCLUDED_FROM_ACQUISITION, get_store
from src.web_app import (
    api_compare,
    api_dashboard_a,
    api_dashboard_b,
    api_methodology,
    api_overview,
    api_player,
    api_rr_squad,
    api_trades,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestScoutingDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = get_store()
        cls.bat3 = pd.read_csv(PROJECT_ROOT / "data/processed/phase3_batting_prelim.csv")
        cls.bowl3 = pd.read_csv(PROJECT_ROOT / "data/processed/phase3_bowling_prelim.csv")

    def test_dashboard_a_preserves_original_rankings_unchanged(self) -> None:
        bat_elig = self.bat3[self.bat3["elig_primary"] == True].sort_values("pri_score_lcb", ascending=False)
        bowl_elig = self.bowl3[self.bowl3["elig_primary"] == True].sort_values("pri_score_lcb", ascending=False)

        da_bat = self.store.get_dashboard_a("batting")
        da_bowl = self.store.get_dashboard_a("bowling")

        self.assertEqual(da_bat["total_eligible"], 188)
        self.assertEqual(da_bowl["total_eligible"], 200)
        self.assertEqual(
            [r["canonical_player_id"] for r in da_bat["records"]],
            bat_elig["canonical_player_id"].astype(int).tolist(),
        )
        self.assertEqual(
            [r["canonical_player_id"] for r in da_bowl["records"]],
            bowl_elig["canonical_player_id"].astype(int).tolist(),
        )
        # Retired players remain in Dashboard A universe
        bat_ids = {r["canonical_player_id"] for r in da_bat["records"]}
        bowl_ids = {r["canonical_player_id"] for r in da_bowl["records"]}
        self.assertIn(1318, bat_ids)  # Andre Russell
        self.assertIn(143, bat_ids)   # Dinesh Karthik
        self.assertIn(305, bowl_ids)  # Ravichandran Ashwin

    def test_dashboard_b_bucket_exclusions_and_separation(self) -> None:
        db = self.store.get_dashboard_b("all")
        buckets = db["buckets"]

        # Zero confirmed auction-available before Nov 17/23 deadlines
        self.assertEqual(len(buckets["confirmed_auction"]), 0)

        # Retired, RR, and contracted players must NEVER appear in acquisition shortlists
        acquisition_pool = buckets["confirmed_auction"] + buckets["pending_registration"]
        for r in acquisition_pool:
            self.assertNotIn(r["availability_status"], EXCLUDED_FROM_ACQUISITION)
            self.assertIn(r["availability_status"], ("CONFIRMED_RELEASED", "Released, auction registration pending/unverified"))

        # Trade watchlist strictly separated from ordinary auction buckets
        watchlist_ids = {r["canonical_player_id"] for r in buckets["trade_watchlist"]}
        pending_ids = {r["canonical_player_id"] for r in buckets["pending_registration"]}
        self.assertEqual(len(watchlist_ids & pending_ids), 0)
        self.assertIn(6396, watchlist_ids)  # Tim David
        self.assertIn(4900, watchlist_ids)  # Wanindu Hasaranga

        # RR players isolated in rr_squad
        rr_ids = {r["canonical_player_id"] for r in buckets["rr_squad"]}
        self.assertIn(6819, rr_ids)   # Yashasvi Jaiswal
        self.assertIn(50284, rr_ids)  # Vaibhav Sooryavanshi
        self.assertEqual(len(rr_ids & pending_ids), 0)
        self.assertEqual(len(rr_ids & watchlist_ids), 0)

        # Retired isolated in retired
        ret_ids = {r["canonical_player_id"] for r in buckets["retired"]}
        self.assertTrue({1318, 143, 305, 23}.issubset(ret_ids))
        self.assertEqual(len(ret_ids & pending_ids), 0)

    def test_verified_ipl_2027_trade_destinations_and_stalled_rumour(self) -> None:
        expected_2027_teams = {
            3644: "Lucknow Super Giants",   # Kuldeep Yadav (DC -> LSG)
            4703: "Delhi Capitals",         # Rishabh Pant (LSG -> DC)
            4265: "Sunrisers Hyderabad",    # T Natarajan (DC -> SRH)
            4824: "Delhi Capitals",         # Zeeshan Ansari (SRH -> DC)
            4391: "Chennai Super Kings",    # Liam Livingstone (SRH -> CSK)
            4816: "Sunrisers Hyderabad",    # Khaleel Ahmed (CSK -> SRH)
            7920: "Sunrisers Hyderabad",    # Nathan Ellis (CSK -> SRH)
            3330: "Punjab Kings",           # Hardik Pandya (MI -> PBKS)
            6464: "Mumbai Indians",         # Arshdeep Singh (PBKS -> MI)
        }
        for cid, expected_team in expected_2027_teams.items():
            detail = self.store.get_player_detail(cid)
            self.assertIsNotNone(detail, f"Missing player {cid}")
            self.assertEqual(detail["current_2027_team"], expected_team)
            self.assertIn(detail["availability_status"], ("CONFIRMED_OTHER_TEAM", "Contracted to another franchise"))

        trades = self.store.get_trades_overview()
        self.assertEqual(len(trades["official_completed_trades"]), 4)
        self.assertEqual(len(trades["reported_negotiations_not_completed"]), 1)
        rumour = trades["reported_negotiations_not_completed"][0]
        self.assertIn("NOT Completed", rumour["transaction_status"])

    def test_zero_hp_warning_banner_present(self) -> None:
        for cid in (50284, 4847, 25669):  # Sooryavanshi, Finn Allen, Priyansh Arya
            detail = self.store.get_player_detail(cid)
            self.assertEqual(detail["batting"]["neff_hp"], 0.0)
            self.assertIn("NOT been demonstrated", detail["batting"]["hp_banner"])

    def test_fastapi_endpoints_and_live_server(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        proc = subprocess.Popen(
            [sys.executable, "scripts/run_dashboard.py", "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            ready = False
            for _ in range(30):
                try:
                    with urllib.request.urlopen(f"{base}/api/overview", timeout=2) as r:
                        if r.status == 200:
                            ready = True
                            break
                except Exception:
                    time.sleep(0.3)
            self.assertTrue(ready, "Uvicorn server did not become ready in time")

            with urllib.request.urlopen(f"{base}/", timeout=3) as r:
                html = r.read().decode("utf-8")
                self.assertEqual(r.status, 200)
                self.assertIn("Dashboard B: RR Auction Scouting", html)
                self.assertIn("Dashboard A: Performance Analytics", html)

            def get_json(path: str):
                with urllib.request.urlopen(f"{base}{path}", timeout=3) as r:
                    self.assertEqual(r.status, 200)
                    return json.loads(r.read().decode("utf-8"))

            self.assertIn("universe_counts", get_json("/api/overview"))
            self.assertEqual(get_json("/api/dashboard-a?discipline=batting")["total_eligible"], 188)
            self.assertEqual(get_json("/api/dashboard-a?discipline=bowling")["total_eligible"], 200)
            self.assertEqual(get_json("/api/dashboard-b")["counts"]["confirmed_auction"], 0)
            self.assertIn("roster", get_json("/api/rr-squad"))
            self.assertEqual(len(get_json("/api/trades")["official_completed_trades"]), 4)
            self.assertEqual(get_json("/api/player/1699")["player_name"], "Suryakumar Yadav")
            self.assertEqual(len(get_json("/api/compare?ids=479,2636")["players"]), 2)
            self.assertIn("accounting", get_json("/api/methodology"))

            off = get_json("/api/rankings/official-top10")
            self.assertEqual(len(off["deliverable_01"]["batters"]), 10)
            self.assertEqual(len(off["deliverable_01"]["bowlers"]), 10)
            self.assertEqual(off["deliverable_01"]["batters"][0]["player_name"], "Suryakumar Yadav")
            self.assertEqual(off["deliverable_01"]["bowlers"][0]["player_name"], "Sunil Narine")
        finally:
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()

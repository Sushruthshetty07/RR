"""Runner script for the independent Claude review audit (Items 1-4 & 7)."""

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.claude_audit import run_complete_claude_audit


def main() -> None:
    res = run_complete_claude_audit(save_outputs=True)

    print("=== 1. ACCOUNTING & LEAKAGE ===")
    print(json.dumps({"accounting": res["accounting"], "leakage": res["leakage"]}, indent=2))

    print("\n=== 2. PI BY ORDER BUCKET & PHASE ===")
    print(res["hp_audit"]["by_order"].to_string(index=False))
    print(res["hp_audit"]["by_phase"].to_string(index=False))

    print("\n=== 3. TOP-15 BATTER PI PROFILES ===")
    print(res["hp_audit"]["top15_pi_profiles"].to_string(index=False))

    print("\n=== 4. EXPLORATORY PP SENSITIVITY ===")
    print(json.dumps(res["hp_audit"]["pp_alt_summary"], indent=2))
    print(res["hp_audit"]["pp_alt_top15_cmp"].to_string(index=False))

    print("\n=== 5. HP WEIGHT x GATE GRID ===")
    print(res["hp_audit"]["hp_grid"].to_string(index=False))

    print("\n=== 6. PRESSURE DELTA AUDIT ===")
    print(json.dumps(res["delta_audit"]["summary"], indent=2))
    print(res["delta_audit"]["bowl_partition_residuals"].to_string(index=False))
    print(res["delta_audit"]["bowl_top15_delta"].to_string(index=False))

    print("\n=== 7. FULL-COMPOSITE BOOTSTRAP RANK INTERVALS (TOP 15) ===")
    print(json.dumps(res["boot_meta"], indent=2))
    print("-- BATTING --")
    print(res["bat_boot"].head(15).to_string(index=False))
    print("-- BOWLING --")
    print(res["bowl_boot"].head(15).to_string(index=False))

    print("\n=== 8. T20I WEIGHT SENSITIVITY ===")
    print(res["t20i_audit"]["summary"].to_string(index=False))


if __name__ == "__main__":
    main()

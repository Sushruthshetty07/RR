"""
Launch script for the Rajasthan Royals SupeRR Selector IPL 2027 Scouting Dashboard.
"""

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RR SupeRR Selector IPL 2027 Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("src.web_app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

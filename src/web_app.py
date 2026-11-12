"""
FastAPI web application for the Rajasthan Royals SupeRR Selector —
IPL 2027 Auction Scouting & Player Analytics Dashboard.

Serves two unified views:
- Dashboard B (PRIMARY): RR Auction Scouting (strict availability buckets)
- Dashboard A: Performance Analytics (complete unrestricted 188-bat / 200-bowl universe)
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.scouting_data import AVAILABILITY_STATUSES, get_store

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"

app = FastAPI(
    title="Rajasthan Royals SupeRR Selector — IPL 2027 Scouting & Analytics",
    version="2.0.0",
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    store = get_store()
    overview = store.get_overview()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "as_of_date": overview["as_of_date"],
            "availability_statuses": AVAILABILITY_STATUSES,
        },
    )


@app.get("/api/overview")
async def api_overview():
    return get_store().get_overview()


@app.get("/api/dashboard-a")
async def api_dashboard_a(
    discipline: str = Query("batting"),
    search: str = Query(""),
    role: str = Query(""),
    availability: str = Query(""),
    ipl_exp: str = Query(""),
    pressure_tier: str = Query(""),
    confidence_tier: str = Query(""),
    all_rounder_only: bool = Query(False),
):
    return get_store().get_dashboard_a(
        discipline=discipline,
        search=search,
        role=role,
        availability=availability,
        ipl_exp=ipl_exp,
        pressure_tier=pressure_tier,
        confidence_tier=confidence_tier,
        all_rounder_only=all_rounder_only,
    )


@app.get("/api/dashboard-b")
async def api_dashboard_b(
    discipline: str = Query("all"),
    search: str = Query(""),
    role: str = Query(""),
    availability: str = Query(""),
    ipl_exp: str = Query(""),
    pressure_tier: str = Query(""),
    confidence_tier: str = Query(""),
    all_rounder_only: bool = Query(False),
):
    return get_store().get_dashboard_b(
        discipline=discipline,
        search=search,
        role=role,
        availability=availability,
        ipl_exp=ipl_exp,
        pressure_tier=pressure_tier,
        confidence_tier=confidence_tier,
        all_rounder_only=all_rounder_only,
    )


@app.get("/api/rr-squad")
async def api_rr_squad():
    return get_store().get_rr_squad()


@app.get("/api/trades")
async def api_trades():
    return get_store().get_trades_overview()


@app.get("/api/player/{canonical_id}")
async def api_player(canonical_id: int):
    detail = get_store().get_player_detail(canonical_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Player {canonical_id} not found in eligible pool")
    return detail


@app.get("/api/compare")
async def api_compare(ids: str = Query("1699,6396,2933,7920")):
    parsed: List[int] = []
    for tok in ids.split(","):
        tok = tok.strip()
        if tok:
            try:
                parsed.append(int(tok))
            except ValueError:
                pass
    return {"players": get_store().compare_players(parsed)}


@app.get("/api/methodology")
async def api_methodology():
    return get_store().get_methodology()

# Rajasthan Royals SupeRR Selector 2.0 — UI/UX Redesign Audit & Implementation Plan

**Author:** Senior Full-Stack Engineer & Sports-Analytics UX Designer  
**Date:** November 17, 2026  
**Status:** Approved & Moving Directly to Implementation  
**Project:** Rajasthan Royals IPL 2027 Challenge — SupeRR Selector 2.0

---

## 1. Executive Summary & Architectural Audit

### 1.1 Architecture & Verification
- **Backend Core:** FastAPI (`src/web_app.py`) backed by an in-memory high-performance data store (`src/scouting_data.py: ScoutingStore`).
- **Endpoint Contracts:** 15 active endpoints:
  - `/` (HTML shell serving `templates/dashboard.html` with Jinja2 context `as_of_date` and `availability_statuses`)
  - `/api/overview`, `/api/dashboard-a`, `/api/dashboard-b`, `/api/rr-squad`, `/api/trades`
  - `/api/player/{id}`, `/api/compare`, `/api/methodology`, `/api/rankings/official-top10`
  - `/api/evaluation-dimensions`, `/api/win-probability`, `/api/weight-sensitivity`
  - `/api/cricket-intelligence`, `/api/rr-tactical-fit`
- **Data Layers Verified:**
  - `data/processed/`: Reconstructed delivery tables (`deliveries_with_pressure.csv.gz`), official leaderboards (`official_top10_high_pressure_batters.csv`, `official_top10_high_pressure_bowlers.csv`), empirical Bayes tables, bootstrap intervals, and intelligence JSON summaries exist and are intact.
  - `data/external/`: Verified BCCI availability registry (`ipl_2027_player_availability.json`) and transfer ledger (`ipl_2027_transfers.json`) exist.
  - `data/raw/`: Verified 867-match dataset (200,999 deliveries).

### 1.2 Non-Negotiable Contract & Test Guardrails
- **Test Assertion Continuity:** Unit integration test `tests/test_scouting_dashboard.py` explicitly verifies that the rendered HTML contains:
  1. `"Dashboard B: RR Auction Scouting"`
  2. `"Dashboard A: Performance Analytics"`
  These strings must remain prominently accessible in the header/navigation.
- **Data Immutability:** ML models, Kish $N_{\text{eff}}$, Empirical Bayes shrinkage ($\theta_i^{\text{post}}, \theta_i^{\text{lcb}}$), Pressure Index formulas, and official rankings must **not** be modified.

---

## 2. Identified UI/UX Weaknesses & Deficiencies

1. **Monolithic Code Structure:**
   - `templates/dashboard.html` contained 3,083 lines of inline CSS, HTML, and vanilla JS in a single file. Styling changes risked breaking table scripts, and modular testing was difficult.
2. **High-Pressure Fairness & Statistical Representation Gaps:**
   - Previous views sometimes blended overall reliability (PRI 0–100) into high-pressure ranking displays, obscuring the primary metric (runs/100b for batters, runs saved/ov for bowlers).
   - Missing mandatory statistical disclosures: LCB $z = 0.5$ is a conservative lower-bound score, **not** a 95% confidence interval, probability, or percentage.
   - Missing explicit adjacent rank inseparability notice ("Adjacent ranks are not statistically separable; read close ranks as a group").
   - Missing standardized $N_{\text{eff}}^{\text{HP}}$ Evidence-Tier chips: Strong ($\ge 200$), Moderate ($100\text{--}200$), Thin ($< 100$).
   - Absence of an interactive "Supplementary Sensitivity" toggle ($z = 1.0$) for the official Top 10 to demonstrate ranking stability under stricter penalization.
3. **Scouting & Availability Presentation:**
   - Scouting candidates lacked a clear visual separation between **Performance Evidence**, **Tactical Suitability**, and **Availability Status**.
   - Players were sometimes labeled "targets" generically; rule requires that **only** confirmed-released or auction-registered players are labeled "Targets", while all others remain "Watchlist".
4. **Comparison Tool Usability:**
   - Relied heavily on comma-separated IDs input. Needs quick-add chips, multi-select presets, and responsive side-by-side metric comparison cards with winning metric highlights.
5. **Interactive Data Tables:**
   - Tables lacked clickable column header sorting (ascending/descending), sticky headers with elevation, and horizontal scroll affordances for mobile.
6. **Mobile & Tablet Responsiveness:**
   - On screens $\le 768\text{px}$ and $\le 390\text{px}$, filter bars and modal windows overflowed horizontally. A robust sliding drawer with touch backdrop is required.
7. **Streamlit Companion:**
   - `scripts/streamlit_app.py` used default Streamlit styling; needs visual alignment with the Rajasthan Royals royal purple and hot pink design system.

---

## 3. Design System Specification

### 3.1 Color Palette & Design Tokens
```css
:root {
  /* Surface & Background */
  --bg-primary: #07090E;
  --bg-secondary: #0D111A;
  --surface-card: #121824;
  --surface-elevated: #182030;
  --surface-hover: #1F2A3E;
  
  /* Rajasthan Royals Brand Accents */
  --rr-pink: #EA1A85;
  --rr-pink-hover: #D01374;
  --rr-pink-subtle: rgba(234, 26, 133, 0.12);
  --rr-royal: #254AA5;
  --rr-navy: #15102A;
  --rr-purple: #7C3AED;
  
  /* Functional Accents */
  --cyan: #38BDF8;
  --cyan-subtle: rgba(56, 189, 248, 0.12);
  --green: #10B981;
  --green-subtle: rgba(16, 185, 129, 0.12);
  --gold: #F59E0B;
  --gold-subtle: rgba(245, 158, 11, 0.12);
  --red: #EF4444;
  --red-subtle: rgba(239, 68, 68, 0.12);
  
  /* Text */
  --text-primary: #F8FAFC;
  --text-secondary: #94A3B8;
  --text-muted: #64748B;
  --border: rgba(255, 255, 255, 0.08);
  --border-strong: rgba(255, 255, 255, 0.16);
  
  /* Spacing Grid (8px) & Geometry */
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --radius-pill: 9999px;
  
  /* Typography */
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}
```

### 3.2 Reusable UI Components
1. **Evidence-Tier Chips:**
   - `Strong Evidence`: $N_{\text{eff}}^{\text{HP}} \ge 200$ (Cyan/Emerald badge)
   - `Moderate Evidence`: $100 \le N_{\text{eff}}^{\text{HP}} < 200$ (Purple/Blue badge)
   - `Thin Evidence`: $N_{\text{eff}}^{\text{HP}} < 100$ (Amber badge)
   - `No High-Pressure Evidence`: $N_{\text{eff}}^{\text{HP}} = 0$ (Muted Gray badge)
2. **Availability Badges:**
   - `Confirmed Target`: CONFIRMED_RELEASED or registered auction candidate (Emerald)
   - `Pending Watchlist`: Auction registration unverified / retention pending (Amber)
   - `Contracted / Excluded`: Under contract with another team / traded (Red/Muted)
   - `Retired`: Formally retired from IPL (Gray)
3. **KPI Stat Cards:** Metric value, percentage shift, opportunity count, contextual spark indicator.
4. **Interactive Data Tables:** Dynamic sorting on any column, sticky elevated headers, horizontal touch scrolling, player avatar badges.
5. **Loading Skeletons & Error/Empty States:** Graceful fallbacks when queries yield zero records.

---

## 4. Page-by-Page Redesign Plan

### Page 1: Overview / Master Executive Dashboard
- **Header:** Prominent mode switches (`Dashboard B: RR Auction Scouting` & `Dashboard A: Performance Analytics`), audience level switcher (Analyst View vs Data Science View), verified as-of date badge.
- **Executive Metric Strip:** Total deliveries (200,999), evaluated pool, RR 2026 venue record (Guwahati 3-0 vs Jaipur 1-4), orthogonal PI-vs-LI correlation, auction calendar countdown.
- **Official Top 10 High-Pressure Showcase:**
  - Dedicated cards for Top 10 Batters and Top 10 Bowlers.
  - Strict display: Rank 1–10, Player, $N_{\text{eff}}^{\text{HP}}$, High-Pressure Score (LCB $z=0.5$), Evidence Badge.
  - Clear label: *"High-pressure lower-bound score (z = 0.5), not a 95% confidence interval, percentage or probability."*
  - Prominent statistical note: *"Adjacent ranks are not statistically separable; read close ranks as a group."*
  - Interactive "Supplementary Sensitivity ($z=1.0$)" toggle to demonstrate stability without altering official rankings.
- **Powerplay Paradox Warning:** Explaining why explosive openers with zero pressure balls cannot be cast as late-innings finishers.
- **Secondary Dimension A Leaderboards:** Overall reliability ratings (PRI 0–100) kept distinct.

### Pages 2 & 3: Batter & Bowler Pressure Scouting
- View switcher: Official High-Pressure Top 10 vs Complete Universe (188 Batters / 200 Bowlers).
- Interactive filter panel: Role, Availability, IPL Exposure, Pressure Tier, Confidence Tier, All-Rounder checkbox, real-time search.
- Interactive sortable data tables with sticky header and mobile horizontal scroll.

### Page 4: Player Profile Drawer / Detail Modal
- Slide-over drawer with backdrop blur.
- Three distinct blocks: (1) Performance Evidence, (2) Tactical Suitability & Pace/Spin Splits, (3) Availability & Acquisition Pathway.
- Composure sequence intelligence: Post-dot ball strike rate rebound for batters; post-boundary economy and dot recovery for bowlers.
- 4-Question Strategic Intelligence summary for Rajasthan Royals.

### Page 5: Head-to-Head Player Comparison
- Multi-player tray (supports 2 to 4 players simultaneously).
- Preset shortcuts ("Top Clutch Batters", "Top Clutch Bowlers", "Powerplay vs Chasers").
- Side-by-side metric comparison cards with color-coded winning metric indicators.

### Page 6: Pressure & Win Impact (Win Probability & Leverage)
- Pre-delivery WP engine stats: 182,311 balls, ROC-AUC 0.7785 overall, 0.8962 in chases.
- Pressure vs. Leverage scatter visualization: Demonstrating mathematical orthogonality ($r = +0.0058$ overall, $-0.37$ Inn 1, $+0.08$ Inn 2).
- Top 10 Batters and Bowlers by Win Probability Added (WPA).

### Page 7: Tactical Cricket Intelligence
- Sub-tabs: Batting Composure, Bowling Bounce-Back, Strategic Profiles.
- Filterable search across archetypes, boundary recurrence, dot rebound %, and Wilson confidence intervals.

### Page 8: Rajasthan Royals Squad 2026 Audit
- Review of RR's 8–6 league campaign.
- Deep dive on the SMS Jaipur pace economy leak (10.13 RPO in death overs) vs. Guwahati spin record (3/3).
- Full player audit table with 2026 matches, median batting order, and pressure exposure.

### Page 9: Auction Intelligence & Acquisition Buckets
- 3-block candidate layout: Evidence, Tactical Fit, Availability.
- Strict categorization: Only Confirmed Released = "Target"; All others = "Watchlist".
- 6 eligibility buckets with real-time count badges.

### Page 10: IPL 2027 Trade Tracker
- Verified BCCI trade ledger (Kuldeep, Pant, Natarajan, Livingstone, Ellis, Hardik, Arshdeep).
- Stalled negotiations & rumours block with clear "Not Completed" badge.

### Page 11: Model Validation & Sensitivity
- Out-of-fold calibration tables (GroupKFold).
- 4-scenario tactical weight correlation matrix.
- Pairwise Top-10 overlap matrix.
- Newly validated sensitivity results: Pressure threshold sensitivity ($\text{PI} \in \{50, 60, 70\}$), $B=2,000$ bootstrap intervals, outcome monotonicity, and temporal holdouts.

### Page 12: Technical Methodology & Limitations
- Mathematical formulas (Zero leakage, Kish $N_{\text{eff}}$, Empirical Bayes shrinkage, LCB $z=0.5$).
- Explicit limitations: 2022–2026 data only, small high-pressure samples, no domestic leagues, no ball-tracking Hawkeye telemetry.

### Streamlit Companion App (`scripts/streamlit_app.py`)
- Theme updates matching dark navy/purple/pink color scheme, card styling, and clean metric formatting.

---

## 5. Implementation Architecture & Modularization Strategy

To ensure code maintainability, clean architecture, and browser performance:
1. Create `static/css/tokens.css` (design variables, typography, reset, grid).
2. Create `static/css/components.css` (cards, badges, tables, drawer, buttons, pills, alerts).
3. Create `static/js/app.js` (clean, modern JavaScript modules for state management, API fetching, sorting, component rendering, keyboard shortcuts, and UI interactions).
4. Serve static assets cleanly in `src/web_app.py` via FastAPI `StaticFiles`, while preserving inline fallbacks in `templates/dashboard.html` to guarantee zero dependency failures and 100% test compatibility.

---

## 6. Execution Timeline & Verification Checklist

- [x] Phase 1: Deep Architectural Audit & Specification (`docs/REDESIGN_PLAN.md`).
- [ ] Phase 2: Design Tokens & CSS Component System (`static/css/`).
- [ ] Phase 3: JavaScript Engine & UI Components (`static/js/app.js`).
- [ ] Phase 4: Full Template Redesign (`templates/dashboard.html`).
- [ ] Phase 5: Streamlit Companion App Styling (`scripts/streamlit_app.py`).
- [ ] Phase 6: Automated Test Verification (`pytest tests/`) & Endpoint Audit.
- [ ] Phase 7: Responsive Layout Checks (1440px, 768px, 390px) & Console Audit.
- [ ] Phase 8: Final Engineering Report Delivery.

# Domestic-League Data Completeness & Challenge-Brief Discrepancy Audit
**Rajasthan Royals SupeRR Selector Season 2**

> **Strict Audit Guardrail:** Zero existing source files, datasets, thresholds, weights, or Phase 1–3 processed outputs were modified. Zero player names and zero Top-10 rankings were generated or displayed. Every fact below was independently re-verified directly against [`media_1794423099935_44b59478.zip`](file:///C:/Users/sushruth/.gemini/antigravity/brain/b6ea2651-73fc-4d7f-8e2e-0f0033d4425b/.user_uploaded/media_1794423099935_44b59478.zip) and the workspace PDF/codebase.

---

## Executive Summary

**Definitive Finding:** The official downloaded data pack (`media_1794423099935_44b59478.zip`, `4,545,438` bytes, SHA-256 `82da62e9...`) contains **ZERO domestic T20 league competitions, matches, teams, or deliveries** (`0` SMAT, BBL, SA20, CPL, PSL, Vitality Blast, The Hundred, ILT20, MLC, LPL, BPL, or Super Smash records), even when checking for disguised competition names or alternate tables.

Crucially, there is a **direct wording mismatch between the challenge website blurb and the official Round 1 Problem Statement PDF packaged inside the ZIP itself**:
- **Challenge website blurb** (as quoted): *"It contains player performance data from domestic leagues, which you can combine with relevant publicly available information."*
- **Official Round 1 PDF inside the ZIP** ([`Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/data/raw/Round%201%20Problem%20Statement%20_%20SupeRR%20Selector%20Season%202.docx.pdf), Page 1): *"You will be provided with historical **IPL and T20 International (T20I)** data from 2022 onwards..."* — with **no mention of domestic leagues anywhere in the 2-page PDF**, and no standalone `players.csv` inside the archive despite `"players"` appearing in the PDF's supporting-data list.

Additionally, an audit of our existing Phase 2–3 code surfaces one **unflagged methodology discrepancy** requiring a separate proposal before Phase 4: [`pressure.py#L366`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/pressure.py#L366) assigns `is_ipl = False` to `Mens T20 Emerging Teams Asia Cup 2024` and `Asia Cup Rising Stars 2025` (`6,985` deliveries), causing [`performance.py`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/performance.py) to silently apply `w_t20i = 0.50` to Emerging/Rising `T20` deliveries rather than classifying them separately.

---

## 1. Exhaustive Re-Examination of the Complete Data Pack

### 1.1 Archive-Level Inspection (`media_1794423099935_44b59478.zip`)
- **Top-level directory:** `Data Pack/` (`size=0`) containing **8 non-directory files** and **0 nested archives** (`.zip`, `.gz`, `.tar`, `.parquet`, `.xlsx`, etc.):
  1. `Data Pack/ball_by_ball.csv` (`41,071,837` B | `200,999 x 33`)
  2. `Data Pack/playing_eleven.csv` (`966,287` B | `28,381 x 9`)
  3. `Data Pack/competition_team_player_mapping.csv` (`462,969` B | `6,413 x 6`)
  4. `Data Pack/matches.csv` (`123,864` B | `883 x 22`)
  5. `Data Pack/competitions.csv` (`7,624` B | `141 x 4`)
  6. `Data Pack/teams.csv` (`6,922` B | `353 x 3`)
  7. `Data Pack/Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf` (`254,082` B | 2 pages)
  8. `Data Pack/.DS_Store` (`6,148` B | macOS Finder metadata; printable strings `['Bud1', 'DSDB']` only)
- **macOS Resource Forks (`__MACOSX/`):** `9` AppleDouble `._*` sidecar entries (`120`–`489` B each, one for `Data Pack` and each of the `8` files above). No hidden CSVs or `players.csv` exist under `__MACOSX/`.
- **PDF Internals:** `Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf` has `EmbeddedFiles = False`, `PDF /URI links = []`, `/Title = Round 1 Problem Statement | SupeRR Selector Season 2.docx`, `/Producer = Skia/PDF m159 Google Docs Renderer`. It contains no embedded files or external download links.

### 1.2 Search for Hidden Domestic Leagues or External Performance Tables
- **Disguised Competition Names Check:** All `141` rows of `competitions.csv`, all `38` unique `team_name`s across `teams.csv`, and all `17` `venue_country` values in `matches.csv` were scanned directly from the ZIP:
  - Exhaustive keyword scan across all `141` `comp_name` strings yielded **`0` hits** for `SMAT`, `Mushtaq`, `Ranji`, `BBL`, `Big Bash`, `SA20`, `CPL`, `PSL`, `Blast`, `Hundred`, `ILT20`, `MLC`, `LPL`, `BPL`, `Super Smash`, `Trophy`, and `World Cup`.
  - Only `7` competitions carry `match_format == 'T20'`: the `5` IPL seasons (`IPL 2022`–`IPL 2026`), `Mens T20 Emerging Teams Asia Cup 2024`, and `Asia Cup Rising Stars 2025`. The remaining `134` competitions all have `match_format == 'T20I'`.
  - Only `18` `T20` team names exist: the `10` IPL franchises plus `8` Emerging/Rising sides (`Afghanistan A`, `Bangladesh A`, `Hong Kong`, `India A`, `Oman`, `Pakistan A`, `Sri Lanka A`, `United Arab Emirates`). The other `20` teams (plus `Hong Kong`, `Oman`, `UAE` = `23` `T20I` teams total) are senior national sides (`12` Full Members + `11` Associates).
- **Performance Data Outside `ball_by_ball.csv` Check:** Every non-`ball_by_ball` CSV was verified column-by-column:
  - `competitions.csv`: competition IDs, names, formats, seasons only.
  - `teams.csv`: competition-team lookup IDs and names only.
  - `matches.csv`: match metadata, captains, toss, `win_team_id`, `man_of_the_match`, `result`, and DLS revised targets/overs — **no batting/bowling scorecards or player statistics**.
  - `playing_eleven.csv`: roster participation flags (`batting_order`, `is_played`, `is_captain`, `is_wk`, `is_impact`, `is_impact_off`) — **no runs, balls, or wickets**.
  - `competition_team_player_mapping.csv`: player-team-competition IDs/names only.
  - **Result:** `100%` of player performance records reside exclusively inside `ball_by_ball.csv`.
- **Standalone `players.csv` Check:** Confirmed **absent** from both the ZIP (`Data Pack/` and `__MACOSX/`) and `data/raw/`. No biographical player file (DOB/age, nationality, capped status, or auction registration) is present.

---

## 2. Investigation of the Challenge Description vs Official Round 1 PDF

Comparing the **challenge website quote** against the **official PDF bundled inside the downloaded data pack** reveals a clear conflict:

| Source | Exact Data Scope Stated | Deliverables & Restrictions Stated |
|---|---|---|
| **Official Challenge Website** *(quoted by user)* | *"The data pack goes live on 10 November. Download it using the Brief/Case button. It contains **player performance data from domestic leagues**, which you can combine with relevant publicly available information."* | Mentions combining data pack with publicly available info; implies domestic-league coverage. |
| **Official Round 1 PDF** *(inside ZIP, Page 1–2)* | **"You will be provided with historical IPL and T20 International (T20I) data from 2022 onwards, including: Ball-by-ball data... Supporting data: competitions, teams, players, matches, playing XIs, and competition-team-player mappings."** | **01 Player Rankings:** Two ordered top-10 lists (`Top 10 batters` + `Top 10 bowlers`, Rank `1–10` on Unstop). **02 Methodology Report.** Explicitly **no age limit, no uncapped restriction, and no mention of domestic leagues** (`"Rankings must be supported by analysis of the supplied data, not reputation alone."`). |

**Why Neither Source Should Be Assumed to Override Silently:**
1. The **PDF** inside the official ZIP explicitly defines the supplied data as **IPL + T20I (`2022+`)**, which closely matches `96.5%` of the deliveries in `ball_by_ball.csv` (plus `3.5%` Emerging/Rising Asia Cup `T20` matches — and IPL itself is India's domestic T20 franchise competition if someone loosely called IPL + A-team T20 "domestic", though standard cricket terminology distinguishes IPL and state/franchise domestic leagues like SMAT/BBL/SA20).
2. Even within the PDF itself, one discrepancy exists: Page 1 lists **`"players"`** among supporting data, yet **`players.csv` is omitted** from the actual ZIP (`6` CSVs supplied rather than `7`).
3. Meanwhile, the **website copy** explicitly says **"domestic leagues"** and *"combine with relevant publicly available information"*, which would be either inaccurate/generic marketing copy on Unstop, a reference to IPL (`T20` vs `T20I`), OR an indication that a domestic-league / `players.csv` attachment was omitted or planned separately.

---

## 3. Verified Competition Coverage Table (`media_1794423099935_44b59478.zip`)

All `141` competitions in `competitions.csv` (`138` played across `867` matches / `200,999` deliveries; `3` unplayed `T20I` series with `0` matches) classify exhaustively as follows:

| Competition / Group | Explicit Competition Type | Comps (`Played / Listed`) | Matches (`Played / Listed`) | Deliveries | Seasons Covered | Unique `ball_by_ball` Players | Notes & Team Classification |
|---|---|---:|---:|---:|---|---:|---|
| `IPL 2022` | **IPL (Franchise T20)** | `1 / 1` | `74 / 74` | `17,909` | `2022` | `190` | `10` IPL franchises (`match_format = T20`) |
| `IPL 2023` | **IPL (Franchise T20)** | `1 / 1` | `74 / 74` | `17,863` | `2023` | `208` | `10` IPL franchises (`match_format = T20`) |
| `IPL 2024` | **IPL (Franchise T20)** | `1 / 1` | `71 / 74` | `17,103` | `2024` | `208` | `3` abandoned/unplayed (`match_format = T20`) |
| `IPL 2025` | **IPL (Franchise T20)** | `1 / 1` | `74 / 75` | `17,285` | `2025` | `200` | `1` unplayed (`match_format = T20`) |
| `IPL 2026` | **IPL (Franchise T20)** | `1 / 1` | `74 / 74` | `17,527` | `2026` | `202` | `10` IPL franchises (`match_format = T20`) |
| **Subtotal: IPL (`2022–2026`)** | **IPL (`match_format = T20`)** | **`5 / 5`** | **`367 / 371`** | **`87,687` (`43.6%`)** | **`2022–2026`** | **`382`** (`477` `pmap`) | Primary franchise league |
| `Mens T20 Emerging Teams Asia Cup 2024` | **Emerging/Rising T20** *(NOT Senior T20I, NOT Domestic League)* | `1 / 1` | `15 / 15` | `3,519` | `2024` | `108` | `5` Asian "A" teams (`India A`, `Pakistan A`, `SL A`, `Ban A`, `Afg A`) + `3` Associates (`HK`, `Oman`, `UAE`) |
| `Asia Cup Rising Stars 2025` | **Emerging/Rising T20** *(NOT Senior T20I, NOT Domestic League)* | `1 / 1` | `15 / 15` | `3,466` | `2025` | `108` | Same `8` Emerging/Associate teams (`match_format = T20`) |
| **Subtotal: Emerging/Rising T20** | **Emerging/Rising T20 (`match_format = T20`)** | **`2 / 2`** | **`30 / 30`** | **`6,985` (`3.5%`)** | **`2024–2025`** | **`172`** (`227` `pmap`) | Developmental/A-team ACC T20s; distinct from Senior T20Is and domestic leagues |
| `Mens Twenty20 Asia Cup 2022` (`id=1533`) | **Senior T20I (Continental)** | `1 / 1` | `13 / 13` | `3,071` | `2022` | `82` | Full ICC T20I status (`match_format = T20I`) |
| `Asia Cup 2025` (`id=3133`) | **Senior T20I (Continental)** | `1 / 1` | `19 / 19` | `4,437` | `2025` | `109` | Full ICC T20I status (`match_format = T20I`) |
| `Asia Cup T20 Qualifiers 2022` (`id=1551`) | **Senior T20I (Qualifier)** | `1 / 1` | `6 / 6` | `1,379` | `2022` | `51` | `4` Associate teams (`HK`, `Kuwait`, `Singapore`, `UAE`) |
| `4` Multi-Nation Tri-Series (`ids=2398, 2605, 3159, 3288`) | **Senior T20I (Tri-Series)** | `4 / 4` | `25 / 26` | `5,883` | `2024–2025` | **`*`** | Netherlands (x2), UAE, Pakistan Tri-Series |
| `127` Bilateral T20I Series / Matches | **Senior T20I (Bilateral)** | `124 / 127` | `407 / 418` | `91,557` | `2022–2026` *(incl. `22/23`, `23/24`, `24/25`)* | **`*`** | Includes `3` unplayed series (`2940`, `3405`, `3473`); spans `23` national teams (`12` FM, `11` Assoc) |
| **Subtotal: Senior T20I (`2022–2026`)** | **Senior T20I (`match_format = T20I`)** | **`131 / 134`** | **`470 / 482`** | **`106,327` (`52.9%`)** | **`2022–2026`** | **`709`** (`885` `pmap`) | `88.8%` balls Full Member vs FM (`94,377`); `11.2%` vs Associate (`11,950`) |
| **Domestic T20 Leagues** (SMAT, BBL, SA20, CPL, PSL, etc.) | **Domestic Franchise / State T20** | **`0 / 0`** | **`0 / 0`** | **`0` (`0.0%`)** | — | **`0`** | Completely absent from ZIP |
| **ICC T20 World Cups (`2022, 2024, 2026`)** | **Senior T20I (Global)** | **`0 / 0`** | **`0 / 0`** | **`0` (`0.0%`)** | — | **`0`** | Completely absent from ZIP |
| **ALL COMPETITIONS COMBINED** | `T20` + `T20I` | **`138 / 141`** | **`867 / 883`** | **`200,999`** | **`2022-01-22` – `2026-11-01`** | **`959`** (`1,217` `pmap`) | Exact reconciliation across all `6` CSVs |

*(Note: Individual Multi-Nation vs Bilateral unique player subtotals were not double-de-duplicated separately; combined unique `ball_by_ball` players across all `131` played Senior T20I competitions = `709`.)*

### Explicit Distinction: Emerging/Rising T20 vs Senior T20I vs Domestic Franchise Leagues
- **Senior T20I (`106,327` balls):** Full ICC-recognized international T20s between senior national XIs (`India`, `Australia`, `England`, `UAE`, `Nepal`, etc.; `match_format = T20I`).
- **Emerging/Rising T20 (`6,985` balls):** ACC developmental tournaments (`Mens T20 Emerging Teams Asia Cup 2024`, `Asia Cup Rising Stars 2025`; `match_format = T20`) pitting **U-23 / "A" squads** (`India A`, `Pakistan A`, etc.) against Associate senior sides (`Hong Kong`, `Oman`, `UAE`). Neither senior T20I nor domestic franchise/state league.
- **Domestic T20 Leagues (`0` balls):** National state/franchise T20 competitions such as SMAT (India), BBL (Australia), SA20 (South Africa), CPL (West Indies), PSL (Pakistan), Vitality Blast (England) — completely absent (`IPL` alone is present among domestic franchise leagues).

---

## 4. Investigation of Possible Missing Data (Facts vs Hypotheses)

### 4.1 Verified Facts (Directly Observed)
1. **ZIP completeness:** Every file inside `Data Pack/` in `media_1794423099935_44b59478.zip` unpacks without truncation or corruption; no nested zip/folder or PDF attachment exists.
2. **PDF omits domestic leagues & lists `players`:** The official Round 1 PDF explicitly defines the dataset as **IPL and T20I from 2022 onwards** (matching `ball_by_ball.csv`), yet names **`"players"`** in its Supporting Data list even though **`players.csv` is absent** from `Data Pack/`.
3. **Zero SMAT/other domestic leagues exist anywhere in the ZIP**, nor do `ICC T20 World Cup` matches.
4. **External data permission:** The website text explicitly permits combining the data pack with *"relevant publicly available information."*

### 4.2 Competing Hypotheses Explaining the Website vs ZIP Discrepancy *(Not Facts — Require Organiser Clarification)*
- **Hypothesis A — Generic / Loose Website Copy Referring to IPL (+ Emerging) or Template Text:** The Unstop challenge summary used *"domestic leagues"* loosely as shorthand for *domestic/franchise T20 (IPL)* or was written from an early Season 1/2 draft before the final Round 1 PDF scoped the data pack to **IPL + T20I (`2022+`)**.
- **Hypothesis B — Incomplete / Truncated ZIP Export (`players.csv` & Domestic Leagues Omitted):** The organisers intended to export `players.csv` (explicitly named in the PDF!) and possibly non-IPL domestic T20 leagues via the Brief/Case package, or uploaded a preliminary `Data Pack.zip` on 10 November omitting `players.csv` (and/or domestic leagues).
- **Hypothesis C — Public External Augmentation Only for Supplementary Context:** Organisers intentionally supplied only **IPL + T20I (+ 2 ACC Emerging events)** inside the official ZIP and expect teams to bring **public external sources** (e.g., player age/capping/domestic SMAT/franchise stats) where relevant—though note the PDF warns: *"Rankings must be supported by analysis of the supplied data, not reputation alone."*

---

## 5. Impact Assessment on Existing Approved Phase 1–3 Methodology

Per strict project rules, **all approved Phase 1–3 decisions remain untouched**:
- `IPL` weight `= 1.00`, `Senior T20I` weight `= 0.50`
- Confirmed DLS, suspected DLS, ambiguous DLS (`143` excluded), and Super Over (`38` separate) policies intact
- Pre-delivery pressure models, 5-fold `GroupKFold` OOF baselines, disjoint `PI < 60` / `PI >= 60` partition, Empirical Bayes shrinkage (`k0_waa = 3,182.6`, `55/15/15/15` bowling / `45/25/15/15` batting), `LCB` (`lcb_z = 0.50`), `PRI`, and `(Neff >= 30, IPL balls >= 30)` eligibility intact.

### 5.1 Critical Code Audit Finding: `Emerging/Rising T20` Classification
- **What Currently Happens in Code (Unchanged During Audit):**
  - [`src/pressure.py#L366`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/pressure.py#L366) computes `is_ipl = comp_meta['comp_name'].str.contains('IPL', case=False, na=False)`.
  - Consequently, `Mens T20 Emerging Teams Asia Cup 2024` and `Asia Cup Rising Stars 2025` (`6,985` deliveries = `3.5%` of corpus) receive `is_ipl = False`.
  - In [`src/performance.py#L460`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/performance.py#L460) and [`#L656`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/performance.py#L656) (`np.where(is_ipl, 1.0, config.w_t20i)`), **Emerging/Rising T20 deliveries currently receive the Senior `T20I` weight `0.50` silently**, and share the `T20I` OOF baseline `is_ipl_int = 0` and LOO `is_ipl = False` par curve!
- **Quantified Sensitivity Evidence (Without Editing Any Code/Outputs):**
  Testing complete exclusion (`w_emerging = 0.00`) against current silent inclusion (`w_emerging = 0.50`) on the primary OOF pool shows:
  - **Batting:** `elig_primary` changes `189 -> 188`; **Top-10 overlap = `10/10`**; `Spearman rho = 0.999021`; `max |ΔPRI| = 7.9822` (`16` eligible batters affected, max ball share `26.8%`).
  - **Bowling:** `elig_primary` stays `200 -> 200`; **Top-10 overlap = `10/10`**; `Spearman rho = 0.994964`; `max |ΔPRI| = 23.9175` (`17` eligible bowlers affected, max ball share `47.4%`).
- **Evidence-Based Proposal Required Before Phase 4:** Because `Emerging/Rising T20` must remain separately classified and not silently inherit `w_t20i = 0.50`, we propose adding an explicit `comp_tier` (`'IPL'`, `'T20I'`, `'Emerging_Rising'`) and explicit `w_emerging` parameter (comparing `0.00` vs `0.25` vs `0.50` in sensitivity) via a separate approved Phase 3 patch—preserving `IPL = 1.00`, `Senior T20I = 0.50`, and all Top-10 stability intact. *(Meanwhile, since `0` domestic leagues exist in the ZIP, no domestic league is silently inheriting `w_t20i = 0.50`.)*

### 5.2 Material Impact of Missing Domestic-League Data & `players.csv`
1. **On the Official Round 1 PDF Challenge (`Top 10 Batters` + `Top 10 Bowlers`, unrestricted IPL + T20I):**
   - **Minimal impact on pressure modeling, OOF baselines, EB shrinkage, or PRI rankings:** `200,999` deliveries across `5` IPL seasons (`87,687` balls) and `131` T20I series (`106,327` balls) provide deep pressure calibration and clean `189` / `200` eligible pools. Top-10 membership is unchanged (`10/10`) even if Emerging/Rising is excluded.
   - **Moderate player-identity caveat:** Lack of `players.csv` leaves one unmerged `bbb`-vs-`p11` UAE ID pair (`36153` vs `8729` across matches `37087, 37092, 37095`), cleanly handled by keeping raw IDs separate per project rules.
2. **On Youngest / Uncapped / Domestic Scouting & RR IPL 2027 Auction Recommendations:**
   - **Severe material impact:**
     - **Player identification & eligibility:** Uncapped Indian players with `< 30` IPL balls (`139` batters, `50` bowlers) or only `India A`/domestic prospects (`139` Emerging-only batters, `92` Emerging-only bowlers, plus all SMAT-only players) are completely invisible or fail `(Neff >= 30, IPL balls >= 30)`.
     - **Domestic-to-IPL translation:** Without SMAT/BBL/SA20 ball-by-ball records, pressure reliability cannot be quantified on pre-IPL domestic performances.
     - **Auction & RR squad fit:** Missing `players.csv` leaves DOB/age, nationality, pre-2022/Test/ODI caps, and IPL 2027 retention status unverifiable solely from the official ZIP.

---

## 6. Final Conclusions & Actionable Recommendations

1. **Definitive Conclusion on Domestic-League Data:**
   The uploaded official data pack [`media_1794423099935_44b59478.zip`](file:///C:/Users/sushruth/.gemini/antigravity/brain/b6ea2651-73fc-4d7f-8e2e-0f0033d4425b/.user_uploaded/media_1794423099935_44b59478.zip) contains **NO domestic T20 league performance data** (`0` matches, `0` deliveries outside IPL, Senior T20Is, and two Asian Emerging/Rising `T20` tournaments).

2. **Exact Supporting Evidence:**
   - Exhaustive check of all `8` files in `Data Pack/` (SHA-256 identical to `data/raw/`), `9` `__MACOSX` entries, `.DS_Store`, and PDF internals confirms no hidden CSVs, links, or `players.csv` exist.
   - All `141` `competitions.csv` rows consist exclusively of `5` IPL editions (`367/371` matches, `87,687` balls), `2` Emerging/Rising Asia Cups (`30/30` matches, `6,985` balls), and `134` T20I series/tournaments (`470/482` played matches, `106,327` balls); `0` matches from SMAT, BBL, SA20, CPL, PSL, Blast, or other domestic T20 leagues exist.
   - `100%` of player delivery performance data lives inside `ball_by_ball.csv` (`200,999 x 33`).
   - Inside the ZIP, `Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf` itself states *"historical IPL and T20 International (T20I) data from 2022 onwards"*, directly explaining why SMAT/BBL/SA20 are absent while still leaving a `players.csv` omission inside the ZIP vs Page 1's supporting-files list.

3. **Should We Contact the Organisers?**
   **Yes — a brief clarification message on Unstop/official channel is recommended before final submission**, *without blocking technical prep on the core `IPL + T20I` Round 1 methodology*.

4. **Precise Questions to Ask the Organisers:**
   1. *"The Round 1 Problem Statement PDF states the dataset covers historical IPL and T20I data from 2022 onwards (matching the 5 IPL seasons, 134 T20Is, and 2 Emerging/Rising Asia Cups in the ZIP), whereas the Unstop Brief page states the data pack contains 'player performance data from domestic leagues'. Could you confirm whether non-IPL domestic leagues (e.g., SMAT, BBL, SA20) were intentionally excluded from Round 1, or whether an updated data pack / additional file is pending under Brief/Case?"*
   2. *"Page 1 of the Round 1 PDF lists `'players'` under Supporting data (`competitions, teams, players, matches, playing XIs, and competition-team-player mappings`), but `players.csv` is not present in the downloaded `Data Pack.zip`. Was `players.csv` omitted intentionally, or can it be uploaded?"*
   3. *"Are Round 1 Top 10 Batters and Top 10 Bowlers evaluated strictly across all eligible IPL + T20I players (unrestricted, per the PDF), or is there any unstated age/uncapped/auction eligibility filter? Also, which Unstop dropdown player list should we align any Emerging/Rising Asia Cup players with?"*

5. **Can Phase 4 Safely Proceed Now, and With What Limitations?**
   - **Yes, for the Official Round 1 PDF Deliverable (`Top 10 Batters` + `Top 10 Bowlers`, unrestricted IPL + T20I)** — provided we first receive your explicit sign-off on:
     - Keeping or explicitly re-weighting `Emerging/Rising T20` (`6,985` balls) via a clean `comp_tier` patch (so it does not silently inherit `w_t20i = 0.50` — sensitivity shows `10/10` Top-10 stability in both disciplines even at `w_emg = 0.00`), and
     - Proceeding with **mandatory transparent disclosures** in the Phase 4 report:
       1. `0` non-IPL domestic T20 league data exists in the official ZIP (`Emerging/Rising T20` separated from `Senior T20I`).
       2. Missing `players.csv` means DOB/age, nationality (for IPL-only players), pre-2022 international caps, and IPL 2027 retention status cannot be verified from the official data pack alone (and therefore any RR IPL 2027 auction / uncapped commentary must either await verified public external augmentation or be explicitly omitted/flagged).
       3. Unresolved raw ID conflict `36153` vs `8729` (retained unmerged per audit policy), `neff_hp < 15` low-HP-support flags (`3/10` batting candidates in `Top10(post)`), Associate T20I exposure (`11.2%`), and `2022–2024` IPL staleness (`~20%`) disclosed cleanly on every candidate profile.
   - **No, NOT without additional external verified data** if your intended Phase 4 output is specifically a **"youngest uncapped / domestic-league RR 2027 auction prospect list"**.

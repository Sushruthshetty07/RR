# Final Uploaded Data Pack Verification & Readiness Audit
**Rajasthan Royals SupeRR Selector Season 2**

> **Strict Audit Guardrail:** This is a read-only verification audit directly against the uploaded archive [`media_1794423099935_44b59478.zip`](file:///C:/Users/sushruth/.gemini/antigravity/brain/b6ea2651-73fc-4d7f-8e2e-0f0033d4425b/.user_uploaded/media_1794423099935_44b59478.zip). Existing code, datasets, thresholds, weights, and Phase 1–3 outputs remain unmodified. Zero player names and zero Top-10 rankings are generated or displayed.

---

## 1. Uploaded Dataset Inventory

### 1.1 Archive Location & Contents
- **Authoritative Uploaded Archive:** `data_pack.zip` (original competition data archive)
  - **Size:** `4,545,438` bytes (`4.33 MiB`)
  - **SHA-256:** `82da62e94840e614ef136f0bb24168fe72b567c3eff71bf6b58488ac9b2b7f7d`
  - **Duplicate Upload Check:** A second archive in `.user_uploaded\` (`media_1794413084545_ef360f5f.zip`, `4,545,438` bytes) has the **exact same SHA-256** (`82da62e94840e614ef136f0bb24168fe72b567c3eff71bf6b58488ac9b2b7f7d`).
- **Archive Structure:** Top-level folder `Data Pack/` (`8` non-directory entries) plus `9` macOS resource-fork / AppleDouble entries under `__MACOSX/` (`120`–`489` bytes each) and `1` `Data Pack/.DS_Store` (`6,148` bytes). No nested archives (`.zip`/`.gz`/`.tar`) exist inside the zip.

### 1.2 File-by-File Inventory (`Data Pack/`)

| Archive Entry (`Data Pack/...`) | Format | Uncompressed Size (Bytes) | SHA-256 Prefix | Shape `(Rows x Cols)` | Date Range / Key Identifiers |
|---|---|---:|---:|---:|---|
| `ball_by_ball.csv` | CSV | `41,071,837` | `01e2b4379edfbab6` | `200,999 x 33` | `867` unique `match_id`, `888` strikers, `867` non-strikers, `647` bowlers (`959` combined unique `player_id`s); timestamps all parse cleanly (`0` errors) |
| `playing_eleven.csv` | CSV | `966,287` | `7517c73722a28d5d` | `28,381 x 9` | `880` unique `match_id`, `38` `team_id`, `1,082` unique `player_id` |
| `competition_team_player_mapping.csv` | CSV | `462,969` | `1ded4e42a9336185` | `6,413 x 6` | `140` `comp_id`, `38` `team_id`, `1,217` unique `player_id` (`1,210` unique `player_name`) |
| `matches.csv` | CSV | `123,864` | `bb100ab84b8b9a15` | `883 x 22` | `883` `match_id`, `138` `comp_id`, `98` `venue_id`, `36` home/away teams; `match_date` **`2022-01-22` to `2026-11-01`** (`0` invalid dates) |
| `competitions.csv` | CSV | `7,624` | `507ceb7189f972d4` | `141 x 4` | `141` `comp_id`, `2` formats (`T20I`, `T20`), `8` season strings (`2022`–`2026` plus `2022/23`, `2023/24`, `2024/25`) |
| `teams.csv` | CSV | `6,922` | `a65e5922a9b3bf49` | `353 x 3` | `141` `comp_id`, `38` unique `team_id` / `team_name` |
| `Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf` | PDF | `254,082` | `8f7965c616d4a64b` | `2 pages` | Official Unstop Round 1 challenge specification |
| `.DS_Store` | macOS | `6,148` | `d65165279105ca67` | — | Finder metadata (non-data artifact) |

### 1.3 Column Inventory Across All 6 CSVs
- **`competitions.csv` (`4`):** `comp_id`, `comp_name`, `match_format`, `season`.
- **`teams.csv` (`3`):** `comp_id`, `team_id`, `team_name`.
- **`matches.csv` (`22`):** `match_id`, `comp_id`, `match_number`, `match_date`, `stage`, `venue_id`, `venue_country`, `home_team_id`, `home_team`, `away_team_id`, `away_team`, `home_team_captain_id`, `away_team_captain_id`, `toss_win_team_id`, `toss_opted`, `man_of_the_match`, `first_inn_revised_overs`, `second_inn_revised_overs`, `revised_target`, `knock_out`, `win_team_id`, `result`.
- **`playing_eleven.csv` (`9`):** `match_id`, `team_id`, `player_id`, `batting_order`, `is_played`, `is_captain`, `is_wk`, `is_impact`, `is_impact_off`.
- **`competition_team_player_mapping.csv` (`6`):** `comp_id`, `comp_name`, `team_id`, `team_name`, `player_id`, `player_name`.
- **`ball_by_ball.csv` (`33`):** `match_id`, `innings`, `overid`, `over_ball_num`, `innings_ball_num`, `batting_team_id`, `bowling_team_id`, `striker_id`, `striker_name`, `non_striker_id`, `non_striker_name`, `bowler_id`, `bowler_name`, `batting_type`, `bowling_type`, `batting_order`, `runs`, `is_four`, `is_six`, `is_wicket`, `dismissal_type_id`, `dismissal_type`, `dismissed_player_id`, `extras_type`, `extras1_runs`, `extras_2_runs`, `pace_or_spin`, `bowl_speed`, `stroke`, `delivery_style`, `over_or_around`, `on_or_off`, `delivery_ist_time`.

---

## 2. Dataset Completeness and Authenticity Assessment

1. **Authenticity Verified:** All CSV files and the 2-page PDF unpack cleanly with zero corruption, zero empty CSVs, zero unparsed timestamps or dates, and complete delivery logs across all `867` played matches (`200,999` balls).
2. **Exact Byte-for-Byte Identity With Existing Local Data:** Hashing confirms that the newly uploaded archive (`media_1794423099935_44b59478.zip`) is the exact archive already unpacked in [`data/raw/`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/data/raw).
3. **PDF "Supporting Data" Discrepancy (`players` vs `players.csv`):** Page 1 of `Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf` reads verbatim:
   - *"You will be provided with historical IPL and T20 International (T20I) data from 2022 onwards, including: Ball-by-ball data: delivery-level match records in CSV format. Supporting data: competitions, teams, players, matches, playing XIs, and competition-team-player mappings."*
   - **Critical Finding:** No standalone `players.csv` file exists inside the official downloaded archive (`Data Pack/`), nor inside `__MACOSX/`. Player biographical attributes (`player_name`, `batting_type`, `bowling_type`, `pace_or_spin`, `is_wk`, `batting_order`) are instead embedded inside `ball_by_ball.csv`, `playing_eleven.csv`, and `competition_team_player_mapping.csv`. Age/DOB, nationality, and capped/uncapped metadata were not included in the data pack.

---

## 3. Competition-Wise Coverage

### 3.1 Summary Across Competition Buckets (`867` Played Matches / `883` Listed; `200,999` Deliveries)

| Competition Bucket | Format (`competitions.csv`) | Comps (`Played / Listed`) | Matches (`Played / Listed`) | Deliveries | Unique `ball_by_ball` Players |
|---|---|---:|---:|---:|---:|
| **IPL (`2022–2026`)** | `T20` | `5 / 5` | `367 / 371` | `87,687` (`43.6%`) | `382` |
| **Emerging / Rising T20** (`Mens T20 Emerging Teams Asia Cup 2024`, `Asia Cup Rising Stars 2025`) | `T20` | `2 / 2` | `30 / 30` | `6,985` (`3.5%`) | `172` |
| **Senior T20 Internationals (`2022–2026`)** | `T20I` | `131 / 134` | `470 / 482` | `106,327` (`52.9%`) | `709` |
| **Domestic T20 Leagues** (SMAT, BBL, SA20, CPL, PSL, Blast, etc.) | — | **`0 / 0`** | **`0 / 0`** | **`0` (`0.0%`)** | **`0`** |
| **Other Formats** (ODI, Test, List A, First-Class, T10, etc.) | — | **`0 / 0`** | **`0 / 0`** | **`0` (`0.0%`)** | **`0`** |
| **Total Corpus** | `T20`, `T20I` | **`138 / 141`** | **`867 / 883`** | **`200,999`** | **`959`** (`1,217` in `pmap`) |

### 3.2 Season & Per-Tournament Breakdown of `T20` Competitions (`7` Total)
- `comp_id=1358` | `IPL 2022` (`2022`): `74/74` matches, `17,909` deliveries, `190` unique players.
- `comp_id=1715` | `IPL 2023` (`2023`): `74/74` matches, `17,863` deliveries, `208` unique players.
- `comp_id=2193` | `IPL 2024` (`2024`): `71/74` matches (`3` unplayed), `17,103` deliveries, `208` unique players.
- `comp_id=2670` | `Mens T20 Emerging Teams Asia Cup 2024` (`2024`): `15/15` matches, `3,519` deliveries, `108` unique players.
- `comp_id=2725` | `IPL 2025` (`2025`): `74/75` matches (`1` unplayed; `75` rows in `matches.csv`), `17,285` deliveries, `200` unique players.
- `comp_id=3289` | `Asia Cup Rising Stars 2025` (`2025`): `15/15` matches, `3,466` deliveries, `108` unique players.
- `comp_id=3298` | `IPL 2026` (`2026`): `74/74` matches, `17,527` deliveries, `202` unique players.

### 3.3 Teams & Temporal Distribution
- **T20 Teams (`18` total):** `10` IPL franchises plus `8` Emerging/Rising teams (`India A`, `Pakistan A`, `Sri Lanka A`, `Bangladesh A`, `Afghanistan A`, `Hong Kong`, `Oman`, `United Arab Emirates`).
- **T20I Teams (`23` total):** `12` Full Members (`88.8%` of T20I deliveries = `94,377` balls in Full-Member-only clashes) and `11` Associates (`11.2%` = `11,950` balls across `54` listed matches with `>=1` Associate side).
- **Played Matches by Calendar Year:** `2022`: `180`, `2023`: `140`, `2024`: `192`, `2025`: `211`, `2026`: `144` (`total = 867`).
- **Deliveries by Innings:** Innings `1`: `105,727`; Innings `2`: `95,234`; Super Over Innings `3`: `25`; Super Over Innings `4`: `13` (across `6` matches).

### 3.4 Resolution Check on Previous Audit
- **Does the newly uploaded official archive resolve the `0` domestic-league issue? No.** It contains **zero domestic T20 leagues**. Furthermore, inspecting Page 1 of the official PDF confirms why: the PDF explicitly scopes the challenge dataset to *"historical IPL and T20 International (T20I) data from 2022 onwards"*, not domestic leagues (aside from the two Asian Emerging/Rising `T20` competitions).

---

## 4. Comparison With Existing Local `data/raw` Files

SHA-256 digests were computed for every non-directory file inside `Data Pack/` in `media_1794423099935_44b59478.zip` and compared against `data/raw/`:

| Check Dimension | Uploaded `Data Pack/` vs `data/raw/` | Finding |
|---|---|---|
| **Byte-for-Byte SHA-256 Match** | `8 / 8` files **IDENTICAL SHA-256** (`competitions.csv`, `teams.csv`, `matches.csv`, `ball_by_ball.csv`, `competition_team_player_mapping.csv`, `playing_eleven.csv`, `Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf`, `.DS_Store`) | Zero byte differences |
| **New or Missing Files** | Extra in `data/raw`: `[]`; Missing from `data/raw`: `[]` | Exact `1:1` file parity |
| **Added or Removed Columns** | All 6 CSV schemas (`4, 3, 22, 9, 6, 33` columns) match | Zero schema changes |
| **Row Counts & Date Range** | Exact same row counts (`141`, `353`, `883`, `28,381`, `6,413`, `200,999`) and date span (`2022-01-22` to `2026-11-01`) | Zero data updates |
| **New Player IDs / Competitions** | None (`1,217` raw IDs in `pmap`, `141` `comp_id`s) | Zero new entities |

**Conclusion:** The existing workspace `data/raw/` dataset was **NOT incomplete or stale** compared to the official hackathon upload—it is an exact bit-for-bit extract of the uploaded archive.

---

## 5. Player Metadata Coverage

| Metadata Field | Availability | Source & Precision | Unreliable / Missing Edge Cases |
|---|---|---|---|
| `player_id`, `player_name` | **Available (`100%`)** | `pmap` (`1,217` IDs, `1,210` names), `bbb`, `p11` | `1` verified cross-source alias (`39444`/`8638`); `7` shared-name clusters in raw `pmap` (`6` distinct pairs kept separate in `player_identity.py`); `1` remaining `bbb`-vs-`p11` ID mismatch in Emerging Cup matches `37087, 37092, 37095` (`36153` vs `8729`) |
| `team_id`, `team_name`, `comp_id` | **Available (`100%`)** | `pmap` (`6,413` rows), `teams.csv` (`38` teams) | `0` missing values; `1` listed unplayed T20I comp (`comp_id=2940`) absent from `pmap` |
| `batting_type` (LHB/RHB) | **Available (`100%` of `bbb`)** | `ball_by_ball.csv` (`0` conflicts per `striker_id`) | Only populated for players who faced `>= 1` delivery |
| `pace_or_spin` | **Available (`100%` of `bbb`)** | `ball_by_ball.csv` (`Pace` / `Spin`, `0` conflicts per `bowler_id`) | Only populated for bowlers who bowled `>= 1` delivery |
| `bowling_type` (`12` sub-styles) | **Available (`100%` of `bbb`)** | `ball_by_ball.csv` | **`47` bowlers** have `>1` distinct `bowling_type` label across deliveries (e.g., speed-band reclassification like medium vs fast-medium) |
| `bowl_speed` | **Partially Unreliable** | `ball_by_ball.csv` (`0.0%` NaN, but sentinel-coded) | **`77,862` rows (`38.74%`) are `<= 0` (`-1.0` or `0.0`)**—including **`100%`** of Emerging/Rising balls and `45.3%` of T20Is; **`20` rows exceed `165 kph`** (sensor spikes up to `1456.1 kph`); `22` rows sit in `(0, 40)` kph |
| `batting_order`, `is_wk`, `is_captain`, `is_impact`, `is_impact_off` | **Available (`~100%` when `is_played == True`)** | `playing_eleven.csv`, `ball_by_ball.csv` (`batting_order` `1–13`) | `30.5%–30.6%` NaNs in `p11` (`8,667–8,678` rows) correspond to bench players (`is_played == False`); `35` `(match_id, team_id, player_id)` keys (`75` rows) have duplicate entries differing only on `batting_order` |
| **DOB / Age** | **MISSING (`0%`)** | Not present anywhere in archive | Required only if an age cutoff is imposed (NOT required by PDF brief) |
| **Nationality / Capped Status / IPL Auction Pool & Retentions** | **MISSING (`0%`)** | Not present anywhere in archive | Cannot verify Indian domestic vs overseas passport for IPL-only players, pre-2022/Test/ODI caps, or IPL 2027 retention/auction registration |

---

## 6. Data-Quality Audit

### 6.1 Key Uniqueness & Referential Integrity
- **Exact Duplicate Rows:** `0` across all 6 CSVs.
- **Primary Key Uniqueness:** `0` duplicate keys in `competitions` (`comp_id`), `teams` (`comp_id, team_id`), `matches` (`match_id`), `pmap` (`comp_id, team_id, player_id`), or `bbb` (`match_id, innings, overid, over_ball_num` and `match_id, innings, innings_ball_num`).
- **Foreign-Key Joins:**
  - `0` orphan `comp_id` in `matches` or `pmap`; `0` orphan `match_id` in `p11` or `bbb`.
  - `0` orphan `team_id` in `matches` or `bbb`; `0` invalid `win_team_id` (`41` nulls correspond to no-results/ties/abandoned).
  - `0` `bbb` (`striker_id`, `non_striker_id`, `bowler_id`, `dismissed_player_id`) or `p11` `player_id` values missing from `pmap`.

### 6.2 Audited Quirks & Anomalies Requiring Awareness (Raw IDs Preserved)
1. **`playing_eleven.csv` Duplicate Keys (`40` excess rows across `35` keys / `75` rows):** In `3` T20I series (*Sri Lanka in Australia 2022*, *Sri Lanka in New Zealand 2023*, *Afghanistan in India 2024*), the same `(match_id, team_id, player_id)` appears twice with different `batting_order` values (e.g., `5.0` vs `6.0`). Deduplicating on `(match_id, team_id, player_id)` is necessary before any `p11` join.
2. **Unplayed Matches (`16` of `883` in `matches.csv`):** `13` unplayed matches have `playing_eleven.csv` rows but `0` deliveries in `ball_by_ball.csv`, while `3` abandoned T20I matches (`match_id`s `36210`, `41503`, `48513`) have `0` `playing_eleven.csv` rows and `0` deliveries. Additionally, `3` listed T20I series (`comp_id`s `2940`, `3405`, `3473`) have `0` matches (`2940` also absent from `pmap`).
3. **`man_of_the_match == 0` Sentinel (`119` matches):** `119` rows in `matches.csv` use `0` (not in `pmap`), primarily in abandoned/incomplete or non-award matches.
4. **`ball_by_ball` vs `playing_eleven` Identity Conflicts in Matches `37087, 37092, 37095`:**
   - Raw comparison finds `6` `(match_id, player_id)` pairs active in `bbb` yet absent from `p11` (`[36153, 39444]` across `37087, 37092, 37095`), while `[8638, 8729]` appear in `p11` with `is_played == True` yet `0` `bbb` actions (`22` played rows in `p11`, `22` unique raw players in `bbb`).
   - Verified alias `39444 -> 8638` in [`player_identity.py`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/player_identity.py) resolves `3` of the `6` rows.
   - The remaining `3` rows correspond to `36153` (`bbb_bat = 49`, `p11_rows = 0`) vs `8729` (`p11_rows = 6`, `bbb = 0`) in the same `3` UAE matches. Consistent with the project rule *never merge players based solely on similar names*, raw IDs (`36153` and `8729`) remain untouched and unmerged unless independent corroboration is explicitly approved.
   - In raw `pmap`, `7` `player_name` strings map to `>1` distinct `player_id` (`6` distinct pairs after `39444 -> 8638`).
5. **Wicket & Extras Schema Cases:**
   - `1` delivery has `is_wicket == True` and `dismissed_player_id == NaN` (`dismissal_type == 'Retire Out'`).
   - `12` deliveries have `is_wicket == False` and non-null `dismissed_player_id` (`retired hurt` events).
   - `extras_type` counts: `NaN` (`189,167`), `wide` (`7,714`), `legbyes` (`2,748`), `Overstep` (`576`), `byes` (`529`), `other` (`264`), `Penalty` (`1`).

---

## 7. Compatibility With Completed Phase 1–3 Work

Because the uploaded `Data Pack/` is SHA-256 identical to `data/raw/`, every completed artifact in [`match_state.py`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/match_state.py), [`pressure.py`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/pressure.py), [`performance.py`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/performance.py), and [`phase3_methodology_report.md`](file:///C:/Users/sushruth/.gemini/antigravity/brain/b6ea2651-73fc-4d7f-8e2e-0f0033d4425b/phase3_methodology_report.md) aligns `100%` with the authoritative upload. All `30` automated tests pass (`Ran 30 tests in 40.732s — OK`).

| Phase 1–3 Component | Status vs Uploaded Data Pack | Specific Nuance / Observation |
|---|---|---|
| **IPL vs Senior T20I Classification** | **Preserve Approved (`1.00` vs `0.50`)** | [`pressure.py#L366`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/pressure.py#L366) assigns `is_ipl = True` strictly to the `5` IPL competitions (`87,687` balls) and `is_ipl = False` to everything else (`113,312` balls). |
| **Domestic Leagues vs Emerging/Rising vs T20I Grouping** | **No Domestic Leagues Present; Emerging/Rising Grouped with T20I** | There are **0 domestic league matches** to be misclassified. However, `2` Asian `T20` tournaments (`Emerging Teams Asia Cup 2024` + `Asia Cup Rising Stars 2025`, `6,985` balls = `3.5%` of corpus) have `is_ipl = False` and share the `w_t20i = 0.50` bucket and `is_ipl_int = 0` OOF baseline tier with senior T20Is (`106,327` balls). |
| **Pre-Delivery State, DLS & Super Overs** | **Fully Preserved & Compatible** | `143` ambiguous DLS balls set to `NaN` and excluded; `701` suspected DLS isolated for sensitivity (`9/10` bat, `10/10` bowl Top-10 overlap); `38` Super Over balls across `6` matches kept strictly separate (`innings >= 3`); `conc_i` cleanly charges `wide`/`Overstep`/`other` (`8,554` balls) while excluding `byes`/`legbyes`/`Penalty` (`3,278` balls). |
| **Disjoint General vs High-Pressure Partition** | **Fully Preserved & Compatible** | `PI < 60` (`raa`, `surv`, `cons`, `rsaa`, `waa`) and `PI >= 60` (`hp_raa`, `hp_impact`) have `0` shared deliveries (`max weight conservation diff = 5.68e-14`). |
| **Empirical Bayes Shrinkage, `WAA` (`55/15/15/15`), LCB & PRI** | **Fully Preserved & Compatible** | Heavily shrunk `WAA` (`k0 = 3,182.6`) remains capped at `15%` weight (`6.7%` PRI variance share); `lcb_z = 0.50` and Normal-CDF `PRI` require no modification. |
| **Eligibility `(Neff >= 30, IPL balls >= 30)`** | **Aligned With Official PDF** | Produces `189` eligible batters and `200` eligible bowlers (`84` eligible in both disciplines) appropriate for the unrestricted IPL + T20I Round 1 brief. (Under a hypothetical uncapped domestic-only mandate, it would exclude `139` Emerging-only batters and `92` Emerging-only bowlers.) |

---

## 8. Official Challenge Requirements Verification

Inspecting the official [`Round 1 Problem Statement _ SupeRR Selector Season 2.docx.pdf`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/data/raw/Round%201%20Problem%20Statement%20_%20SupeRR%20Selector%20Season%202.docx.pdf) inside the uploaded archive confirms **zero ambiguity in the hackathon's official rules**:

1. **Exact Deliverables Required:**
   - **Component 01 — Player Rankings (Unstop Dropdowns):** **Two separate ordered Top-10 lists**:
     - `Top 10 batters`: Rank `1` through Rank `10` (no repeats).
     - `Top 10 bowlers`: Rank `1` through Rank `10` (no repeats).
   - **Component 02 — Methodology Report:** Document covering pressure definition, performance measurement, fair comparisons (sample sizes, eligibility, IPL vs T20I), evidence for Top 10, and reliability/limitations.
2. **Population & Eligibility Restrictions in Official Rules:**
   - **NONE.** The official PDF asks to rank the most reliable batters and bowlers under pressure across *"historical IPL and T20 International (T20I) data from 2022 onwards"*. **It does NOT restrict eligibility by age, uncapped status, domestic league status, nationality, or IPL 2027 auction pool.**

### Missing Information and Limitations (If Evaluated Against Non-PDF / Auction-Filter Framing)
- **No `players.csv`:** No DOB, age, nationality, or international cap ledger exists in the official upload.
- **No Domestic T20 Leagues:** `0` matches from SMAT, BBL, SA20, CPL, or state competitions exist in the official upload.
- **Residual High-Pressure Sparsity:** Inside the official eligible pool, `3/10` batters in `Top10(post)` (`2/10` in `Top10(LCB)`) have `neff_hp < 15`, requiring transparent uncertainty flags in the Phase 4 report.
- **Associate T20I Exposure (`11.2%`) & Staleness (`~20%`):** Phase 4 player cards should display IPL share, Emerging/Rising share, Associate-opponent share, and last IPL season (`2022–2026`).

---

## 9. Required Corrections Before Phase 4

No changes are required to Phase 1–3 pressure formulas, DLS exclusions, Super Over separation, disjoint `PI < 60 / PI >= 60` partition, shrinkage priors, competition weights (`1.00 / 0.50`), or eligibility `(Neff >= 30, IPL >= 30)` for the official Round 1 submission. Before generating Phase 4 deliverables, implement only these **lightweight Phase 4 presentation/hygiene controls** (upon user sign-off):

1. **Confirm Governing Scope:** Execute Phase 4 strictly to the official PDF deliverable (**two ordered Top-10 lists: Batters `1–10` and Bowlers `1–10`**, unrestricted across `IPL + T20I`), preserving `match_state.py`, `pressure.py`, and `performance.py` decisions unchanged. *(If a separate "youngest uncapped / RR 2027 auction" overlay is still desired, verified external DOB/uncapped/auction data must first be provided.)*
2. **Clean Role & Provenance Diagnostics in Phase 4 Tables:**
   - Deduplicate `playing_eleven.csv` on `(match_id, team_id, player_id)` before joining `batting_order`, `is_wk`, and `is_impact`.
   - Filter `bowl_speed` with `(bowl_speed > 0) & (bowl_speed <= 165)` before summarizing bowling pace.
   - Surface per-candidate diagnostics on every ranked card: `ipl_balls`, `t20i_balls`, `emerging_balls` (split visibly from senior `T20I`), `neff`, `neff_hp`, `hp_low_support_flag` (`neff_hp < 15`), `last_ipl_season`, role (`LHB/RHB`, `pace/spin`, `PP/Mid/Death` shares, `two_way_allrounder`), and posterior uncertainty/`LCB`.
3. **Unstop Selection Reconciliation:** Verify exact spelling between `canonical_player_id` names and Unstop player-selection dropdown entries prior to final submission (`player_identity.py` `CANONICAL_ID_MAP` preserved).

---

## 10. Final Readiness Verdict

**Verdict: `READY WITH REQUIRED FIXES`** *(for the Official Unstop Round 1 SupeRR Selector Season 2 Challenge)*

- **Why `READY WITH REQUIRED FIXES`:** The newly uploaded archive [`media_1794423099935_44b59478.zip`](file:///C:/Users/sushruth/.gemini/antigravity/brain/b6ea2651-73fc-4d7f-8e2e-0f0033d4425b/.user_uploaded/media_1794423099935_44b59478.zip) is `100%` genuine and **byte-for-byte identical** to `data/raw/`, and it **completely satisfies the official Round 1 PDF requirement** (*"historical IPL and T20 International (T20I) data from 2022 onwards"* -> **Top 10 Batters + Top 10 Bowlers**). Phase 1–3 requires zero rework. Only lightweight Phase 4 reporting hygiene (deduplicating the `35` `playing_eleven` keys, filtering `bowl_speed` sentinels/spikes, separating `Emerging/Rising` vs `Senior T20I` ball counts, and flagging `neff_hp < 15`) and explicit user sign-off on the PDF scope are required before generating final rankings.
- *(Note on Uncapped / Domestic-League Framing: Relative to an external "youngest uncapped domestic-league RR 2027 auction" filter—which is NOT in the official Round 1 PDF—the dataset would be **NOT READY** because `players.csv`, age/DOB, nationality, capped status, auction availability, and domestic leagues (`0` matches) do not exist in the official data pack.)*

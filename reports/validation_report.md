# Audit & Validation Report: SupeRR Selector 2.0

## Executive Summary
An independent, delivery-level audit was conducted on `src/match_state.py`, the 6 competition CSV datasets, and all statistical claims in `eda_report.md`.

All original datasets in `data/raw/` and existing EDA outputs (`notebooks/01_initial_eda.ipynb`, `reports/figures/`, `scripts/run_eda.py`) were preserved unmodified. Four critical logic bugs were discovered and fixed in `src/match_state.py`, several statistical inaccuracies in `eda_report.md` were identified, and a 10-test verification suite (`tests/test_match_state.py`) passes 100%.

---

## 1. Audit of `src/match_state.py` & Corrections Made

### 1.1 Pre-Delivery State Isolation (Verified)
- **Sorting Order**: Verified across all 200,999 rows that sorting by `['match_id', 'innings', 'overid', 'over_ball_num']` matches `['match_id', 'innings', 'innings_ball_num']` with 0 discrepancies.
- **Pre-Delivery Arithmetic**: `score_before`, `wickets_before`, `legal_balls_before`, `wickets_in_hand_before`, `legal_balls_remaining_before`, `crr_before`, `runs_needed_before`, and `rrr_before` are computed strictly before the current delivery is bowled. Verified that ball 1 of every `(match_id, innings)` starts at `score_before = 0`, `wickets_before = 0`, `legal_balls_before = 0`, `crr_before = 0.0`.

### 1.2 Critical Bugs Found & Fixed
1. **Revised Overs Stored in Balls (not Overs)**: `first_inn_revised_overs` and `second_inn_revised_overs` in `matches.csv` are stored in legal balls (e.g., `90.0` for 15 overs in IPL 2023 Final `21586`, `66.0` for 11 overs). Original code multiplied by `6` (turning 15 overs into `540` balls). Fixed by removing `* 6`.
2. **Pre-Start Shortened Matches Innings 2 Fallback**: Matches `16753` and `19959` set `first_inn_revised_overs = 66.0` (11 overs) and left `second_inn_revised_overs = NaN`. Original code defaulted Innings 2 to `120` balls instead of inheriting `66`. Fixed so Innings 2 falls back to `first_inn_revised_overs` when `second_inn_revised_overs` is null.
3. **`extras_type == 'other'` (264 deliveries) Must Not Count as Legal Ball**: Excluding only `wide` and `Overstep` left 150 innings with 121+ legal balls. `'other'` represents non-overstep no-balls that are re-bowled (excluding `{'wide', 'Overstep', 'other'}` reduces $>120$-ball innings from 150 to 0). Also added `is_batter_ball_faced` (`!= 'wide'`) and `bowler_runs_conceded`.
4. **Super Over (`innings >= 3`) Wickets & Phase**: Super Overs allow 2 wickets (max 2 observed), not 10, and were mislabeled as `'Powerplay'` because `overid == 1`. Fixed `max_wickets = 2` and `match_phase = 'Super Over'` for `innings >= 3`.

---

## 2. Verification of `overid`, Score Totals, Extras & Wickets
- `overid` is **1-indexed** (`1..20`).
- Score totals reconciled against `matches.csv['result']` across all 372 non-DLS wicket wins and 361 non-DLS run wins with **0 anomalies**.
- Wickets: `10,881` total (`10,272` bowler-credited, `609` field/other).

---

## 3. Data Quality Issues: Duplicates, Missing Records & Identifiers
- **0 duplicate deliveries** across all 200,999 rows.
- **16 missing matches** in `ball_by_ball.csv` are all `Match Abandoned` without a ball bowled; **3 matches** missing from `playing_eleven.csv` (`36210`, `41503`, `48513`) are abandoned matches. **3 competitions** (`3405`, `3473`, `2940`) have 0 matches in `matches.csv`. **29** NR/Abandoned/Tied matches have rows in `ball_by_ball.csv`.
- **12 rows** have `is_wicket == False` with `dismissed_player_id` non-null; **1 row** (`127899`, `Retire Out`) has `is_wicket == True` with `dismissed_player_id` null; **2 rows** (`36854`, `107489`) have extras runs with `extras_type == NaN`; **23 rows** have `runs` of 4 or 6 with `is_four == False` / `is_six == False`.
- **Player ID/Name Collisions**: Never group by `player_name`. Homonyms exist (`Abdul Samad`, `Haider Ali`, `Rashid Khan`), likely duplicate IDs exist (`Basil Hameed`, `Bashir Ahmad`, `Mohammad Ali`, `Muhammad Farooq`), and 2 BBB players (`36153`, `39444`) are absent from `playing_eleven.csv`.

---

## 4. Independent Fact-Check of `eda_report.md`
- **Unique batters**: `eda_report.md` stated `958` unique batters; actual unique `striker_id` is **888** (`striker | non_striker` = **893**, total unique BBB players = **959**).
- **Dot balls**: `eda_report.md` stated `79,387` (`39.5%`) dot balls; that is off-bat `runs == 0`. True dot balls (`ball_total_runs == 0`) = **68,027** (**33.84%**).
- **Boundaries**: `is_four | is_six` = **36,094** (**17.96%**), vs `36,117` (`runs in {4, 6}`).
- **Phase stats**: Actual boundary % is PP **21.83%**, Mid **15.39%**, Death **21.43%** (vs `19.4%`, `14.1%`, `23.8%` in `eda_report.md`). Actual mean runs/over is PP `6.86–9.41`, Mid `7.43–8.96`, Death `8.94–11.80`.

---

## 5. Test Results
`python -m unittest discover -s tests -v`: **10 tests ran in 2.057s — OK**.

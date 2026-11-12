# Phase 2 Final Audit Report (Pre-Phase 3 Verification)

> **Scope**: Focused audit of DLS ambiguity gating, LOO contributing match support (`0–119`), opportunity exclusions, and regression tests.  
> **Constraints Preserved**: Zero changes to any pressure formulas or weights (`PressureConfig`); no player performance scoring or Top 10 rankings; raw CSVs, Phase 1 EDA outputs (`01_`–`06_`), and prior `reports/phase2_pressure_report.md` preserved.

---

## 1. DLS Ambiguity Audit & Correction (143 Pre-Stoppage Deliveries)

### Finding Before Fix
`detect_dls_regimes` (`src/pressure.py`) flagged all **143 `dls_pre_stoppage_suspected` deliveries** (4 matches where mid-innings interruption timing was reconstructed via inter-ball timestamp gaps `>= 12.0` min) with `dls_ambiguous = True` and prevented future `revised_target` leakage. However, `compute_pressure_scores` previously gated Innings 2 using:
```python
inn2_mask = (df['innings'] == 2) & (df['dls_status'] != 'dls_ambiguous_timing') & df['rrr_eff'].notna()
```
As a result, all **143 `dls_pre_stoppage_suspected` deliveries produced non-NaN regular pressure scores (`143 / 143`)** and entered regular Innings 2 distributions (`95,234` rows) and player opportunity totals (`max total_balls = 2,829`).

| Match ID | Competition | Pre-Stoppage Balls (`dls_ambiguous=True`) | Post-Stoppage Balls (`dls_ambiguous=False`) | Detected Gap & Location | Before Fix (`pressure_bat`/`bowl`) | After Fix (`pressure_bat`/`bowl`) |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| `13612` | Sri Lanka in Australia T20I 2022 | `93` | `25` | `13.18m` at over `15.7` | Non-NaN (`93/93`) | **`NaN` (`93/93`)** |
| `21586` | IPL 2023 (Final) | `3` | `91` | `140.72m` at over `1.4` | Non-NaN (`3/3`) | **`NaN` (`3/3`)** |
| `34428` | India in Sri Lanka T20I 2024 | `3` | `36` | `69.07m` at over `1.4` | Non-NaN (`3/3`) | **`NaN` (`3/3`)** |
| `48611` | England in Sri Lanka T20I 2026 | `44` | `62` | `74.17m` at over `8.3` | Non-NaN (`44/44`) | **`NaN` (`44/44`)** |
| **Total** | **4 matches** | **`143`** | **`214`** | `gap >= 12.0 min` | Included (`143`) | **Excluded (`143 / 143 NaN`)** |

### Correction Applied
Updated `src/pressure.py` (`inn2_mask`) to gate on **all** ambiguous DLS rows:
```python
inn2_mask = (df['innings'] == 2) & (~df['dls_ambiguous'].fillna(False)) & df['rrr_eff'].notna()
```
Confirmed vs suspected DLS statuses remain cleanly separated (`0` overlap):
- `dls_pre_innings_confirmed` (**12 matches, `797` balls**, `dls_ambiguous=False`, finite scores)
- `dls_innings_break_suspected` (**7 matches, `487` balls**, `dls_ambiguous=False`, finite scores)
- `dls_post_stoppage_suspected` (**4 matches, `214` balls**, `dls_ambiguous=False`, finite scores)
- `dls_pre_stoppage_suspected` (**4 matches, `143` balls**, **`dls_ambiguous=True`, strictly `NaN`**)
- `dls_ambiguous_timing` (`0` dataset balls; synthetic fallback test coverage, **`dls_ambiguous=True`, strictly `NaN`**)

Post-fix Innings 2 trusted deliveries: `95,091` (`95,234 - 143`), with `0` missing values among trusted regular deliveries (`~is_super_over & ~dls_ambiguous`). Processed output `data/processed/deliveries_with_pressure.csv.gz` was regenerated cleanly.

---

## 2. LOO Support Counts Across Ball Indices (`0–119`) & Fallback Documentation

In standard 20-over Innings 1 (`max_legal_balls == 120`), every match within the same competition (`IPL` vs `T20I`) at ball index `b` sees the same leave-one-match-out pool size `n_c_loo(b) = c_n(b) - 1`, so at any single ball index `b`, `min = median = max`.

- **IPL (`is_ipl == True`) across `b = 0..119`**: **min `335`**, **median `362`**, **max `363`** (shortened Inn 1 `3` matches / `276` balls: `336–364`).
- **T20I (`is_ipl == False`) across `b = 0..119`**: **min `408`**, **median `474`**, **max `477`** (shortened Inn 1 `22` matches / `1,353` balls: `409–478`).
- **Stability & Fallback**: `0 / 105,727` Innings 1 deliveries fall below `min_loo_support = 30`. Tail curves (`b = 110..119`) remain monotonic and smooth (`loo_exp_score` `168.21 -> 187.75` IPL; `154.82 -> 174.34` T20I). Below `min_support = 30`, `compute_loo_inn1_baselines` blends toward the pooled LOO curve (`alpha = clip(n_c_loo / 30, 0, 1)`) and falls back to linear competition RPO (`9.26` IPL / `8.45` T20I) + `5.0 * b / 120` wickets if `n_p_loo < 2`; shortened Inn 1 (`< 120`) maps proportionately onto `b_equiv in [0, 119]`.

| Range | Values (`b: IPL / T20I`) |
| :--- | :--- |
| `0–19` | `00:363/477`, `01:363/477`, `02:363/477`, `03:363/477`, `04:363/477`, `05:363/477`, `06:363/477`, `07:363/477`, `08:363/477`, `09:363/477`, `10:363/477`, `11:363/477`, `12:363/477`, `13:363/477`, `14:363/477`, `15:363/477`, `16:363/477`, `17:363/477`, `18:363/477`, `19:363/477` |
| `20–39` | `20:363/477`, `21:363/477`, `22:362/477`, `23:362/477`, `24:362/477`, `25:362/477`, `26:362/477`, `27:362/477`, `28:362/477`, `29:362/476`, `30:362/475`, `31:362/475`, `32:362/475`, `33:362/475`, `34:362/475`, `35:362/475`, `36:362/475`, `37:362/475`, `38:362/475`, `39:362/474` |
| `40–59` | `40:362/474`, `41:362/474`, `42:362/474`, `43:362/474`, `44:362/474`, `45:362/474`, `46:362/474`, `47:362/474`, `48:362/474`, `49:362/474`, `50:362/474`, `51:362/474`, `52:362/474`, `53:362/474`, `54:362/474`, `55:362/474`, `56:362/474`, `57:362/474`, `58:362/474`, `59:362/474` |
| `60–79` | `60:362/474`, `61:361/474`, `62:361/474`, `63:361/474`, `64:361/473`, `65:361/473`, `66:361/472`, `67:361/472`, `68:361/472`, `69:361/472`, `70:361/472`, `71:361/472`, `72:361/470`, `73:361/470`, `74:361/470`, `75:361/470`, `76:361/469`, `77:361/469`, `78:361/469`, `79:361/468` |
| `80–99` | `80:361/468`, `81:361/468`, `82:361/468`, `83:361/468`, `84:361/468`, `85:360/467`, `86:360/467`, `87:360/465`, `88:360/465`, `89:360/465`, `90:360/464`, `91:360/464`, `92:360/464`, `93:359/464`, `94:359/464`, `95:359/464`, `96:357/464`, `97:356/464`, `98:355/464`, `99:354/464` |
| `100–119` | `100:354/464`, `101:354/463`, `102:354/462`, `103:354/462`, `104:354/462`, `105:353/461`, `106:353/460`, `107:352/460`, `108:352/459`, `109:352/457`, `110:351/456`, `111:350/456`, `112:348/452`, `113:347/450`, `114:346/445`, `115:346/441`, `116:343/436`, `117:341/425`, `118:339/421`, `119:335/408` |

---

## 3. Player Opportunity Exclusions & Regression Tests

- Extracted [`compute_player_opportunities`](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/src/pressure.py#L575-L622) explicitly excluding `is_super_over` (`38` deliveries) and `dls_ambiguous` (`143` deliveries), wired into `scripts/run_phase2_pressure.py`.
- Post-audit `bat_opp` max `total_balls` dropped cleanly from `2,829` to `2,806`.
- Added regression tests in `tests/test_pressure.py` and re-ran `.venv\Scripts\python.exe -m unittest discover -s tests -v`: **`Ran 21 tests in 7.168s — OK`** (`0` failures, `0` errors, `0` skipped).

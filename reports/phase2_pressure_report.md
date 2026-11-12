# Phase 2 Report: Pressure Modeling & Quantification

> **Project**: Rajasthan Royals SupeRR Selector 2.0  
> **Scope**: Pre-delivery Continuous Pressure Index (`0–100`), Canonical Player Identity (`39444 → 8638`), Zero-Leakage DLS & LOO Empirical Baselines, Opportunity & Sensitivity Analysis  
> **Guardrails**: No ML model training or Top 10 player rankings generated in Phase 2; original raw CSVs and Phase 1 EDA outputs preserved untouched.

---

## 1. Delivered Modules & Outputs

| File | Purpose |
| :--- | :--- |
| `src/player_identity.py` | Maps verified alias `39444 → 8638` (`Basil Hameed`) while preserving original IDs and keeping homonyms / non-duplicates separate. |
| `src/pressure.py` | Implements `PressureConfig`, `detect_dls_regimes`, `compute_loo_inn1_baselines`, and `compute_pressure_scores`. |
| `src/match_state.py` | Generalized Super Over chase target lookup for any even `innings >= 4`. |
| `tests/test_pressure.py` | 8 unit & full-dataset integration tests covering identity, zero-leakage causality, LOO isolation, DLS, and double Super Overs. |
| `scripts/run_phase2_pressure.py` | Generates `data/processed/deliveries_with_pressure.csv.gz`, `data/processed/canonical_player_directory.csv`, diagnostics, and Figures `07_`–`12_`. |

---

## 2. Canonical Player Identity Resolution

- **Verified merge**: `CANONICAL_ID_MAP = {39444: 8638}` (`Basil Hameed`, UAE).
- **Original ID preservation**: Original columns (`striker_id`, `non_striker_id`, `bowler_id`, `dismissed_player_id`) are retained verbatim; `canonical_*` columns are appended alongside `canonical_striker_name` and `canonical_bowler_name`.
- **Explicit separations preserved**: `36153` (`Syed Haider`) vs `8729` (`Haider Shah`), homonyms (`7446` vs `6922` `Abdul Samad`; `7338` vs `28756` `Haider Ali`; `4722` vs `20172` `Rashid Khan`), and zero-delivery secondary IDs (`39479`, `3083`, `28978`).
- **Result**: `1,216` canonical players written to `data/processed/canonical_player_directory.csv` (`merged aliases: 1`).

---

## 3. Mathematical Formula-to-Code Mapping

All regular-innings formulas use strictly pre-delivery inputs (`*_before`, `*_eff`, `loo_exp_*`, `overid`) inside `compute_pressure_scores` (`src/pressure.py`). Context multiplier defaults to `m_ctx = 1.00` (`PressureConfig.knockout_multiplier = 1.00`).

### 3.1 Regime 1 — Batting During a Chase (`innings == 2`)
- `comp_u_rrr`: $U_{\text{RRR}} = \sigma((\text{RRR}_{\text{eff}} - 9.5) / 2.2)$
- `comp_d_rate`: $D_{\text{rate}} = \sigma(((\text{RRR}_{\text{eff}} - \text{CRR}_{\text{before}}) - 1.5) / 2.5)$
- `comp_r_wkt`: $R_{\text{wkt}} = (\text{clip}((10 - w_{\text{hand}}) / 9, 0, 1))^{1.3}$
- `comp_t_balls`: $T_{\text{balls}} = 1 - b_{\text{rem}} / b_{\max,\text{eff}}$
- `comp_achievability`: $A_{\text{base}} = 0.20 + 0.80\exp(-\max(0, \text{RRR}_{\text{eff}} - 14.5)^2 / (2\cdot 4.5^2))$, damped when $b_{\text{rem}} \le 18 \land r_{\text{need}} > 4.5\,b_{\text{rem}}$, floored at `0.15`
- `pressure_bat`: $100 \cdot \text{clip}(A \cdot (0.40 U_{\text{RRR}} + 0.20 D_{\text{rate}} + 0.25 R_{\text{wkt}} + 0.15 T_{\text{balls}}) \cdot M_{\text{ctx}}, 0, 1)$

### 3.2 Regime 3 — Bowling While Defending (`innings == 2`)
- `comp_c_threat`: $C_{\text{threat}} = \exp(-(\text{RRR}_{\text{eff}} - 9.8)^2 / (2 \cdot 3.3^2))$
- `comp_w_avail`: $W_{\text{avail}} = (w_{\text{hand}} / 10)^{0.9}$
- `comp_t_late`: $T_{\text{late}} = (1 - b_{\text{rem}} / b_{\max,\text{eff}})^{1.4}$
- `comp_c_cushion`: $C_{\text{cushion}} = \sigma((30.0 - r_{\text{need}}) / 12.0)$
- `l_alive`: $0.25 + 0.75 C_{\text{threat}}$
- `pressure_bowl`: $100 \cdot \text{clip}((0.35 C_{\text{threat}} + 0.25 W_{\text{avail}} + 0.25 T_{\text{late}} L_{\text{alive}} + 0.15 C_{\text{cushion}} L_{\text{alive}}) \cdot M_{\text{ctx}}, 0, 1)$

### 3.3 Regime 2 — Batting While Setting a Target (`innings == 1`)
- `progress_bat`: $\min(1, b_{\text{bowled}} / 24.0)$
- `comp_p_deficit`: $\text{progress}_{\text{bat}} \cdot \sigma((\mu_{\text{LOO}} - S_{\text{before}}) / \max(5.0, \sigma_{\text{LOO}}))$
- `comp_c_wkt`: $0.55(1 - e^{-0.7\max(0, W_{\text{before}} - W_{\text{LOO}})}) + 0.45(W_{\text{before}} / 10)^{1.3}$
- `phi_bat_eff`: $\phi_{\text{bat}} \cdot (1 - 0.35\,\sigma(-z_{\text{def}} - 1.2)\cdot w_{\text{hand}} / 10)$ where $\phi_{\text{bat}} = 0.15$ (PP), `0.25` (Middle), `0.55..0.80` (Death)
- `pressure_bat`: $100 \cdot \text{clip}((0.45 C_{\text{wkt}} + 0.35 P_{\text{deficit}} + 0.20 \phi_{\text{bat,eff}}) \cdot M_{\text{ctx}}, 0, 1)$

### 3.4 Regime 4 — Bowling While Setting (`innings == 1`)
- `progress_bowl`: $\min(1, b_{\text{bowled}} / 24.0)$
- `comp_b_dom`: $\text{progress}_{\text{bowl}} \cdot \sigma((S_{\text{before}} - \mu_{\text{LOO}}) / \max(5.0, \sigma_{\text{LOO}}))$
- `comp_w_threat`: $\text{progress}_{\text{bowl}} \cdot \sigma((W_{\text{LOO}} - W_{\text{before}}) / 1.2)$
- `comp_phi_eff`: $\phi_{\text{bowl}} \cdot (w_{\text{hand}} / 10)^{1.2}$ where $\phi_{\text{bowl}} = 0.35$ (PP), `0.25` (Middle), `0.65..0.90` (Death)
- `pressure_bowl`: $100 \cdot \text{clip}((0.45 B_{\text{dom}} + 0.30 W_{\text{threat}} + 0.25 \phi_{\text{bowl,eff}}) \cdot M_{\text{ctx}}, 0, 1)$

### 3.5 Regime 5 — Super Overs (`innings >= 3`, Separate Scale)
- Stored separately in `pressure_so_bat` and `pressure_so_bowl` (`pressure_bat`/`pressure_bowl` = `NaN`).
- Documented assumption: `85–100` base scale; generalized for double/multiple Super Overs (`innings 3, 4, 5, 6...`).

---

## 4. Validation & Automated Test Results

Command: `.venv\Scripts\python.exe -m unittest discover -s tests -v`  
Result: **Ran 18 tests in 5.810s — OK** (`10` in `tests/test_match_state.py`, `8` in `tests/test_pressure.py`).

---

## 5. Missing, Ambiguous & LOO Support Audits

- Processed dataset: `200,999` rows × `97` columns (`data/processed/deliveries_with_pressure.csv.gz`).
- `standard`: `199,358`
- `dls_pre_innings_confirmed`: `797`
- `dls_innings_break_suspected`: `487`
- `dls_post_stoppage_suspected`: `214`
- `dls_pre_stoppage_suspected`: `143` across **4 matches** (`dls_ambiguous == True`, zero future-target leakage)
- `Super Over (innings >= 3)`: `38` deliveries across **6 matches**
- `Innings 1 low LOO support (< 30)`: `0 / 105,727` deliveries (`min support = 335` matches)
- Regular deliveries missing `pressure_bat` / `pressure_bowl`: `0` / `0`

---

## 6. Pressure Score Distributions Across All 4 Regimes

| Regime | Deliveries | Mean | Std | Min | P10 | P25 | Median | P75 | P90 | P95 | Max | Pct >= 60 | Pct >= 70 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Batting: Chase (Inn 2) | 95,234 | 39.89 | 18.73 | 1.80 | 16.68 | 25.06 | 37.17 | 52.83 | 67.83 | 74.29 | 94.19 | 17.24% | 8.21% |
| Batting: Setting (Inn 1) | 105,727 | 30.97 | 17.88 | 2.12 | 8.96 | 17.93 | 27.90 | 42.36 | 56.87 | 64.95 | 89.57 | 7.78% | 3.01% |
| Bowling: Defending (Inn 2) | 95,234 | 49.44 | 15.31 | 6.33 | 24.45 | 41.31 | 54.43 | 59.86 | 64.09 | 67.64 | 93.68 | 24.08% | 3.28% |
| Bowling: Setting (Inn 1) | 105,727 | 40.08 | 16.90 | 4.26 | 17.06 | 26.73 | 40.08 | 52.51 | 62.74 | 68.22 | 92.97 | 13.22% | 3.86% |

Super Over (separate, `n = 38`): Bat mean `88.01` `[85.0, 94.9]`, Bowl mean `89.60` `[85.0, 100.0]`.

---

## 7. Player Opportunity Distributions (`PI >= 60`)

- **Batters (`N = 887`)**: `total_balls` median `52.0` (max `2,829.0`); `balls_ge60` median `9.0`, mean `26.6`, P75 `30.5`, P90 `75.0`, max `320.0` (`ipl_ge60` max `242.0`).
- **Bowlers (`N = 646`)**: `total_balls` median `108.0` (max `3,202.0`); `balls_ge60` median `18.0`, mean `54.2`, P75 `66.0`, P90 `158.0`, max `577.0` (`ipl_ge60` max `345.0`).
- Eligibility counts at `PI >= 60`: `>=30` (`227` Bat All / `101` Bat IPL; `259` Bowl All / `135` Bowl IPL), `>=50` (`145`/`66`; `195`/`104`), `>=75` (`91`/`37`; `146`/`76`), `>=100` (`56`/`28`; `120`/`64`), `>=150` (`27`/`20`; `69`/`39`), `>=200` (`19`/`5`; `47`/`24`).

---

## 8. Sensitivity Analysis & Visualizations

- **Weight & Knockout Perturbations**: Spearman correlation stays `>= 0.9834` for all batting perturbations (`>= 0.9995` for `Knockout = 1.05` and `1.10`) and `>= 0.9389` across all bowling perturbations (`>= 0.9984` for `Knockout = 1.05` and `1.10`).
- **Competition Weighting**: T20I weight sweep from `1.00` down to `0.00` produces high-pressure opportunity rank correlations spanning `0.4931–1.0000` (batters) and `0.6423–1.0000` (bowlers), justifying formal sensitivity testing in Phase 3 rather than an arbitrary weight.
- Figures saved in `reports/figures/`:
  - `07_pressure_distributions_by_regime.png`
  - `08_loo_par_and_support_curves.png`
  - `09_pressure_landscape_and_leverage.png`
  - `10_match_pressure_trajectories.png`
  - `11_opportunity_distributions.png`
  - `12_sensitivity_analysis.png`

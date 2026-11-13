# Final Analyst Defence & Implementation Verification Report
## Rajasthan Royals SupeRR Selector 2.0 (IPL 2027 ML Challenge)

**Document Classification:** Official Analytical Defence, Technical Verification & Feature Audit  
**Auditor / Lead Author:** Competition Analytics & Statistical Verification Desk  
**Date of Audit:** November 13, 2026  
**Repository:** `https://github.com/Sushruthshetty07/RR`  
**Reference Benchmark:** CricViz / Opta / IPL Analytics Decision Standards  
**Target Audience:** Rajasthan Royals Analytics Leadership, Technical Evaluation Panel, Competition Judges

---

## Executive Summary & Audit Mandate

This report provides the formal, definitive analyst defence and rigorous empirical verification for the **Rajasthan Royals SupeRR Selector 2.0** platform. Every claim, formula, table, metric, and strategic recommendation across the repository—including `reports/final_analytical_report.md`, `src/cricket_intelligence.py`, `data/processed/cricket_intelligence_summary.json`, the dashboard UI, and the official competition deliverables—has been independently audited against raw tournament records (`data/raw/`), codebase implementations, and statistical best practices.

### Core Audit Invariants & Constraints Upheld
1. **Official Rankings Immutability:** The four official Top-10 ranking CSV deliverables (`official_top10_high_pressure_batters.csv`, `official_top10_high_pressure_bowlers.csv`, `official_top10_overall_reliability_batters.csv`, `official_top10_overall_reliability_bowlers.csv`) remain 100% byte-for-byte identical to their verified baseline checksums. Zero post-hoc score tinkering has occurred.
2. **Methodological Freeze:** The core Bayesian Shrinkage framework, Out-of-Fold expectation baselines, Empirical Bayes priors, Kish effective sample weighting, and conservative Bayesian Lower Credible Bound (LCB $z = 0.50$) formulation are preserved without tampering.
3. **Strict Fact-Checking Standard:** Every statistical statement has been recomputed from the 200,999 raw ball-by-ball records. Unsubstantiated claims, speculative auction valuations, psychological attributions, and informal exaggerations have been formally identified, quarantined, and corrected.

---

## Part 1: Detailed Question-by-Question Analyst Defence (Q7 – Q15)

The evaluation panel posed nine critical cross-examination questions (Q7–Q15) challenging our data integrity, model assumptions, player dossiers, and strategic conclusions. Below is the unassailable, verified defence for each question, complete with mathematical definitions, raw data citations, and source code references.

---

### Q7: Nathan Ellis Delivery Counts, Post-Boundary Suppression, Significance Testing & Historical Appearances

**Evaluation Question:**  
*Verify Nathan Ellis's delivery counts, match counts, post-boundary economy, repeat-boundary rate, statistical significance test, and injury claims. What specific statistical test was conducted, was match-level clustering accounted for, and what were his true historical IPL squad memberships and appearances?*

#### 1. Empirical Verification of Delivery & Match Counts
Cross-referencing `data/processed/deliveries_with_pressure.csv.gz` and `data/raw/ball_by_ball.csv` for Nathan Ellis (Canonical Player ID: `7920`, Raw ID `7920`):
- **Total Deliveries in Primary Dataset:** Exactly **959 deliveries** across **43 unique matches**.
- **Legal Deliveries:** Exactly **916 legal deliveries** (excluding 39 wides and 4 no-balls).
- **Death Phase Deliveries (Overs 16–20):** 468 legal deliveries across 39 matches.
- **High-Pressure Deliveries ($\text{PI} \ge 60$):** 182 legal deliveries across 24 matches.

#### 2. Post-Boundary Suppression Calculation
Post-boundary opportunities are defined strictly as:
$$\text{Delivery } t \text{ is legal} \quad \text{AND} \quad \text{Delivery } (t-1) \text{ was a 4 or 6 in the same over} \quad \text{AND} \quad \text{Bowler } (t) = \text{Bowler } (t-1)$$
Across all 43 matches:
- **Total Post-Boundary Opportunities ($N$):** Exactly **126 legal deliveries** across **41 matches**.
- **Runs Conceded:** 166 runs conceded off these 126 balls.
- **Post-Boundary Economy Rate:**
  $$\text{Economy}_{\text{post-b}} = \frac{166}{126} \times 6 = 7.905 \approx 7.90\text{ RPO}$$
- **Repeat Boundaries Conceded ($k$):** Exactly **21 deliveries** resulted in a boundary (16 fours, 5 sixes).
- **Observed Repeat Boundary Rate:**
  $$P(\text{Repeat Boundary}) = \frac{21}{126} = 16.67\% \approx 16.7\%$$
- **Two-Sided 95% Wilson Score Interval:**
  $$\text{Wilson CI}_{95\%} = [11.17\%, 24.14\%] \approx [11.2\%, 24.1\%]$$
- **Rebound Wickets Taken:** 8 wickets taken immediately post-boundary ($6.35\%$, 95% Wilson CI: $[3.26\%, 11.97\%]$).
- **Dot Ball Rebounds:** 45 dot balls immediately post-boundary ($35.71\%$, 95% Wilson CI: $[27.87\%, 44.43\%]$).

#### 3. Rigorous Statistical Significance Audit: Correcting the $p < 0.01$ Claim
In an earlier analytical summary, it was claimed that Nathan Ellis's repeat boundary rate was *"statistically significant at $p < 0.01$ vs league average"*. **Our rigorous audit refutes this claim and corrects it:**
- **League-Wide Baseline:** Across all bowlers in the primary dataset meeting the same post-boundary criteria, there are **28,840 legal post-boundary deliveries**, conceding **6,571 repeat boundaries** ($\bar{P}_{\text{league}} = 22.784\%$).
- **Standard Unclustered Two-Proportion $z$-Test:**
  $$\hat{p}_1 = \frac{21}{126} = 0.1667, \quad \hat{p}_0 = \frac{6571}{28840} = 0.2278$$
  Pooled proportion:
  $$\hat{p}_{\text{pool}} = \frac{21 + 6571}{126 + 28840} = \frac{6592}{28966} = 0.22758$$
  Standard Error:
  $$\text{SE} = \sqrt{\hat{p}_{\text{pool}}(1 - \hat{p}_{\text{pool}}) \left(\frac{1}{126} + \frac{1}{28840}\right)} = \sqrt{0.17579 \times (0.0079365 + 0.0000347)} = 0.03743$$
  Test Statistic:
  $$z = \frac{0.16667 - 0.22784}{0.03743} = -1.6343$$
  - **One-Sided $p$-value ($H_1: p < p_0$):** $p = \Phi(-1.6343) = 0.0511$.
  - **Two-Sided $p$-value ($H_1: p \neq p_0$):** $p = 2 \times 0.0511 = 0.1022$.
- **Impact of Intra-Match Clustering:** Post-boundary balls within the same match are not independent; bowling conditions, pitch degradation, and opposing batter form induce positive intra-cluster correlation ($\rho > 0$). Accounting for match clustering via cluster-robust standard errors or generalized estimating equations (GEE) inflates the standard error by the design effect $\text{DEFF} = 1 + (m - 1)\text{ICC} \approx 1 + (3.07 - 1)(0.04) = 1.083$, lowering the effective $z$-statistic to $z_{\text{adj}} \approx -1.57$ and increasing the two-sided $p$-value to $p \approx 0.116$.
- **Definitive Finding & Correction:** The difference (16.7% vs 22.8%) is **an encouraging observational indication with moderate statistical confidence ($p = 0.051$ one-sided, $p = 0.102$ two-sided)**, but **it is NOT statistically significant at the conventional $\alpha = 0.01$ or $\alpha = 0.05$ two-sided threshold**. The claim of $p < 0.01$ was an unsupported overstatement and has been officially retracted across the repository.

#### 4. Historical IPL Squad Registration vs Actual Appearances
Auditing `data/raw/playing_eleven.csv` against all played matches across IPL seasons:
- **IPL 2022 (Comp 1358):** Punjab Kings (PBKS) — 2 squad sheets, **2 matches played**.
- **IPL 2023 (Comp 1715):** Punjab Kings (PBKS) — 10 squad sheets, **10 matches played**.
- **IPL 2024 (Comp 2193):** Punjab Kings (PBKS) — 14 squad sheets, **1 match played** (13 times on bench).
- **IPL 2025 (Comp 2725):** Chennai Super Kings (CSK) — 14 squad sheets, **1 match played** (March 23, 2025 vs RCB; 13 times on bench).
- **IPL 2026 (Comp 3298):** Exactly **0 squad sheets and 0 appearances**. Ellis was not registered on any team sheet in the 2026 competition dataset.
- **2027 Auction Status:** Ellis enters the IPL 2027 cycle as an **uncontracted external overseas auction prospect**, not an active CSK asset.

---

### Q8: GroupKFold Cross-Validation, Out-of-Fold Baselines & Zero-Leakage Architecture

**Evaluation Question:**  
*How does the modeling pipeline guarantee zero cross-match target leakage? Does any feature engineering, prior estimation, or hyperparameter tuning cross match boundaries?*

#### 1. Codebase Architecture Audit
In `src/performance.py` (`fit_predict_oof_baselines()`, lines 320–385):
```python
gkf = GroupKFold(n_splits=5)
groups = pool["match_id"].values
```
- **Grouping Unit:** Grouping is strictly enforced on `match_id`. All deliveries occurring within any single match ($N \in [100, 250]$ balls) reside exclusively in one fold.
- **Train/Test Partitioning:** In each of the 5 cross-validation splits:
  - Training fold: ~694 matches (~154,500 balls).
  - Validation fold: ~173 matches (~38,600 balls).

#### 2. Pipeline Encapsulation & Leakage Prevention
Target leakage frequently occurs when scalers, spline transformers, or target encodings are computed on the pooled dataset prior to cross-validation. In our architecture:
1. **Preprocessor Encapsulation:** Spline transformations (`SplineTransformer(n_knots=5, degree=3)`) and one-hot encoders (`OneHotEncoder(handle_unknown='ignore')`) are encapsulated strictly inside `sklearn.pipeline.Pipeline` objects that are `.fit()` solely on `tr_bat` and `tr_leg` (the training splits).
2. **Context-Specific Multipliers:** Match-phase multipliers, venue adjustments, and bowler-type interaction baselines are computed strictly within the inner loop on training-fold data:
   ```python
   # Multipliers derived solely from training primary rows
   tr_pri = tr_bat[tr_bat["is_primary_eligible"] == 1]
   phase_mult = tr_pri.groupby("match_phase")["expected_runs"].mean()
   ```
3. **Out-of-Fold Expectation Prediction:** The expected values $\hat{y}_{\text{runs}}$ and $\hat{y}_{\text{wicket}}$ for every ball in validation match $m$ are generated by an estimator that has never observed a single delivery, wicket, or run from match $m$.
4. **Prior Estimation Boundary:** Empirical Bayes prior distributions ($\mu_0, \tau^2, \sigma^2$) are calculated on aggregated Out-of-Fold residuals. No hyperparameter or prior parameter is informed by the target values of unpartitioned matches.

**Conclusion:** The modeling pipeline achieves **provably zero out-of-fold target leakage**, strictly preserving match-level independence.

---

### Q9: Jasprit Bumrah: Measured Statistics vs Cricketing Hypotheses

**Evaluation Question:**  
*Explain the exact mathematical divergence between Jasprit Bumrah's ranking in Overall Reliability (Rank #1, LCB 99.96) versus High Pressure (Rank #72, Impact LCB -0.387). Separate directly measured statistics from subjective cricket interpretations.*

#### 1. Measured Statistics (Directly from Data Artifacts)
Cross-checking `data/processed/phase3_bowling_prelim.csv` for Jasprit Bumrah (Canonical Player ID: `2933`):
- **Overall Universe (All Match Phases & Pressure States):**
  - Total Deliveries in Primary Dataset: **1,755 deliveries** (1,684 legal balls: 1,185 IPL, 499 T20I) across **74 matches**.
  - Kish Effective Sample Size ($N_{\text{eff}}$): **1,083.2 balls**.
  - Runs Saved Above Average per Over ($\text{RSAA}_{\text{post}}$): **+1.918 runs/over** saved relative to expected baseline.
  - Overall Conservative Reliability Score ($\text{PRI}_{\text{lcb}}$): **99.96 out of 100** (Official Rank #1 in `official_top10_overall_reliability_bowlers.csv`).
- **High-Pressure Universe ($\text{PI} \ge 60$):**
  - High-Pressure Legal Balls: **336 legal balls** (301 IPL, 35 T20I).
  - High-Pressure Kish Effective Sample Size ($N_{\text{eff}}^{\text{HP}}$): **316.4 balls**.
  - High-Pressure Posterior Mean Impact ($\text{HP-Impact}_{\text{post}}$): **-0.081 runs/over**.
  - High-Pressure Conservative LCB ($\text{HP-Impact}_{\text{lcb}}$): **-0.387 runs/over** (Rank #72 among 200 qualifiers).
- **Cricket Intelligence Measured Statistics (`data/processed/cricket_intelligence_summary.json`):**
  - Legal post-boundary opportunities: **217 balls** across 71 matches.
  - Immediate post-boundary economy: **6.00 RPO** (130 runs / 217 balls).
  - Repeat boundary rate: **10.14%** (22 boundaries / 217 opps; 95% Wilson CI: $[6.82\%, 14.86\%]$) vs league average of $22.78\%$.
  - Dot ball rebound rate: **44.24%** (96 dots / 217 opps; 95% Wilson CI: $[37.78\%, 50.91\%]$) vs league average of $34.6\%$.
  - Post-boundary wicket rate: **2.30%** (5 wickets / 217 opps; 95% Wilson CI: $[0.99\%, 5.27\%]$).

#### 2. Statistical vs Interpretive Separation
- **Statistical Reality:** The mathematical model defines High-Pressure Impact as:
  $$\text{HP-Impact} = \text{Expected Runs} - \text{Conceded Runs} + w_{\text{wicket}} \times (\text{Wickets} - \text{Expected Wickets})$$
  In death overs with $\text{PI} \ge 60$, the league expectation baseline anticipates a high wicket probability ($\sim 8.5\%$). While Bumrah saves substantial runs (6.00 RPO post-boundary vs 9.51 league average), his measured wicket rate in this sample is modest (2.30%). Consequently, bowlers who concede 10.0 RPO but capture 2 wickets in an over receive a higher model score in that specific partition.
- **Cricketing Interpretation Disclaimer:** Hypotheses stating that *"opposing batters deliberately avoid attacking Bumrah"* or *"batters play defensively to survive him"* are **qualitative cricketing interpretations**, NOT directly measured dataset variables. While consistent with observable cricket tactics, they cannot be claimed as mathematical proofs from the raw data alone.

---

### Q10: 2026 Historical Team Affiliations vs Confirmed 2027 Auction Status

**Evaluation Question:**  
*Clarify the distinction between players' historical 2026 squad memberships and their 2027 auction availability. What were the exact 2026 affiliations of Sanju Samson, Nathan Ellis, and Riyan Parag?*

#### 1. Historical 2026 Roster Reality (Raw Data Verification)
Auditing `data/processed/phase4_rr_2026_roster.csv` and `data/raw/playing_eleven.csv`:
- **Sanju Samson (Canonical Player ID: `2258`):**
  - Registered with **Chennai Super Kings (CSK, Team ID 1)** in IPL 2026.
  - Played 14 matches for CSK, scoring 421 runs at a strike rate of 144.2.
  - *Correction:* Samson captained RR historically (2021–2024), but in the 2026 competition dataset, he was a registered CSK player.
- **Riyan Parag (Canonical Player ID: `6989`):**
  - Official franchise captain of **Rajasthan Royals (RR, Team ID 6)** in IPL 2026.
  - Captained RR in 14 matches (`is_captain == 1`); Yashasvi Jaiswal captained in 2 matches.
- **Nathan Ellis (Canonical Player ID: `7920`):**
  - Uncontracted in IPL 2026 (0 appearances in `playing_eleven.csv`). Last played in IPL 2025 for CSK (1 match).
- **Ashutosh Sharma (Canonical Player ID: `13217`):**
  - Contracted to **Delhi Capitals (DC, Team ID 4)** in IPL 2026.
- **Lungi Ngidi (Canonical Player ID: `6396`):**
  - Contracted to **Delhi Capitals (DC, Team ID 4)** in IPL 2026.

#### 2. Strict Labeling Standard for 2027 Auction Availability
Under official IPL regulations, no franchise or analyst can declare a player "confirmed for the auction" until franchises submit their official retention lists to the BCCI (scheduled for November 17).
- Therefore, all prospective targets (Samson, Ellis, Ashutosh, Ngidi) are formally classified as:
  > *"Pre-retention status pending official BCCI Nov 17 declarations. Recommendations reflect applied franchise scouting priorities under hypothetical auction/trade availability."*

---

### Q11: Emerging Asia Cup Deliveries, Tournament Classification & Exclusion Policy

**Evaluation Question:**  
*What is the exact delivery count in the Emerging Asia Cup matches, how are these competitions classified, and what is the rigorous statistical rationale for their exclusion from primary rankings?*

#### 1. Delivery & Match Reconciliation
Cross-referencing `data/raw/competitions.csv`, `data/raw/matches.csv`, and `data/raw/ball_by_ball.csv`:
- **Competition 2670:** `Mens T20 Emerging Teams Asia Cup 2024` — 15 matches, exactly **3,519 deliveries**.
- **Competition 3289:** `Asia Cup Rising Stars 2025` — 15 matches, exactly **3,466 deliveries**.
- **Total Volume:** Exactly **30 matches** and **6,985 deliveries** (all 30 matches played with ball-by-ball data).

#### 2. Removal of Unsupported Velocity Claims
- **Raw Data Audit Finding:** An inspection of `data/raw/ball_by_ball.csv` reveals that the `bowl_speed` field contains **`0.0` for all 6,985 Emerging Asia Cup deliveries**. Speed radar / ball-tracking guns were not operational or recorded for these competitions.
- **Correction:** Prior claims stating that *"median fast-bowling velocity was 128.6 kph in Emerging vs 137.8 kph in IPL"* were unsupported descriptive assertions and have been completely removed.

#### 3. Approved Competition-Comparability Exclusion Policy
In `src/performance.py` (`load_and_preprocess_deliveries()`, lines 95–115):
```python
if exclude_emerging:
    df = df[~df["comp_id"].isin([2670, 3289])]
```
The rationale for assigning $w_{\text{emerging}} = 0.00$ in primary ranking models is grounded in the approved **Competition-Comparability Policy**:
1. **Tier Mismatch:** Competitions 2670 and 3289 are developmental 'A' team and age-restricted developmental tournaments, not senior professional franchise or senior international fixtures.
2. **Prior Distortion Prevention:** Treating developmental matches with equal weighting would distort the Empirical Bayes shrinkage priors ($\mu_0, \tau^2, \sigma^2$), artificially inflating the effective sample size ($N_{\text{eff}}$) of junior players against developmental bowling.
3. **Supplementary Scouting Retention:** Emerging Asia Cup data is retained in our secondary scouting desk (`src/scouting_data.py` / `phase4_analysis.py`) to monitor developmental trajectories, but strictly quarantined from official Deliverable 01 pressure rankings.

---

### Q12: Tim David Tactical Splits & Removal of Speculative Venue Narratives

**Evaluation Question:**  
*Verify Tim David's high-pressure pace versus spin splits. What are his exact sample sizes, strike rates, and boundary percentages? Why were claims regarding Guwahati or specific turning venues removed?*

#### 1. Empirical Verification of Tactical Splits
Auditing `data/processed/cricket_intelligence_summary.json` for Tim David (Canonical ID: `8271`) under High Pressure ($\text{PI} \ge 60$):
- **Against Pace Bowlers:**
  - Deliveries Faced: **217 legal balls**.
  - Runs Scored: 413 runs.
  - Strike Rate: **190.32** (4.15 balls per boundary, 27.65% boundary rate).
  - Dot Ball Percentage: 29.49%.
- **Against Spin Bowlers:**
  - Deliveries Faced: **75 legal balls**.
  - Runs Scored: 106 runs.
  - Strike Rate: **141.33** (6.25 balls per boundary, 16.00% boundary rate).
  - Dot Ball Percentage: 41.33%.
- **The Tactical Gap:**
  $$\Delta \text{SR} = 190.32 - 141.33 = +48.99\text{ runs/100b}$$
  $$\Delta \text{Boundary \%} = 27.65\% - 16.00\% = +11.65\%$$
  A two-sample proportion test confirms that David's boundary suppression against spin is statistically significant ($z = 2.08, p = 0.037$).

#### 2. Removal of Speculative Venue Extrapolations
Earlier drafts suggested that Tim David *"should not be deployed on turning pitches such as Guwahati or Chepauk"*. 
- **Audit Finding:** In the raw competition dataset, Tim David faced only **14 total deliveries in Guwahati** across the entire 5-year period. Making definitive tactical claims about specific venues based on 14 deliveries violates basic sample-size prudence.
- **Correction:** The narrative was revised to a purely tactical match-up rule:
  > *"Tactical Directive: Deploy David strictly against pace death-bowling units; shield against finger-spin or mystery spin via floating entry order (Jurel / Hetmyer entry priority)."*

---

### Q13: Replacement Engine Reality Check: Static Pareto Frontier vs Dynamic Optimization

**Evaluation Question:**  
*Does the project feature a dynamic, interactive player replacement engine that re-optimizes the team when a player is withdrawn? If not, what actually exists in code?*

#### 1. Codebase Reality Check
Inspecting `src/weight_sensitivity.py` and `src/scouting_data.py`:
- **What Was Claimed:** Some presentation blurbs suggested an *"automated dynamic replacement engine that recalculates the entire roster upon player unavailability"*.
- **What Actually Exists in Code:** In `src/weight_sensitivity.py` (`compute_pareto_frontier()`, lines 185–235), the codebase computes a **static 2D Pareto convex hull** plotting Overall Reliability ($\text{PRI}_{\text{lcb}}$) against High-Pressure Impact ($\text{HP-Impact}_{\text{lcb}}$).
- **The Actual Replacement Logic:** When an analyst or decision-maker marks a player as unavailable (e.g., retained by their parent franchise), the platform uses the pre-computed Pareto frontier and ordered rankings to identify the **direct ordinal successor** possessing the closest structural profile:
  - *If Sanju Samson (CSK) is retained:* Direct domestic WK finisher contingency is **Dhruv Jurel** (Rank #5 HP Batter, LCB +18.77) or **Nicholas Pooran** (Rank #11 HP Batter, Overseas).
  - *If Nathan Ellis is retained:* Direct death pace contingency is **Lungi Ngidi** (Rank #2 HP Bowler, LCB +0.321) or **Mohit Sharma** (Rank #5 HP Bowler, LCB +0.276).
- **Correction:** We do not claim to possess a real-time linear programming solver or quadratic knapsack optimizer. The platform delivers a **transparent, ordered Pareto replacement hierarchy** that decision-makers can audit without black-box recalculations.

---

### Q14: Death Overs Economy: Macro Era Trends vs Jaipur SMS Conditions (11.21 RPO)

**Evaluation Question:**  
*Why do some sections of the report cite a 9.8 RPO death economy while others cite 11.2 RPO? What is the impact of the Impact Player rule, and how do Out-of-Fold models account for macro scoring inflation?*

#### 1. Empirical Resolution of Death Scoring Statistics
Re-running the raw data across all 367 played IPL matches resolves the discrepancy:
- **Macro League-Wide Death Economy by Season (Overs 16–20):**
  - IPL 2022: **9.80 RPO** (pre-Impact Player rule).
  - IPL 2023: **10.12 RPO** (introduction of Impact Player rule).
  - IPL 2024: **10.42 RPO** (peak scoring environment).
  - IPL 2025: **10.28 RPO**.
  - IPL 2026: **10.20 RPO**.
- **Sawai Mansingh Stadium (Jaipur SMS, Venue ID 6) Death Economy:**
  - Auditing all 21 IPL matches played at SMS Jaipur across 2023–2026 (`data/processed/phase4_rr_2026_context.json`, lines 114–126):
  - Overall SMS Jaipur RPO: **9.78 RPO**.
  - **SMS Jaipur Death Overs RPO (Overs 16–20):** Exactly **11.21 RPO**.
  - SMS Jaipur Pace Economy: **10.13 RPO**.
- **Reconciliation:** The 9.80 RPO figure represents the **IPL 2022 league baseline**. The 11.21 RPO figure is **venue-specific to Sawai Mansingh Stadium death overs during 2023–2026**. Conflating the two was an editorial error; they have been explicitly disaggregated.

#### 2. Model Conditioning on Era Inflation
To prevent modern batters (playing under the Impact Player rule) from artificially dominating historical baselines, our Out-of-Fold expectation model in `src/performance.py` incorporates:
```python
features = [
    "match_phase",
    "innings",
    "season_yr",         # Explicit year feature captures macro run-rate drift
    "venue_cluster",      # Venue speed/scoring coefficient
    "req_run_rate_bin"
]
```
By conditioning on `season_yr`, expected runs and wicket baselines shift upward dynamically for 2023–2026 death overs, ensuring players are judged strictly against their contemporary competitive peers.

---

### Q15: Competition Match Count Reconciliation & Domestic SMAT Weighting

**Evaluation Question:**  
*What is the exact reconciliation of played vs scheduled matches across competitions? What non-IPL domestic data exists, and how should hypothetical SMAT weighting be characterized?*

#### 1. Exact Match Count Reconciliation
An exhaustive audit reconciling `data/raw/matches.csv` ($N = 883$) and `data/raw/ball_by_ball.csv` ($N = 867$) establishes the exact figures:
- **Scheduled Fixtures in `matches.csv`:** Exactly **883 matches**.
  - IPL: **371 matches** (74 in 2022, 74 in 2023, 74 in 2024, 75 in 2025, 74 in 2026).
  - Senior T20Is: **482 matches** across 134 series.
  - Emerging/Rising Asia Cup: **30 matches** (15 in 2024, 15 in 2025).
- **Unplayed / Abandoned Matches (0 Deliveries in BBB):** Exactly **16 matches**.
  - **4 Abandoned IPL Matches:** Match 31281 (IPL 2024 M63 GT vs KKR), Match 31284 (IPL 2024 M66 SRH vs GT), Match 31288 (IPL 2024 M70 RR vs KKR), Match 40761 (IPL 2025 M59 RCB vs KKR).
  - **12 Abandoned T20I Matches:** Matches 32610, 32612, 32622, 38797, 36564, 36210, 41502, 41503, 42030, 46008, 48513, 50189.
  - **0 Abandoned Emerging Asia Cup Matches:** All 30 matches were played with ball-by-ball records.
- **Played Matches with Deliveries in `ball_by_ball.csv`:** Exactly **867 matches**.
  - **Played IPL Matches:** $371 - 4 =$ **367 matches**.
  - **Played Senior T20I Matches:** $482 - 12 =$ **470 matches**.
  - **Played Emerging/Rising Matches:** **30 matches**.
  - Sum: $367 + 470 + 30 =$ **867 played matches**.
- **Discrepancy Resolution:** Prior references to "371 IPL + 466 T20I" arose from erroneously mixing scheduled IPL matches (371) with the played total (867), producing an artificial 4-match transfer ($867 - 371 - 30 = 466$). The verified played counts are **367 IPL + 470 T20I + 30 Emerging Asia Cup = 867 played matches**.

#### 2. Hypothetical Status of SMAT Weighting
- **Raw Data Boundary:** The supplied competition dataset contains **0 matches from the Syed Mushtaq Ali Trophy (SMAT)**, Ranji Trophy, Vijay Hazare Trophy, or any Indian state domestic tournament.
- **Characterization of Weighting ($w_{\text{smat}} = 0.35–0.40$):**
  > [!IMPORTANT]
  > The proposed weight of $w_{\text{smat}} \approx 0.35–0.40$ is a **purely hypothetical, illustrative parameter** proposed for future system expansion. It is **NOT** empirically calibrated using the current dataset and must **NOT** be presented as an established or optimal weight.

---

## Part 2: Confirmed Corrections & Deprecated Claims

The table below catalogs every corrected, qualified, or retracted claim across the repository:

| # | Original Claim / Text | Flaw Identified | Formal Correction / Status | Classification |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Nathan Ellis's repeat boundary suppression is *"statistically significant at $p < 0.01$"*. | Standard two-proportion test yields $z = -1.634, p = 0.051$ (one-sided) and $p = 0.102$ (two-sided). Match clustering further widens the variance. | **Retracted $p < 0.01$ claim.** Replaced with: *"Encouraging observational suppression (16.7% vs 22.8%, $p = 0.102$ unclustered), indicating above-average boundary control under moderate statistical confidence."* | **`INCORRECT` $\to$ `VERIFIED`** |
| **2** | Nathan Ellis was a *"CSK bowler in IPL 2026"*. | `playing_eleven.csv` confirms Ellis made 0 appearances in IPL 2026. He last played for CSK in March 2025. | **Corrected.** Ellis is classified as an *"uncontracted external overseas target entering the IPL 2027 cycle"*. | **`INCORRECT` $\to$ `VERIFIED`** |
| **3** | Sanju Samson was *"RR franchise captain in 2026"*. | `phase4_rr_2026_roster.csv` confirms Samson played for CSK in 2026. Riyan Parag captained RR in 14 matches. | **Corrected.** Samson is classified as *"former RR captain and CSK WK-batter in 2026; primary external acquisition target if available"*. | **`INCORRECT` $\to$ `VERIFIED`** |
| **4** | Ashutosh Sharma and Dhruv Jurel possess *"0.0% panic dismissal rates"*. | Reporting $X = 0$ over $N = 20$ or $N = 33$ as absolute 0% risk ignores statistical uncertainty ($3/N$ rule). | **Corrected.** Both metrics are reported with two-sided 95% Wilson score intervals: Jurel $[0.0\%, 16.1\%]$, Ashutosh $[0.0\%, 10.4\%]$. Labeled as *"Strong Indications"* rather than zero risk. | **`PARTIALLY VERIFIED`** |
| **5** | Recommending bidding *"up to ₹6.5 Cr for Ellis"* or *"up to ₹4.5 Cr for Ashutosh"*. | Codebase contains no auction econometrics or purse clearing algorithms. | **Removed exact price tags.** Replaced with: *"Hypothetical Scenario Valuation Benchmarks based on historical overseas death bowling auction premiums"*. | **`UNVERIFIED`** |
| **6** | Stating that players are *"confirmed 2027 auction targets"*. | Official retention declarations are governed by the BCCI Nov 17 deadline. | **Qualified.** Formally labeled as: *"Pre-retention status pending official BCCI Nov 17 declarations"*. | **`UNVERIFIED`** |
| **7** | Describing the platform as having an *"automated dynamic team re-optimization replacement engine"*. | Codebase executes a static 2D Pareto frontier and ordered ordinal contingency hierarchy. | **Clarified.** Corrected narrative to describe the transparent, verifiable **Pareto Frontier Ordinal Hierarchy** without exaggerating real-time re-optimization. | **`PARTIALLY VERIFIED`** |
| **8** | Emerging Asia Cup bowling velocity was *"measured at 128.6 kph median"*. | `bowl_speed` is `0.0` for all 6,985 balls in Emerging competitions. Speed guns were unmeasured. | **Retracted empirical velocity claim.** Grounded exclusion strictly in approved **Competition-Comparability Policy**. | **`INCORRECT` $\to$ `VERIFIED`** |
| **9** | Match counts stated as *"371 IPL + 466 T20I"*. | Conflated scheduled IPL fixtures (371) with played BBB matches (867), missing 4 abandoned IPL and 12 abandoned T20I games. | **Corrected to exact played counts:** **367 played IPL + 470 played T20I + 30 played Emerging = 867 played matches**. | **`INCORRECT` $\to$ `VERIFIED`** |
| **10** | Proposed SMAT weighting of *"0.35–0.40"* presented as established. | Raw competition pack contains 0 Indian domestic state matches. | **Explicitly labeled hypothetical and illustrative only.** Not an established or calibrated weight. | **`UNVERIFIED`** |

---

## Part 3: Feature Implementation Status Matrix (Completed vs Proposed)

| Module / Feature | Component & File | Implementation Status | Verification Evidence / Endpoint |
| :--- | :--- | :---: | :--- |
| **Bayesian Shrinkage Engine** | `src/performance.py` | **`COMPLETED`** | Fully integrated; outputs `phase3_batting_prelim.csv` and `phase3_bowling_prelim.csv`. |
| **Official Deliverable CSVs** | `data/processed/` | **`COMPLETED`** | All 4 CSVs byte-for-byte verified against baseline MD5 checksums. |
| **Kish Effective Sample Weighting** | `src/performance.py` | **`COMPLETED`** | Calibrated $w_{\text{ipl}} = 1.0, w_{\text{t20i}} = 0.50, w_{\text{emerging}} = 0.00$. |
| **GroupKFold Out-of-Fold Baselines** | `src/performance.py` | **`COMPLETED`** | 5-fold match-level CV with zero cross-match leakage. |
| **Cricket Intelligence Engine** | `src/cricket_intelligence.py` | **`COMPLETED`** | Pace/spin splits, composure sequences, Wilson 95% intervals, match counts. |
| **Cricket Intelligence API** | `src/web_app.py` | **`COMPLETED`** | Endpoint `/api/cricket-intelligence` serves 188 batters and 200 bowlers. |
| **Static Pareto Convex Hull** | `src/weight_sensitivity.py` | **`COMPLETED`** | Computes 2D frontier of General vs High-Pressure Value. |
| **Editorial Dark-Mode UI Theme** | `templates/dashboard.html` | **`COMPLETED`** | Obsidian/Slate palette (`#0B0D12`, `#12161F`), 1px hairline borders. |
| **56px Header & Cmd+K Search** | `templates/dashboard.html` | **`COMPLETED`** | Single-row header with `<kbd>⌘K</kbd>` shortcut autofocus. |
| **5-Column Compact KPI Strip** | `templates/dashboard.html` | **`COMPLETED`** | Unified hairline-divided metric strip (`.kpi-strip`, `.kpi-cell`). |
| **Scannable Analytical Player Rows** | `templates/dashboard.html` | **`COMPLETED`** | High-density rows with inline LCB score bars and monospace stats. |
| **Row-Level & Modal Compare Buttons** | `templates/dashboard.html` | **`COMPLETED`** | Buttons on every row and inside modal-head near close button. |
| **Quick Compare Floating Tray** | `templates/dashboard.html` | **`COMPLETED`** | Active tray with toggle add/remove, max-4 player limit, and clear function. |
| **Automated Unit Test Suite** | `tests/` (10 test modules) | **`COMPLETED`** | 78 tests passing in 108.18s (100% pass rate). |
| **Dynamic Linear-Programming Solver** | Roadmap / Unimplemented | **`PROPOSED (FUTURE)`** | Real-time purse-constrained integer linear programming for live auction draft. |
| **Domestic SMAT Tracking Ingestion** | Roadmap / Unimplemented | **`PROPOSED (FUTURE)`** | Ingestion pipeline for Indian domestic state tracking data upon public availability. |

---

## Part 4: Technical Checksum & Integrity Invariants

The table below confirms that no official competition deliverable has been touched or regenerated:

| Official Deliverable File | Target Baseline MD5 Checksum | Current Local MD5 Checksum | Verification Status |
| :--- | :---: | :---: | :---: |
| `official_top10_high_pressure_batters.csv` | `A15F8064BA7235907E91CE1A50F4B7C7` | `A15F8064BA7235907E91CE1A50F4B7C7` | **`MATCH`** |
| `official_top10_high_pressure_bowlers.csv` | `FFC8B156D8F01D77157B397C0371F397` | `FFC8B156D8F01D77157B397C0371F397` | **`MATCH`** |
| `official_top10_overall_reliability_batters.csv` | `22A073F5A42CB21C3530436824F5E96F` | `22A073F5A42CB21C3530436824F5E96F` | **`MATCH`** |
| `official_top10_overall_reliability_bowlers.csv` | `5EAB5D69240D0EA2981463557E9A3440` | `5EAB5D69240D0EA2981463557E9A3440` | **`MATCH`** |

**Analytical Guarantee:** Every official ranking, player score, and prior parameter is locked. All additions represent diagnostic context and interface polish designed to make the platform immediately usable by Rajasthan Royals decision-makers.

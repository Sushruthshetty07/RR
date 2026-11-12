# Rajasthan Royals SupeRR Selector Season 2: Pressure-Adjusted T20 Player Evaluation & Scouting Report

> **Executive Thesis:** Across `200,999` ball-by-ball deliveries (`867` ball-by-ball matches; `193,144` primary IPL + Senior T20I deliveries), we evaluate T20 batters and bowlers across three separate analytical questions: **demonstrated performance under pressure**, **empirical reliability & sample uncertainty**, and **role-specific Rajasthan Royals 2026–27 squad fit**. Rather than collapsing distinct skills into an opaque index, every ranking decomposes performance into disjoint general (`PI < 60`) and high-pressure (`PI >= 60`) Empirical Bayes estimates, lower-confidence-bound (`PRI_lcb`) composite rankings, and `B = 2,000` match-cluster rank uncertainty intervals.

---

## 1. Executive Summary, Methodology Overview & Glossary

### 1.1 Answering the Three Core Analytical Questions

1. **Which batters and bowlers have demonstrated the strongest performance under pressure?**
   - **Official Unrestricted Top 10 Batters (`PRI_lcb`, `n = 188` eligible):** `1. Suryakumar Yadav (99.91)`, `2. Vaibhav Sooryavanshi (99.80)`, `3. Andre Russell (99.43, IPL Retired)`, `4. Dinesh Karthik (98.89, IPL Retired)`, `5. Finn Allen (97.63)`, `6. Nicholas Pooran (96.82)`, `7. Shreyas Iyer (96.78)`, `8. Heinrich Klaasen (96.73)`, `9. Tim David (96.60)`, `10. Phil Salt (96.53)`.
   - **Active-Player Top 10 Batters (`#3 Russell` and `#4 Karthik` excluded for franchise scouting):** Advances `Pathum Nissanka (96.16, Active #9)` and `Rajat Patidar (94.81, Active #10)` into the active Top 10.
   - **Official Unrestricted & Active Top 10 Bowlers (`PRI_lcb`, `n = 200` eligible; all Top 10 active):** `1. Jasprit Bumrah (99.96)`, `2. Sunil Narine (99.96)`, `3. Varun Chakravarthy (99.87)`, `4. Rashid Khan (99.66)`, `5. Axar Patel (98.65)`, `6. Bhuvneshwar Kumar (98.18)`, `7. Nathan Ellis (97.73)`, `8. Josh Hazlewood (96.74)`, `9. Adil Rashid (95.94)`, `10. Wanindu Hasaranga (95.60)`. *(Next active candidates `#11–#13`: `Adam Zampa 94.99`, `Kuldeep Yadav 94.80`, `Mitchell Santner 94.79`; retired `#14 Ravichandran Ashwin 94.56`.)*
   - **Critical Separation — Battle-Tested HP Leaders vs Powerplay Pace-Setters:** `Suryakumar Yadav` leads verified high-pressure batting (`HP-RAA = +29.15 runs/100b`, `Neff_HP = 231.1`), joined by `Tim David (+21.50)`, `Nicholas Pooran (+18.63)`, `Andre Russell (+18.18)`, and `Heinrich Klaasen (+14.47)`. By contrast, `#2 Vaibhav Sooryavanshi` and `#5 Finn Allen` earn elite `PRI_lcb` from dominant Powerplay output (`+41.95` and `+28.28 runs/100b` at `PI < 60`) with **zero high-pressure exposure (`Neff_HP = 0.0`)** and must never be mislabeled as late-innings crunch closers. Among bowlers, `Sunil Narine (+0.64 runs/over)`, `Nathan Ellis (+0.59)`, `Rashid Khan (+0.55)`, and `Wanindu Hasaranga (+0.55)` deliver the strongest positive high-pressure bowling impact (`PI >= 60`).

2. **How reliable is the evidence supporting each player's performance?**
   - **Match-Cluster Bootstrap Tiers (`B = 2,000` replicates):** Resampling entire matches explicitly accounts for within-spell and within-partnership correlation (`median DEFF = 1.08` batting; `1.26` exact bowling including wides/no-balls). Four bowlers (`Narine 98.4%`, `Bumrah 97.2%`, `Varun 93.0%`, `Rashid Khan 83.6%`) and two batters (`Suryakumar Yadav 97.1%` Tier 1A Proven HP Lock; `Vaibhav Sooryavanshi 78.9%` Tier 1B Low-HP Statistical Lock) anchor Top-10 inclusion reliably. Ranks `#5–#15` form a tightly bunched **Contender / Bubble Tier (`24%–62%` Top-10 inclusion)** where single-metric point ordering should always be paired with `95%` intervals and `Neff_HP`.

3. **Which players offer the highest tactical value to Rajasthan Royals for 2026–27?**
   - **RR 2026 Baseline:** RR finished IPL 2026 at **`8–6` in the league stage (4th, 16 pts, NRR `+0.189`)** and **`1–1` in the playoffs (`9–7` overall, 3rd place)**—unbeaten in spin-friendly Guwahati (`3/3`) but exposed at high-scoring Jaipur (`1/4`).
   - **Asymmetric Roster Needs:** RR already holds elite opening firepower (`#2 Sooryavanshi 99.80` capped July 2026, `#22 Yashasvi Jaiswal 90.90`) alongside middle-order finishers (`Dhruv Jurel`, `Riyan Parag`, `Shimron Hetmyer`), **but fielded zero bowlers inside the Top 35**. Upgrading Death/Powerplay pace (`Nathan Ellis` trade check / `Josh Hazlewood` / uncapped sleeper `Sakib Hussain`) and attacking/containment spin (`Mitchell Santner` / `Wanindu Hasaranga` injury check / free-agent `Adil Rashid` or `Adam Zampa`), plus an RHB middle-order closer (`Tim David` / `Shashank Singh` / spin-hitter `Rajat Patidar`), directly addresses Jaipur's `10.13 RPO` pace leak and Death batting lag (`163.4` vs `169.4` league SR). *Note: All active 2026 franchise players remain subject to the November 17, 2026 IPL retention list ahead of the December 16, 2026 mini-auction in Goa.*

---

### 1.2 One-Page Methodology Pipeline & Glossary

```
[Raw Ball-by-Ball (200,999)] --> [Strict Pre-Delivery State Reconstruction (0 Outcome Leakage)]
         |
         +--> Exclusions: 844 ambiguous DLS balls + 38 Super Over balls (audited separately)
         +--> Tiering: IPL (w = 1.00) + Senior T20I (w = 0.50) -> 193,144 primary deliveries
         |             Emerging/Rising (w = 0.00) -> isolated supplementary scouting pool
         v
[Out-of-Fold Contextual Baselines (5-Fold Match-Grouped CV)]
         |--> Expected Runs & Expected Wicket Prob given (phase, over, PI, wkts, venue, year)
         |
         +--> Disjoint Partition: General (PI < 60) vs High Pressure (PI >= 60, weight * (PI/100)^2.0)
         |
         v
[Kish Effective Sample Size & Empirical Bayes Shrinkage + LCB (z = 0.50)]
         |--> Bat PRI_lcb:  45% RAA + 25% Survival + 15% Consistency + 15% HP-RAA
         |--> Bowl PRI_lcb: 55% RSAA + 15% WAA + 15% Consistency + 15% HP-Impact
         v
[B = 2,000 Match-Cluster Bootstrap Intervals + 7 Tactical Archetypes + RR Squad Fit]
```

| Term | Exact Definition & Natural Unit | Role in Decision Framework |
| :--- | :--- | :--- |
| **`PI`** (`[0, 100]`) | Pre-delivery Pressure Index built from innings, balls remaining, wickets in hand, required run rate, par chase benchmark, knockout context, and collapse state. | Partitions every delivery cleanly before the ball is bowled; `PI >= 60` defines High Pressure. |
| **`RAA`** (`runs/100b`) | Posterior runs above expected per 100 batting balls faced (`PI < 60` general partition). | Core batting scoring volume & strike-rate value (`45%` of Batting `PRI_lcb`). |
| **`Surv`** (`log-odds/pct`) | Posterior batting wicket-avoidance vs OOF expected dismissal rate (`PI < 60`). Positive = gets out less often than context expects. | Prevents reckless slogging without wicket preservation (`25%` of Batting `PRI_lcb`). |
| **`HP-RAA`** (`runs/100b`) | Posterior runs above expected per 100 batting balls strictly on `PI >= 60`, severity-weighted by `(PI / 100)^2`. | Measures proven crunch-time batting impact (`15%` of Batting `PRI_lcb`). |
| **`RSAA`** (`runs/over`) | Posterior bowling runs saved above expected per 6 legal balls (`PI < 60`), charging wides and overstep/other no-ball penalties to the bowler while excluding byes/leg-byes/penalty runs. | Primary bowling run-prevention pillar (`55%` of Bowling `PRI_lcb`). |
| **`WAA`** (`wkts/4-ov`) | Posterior bowler-credited wickets above expected per 24 legal balls (`PI < 60`); excludes run-outs, retired hurt/out, and obstructing the field. | Separates genuine strike bowlers from passive dart-throwers (`15%` of Bowling `PRI_lcb`). |
| **`HP-Impact`** (`runs/over`) | Unified high-pressure bowling value combining runs saved and wicket-value saved above expected strictly on `PI >= 60` (runs/over). | Primary clutch bowling metric (`15%` of Bowling `PRI_lcb`); replaces retired bowling Pressure Delta. |
| **`Cons`** (`[0, 1]`) | Shrinkage-adjusted inverse match-to-match variance consistency across appearances. | Rewards repeatable delivery across matches (`15%` of both `PRI_lcb` composites). |
| **`Neff` / `Neff_HP`** | Kish effective sample size `((sum w)^2 / sum w^2)` across all primary balls / `PI >= 60` balls, reflecting `IPL 1.0` vs `T20I 0.5` weights. | Quantifies evidence depth; distinguishes proven `PI >= 60` performers from `Neff_HP = 0` Powerplay specialists. |
| **`DEFF`** | Design Effect = `Cluster Bootstrap Variance / Kish Analytical Variance` when resampling whole matches (`B = 2,000`). | Quantifies intra-match correlation (spells/partnerships) omitted by ball-independent formulas. |
| **`SS-PDI`** | Same-Season Same-Competition Phase Dominance Index: residual skill inside exact `(comp_tier, season_year, match_phase)` cells. | Isolates Powerplay, Middle, Death, and Spin-Hitting ability free of era scoring inflation (2023–26). |
| **`PRI_lcb`** (`[0, 100]`) | Normal-CDF percentile transformation of standardized composite `post - 0.50 * SD` across eligible primary players (`Neff >= 30`, `IPL >= 30`). | Official ranking index; note `99.9` vs `99.8` is a tail percentile compression, not linear natural units. |

---

## 2. Defining & Validating Pressure (`PI >= 60`)

### 2.1 Four Match Regimes, Hopeless-Chase Guard & Collapse Calibration

Because batting and bowling pressures peak in fundamentally different cricket states, `PI` models four distinct pre-delivery game-state regimes: **Innings-1 Batting**, **Innings-2 Chasing**, **Innings-1 Bowling**, and **Innings-2 Defending**.

- **DLS & Super Over Discipline:** `38` Super Over deliveries are scored on an elevated tie-breaker scale (`PI 85–100`) and kept out of regular-overs modeling. Matches where pre-innings DLS targets are explicitly confirmed (`797` balls) are included cleanly using official revised overs and targets, whereas `844` deliveries in mid-innings stoppage matches with ambiguous target timing are flagged and excluded (`193,144` non-Emerging regular primary deliveries retained).
- **Innings-2 Hopeless-Chase Guard:** Naively scaling `PI` monotonically with Required Run Rate (`RRR`) misclassifies hopeless garbage time (`RRR >= 24`, target unreachable) as maximum pressure. Empirical `PI` peaks exactly where outcomes hinge on every ball and collapses safely when matches are decided:
  - `RRR < 8` (`25,168` balls): Batting mean `PI = 22.77` (`0.0%` HP); Bowling mean `PI = 46.33` (`6.4%` HP) — defending bowlers feel early jeopardy defending par totals.
  - `RRR 8–12` (`40,390` balls): Batting mean `PI = 41.37` (`6.1%` HP); Bowling mean `PI = 60.67` (**`47.4%` HP**) — peak crunch-time equilibrium for defending bowlers.
  - `RRR 12–16` (`13,601` balls): Batting mean `PI = 66.55` (**`72.8%` HP**, p90 `80.44`); Bowling mean `PI = 42.90` (`2.3%` HP) — peak chase pressure for batters.
  - `RRR 16–24` (`5,170` balls): Batting mean `PI = 59.34` (`55.2%` HP); Bowling mean `PI = 18.26` (`0.0%` HP).
  - `RRR >= 24` (`3,294` balls): Batting mean `PI` drops cleanly to **`16.34`** (`0.0%` HP); Bowling drops to **`14.27`** (`0.0%` HP).
- **Wicket-Collapse Monotonicity:** In Innings 1, batting pressure rises smoothly from `0–2` wickets lost (`mean PI 18.96, 0% HP`) -> `3–5` wickets (`38.82, 2.7% HP`) -> `6+` wickets (`63.08, 63.2% HP`). In Chases, it escalates from `31.47 (1.4% HP)` -> `48.59 (29.8% HP)` -> `53.42 (50.6% HP)`.

![Pressure Index Calibration Across Regimes, Phases and Wicket States](figures/33_final_pressure_methodology.png)

### 2.2 Why Elite Openers Have `Neff_HP = 0.0` (Structural Exposure & Sensitivity Audit)

A crucial diagnostic finding explains why high-impact Powerplay openers (`#2 Vaibhav Sooryavanshi`, `#5 Finn Allen`, `#14 Priyansh Arya`) register `Neff_HP = 0.0`: **`PI >= 60` structurally targets late-innings chases and wicket-collapse rebuilds rather than overs 1–6.**

| Stratification Bucket | Primary Batting Balls | Mean `PI` | `p90 PI` | `% PI >= 40` | `% PI >= 50` | **`% PI >= 60` (HP)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Opener (Positions 1–2)** | `63,920` | `24.08` | `43.10` | `13.46%` | `4.68%` | **`1.52%`** |
| **Top Order (Position 3)** | `28,811` | `29.16` | `48.25` | `20.32%` | `8.43%` | **`2.70%`** |
| **Middle Order (Positions 4–5)** | `49,743` | `38.02` | `59.36` | `41.08%` | `22.07%` | **`9.45%`** |
| **Finisher (Positions 6+)** | `43,128` | `52.55` | `75.87` | `74.17%` | `57.22%` | **`37.83%`** |
| **Powerplay Phase (Overs 1–6)** | `59,147` | `25.03` | `44.77` | `16.52%` | `5.00%` | **`0.66%`** |
| **Middle Phase (Overs 7–15)** | `86,067` | `37.52` | `63.25` | `41.16%` | `26.33%` | **`13.48%`** |
| **Death Phase (Overs 16–20)** | `40,488` | `45.16` | `74.76` | `53.57%` | `38.19%` | **`26.62%`** |

Because `Sooryavanshi` (`67.0%` Powerplay balls, `max PI = 48.18`), `Finn Allen` (`70.7%` PP balls, `max PI = 54.43`), and `Priyansh Arya` (`75.4%` PP balls, `max PI = 53.66`) bat predominantly with fields up in overs 1–6 and rarely remain at the crease past over 12 unless pacing a blowout, they face **zero** `PI >= 60` balls. Two empirical sensitivity checks tested whether `PI >= 60` or the `15%` HP weight should be altered:

1. **Exploratory Powerplay Early-Pressure Variant (`PI >= 62` when Chase `RRR >= 10.5` or `>= 2` Powerplay wickets lost):** Reclassifies `16,969` Powerplay deliveries (`9.14%` of primary batting balls). Under this exploratory definition, `Sooryavanshi` earns `70` HP balls and moves `#2 -> #1` (`99.97`), `Phil Salt` jumps `#10 -> #4` (`154` HP balls), `Finn Allen` drops `#5 -> #11` (`66` HP balls), while full-pool ranking stability remains strong (**Spearman `rho = 0.9642`, `9/10` Top-10 overlap**). Because early Powerplay hitting still differs tactically from death-over wicket-depleted crunch play, we preserve `PI >= 60` as the single clean primary standard and explicitly distinguish **Powerplay Pace-Setters (`Tier 1B`)** from **Battle-Tested HP Leaders (`Tier 1A`)**.
2. **HP Component Weight (`15% -> 40%`) & `Neff_HP` Eligibility Gate (`None, 15, 30`) Grid:**
   - Raising `w_hp` from `15% -> 25%` preserves **`9/10` Top-10 overlap** in both disciplines (`Bat rho = 0.9857`, `Bowl rho = 0.9879`); at `w_hp = 35%`, overlap eases to `8/10` (`rho = 0.9448` Bat, `0.9513` Bowl).
   - Imposing a hard `min Neff_HP >= 15` or `>= 30` gate disqualifies **`36` (`19.1%`)** and **`64` (`34.0%`)** of the `188` eligible batters (removing Powerplay elites like `Sooryavanshi` and `Allen` outright rather than surfacing their `Neff_HP = 0` caveat transparently). Keeping `w_hp = 15%` without a destructive eligibility cutoff preserves full-role comparability while reporting `Neff_HP` prominently next to every rank.

![Structural Powerplay Exposure, HP Sensitivity Grid & Unified Runs/Over Scale](figures/37_claude_pressure_explainer_and_hp_exposure.png)

---

## 3. Measuring Performance, Uncertainty & Retiring Bowling Pressure Delta

### 3.1 Empirical Bayes Shrinkage & Composite Tail Behavior

General (`PI < 60`) and High-Pressure (`PI >= 60`) metrics are estimated separately with empirical Bayes shrinkage (`Normal-Normal` for `RAA`, `RSAA`, `HP-RAA`, `HP-Impact`; `Beta-Binomial` for `Survival` and `WAA`) weighted by Kish `Neff` and lowered by `0.50 * SD_post` before Normal-CDF scaling onto `[0, 100]`. Consequently:
- Small-sample spikes (`Priyansh Arya Neff = 218.6`, `Sakib Hussain Neff = 183.6`) are damped proportionally toward league expectation, and zero-HP players (`Neff_HP = 0`) default exactly to the HP prior (`-1.54 runs/100b`) with wide empirical CIs (`[-37.0, +33.9]`).
- Because `PRI_lcb` maps standardized LCB Z-scores through the Normal CDF, fractional score gaps at the extreme right tail (`99.91` vs `99.80`, or `99.96` vs `99.96`) correspond to large differences in raw Z-score (`~0.25–0.35 SD`), whereas ranks `#5–#12` occupy a narrow band (`97.6` down to `94.8`). Natural units (`runs/100b`, `runs/over`, `wickets/4-over`) must therefore always be read alongside `PRI_lcb`.

### 3.2 Audit Decision: Retire Bowling `Pressure Delta`, Standardize on `runs/over`

An internal audit of `Pressure Delta` (`HP minus General`) uncovered a critical methodological flaw in **bowling** `Pressure Delta`:

1. **Scale Mismatch:** Earlier tables printed `RSAA_post` and `HP_Impact_post` in `runs/over` alongside `Pressure Delta` in `runs/100 balls` (`6x` multiplier), making modest per-over drops appear massive (`-33.75` for Bumrah, `-37.76` for Varun).
2. **Structural Negative Bias (`55.5%` of `200` primary bowlers negative; median `-0.085 runs/over`):** Across the primary dataset, `PI >= 60` overs for bowlers occur disproportionately in tougher 2nd-innings defense and death overs (Inn-1 HP actual `11.43 RPO` vs `10.95` exp; Inn-2 HP weighted residual `+0.35/ov` vs Gen `+0.44/ov`) and carry `15x` larger prior variance. Comparing HP minus a world-class `PI < 60` baseline (`+1.94/ov` for Bumrah, `+1.42/ov` for Varun) systematically penalizes elite general-overs dominance.
3. **Failure of Temporal Predictive Validity (`2022–24 -> 2025–26`):** Across `N = 131` primary bowlers active in both eras, **Bowling `Pressure Delta` correlates negatively with future high-pressure performance (`r = -0.2197`)**, whereas **absolute `HP-Impact` correlates positively (`r = +0.1220`)**. By contrast, **Batting `Pressure Delta` generalizes positively (`r = +0.2987`, `N = 97`)** alongside `HP-RAA` (`r = +0.3566`).

> **Formal Decision:** **Bowling `Pressure Delta` is retired** from all rankings and scouting comparisons. Every bowler evaluation now compares **General Net Saved (`PI < 60`)** and **High-Pressure Impact (`PI >= 60`)** side-by-side on the exact same **`runs/over`** scale (Figure 37D). Batting `Pressure Delta` is retained strictly as a secondary scouting diagnostic subject to `Neff_HP` and `HP-RAA` level guardrails.

### 3.3 Full-Composite Rank Uncertainty (`B = 2,000` Cluster Bootstrap Tiers)

To measure true rank precision rather than single-component CIs, we re-ran the full end-to-end composite ranking (recomputing OOF component posteriors, LCBs, and Normal-CDF `PRI_lcb` scores inside each bootstrap draw) across **`B = 2,000` match-cluster replicates**:

- **Design Effects & Multi-Seed Consistency:** Match clustering inflates variance modestly in batting (`median DEFF = 1.076` general, `1.084` HP) and materially in bowling (`median exact DEFF = 1.256` including wides/no-balls vs `1.198` legal-only). Across three independent `B = 2,000` random seeds (`42, 101, 2027`), both Top-10 sets remain **100% identical** (`rho = 0.9954` batting, `0.9917` bowling).
- **Performance Tiers (Full-Composite Bootstrap Distribution):**
  - **Tier 1A — Proven HP Core Locks (`>80%` Top-10 Inclusion & Proven `Neff_HP >= 100`):** `Suryakumar Yadav` (`97.1%`, rank `[1–8]`), `Sunil Narine` (`98.4%`, `[1–7]`), `Jasprit Bumrah` (`97.2%`, `[1–8]`), `Varun Chakravarthy` (`93.0%`, `[1–12]`), and `Rashid Khan` (`83.6%`, `[1–18]`).
  - **Tier 1B — Low-HP Statistical Lock (`>75%` Top-10, Zero HP Exposure):** `Vaibhav Sooryavanshi` (`78.9%` unrestricted / `81.4%` active, rank `[1–34]`, `Neff_HP = 0.0`) — statistically unambiguous elite Powerplay hitter, structurally untested at `PI >= 60`.
  - **Tier 2 — Top-10 Contenders / Bubble (`23%–62%` Top-10 Inclusion):** Includes retired locks `Andre Russell (71.7%, [1–34])` and `Dinesh Karthik (63.9%, [1–38])`, plus active batters `#5 Finn Allen (48.1%, [2–56])`, `#6 Nicholas Pooran (41.5%, [2–58])`, `#9 Tim David (40.1%, [2–57])`, `#7 Shreyas Iyer (39.6%, [3–52])`, `#10 Phil Salt (38.3%, [3–55])`, `#8 Heinrich Klaasen (36.9%, [4–44])`, `#11 Pathum Nissanka (33.0% unrestricted / 39.0% active)`, and `#12 Rajat Patidar (27.8% / 32.2% active)`, and active bowlers `#5 Axar Patel (62.0%, [3–29])`, `#6 Bhuvneshwar Kumar (52.9%, [2–40])`, `#7 Nathan Ellis (48.8%, [3–37])`, `#8 Josh Hazlewood (41.1%, [3–70])`, `#9 Adil Rashid (33.8%, [4–54])`, `#10 Wanindu Hasaranga (30.3%, [5–47])`, `#13 Mitchell Santner (31.0%)`, `#11 Adam Zampa (29.5%)`, `#12 Kuldeep Yadav (27.9%)`, and retired `#14 Ravichandran Ashwin (23.7%)`.
  - **Tier 3 — Watchlist Fringe (`<20%` Top-10 Inclusion):** High-upside sleepers such as `#15 Sakib Hussain` (`14.9%`, `[7–73]`, `Neff = 183.6`).

![Full-Composite Rank Uncertainty Across B=2,000 Match-Cluster Bootstrap Replicates](figures/38_claude_rank_uncertainty_intervals.png)

---

## 4. Fair Cross-Format Comparisons (`IPL = 1.00`, `Senior T20I = 0.50`, `Emerging = 0.00`)

### 4.1 Empirical Competition-Bridge Calibration & Domestic-League Gap

- **Domestic League Data Gap:** Independent audit of the official competition dataset confirms `IPL`, `Senior T20I`, and `Emerging/Rising Asia Cup` data (`867` ball-by-ball matches), with **zero matches from non-IPL domestic T20 leagues** (SMAT, BBL, SA20, PSL, Vitality Blast). Accordingly, all primary rankings rely exclusively on verified IPL and Senior T20I deliveries, while non-IPL uncapped prospects lacking `>= 30` IPL balls cannot be discovered without external domestic scouting.
- **Why Senior T20Is Must Be Down-Weighted (`0.50` vs `1.00`):** Across `105` bridge batters active in both IPL and Top-8 full-member T20Is, IPL scoring is **`-5.21 runs/100b` tougher** (`r = 0.4624`); across `58` bridge batters facing non-Top-8 / Associate opposition, the inflation gap widens to **`-13.92 runs/100b`** (`r = 0.3976`). Similarly, bowlers save `-0.27 runs/over` more in Top-8 T20Is and **`-1.03 runs/over`** more against non-Top-8 teams relative to IPL baselines. This provides empirical justification for weighting `w_t20i < 1.0` rather than treating all T20Is as IPL-equivalent.
- **Strict Isolation of Emerging/Rising T20 (`w_emerging = 0.00`):** Rising Stars / Emerging Asia Cup matches (`6,985` deliveries) carry weight zero in primary evaluation so A-team spikes never enter IPL/T20I rankings silently.

### 4.2 Full-Composite `w_t20i` Sensitivity Grid (`0.00` to `1.00`)

Because empirical bridge calibration supports `w_t20i < 1.0` conceptually without uniquely pinning down `0.50`, we re-ran the full end-to-end pipeline across `w_t20i in {0.00, 0.25, 0.50, 0.75, 1.00}`:

| T20I Weight (`w_t20i`) | Eligible Bat / Bowl | Bat `rho` vs `0.50` | Bat Top-10 Overlap | Bowl `rho` vs `0.50` | Bowl Top-10 Overlap | Key Sensitive Movers (`0.25 -> 0.50 -> 0.75`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`0.00` (IPL Only)** | `178` / `190` | `0.9154` | `7 / 10` | `0.9106` | `6 / 10` | Excludes T20I-heavy anchors (`Adil Rashid #119`, `Nissanka #49`, `Tim David #29`) |
| **`0.25`** | `187` / `199` | **`0.9853`** | `8 / 10` | **`0.9853`** | `8 / 10` | `Tim David #17 -> #9 -> #5`; `Phil Salt #11 -> #10 -> #6` |
| **`0.50` (Approved)** | `188` / `200` | `1.0000` | `10 / 10` | `1.0000` | `10 / 10` | Balanced plateau (`Nissanka #15 -> #11 -> #9`; `Ellis #11 -> #7 -> #6`) |
| **`0.75`** | `188` / `200` | **`0.9923`** | `9 / 10` | **`0.9928`** | `9 / 10` | `Klaasen #5 -> #8 -> #11`; `Santner #8 -> #13 -> #14`; `Kuldeep #16 -> #12 -> #9` |
| **`1.00` (Equal)** | `189` / `200` | `0.9790` | `8 / 10` | `0.9824` | `9 / 10` | Overweights T20I volume (`Abhishek Sharma #7`, `Naman Dhir #11`) |

Ranks across the moderate-discount plateau (`0.25–0.75`) correlate above **`0.985`** in both disciplines, confirming `w_t20i = 0.50` is stable while flagging cross-format sensitive profiles (`Adil Rashid`, `Pathum Nissanka`, `Tim David`, `Nathan Ellis`) transparently.

---

## 5. Official Unrestricted Top 10 & Active-Player Shortlists

### 5.1 Top 10 Batters (Unrestricted Official Rank #1–#10 + Active #9–#10 Extensions)

Per the Problem Statement, no age, nationality, or retirement filter is applied to the official Top 10 (`#1–#10`). Active replacement ranks (`Active #1–#10`, excluding retired `#3 Russell` and `#4 Karthik`) bring `#11 Pathum Nissanka` and `#12 Rajat Patidar` into the active Top 10. All 2026 franchise contracts are **subject to the November 17, 2026 retention list** ahead of the **December 16, 2026 mini-auction in Goa**.

| Off. Rk | Act. Rk | Player (`Age / Cap / DOB`) | `PRI_lcb` | Top-10 Inc (`[P05-P95]`) | Tier | `Neff` / `Neff_HP` | Gen `RAA` `[95% CI]` | `Surv` / `Cons` | `HP-RAA` `[95% CI]` | Role & 2026 Contract Context |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **1** | **Suryakumar Yadav**<br>`36 \| Capped IND \| 1990-09-14` | **99.91** | `97.1%` `[1-8]` | **1A** | `1284.8` / `231.1` | `+14.62` `[+5.9,+23.3]` | `+0.91` / `0.63` | **`+29.15`** `[+6.4,+51.9]` | **MI** (`INR 16.35 Cr`, *Pending Nov 17*) \| RHB #4 Spin/360 |
| **2** | **2** | **Vaibhav Sooryavanshi**<br>`15 \| Capped IND \| 2011-03-27` | **99.80** | `78.9%` `[1-34]` | **1B** | `280.3` / **`0.0`** | **`+41.95`** `[+21.7,+62.2]` | `-0.62` / `0.57` | `-1.54` `[-37.0,+33.9]` | **RR Incumbent** (`INR 1.10 Cr`, *T20I debut Jul 2026*) \| LHB #2 PP |
| **3** | — | **Andre Russell** *(RET)*<br>`38 \| Capped WI \| 1988-04-29` | **99.43** | `71.7%` `[1-34]` | **2** | `488.3` / `219.2` | `+18.59` `[+3.4,+33.8]` | `+0.43` / `0.59` | `+18.18` `[-10.2,+46.6]` | **IPL Retired** (`Nov 30, 2025`) \| RHB #7 Finisher |
| **4** | — | **Dinesh Karthik** *(RET)*<br>`41 \| Capped IND \| 1985-06-01` | **98.89** | `63.9%` `[1-38]` | **2** | `411.3` / `111.6` | `+11.31` `[-4.3,+26.9]` | **`+1.05`** / `0.58` | `+11.96` `[-26.1,+50.0]` | **IPL Retired** (`Jun 1, 2024`) \| RHB #6 Finisher/WK |
| **5** | **3** | **Finn Allen**<br>`27 \| Capped NZ \| 1999-04-22` | **97.63** | `48.1%` `[2-56]` | **2** | `294.9` / **`0.0`** | `+28.28` `[+10.1,+46.5]` | `-0.77` / `0.58` | `-1.54` `[-37.0,+33.9]` | **KKR** (`INR 2.00 Cr`, *Pending Nov 17*) \| RHB #1 PP Aggressor |
| **6** | **4** | **Nicholas Pooran**<br>`31 \| T20I-Ret WI \| 1995-10-02` | **96.82** | `41.5%` `[2-58]` | **2** | `1014.5` / `191.9` | `+7.64` `[-7.4,+22.7]` | `+0.41` / `0.56` | `+18.63` `[-4.0,+41.2]` | **LSG** (`INR 21.00 Cr`, *T20I ret Jun 2025; IPL active*) \| LHB WK |
| **7** | **5** | **Shreyas Iyer**<br>`31 \| Capped IND \| 1994-12-06` | **96.78** | `39.6%` `[3-52]` | **2** | `1017.9` / `41.1` | `+11.74` `[+1.4,+22.1]` | `+0.52` / `0.54` | `+9.29` `[-24.8,+43.4]` | **PBKS** (`INR 26.75 Cr`, *Pending Nov 17*) \| RHB #4 Anchor/Death |
| **8** | **6** | **Heinrich Klaasen**<br>`35 \| Capped SA \| 1991-07-30` | **96.73** | `36.9%` `[4-44]` | **2** | `805.1` / `209.0` | `+11.27` `[+0.6,+22.0]` | `+0.37` / `0.55` | `+14.47` `[-5.1,+34.0]` | **SRH** (`INR 23.00 Cr`, *Pending Nov 17*) \| RHB #5 Spin/WK |
| **9** | **7** | **Tim David**<br>`30 \| Capped AUS \| 1996-03-16` | **96.60** | `40.1%` `[2-57]` | **2** | `636.2` / `270.5` | `+12.41` `[-3.0,+27.8]` | `+0.32` / `0.52` | **`+21.50`** `[-4.3,+47.3]` | **RCB** (`INR 3.00 Cr`, *Pending Nov 17*) \| RHB #6 Finisher |
| **10** | **8** | **Phil Salt**<br>`30 \| Capped ENG \| 1996-08-28` | **96.53** | `38.3%` `[3-55]` | **2** | `569.8` / `41.2` | `+18.55` `[+5.5,+31.6]` | `-0.26` / `0.56` | `+6.91` `[-40.3,+54.1]` | **RCB** (`INR 11.50 Cr`, *Pending Nov 17*) \| RHB #1 PP/WK |
| *11* | **9** | *Pathum Nissanka*<br>`28 \| Capped SL \| 1998-05-18` | *96.16* | `33.0%` `[4-51]` | **2** | `767.5` / `27.6` | `+9.87` `[-0.6,+20.3]` | `+0.46` / `0.57` | `+5.67` `[-59.0,+70.3]` | **DC** (`INR 4.00 Cr`, *Pending Nov 17*) \| RHB PP Anchor |
| *12* | **10** | *Rajat Patidar*<br>`33 \| Capped IND \| 1993-06-01` | *94.81* | `27.8%` `[4-68]` | **2** | `529.4` / `32.6` | `+15.44` `[+0.9,+30.0]` | `+0.21` / `0.54` | `-2.67` `[-51.9,+46.6]` | **RCB** (`INR 11.00 Cr`, *Pending Nov 17*) \| RHB Spin Hitter (`+32.5`) |
| *13* | *11* | *Shashank Singh*<br>`34 \| Uncapped IND \| 1991-11-21` | *94.33* | `26.2%` `[3-82]` | **2** | `389.8` / `146.1` | `+8.17` `[-9.9,+26.2]` | `+0.65` / `0.55` | `+7.44` `[-16.8,+31.7]` | **PBKS** (`INR 5.50 Cr`, *Pending Nov 17*) \| Uncapped RHB #6 |
| *14* | *12* | *Priyansh Arya*<br>`25 \| Uncapped IND \| 2001-09-18` | *94.16* | `32.2%` `[3-111]` | **2** | `218.6` / **`0.0`** | `+28.04` `[+1.2,+54.9]` | `-1.04` / `0.54` | `-1.54` `[-37.0,+33.9]` | **PBKS** (`INR 3.80 Cr`, *Pending Nov 17*) \| Uncapped LHB PP |
| *15* | *13* | *Sanju Samson*<br>`32 \| Capped IND \| 1994-11-11` | *94.09* | `22.7%` `[5-70]` | **2** | `941.2` / `61.2` | `+7.32` `[-2.1,+16.8]` | `+0.01` / `0.54` | `+25.71` `[-1.7,+53.2]` | **CSK** (`INR 18.00 Cr`, *Traded RR->CSK pre-2026*) \| RHB WK |

### 5.2 Top 10 Bowlers (All Active Top 10 + Watchlist #11–#15, Unified `runs/over` Scale)

All Top-10 bowlers are active players; `#14 Ravichandran Ashwin` (`94.56`, retired Aug 27, 2025) is the highest retired bowler on the watchlist. Bowling `Pressure Delta` is retired; general run prevention (`Gen Net Saved`, combining `RSAA + WAA` run value) and clutch `HP-Impact` are reported directly in unified **`runs/over`** units.

| Off. Rk | Act. Rk | Player (`Age / Cap / DOB`) | `PRI_lcb` | Top-10 Inc (`[P05-P95]`) | Tier | `Neff` / `Neff_HP` | `RSAA_post` (`runs/ov`) | `WAA` / `Cons` | `Gen Net` (`runs/ov`) | `HP-Impact` (`runs/ov`) | Role & 2026–27 Status Context |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **1** | **Jasprit Bumrah**<br>`32 \| Capped IND \| 1993-12-06` | **99.96** | `97.2%` `[1-8]` | **1A** | `1083.2` / `316.4` | **`+1.92`** `[+1.31,+2.53]` | `+0.01` / `0.59` | **`+1.94`** | `-0.08` `[-1.28,+1.12]` | **MI** (`INR 18.00 Cr`, *Pending Nov 17*) \| All-Phase Pace |
| **2** | **2** | **Sunil Narine**<br>`38 \| Capped WI \| 1988-05-26` | **99.96** | `98.4%` `[1-7]` | **1A** | `1174.9` / `293.4` | `+1.73` `[+1.14,+2.33]` | `-0.03` / **`0.63`** | `+1.67` | **`+0.64`** `[-0.58,+1.85]` | **KKR** (`INR 12.00 Cr`, *IPL-only pool*) \| Mystery Spin |
| **3** | **3** | **Varun Chakravarthy**<br>`35 \| Capped IND \| 1991-08-29` | **99.87** | `93.0%` `[1-12]` | **1A** | `1398.6` / `363.4` | `+1.23` `[+0.65,+1.82]` | **`+0.10`** / `0.62` | `+1.42` | `-0.84` `[-2.05,+0.36]` | **KKR** (`INR 12.00 Cr`, *Pending Nov 17*) \| Middle Mystery Spin |
| **4** | **4** | **Rashid Khan**<br>`28 \| Capped AFG \| 1998-09-20` | **99.66** | `83.6%` `[1-18]` | **1A** | `1838.2` / **`450.4`** | `+0.79` `[+0.23,+1.35]` | `+0.07` / `0.56` | `+0.92` | `+0.55` `[-0.55,+1.66]` | **GT** (`INR 18.00 Cr`, *Pending Nov 17*) \| Attacking Leg-Spin |
| **5** | **5** | **Axar Patel**<br>`32 \| Capped IND \| 1994-01-20` | **98.65** | `62.0%` `[3-29]` | **2** | **`1844.9`** / `436.8` | `+0.91` `[+0.42,+1.39]` | `+0.01` / `0.56` | `+0.93` | `+0.29` `[-0.83,+1.40]` | **DC** (`INR 16.50 Cr`, *Pending Nov 17*) \| SLAO All-Rounder |
| **6** | **6** | **Bhuvneshwar Kumar**<br>`36 \| Capped IND \| 1990-02-05` | **98.18** | `52.9%` `[2-40]` | **2** | `1359.5` / `367.0` | `+0.82` `[+0.26,+1.38]` | `+0.05` / `0.53` | `+0.92` | `-0.07` `[-1.23,+1.09]` | **RCB** (`INR 10.75 Cr`, *Pending Nov 17*) \| PP Swing & Death |
| **7** | **7** | **Nathan Ellis**<br>`32 \| Capped AUS \| 1994-09-22` | **97.73** | `48.8%` `[3-37]` | **2** | `572.7` / `203.8` | `+0.90` `[+0.09,+1.71]` | `+0.02` / `0.51` | `+0.94` | **`+0.59`** `[-0.71,+1.89]` | **Traded CSK->SRH pre-2027** *(2026 hamstring inj; 0 IPL 2026)* |
| **8** | **8** | **Josh Hazlewood**<br>`35 \| Capped AUS \| 1991-01-08` | **96.74** | `41.1%` `[3-70]` | **2** | `879.1` / `223.0` | `+0.55` `[-0.31,+1.41]` | `+0.04` / `0.54` | `+0.62` | `+0.32` `[-1.22,+1.87]` | **RCB** (`INR 12.50 Cr`, *Pending Nov 17*) \| PP Hit-the-Deck |
| **9** | **9** | **Adil Rashid**<br>`38 \| Capped ENG \| 1988-02-17` | **95.94** | `33.8%` `[4-54]` | **2** | `941.7` / `174.6` | `+0.79` `[+0.08,+1.51]` | `+0.02` / `0.56` | `+0.82` | `-0.23` `[-1.56,+1.10]` | **Free Agent / Unsold IPL 2026** *(T20I leg-spin; 0 IPL >2023)* |
| **10** | **10** | **Wanindu Hasaranga**<br>`29 \| Capped SL \| 1997-07-29` | **95.60** | `30.3%` `[5-47]` | **2** | `1309.9` / `353.3` | `+0.30` `[-0.27,+0.86]` | `+0.04` / `0.53` | `+0.38` | `+0.55` `[-0.70,+1.80]` | **LSG** (`INR 2.00 Cr`, *2026 T20 WC hamstring inj; Pending Nov 17*) |
| *11* | *11* | *Adam Zampa*<br>`34 \| Capped AUS \| 1992-03-31` | *94.99* | `29.5%` `[4-71]` | **2** | `824.3` / `244.4` | `+0.50` `[-0.22,+1.22]` | `+0.08` / `0.52` | `+0.64` | `-0.25` `[-1.54,+1.03]` | **Not in 2026 Active Squads** *(Opted out / Auction pool)* |
| *12* | *12* | *Kuldeep Yadav*<br>`31 \| Capped IND \| 1994-12-14` | *94.80* | `27.9%` `[4-63]` | **2** | `1300.2` / `361.1` | `+0.50` `[-0.14,+1.14]` | `+0.03` / `0.51` | `+0.56` | `+0.39` `[-0.77,+1.56]` | **DC** (`INR 13.25 Cr`, *Pending Nov 17*) \| Left-Arm Chinaman |
| *13* | *13* | *Mitchell Santner*<br>`34 \| Capped NZ \| 1992-02-05` | *94.79* | `31.0%` `[4-68]` | **2** | `1164.8` / `264.3` | `+0.68` `[-0.10,+1.47]` | `-0.02` / `0.53` | `+0.65` | `+0.49` `[-0.76,+1.73]` | **MI** (`INR 2.00 Cr`, *Pending Nov 17; #7 IPL-only*) \| SLAO AR |
| *14* | — | *Ravichandran Ashwin* *(RET)*<br>`40 \| Capped IND \| 1986-09-17` | *94.56* | `23.7%` `[5-63]` | **2** | `1069.3` / `275.3` | `+0.65` `[+0.02,+1.29]` | `0.00` / `0.54` | `+0.66` | `+0.15` `[-1.08,+1.38]` | **IPL Retired** (`Aug 27, 2025`; ex-RR) \| Off-Spin |
| *15* | *14* | *Sakib Hussain*<br>`21 \| Uncapped IND \| 2004-12-14` | *92.07* | `14.9%` `[7-73]` | **3** | `183.6` / `64.3` | `+0.84` `[-0.26,+1.95]` | `+0.03` / `0.55` | `+0.90` | `-0.67` `[-2.32,+0.98]` | **SRH** (`INR 0.30 Cr`, *Pending Nov 17*) \| Uncapped Young Pacer |

![Official Top 10 Rankings, B=2,000 Cluster Bootstrap Stability & Enriched Context](figures/34_final_top10_validation_uncertainty.png)

### 5.3 Seven Tactical Role Archetypes, Spin-Hitting & Excluded/Caution Names

- **7 Tactical Archetypes (`SS-PDI`, controlling for 2023–26 era inflation):**
  1. **Powerplay Aggressors:** `Sooryavanshi (+34.89)`, `Priyansh Arya (+24.93)`, `Finn Allen (+24.89)`, `Pathum Nissanka (+18.33)`, `Phil Salt (+17.20)`.
  2. **Middle-Over Accelerators:** `Suryakumar Yadav (+22.95)`, `Andre Russell (+14.67)`, `Tim David (+12.96)`, `Heinrich Klaasen (+12.92)`, `Nicholas Pooran (+8.72)`, `Shreyas Iyer (+8.48)`.
  3. **Spin Hitters (`pace_or_spin == 'Spin'`):** Distinct skill (`Spin SS-PDI vs PRI rho = 0.5499`). Active standouts: `Rajat Patidar (+32.48, n=349)`, `Heinrich Klaasen (+27.07, n=607)`, `Finn Allen (+25.58, n=216)`, `Abhishek Sharma (+23.8, n=715)`, `Priyansh Arya (+21.1, n=88)`, `Vaibhav Sooryavanshi (+20.50, n=148)`, `Suryakumar Yadav (+10.84)`.
  4. **Death Finishers:** `Shreyas Iyer (+27.78)`, `Andre Russell (+19.43)`, `Dinesh Karthik (+18.50)`, `Tim David (+18.31)`, `Shashank Singh (+10.01)`.
  5. **Powerplay Bowlers:** `Bhuvneshwar Kumar (+0.52/ov)`, `Josh Hazlewood (+0.34)`, `Jasprit Bumrah (+0.33)`, `Nathan Ellis (+0.31)`, `Axar Patel (+0.22)`.
  6. **Middle-Over Spinners:** `Sunil Narine (+0.62/ov)`, `Rashid Khan (+0.58)`, `Kuldeep Yadav (+0.49)`, `Ravichandran Ashwin (+0.46)`, `Axar Patel (+0.43)`, `Varun Chakravarthy (+0.39)`.
  7. **Death Specialists:** `Sunil Narine (+0.78/ov)`, `Rashid Khan (+0.50)`, `Adam Zampa (+0.44)`, `Wanindu Hasaranga (+0.40)`, `Jasprit Bumrah (+0.37)`, `Nathan Ellis (+0.03)`. *(Crucial usage caveat: `Varun Chakravarthy (-0.90)` and `Axar Patel (-0.61)` lose value when bowled at the death.)*

![Same-Season Same-Competition Phase Dominance & 7 Tactical Archetypes](figures/36_final_role_archetypes_rr_fit.png)

- **Notable Exclusions & High-Reputation Caution Traps:**
  - *Primary `IPL >= 30` Gate:* Senior T20I-only specialists (`Fakhar Zaman 92.61`, `Babar Azam 91.33`, `Mohammad Nawaz 96.25`, `Haris Rauf 95.40`) are excluded cleanly due to zero IPL exposure.
  - *Negative Survival / Negligible HP:* `Virat Kohli (#25, 89.38, RAA -0.17, Surv +1.64, Neff_HP 7.0)` and `Yashasvi Jaiswal (#22, 90.90, RAA +17.51, Surv -0.50, Neff_HP 1.0)` miss the Top 10 due to neutral general scoring/survival and minimal `PI >= 60` evidence.
  - *Negative HP or Negative Run Impact (Scout Caution):* `Ishan Kishan (#56, HP-RAA -11.63)`, `KL Rahul (#58, RAA -2.20, HP -6.05)`, `Rohit Sharma (#108, Surv -1.02, HP -6.19)`, `Mohammed Siraj (#69, HP-Imp -0.86/ov)`, `Kagiso Rabada (#32, HP-Imp -0.63/ov)`, `Avesh Khan (#88, RSAA -0.25/ov)`, `Tushar Deshpande (#142, RSAA -0.52/ov)`.

---

## 6. Rajasthan Royals 2026–27 Squad-Fit Assessment

### 6.1 Season Record, Venue Split & Career PRI vs 2026 Output Separation

To evaluate tactical needs accurately, **career primary `PRI_lcb` (`2022–2026`)** is kept strictly separate from **IPL 2026 single-season team output**:

- **Verified IPL 2026 Record:** Rajasthan Royals went **`8–6` in the 14-match league stage (4th place, 16 points, NRR `+0.189`)** and **`1–1` in the playoffs** (defeated SRH by 47 runs in the Eliminator on May 27; lost Qualifier 2 to GT by 7 wickets on May 29) for a **`9–7` overall record (3rd place)**.
- **Venue Polarity (`2023–2026`):** RR went **`3/3` in spin-friendly Guwahati (`168.9` avg Inn-1, `9.06 RPO`, spin econ `8.07`, chase win `62.5%`)**, but collapsed **`1/4` at high-scoring Jaipur SMS (`194.5` avg Inn-1, `9.78 RPO`, Death `11.21 RPO`, pace econ `10.13`, chase win `42.9%`)**.
- **IPL 2026 Phase Diagnostics vs League Averages:**
  - *Batting:* RR crushed **Powerplay (`187.3 SR` vs `159.1` lg)** and led **Middle (`153.0` vs `148.3`)**, but lagged in **Death (`163.4 SR` vs `169.4` lg; `8.82%` wkt vs `7.44%`)**.
  - *Bowling:* Pace leaked **(`10.21 econ` vs `9.97` lg)** across **Powerplay (`10.15` vs `9.95`)** and **Middle (`9.43` vs `9.22`)**, while spin broke even (`9.29` vs `9.29`).
- **Incumbent RR Career `PRI_lcb` Roster Benchmark:**
  - *Batting Strengths:* `#2 Vaibhav Sooryavanshi (99.80, LHB #2)`, `#22 Yashasvi Jaiswal (90.90, LHB #1)`, `#49 Riyan Parag (73.28, HP +7.55)`, `#61 Donovan Ferreira (67.93, RAA +7.33)`, `#66 Shimron Hetmyer (66.07, HP +16.64)`, `#79 Dhruv Jurel (59.59, HP +24.88)`, `#83 Dasun Shanaka (57.97, HP +14.08)`, `#128 Ravindra Jadeja (41.94, RAA -15.06)`, `#174 Shubham Dubey (19.51)`.
  - *Severe Bowling Deficit (**Zero Top-35 Bowlers**):* `#37 Ravindra Jadeja (78.86, RSAA +0.32, HP -0.21)`, `#44 Ravi Bishnoi (77.49, RSAA +0.20, HP +0.07)`, `#48 Sandeep Sharma (75.75, RSAA +0.55, HP -0.04)`, `#54 Jofra Archer (73.74, RSAA +0.05, HP +0.01)`, `#76 Brijesh Sharma (59.59)`, `#112 Nandre Burger (45.05, RSAA -0.45)`, `#142 Tushar Deshpande (32.79, RSAA -0.41)`.

### 6.2 Targeted 2027 Acquisition Priorities *(Pre-Retention Framework)*

All contracted targets below depend on franchise retention/release decisions due **November 17, 2026** (or bilateral trade availability) ahead of the **December 16, 2026 mini-auction in Goa**.

| Structural Gap | Top Statistical Fit Targets | Key Metrics | Contract / Medical / Auction Caveat |
| :--- | :--- | :--- | :--- |
| **1. Death & PP Pace Spearhead (Critical)** | `Nathan Ellis (#7)` / `Josh Hazlewood (#8)` / Sleeper: `Sakib Hussain (#15)` | Ellis: `HP-Imp +0.59/ov, RSAA +0.90`; Sakib: `RSAA +0.84, Age 21` | **Ellis traded CSK->SRH pre-2027** & missed 2026 (hamstring) — trade/medical check required. Hazlewood (`RCB 12.5 Cr`) & Sakib (`SRH 0.30 Cr`) pending Nov 17 list. |
| **2. Wicket-Taking & Control Spin** | `Mitchell Santner (#13)` / `Wanindu Hasaranga (#10)` / `Adil Rashid (#9)` / `Adam Zampa (#11)` | Santner: `RSAA +0.68, HP +0.49 (#7 IPL-only)`; Hasaranga: `HP +0.55, SR 14.95` | **Adil Rashid** (Unsold 2026) & **Zampa** (free-agent auction pool) are directly reachable in auction if registered. **Hasaranga** (`LSG 2.00 Cr`) & **Santner** (`MI 2.00 Cr`) pending Nov 17 + Hasaranga hamstring recovery check. |
| **3. RHB Death Closer & Spin Hitter** | `Tim David (#9)` / `Rajat Patidar (#12)` / `Shashank Singh (#13)` | David: `HP-RAA +21.50, Death +18.3`; Patidar: `Spin +32.5`; Shashank: `HP +7.44` | Balances LHB-heavy top order (`Jaiswal`, `Sooryavanshi`, `Hetmyer`, `Jadeja`). All 3 currently contracted (`RCB`/`PBKS`) — pending Nov 17 retention or trade. |
| **Core Asset Preservation** | Retain `Sooryavanshi`, `Jaiswal`, `Jurel`, `Parag` | Top-2 PP (`SR 187.3`) + Jurel (`HP +24.88`) | Do **not** spend auction budget on redundant marquee openers (`Phil Salt`, `Finn Allen`). |

---

## 7. Technical Validation, Limitations & Data Accounting Appendix

### 7.1 Complete Raw-to-Modeling Data Accounting

Every single row from the raw pack is accounted for cleanly with zero silent drops:

| Dataset Layer | Exact Count | Reconciliation Notes |
| :--- | :---: | :--- |
| **Raw Matches / BBB Matches** | `883` / `867` | All `867` matches with ball-by-ball records enter match-state reconstruction. |
| **Raw Deliveries Preserved** | `200,999` (`100.0%`) | `199,358` standard (`199,320` reg + `38` SO) + `797` pre-innings DLS + `844` ambiguous DLS. |
| **Excluded Non-Standard Rows** | `38` Super Over + `844` Ambiguous DLS | `38` SO (`14` IPL, `12` T20I, `12` Emg); `844` DLS (`487` inn-break, `214` post-stop, `143` pre-stop). |
| **Emerging/Rising Isolation** | `6,985` (`6,973` reg + `12` SO) | Assigned `w_emerging = 0.00`; excluded from primary pool, kept in supplementary watchlist. |
| **Primary Non-Emerging Pool** | **`193,144` rows** | `87,456` IPL + `105,688` Senior T20I regular non-ambiguous rows. |
| **Primary Batting Balls Faced** | **`185,702`** | `83,914` IPL + `101,788` T20I (`193,144 - 7,442` wides [`3,542` IPL + `3,900` T20I]). |
| **Primary Bowling Legal Balls** | **`184,884`** | `83,538` IPL + `101,346` T20I (`185,702 - 818` no-balls [`376` IPL + `442` T20I]). |
| **Playing XI Deduplication** | `28,381` -> **`28,341`** | `35` duplicate `(match_id, team_id, player_id)` keys (`75` rows) -> `40` extra rows dropped cleanly. |

### 7.2 Scratch Match-State Audit & Extras Attribution

An independent from-scratch reconstruction across 5 diverse audit matches—`IPL 2023 Final DLS (#21586)`, `SL vs IND Super Over (#34429)`, `IRE vs AFG all 6 extras types (#16753)`, `WI vs ENG non-striker run-out (#13513)`, and `GT vs RR IPL 2026 chase (#50944)`—recorded **`0` field mismatches across all 9 reconstructed state columns** and **zero post-delivery outcome columns** in the modeling table. Extras attribution across all `200,999` raw deliveries:
- `wide` (`7,714` balls): excluded from batter balls faced (`0%`) and legal balls (`0%`), penalized to bowler (`1.1893` runs/ball).
- `Overstep` (`576`) & `other` (`264`) no-balls: excluded from legal balls (`0%`), **included** in batter balls faced (`100%`), penalty `+1.0000` charged to bowler (`conc = 2.4792 / 2.5606`, excluding secondary bye/legbye runs).
- `legbyes` (`2,748`), `byes` (`529`), `penalty` (`1`): `100%` legal; bowler concession strictly excludes byes, leg-byes, and penalty runs.

### 7.3 Feature Leakage & Cross-Validation Integrity

- **Strictly Pre-Delivery OOF Predictors:** Both batting and bowling baseline pipelines use *only* `ipl_phase`, `inn_phase`, `overid`, `venue_id`, pre-delivery `pressure_bat`/`pressure_bowl`, `wickets_in_hand_before`, and `season_year`. Outcome columns overlap = **`[]`** (zero).
- **Fold Isolation (`GroupKFold(n_splits=5)` by `match_id`):** Maximum training/validation `match_id` overlap across every fold is **`0`**. Feature encoding (`OneHotEncoder`, `SimpleImputer`, `StandardScaler`) and extras run/wicket multipliers fit strictly inside each training fold; Innings-1 par baselines use Leave-One-Match-Out excluding the current match; pre-stoppage DLS delivers zero future `revised_target` leakage.
- **EB Hyper-Prior Scope Note:** Empirical Bayes pool hyper-priors (`mu0, tau2, M0`) are estimated once empirically across the primary eligible player pool (`n = 188 / 200`) using out-of-fold delivery residuals; strict chronological validation (`Train 2022–24 -> Test 2025–26`) verifies forward out-of-sample generalization without leakage.

### 7.4 Chronological Holdout (`Train 2022–24 -> Test 2025–26`) & Model Comparisons

Training strictly on `2022–2024` (`498` matches, `112,070` bat / `111,564` bowl balls) and predicting unseen `2025–2026` deliveries (`339` matches, `73,632` bat / `73,320` bowl balls) establishes empirical out-of-sample validity:

| Framework Evaluated | Temporal Out-of-Sample Result (`<=2024 -> 2025–26`) | Agreement vs Approved | Verdict |
| :--- | :--- | :--- | :--- |
| **Approved Primary `PRI_lcb`** | Fut bat RMSE `1.77386`; Player fut **`r = 0.2115` (bat)**, **`r = 0.3916` (bowl econ)**, **`rho = 0.3110` (HP bat)** | Reference Baseline (`10/10`) | **Primary Standard** |
| **Enriched Context RoE** (`+pace/spin/score`) | Fut RMSE `1.76911` bat / `1.76912` bowl; Player fut `r = 0.2059` bat, `0.3670` bowl | Bat norm `rho = 0.9949 (10/10)`; Bowl `0.9717 (8/10)` | **Context Sensitivity** (conditions away spin/pace identity) |
| **Pre-Delivery WP & Leverage (`LI >= 10%`)** | Inn-2 AUC `0.8962` vs **Inn-1 AUC `0.6375`** (flat first-innings discrimination) | Bat `rho = 0.2190 (1/10)`; Bowl `0.3469 (4/10)` | **Excluded as Primary** (starves Inn-1 pressure rebuilds) |
| **Two-Way Matchup Ridge (`a = 100 / 500`)** | Fut bat RMSE `1.77117`; lifts future batting `r = 0.3350`, bowl `r = 0.3703` | Bat `rho = 0.7667 (5/10)`; Bowl `0.8154 (4/10)` | **Opponent-Strength Check** (no `PI >= 60` weighting) |
| **Hierarchical Venue + Era Trend** | Fut bat RMSE `1.76982`; max season-phase resid `6.32 -> 2.38` | `rho = 0.9932`, **`10/10` overlap** | **Confirmed Robustness** (linear trend risk if scoring plateaus) |
| **Weighted Wicket Quality (`WQA`)** | Fut WAA `r = 0.0835` vs Approved `r = 0.0821` | `rho = 0.9983`, **`10/10` overlap** | **Negligible Gain** (wicket sparsity dominates) |

![Temporal Out-of-Sample Generalization & Batting Pressure Delta Guardrail](figures/35_final_temporal_holdout_validation.png)

### 7.5 Key Methodological Limitations

1. **Absence of Non-IPL Domestic Leagues:** With zero SMAT/BBL/SA20/PSL data in the pack, uncapped domestic sleepers with `< 30` IPL balls require external domestic scouting (e.g., following up `Nishant Kumar Sindhu` from the Emerging watchlist).
2. **Left-Censoring of Player Tenure:** Because `72.9%` of eligible batters and `62.0%` of bowlers appear in the dataset's `2022` opening window, tenure in this pack correlates weakly with true age (`r = 0.3496`); external verified DOBs and cap status are required.
3. **High-Pressure Sparsity for Openers:** Aggressive Powerplay openers rarely experience `PI >= 60` late-innings overs (`Neff_HP = 0.0`), forcing shrinkage toward the HP prior (`-1.54`) and requiring explicit `Tier 1B` labeling rather than calling them "proven clutch closers".
4. **T20 Wicket Stochasticity:** Single-season dismissal counts carry inherently high randomness (`temporal WAA r = 0.082`), confirming run prevention (`RSAA` `55%` weight) is a far stabler signal than wicket rate (`WAA` `15%` weight).

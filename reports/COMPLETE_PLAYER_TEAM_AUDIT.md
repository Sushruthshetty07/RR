# Complete Player-to-Team Classification Audit
**Rajasthan Royals SupeRR Selector 2.0 — Comprehensive Universe Audit**
**Date:** 13 November 2026  
**Auditor:** AntiGravity Analytics Director & Verification Engine  
**Scope:** Complete Canonical Player Universe ($N = 1,216$ players across all dataset files, processed registries, dashboard APIs, and strategic reports)

---

### Executive Summary

In strict compliance with audit instructions, a complete player-to-team classification audit was conducted across the entire Rajasthan Royals SupeRR Selector 2.0 repository as of **13 November 2026**.

**Key Audit Findings:**
1. **Total Unique Players Audited:** Exactly **1,216 canonical players** from [data/processed/canonical_player_directory.csv](file:///C:/Users/sushruth/.gemini/antigravity/scratch/rr_superr_selector/data/processed/canonical_player_directory.csv).
2. **IPL Match Participation Breakdown:**
   - **265 players** have verified match/squad appearances in **IPL 2026** across the 10 franchises.
   - **171 players** have historical IPL appearances (2022–2025) but zero appearances in IPL 2026.
   - **780 players** have zero IPL match appearances in the supplied 2022–2026 dataset (competing exclusively in Senior Men's T20I bilaterals/ICC tournaments or the Emerging/Rising Asia Cup).
3. **Sanju Samson Franchise Status:** Confirmed as **Chennai Super Kings** in IPL 2026 (14 matches played as wicketkeeper) following the pre-2026 blockbuster swap that sent Ravindra Jadeja and Sam Curran to Rajasthan Royals. In `reports/final_analytical_report.md` Section 5.3 Case 3, a headline mislabeled Samson under "(Internal Core Maximization)"; this has been flagged for retitling to "(External Marquee Pursuit)".
4. **Nathan Ellis Franchise Status:** Retained by **Chennai Super Kings** for IPL 2026 but ruled out prior to the season due to a hamstring tear suffered in the Australian domestic One-Day Cup final (replaced by Spencer Johnson; zero dataset match appearances). Formally traded to **Sunrisers Hyderabad** on **5 November 2026** in Trade `TR-2027-03`.
5. **Confirmed IPL 2027 Trades ($N = 10$ movements across 4 trades):** All 4 trades announced between June 2026 and November 2026 were independently corroborated against official IPL announcements (`iplt20.com`, `delhicapitals.in`, `rajasthanroyals.com`) and established media releases (`The Hindu`, `Hindustan Times`, `LiveMint`).
6. **Classification Coverage Gap Identified:** While `data/external/ipl_2027_player_availability.json` accurately curates 96 players, **181 players who actively participated in IPL 2026** currently fall back to `"Unverified (check pre-Nov 17 lists)"` in the frontend dashboard due to a missing dynamic join against `playing_eleven.csv`.

---

### 1. Complete Player Universe Inventory

| Cohort Classification | Player Count | Dataset Basis & Evidence | Status in Dashboard & Reports |
| :--- | :---: | :--- | :--- |
| **Total Canonical Players** | **1,216** | `canonical_player_directory.csv` | Full universe indexed |
| **Active IPL 2026 Squad Players** | **265** | `playing_eleven.csv` (Comp 3298) | 84 curated/RR; 181 fallback to Unverified |
| — *Rajasthan Royals 2026 Squad* | *26* | `phase4_rr_2026_roster.csv` | Fully verified (Jadeja, Curran, Parag, Jaiswal, etc.) |
| — *Other 9 IPL Franchises (Curated)* | *58* | `ipl_2027_player_availability.json` | Fully verified with 2026/2027 fields |
| — *Other 9 IPL Franchises (Uncurated)* | *181* | `playing_eleven.csv` (10 franchises) | Displayed as "Unverified" (Coverage Gap) |
| **Historical IPL Only (2022–2025)** | **171** | `playing_eleven.csv` (Comps 1358, 1715, 2193, 2725) | No current 2026 team; Last Known IPL team unextracted |
| **Non-IPL Players** | **780** | Senior T20Is ($N=700$), Emerging Asia Cup only ($N=80$) | Correctly excluded from IPL franchise badges |

#### Breakdown of 265 Verified IPL 2026 Players by Franchise
- **Chennai Super Kings (29 players):** Including Sanju Samson, Ruturaj Gaikwad, Shivam Dube, Matheesha Pathirana, MS Dhoni, Noor Ahmad, Khaleel Ahmed, Dewald Brevis, etc.
- **Delhi Capitals (25 players):** Including David Miller, Ashutosh Sharma, Axar Patel, Mitchell Starc, KL Rahul, Kuldeep Yadav, Karun Nair, Mukesh Kumar, etc.
- **Gujarat Titans (26 players):** Including Rashid Khan, Jos Buttler, Shubman Gill, Mohammed Siraj, Sai Sudharsan, Kagiso Rabada, Washington Sundar, Ishant Sharma, etc.
- **Kolkata Knight Riders (26 players):** Including Sunil Narine, Varun Chakravarthy, Rinku Singh, Andre Russell (pre-retirement), Ajinkya Rahane, Quinton de Kock, Venkatesh Iyer, etc.
- **Lucknow Super Giants (26 players):** Including Nicholas Pooran, Rishabh Pant, Mayank Yadav, Mitchell Marsh, Ayush Badoni, Avesh Khan, Wanindu Hasaranga (squad), etc.
- **Mumbai Indians (29 players):** Including Jasprit Bumrah, Suryakumar Yadav, Hardik Pandya, Rohit Sharma, Namboori Tilak Varma, Mitchell Santner, Trent Boult, Deepak Chahar, etc.
- **Punjab Kings (25 players):** Including Shreyas Iyer, Arshdeep Singh, Shashank Singh, Harpreet Brar, Glenn Maxwell, Yuzvendra Chahal, Marcus Stoinis, Prabhsimran Singh, etc.
- **Rajasthan Royals (26 players):** Including Ravindra Jadeja, Sam Curran, Riyan Parag (Captain), Yashasvi Jaiswal, Dhruv Jurel, Jofra Archer, Shimron Hetmyer, Ravi Bishnoi, etc.
- **Royal Challengers Bengaluru (25 players):** Including Virat Kohli, Tim David, Phil Salt, Josh Hazlewood, Bhuvneshwar Kumar, Krunal Pandya, Liam Livingstone, Rajat Patidar, etc.
- **Sunrisers Hyderabad (28 players):** Including Heinrich Klaasen, Patrick Cummins, Travis Head, Abhishek Sharma, Nitish Kumar Reddy, Sakib Hussain, Jaydev Unadkat, T. Natarajan, etc.

---

### 2. Independent Verification of IPL 2027 Player Trades

Each of the following 10 player movements across 4 official trade transactions was independently verified against primary source announcements as of 13 November 2026:

| Trade ID | Date Confirmed | Player Name | Canonical ID | Traded From | Traded To | Transfer Fee | Primary Source Announcement | Confidence |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **TR-2027-01** | 2026-06-23 | **Rishabh Pant** | `4703` | LSG | **DC** | ₹15.00 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4245/) / LiveMint | **Verified (Official)** |
| **TR-2027-01** | 2026-06-23 | **Kuldeep Yadav** | `3644` | DC | **LSG** | ₹13.50 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4245/) / LiveMint | **Verified (Official)** |
| **TR-2027-02** | 2026-10-14 | **T. Natarajan** | `4265` | DC | **SRH** | ₹5.00 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4270/) / Delhi Capitals | **Verified (Official)** |
| **TR-2027-02** | 2026-10-14 | **Zeeshan Ansari** | `4824` | SRH | **DC** | ₹0.40 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4270/) / Delhi Capitals | **Verified (Official)** |
| **TR-2027-03** | 2026-11-05 | **Liam Livingstone** | `4391` | SRH | **CSK** | ₹6.00 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4285/) / Hindustan Times | **Verified (Official)** |
| **TR-2027-03** | 2026-11-05 | **Jack Edwards** | `6166` | SRH | **CSK** | ₹3.00 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4285/) / Hindustan Times | **Verified (Official)** |
| **TR-2027-03** | 2026-11-05 | **Khaleel Ahmed** | `4816` | CSK | **SRH** | ₹4.80 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4285/) / ETV Bharat | **Verified (Official)** |
| **TR-2027-03** | 2026-11-05 | **Nathan Ellis** | `7920` | CSK | **SRH** | ₹2.00 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4285/) / Hindustan Times | **Verified (Official)** |
| **TR-2027-04** | 2026-11-11 | **Hardik Pandya** | `3330` | MI | **PBKS** | ₹18.00 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4291/) / The Hindu | **Verified (Official)** |
| **TR-2027-04** | 2026-11-11 | **Arshdeep Singh** | `6464` | PBKS | **MI** | ₹18.00 Cr | [IPLT20.com Announcement](https://www.iplt20.com/news/4291/) / The Hindu | **Verified (Official)** |

#### Historical Pre-2026 Blockbuster Trade (Contextual Baseline)
- **Sanju Samson (RR $ightarrow$ CSK):** ₹18.00 Cr fee. Formally transferred ahead of IPL 2026 to CSK (`rajasthanroyals.com`, `crex.com`).
- **Ravindra Jadeja (CSK $ightarrow$ RR):** ₹14.00 Cr fee. Homecoming transfer ahead of IPL 2026 to RR.
- **Sam Curran (CSK $ightarrow$ RR):** ₹2.40 Cr fee. Transferred alongside Jadeja to RR.
- *Empirical Evidence in Dataset:* Confirmed by `playing_eleven.csv` Comp 3298: Samson played 14 matches for CSK; Jadeja and Curran played 14 and 16 matches respectively for RR. Riyan Parag (14 matches) and Yashasvi Jaiswal (2 matches) served as RR captains in 2026.

#### Reported / Stalled Rumour (NOT Completed)
- **Rumour `RUM-2027-01` (Yashasvi Jaiswal $\leftrightarrow$ Mohammed Shami / Mohsin Khan):** Media speculation emerged on 11 November 2026 regarding LSG inquiring about Yashasvi Jaiswal. Verified as **stalled and collapsed** (`crex.com`); LSG concluded trades and Jaiswal remains under contract with Rajasthan Royals.

---

### 3. Detailed Audit of Sanju Samson

- **Canonical Player ID:** `2258`
- **Canonical Name:** Sanju Samson
- **IPL Match History in Dataset:**
  - 2022 (Rajasthan Royals): 17 matches (17 played, 17 captain, 17 keeper)
  - 2023 (Rajasthan Royals): 14 matches (14 played, 14 captain, 14 keeper)
  - 2024 (Rajasthan Royals): 16 matches (16 played, 16 captain, 16 keeper)
  - 2025 (Rajasthan Royals): 14 matches (9 played, 6 captain, 6 keeper)
  - **2026 (Chennai Super Kings): 14 matches (14 played, 0 captain, 14 keeper)**
- **IPL 2027 Status:** Contracted to Chennai Super Kings (pending BCCI Nov 17 retention declarations).
- **Audit Findings across Project Files:**
  - `data/external/ipl_2027_player_availability.json`: **Correct** (`ipl_2026_team: "Chennai Super Kings"`, `current_2027_team: "Chennai Super Kings"`, `status: "Contracted to another franchise"`).
  - `data/processed/phase4_rr_2026_roster.csv`: **Correct** (Samson is excluded; Jadeja, Curran, Parag, Jaiswal are listed).
  - `src/cricket_intelligence.py`: **Correct** (Q4 scouting decision states: *"Verified 2026 contract: Chennai Super Kings... Primary marquee acquisition target if available in auction or trade."*).
  - `reports/final_analytical_report.md` Section 3.3.4: **Correct** (*"Verified IPL 2026: Chennai Super Kings; IPL 2027 Retention: Pending official confirmation"*).
  - `reports/final_analytical_report.md` Section 5.3 Case 3: **Incorrect Section Header**. The subsection was titled `Strategic Case 3 — Dhruv Jurel & Sanju Samson (Internal Core Maximization)`. Because Samson plays for CSK, he is an *external* trade/auction target, not an internal core player (only Dhruv Jurel is internal). Recommended header update: `Strategic Case 3 — Dhruv Jurel (Internal Core) & Sanju Samson (External Marquee Pursuit)`.

---

### 4. Detailed Audit of Nathan Ellis

- **Canonical Player ID:** `7920`
- **Canonical Name:** Nathan Ellis
- **IPL Match History in Dataset:**
  - 2022 (Punjab Kings): 2 matches (2 played)
  - 2023 (Punjab Kings): 10 matches (10 played)
  - 2024 (Punjab Kings): 14 matches (1 played, 13 bench)
  - 2025 (Chennai Super Kings): 14 matches (1 played, 13 bench)
  - **2026: 0 squad appearances in dataset.**
- **Verification of 2026 Contract vs. Match Appearances:**
  - Authoritative sources (`cricket.com.au`, `iplt20.com`, `ratopati.com`) confirm that Nathan Ellis was retained by CSK for IPL 2026, but suffered a severe hamstring injury during Tasmania's Marsh One-Day Cup final in March 2026 and was officially ruled out of the season.
  - CSK signed Australian fast bowler Spencer Johnson as an injury replacement for ₹1.5 crore.
  - Therefore, the dataset's zero match appearances reflect an injury replacement rather than an uncontracted free-agent status.
- **IPL 2027 Confirmed Status:** Traded from **Chennai Super Kings** to **Sunrisers Hyderabad** on **5 November 2026** (Trade `TR-2027-03`, ₹2.00 Cr).
- **Statistical Test Audit (Post-Boundary Composure):**
  - Sample: $N_1 = 126$ legal post-boundary balls, $k_1 = 21$ repeat boundaries (16.67%).
  - League baseline: $N_2 = 47,896$ deliveries, $k_2 = 10,911$ repeat boundaries (22.78%).
  - Two-proportion z-test (unclustered):
    $$z = -1.634, \quad p = 0.051 	ext{ (one-sided)}, \quad p = 0.102 	ext{ (two-sided)}$$
  - **Audit Correction:** The test is an **ordinary unclustered two-proportion test** that assumes independent trials. It must **not** be described as "cluster-robust", as intra-match correlation was not modeled in this specific formula. If intra-match clustering were accounted for ($DEFF > 1.0$), the standard error would widen and the significance would be even lower.
  - Wilson 95% Confidence Interval: $[11.2\%, 24.1\%]$ (reproduced and verified).

---

### 5. Categorized Audit of Discrepancies and Gaps

#### Category A: Outdated / Misleading Classifications
1. **Mitchell Santner (ID 4101):**
   - *Current File Label:* `src/scouting_data.py` Line 839 lists Santner as `"CSK / MI"`.
   - *Verified Dataset Fact:* Santner represented **Mumbai Indians** in IPL 2025 and IPL 2026 (7 squad appearances, 4 matches played in 2026; 16 squad, 13 played in 2025). His active franchise is Mumbai Indians. Listing "CSK / MI" conflates his historical 2019–2024 CSK stint.
2. **Tim David (ID 6396):**
   - *Current File Label:* `src/scouting_data.py` Line 846 lists Tim David as `"RCB / MI"`.
   - *Verified Dataset Fact:* Tim David represented **Royal Challengers Bengaluru** in IPL 2025 and IPL 2026 (16 matches played in 2026; 12 played in 2025). His active franchise is Royal Challengers Bengaluru.
3. **Sakib Hussain (ID 35318):**
   - *Current File Label:* `src/scouting_data.py` Line 832 lists Sakib Hussain as `"KKR"`.
   - *Verified Dataset Fact:* Sakib was on KKR's squad in 2024 (0 played), but in IPL 2026 represented **Sunrisers Hyderabad** (15 squad appearances, 11 matches played).
4. **Nathan Ellis (ID 7920):**
   - *Current File Label:* `src/scouting_data.py` Line 829 lists Ellis as `"CSK (Traded to SRH 2027)"` under `pathway: "Trade Target"`.
   - *Verified Dataset Fact:* Having already been traded to SRH on Nov 5, 2026, he cannot be a pending trade target for Rajasthan Royals.

#### Category B: Missing Classifications (Frontend Coverage Gap)
- `data/external/ipl_2027_player_availability.json` covers 96 curated players.
- `data/processed/phase4_rr_2026_roster.csv` covers 26 Rajasthan Royals players.
- **181 active IPL 2026 players** who played or held squad sheets in 2026 (e.g. David Miller [DC], Ashutosh Sharma [DC], Shashank Singh [PBKS], Heinrich Klaasen [SRH], Nicholas Pooran [LSG], Phil Salt [RCB], Shreyas Iyer [PBKS], Rohit Sharma [MI], KL Rahul [DC], Mitchell Starc [DC]) have no record in the static availability JSON.
- Consequently, in `src/scouting_data.py` (lines 377–386), these players fall back to:
  ```json
  {
    "ipl_2026_team": "Unverified (check pre-Nov 17 lists)",
    "current_2027_team": "Unverified",
    "availability_status": "Unknown/unverified"
  }
  ```
  even though their 2026 IPL franchise is definitively recorded in `playing_eleven.csv`.

#### Category C: Status Unknown for IPL 2027
- **BCCI Regulatory Calendar:**
  - Retention/Release Deadline: **17 November 2026** (extended from Nov 15).
  - Player Auction Registration Deadline: **23 November 2026**.
  - Mini-Auction Date: **16 December 2026** (Goa).
- **Rule of Prudence:** For all players not traded in Trades `TR-2027-01` through `TR-2027-04` and not retired, their formal 2027 auction availability or retention status cannot be inferred prior to 17 November 2026. They must remain classified as **Contracted (Pending Nov 17 Retention Announcement)**.

---

### 6. Exact Files and Dashboard/API Locations Containing Discrepancies

| File Path | Line Range | Identified Issue | Recommended Correction |
| :--- | :---: | :--- | :--- |
| `src/scouting_data.py` | L839 | Mitchell Santner labeled `"CSK / MI"` | Update to `"Mumbai Indians"` |
| `src/scouting_data.py` | L846 | Tim David labeled `"RCB / MI"` | Update to `"Royal Challengers Bengaluru"` |
| `src/scouting_data.py` | L832 | Sakib Hussain labeled `"KKR"` | Update to `"Sunrisers Hyderabad"` |
| `src/scouting_data.py` | L829 | Nathan Ellis pathway labeled `"Trade Target"` | Update to `"Traded to SRH (Nov 5, 2026)"` |
| `reports/final_analytical_report.md` | L1215 | Headline: `Case 3 — Dhruv Jurel & Sanju Samson (Internal Core Maximization)` | Update to: `Case 3 — Dhruv Jurel (Internal Core) & Sanju Samson (External Marquee Pursuit)` |
| `src/scouting_data.py` | L377–386 | 181 active 2026 players default to `"Unverified"` | Dynamically resolve `ipl_2026_team` from `p11_2026` before falling back |

---

### 7. Non-Negotiable Submission Governance

In accordance with user instructions:
- **Zero official Top 10 ranking CSV files have been modified.**
- **Zero changes have been applied to ML methodologies, priors, or rankings.**
- **No changes have been committed or pushed.**
- All findings are submitted for user review and approval prior to implementation.

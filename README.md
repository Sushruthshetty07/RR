# Rajasthan Royals SupeRR Selector 2.0 ML Challenge

## Objective
Identify and rank the **Top 10 Most Reliable Batters** and **Top 10 Most Reliable Bowlers** under high-pressure match situations to inform Rajasthan Royals' auction strategy for **IPL 2027**.

## Repository Structure
```
rr_superr_selector/
├── data/
│   ├── raw/                 # Unmodified competition CSV datasets
│   └── processed/           # Reconstructed match states and enriched tables
├── notebooks/
│   └── 01_initial_eda.ipynb # Initial Exploratory Data Analysis notebook
├── reports/
│   └── figures/             # High-resolution statistical plots
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Safe raw data ingestion & validation
│   └── match_state.py       # Ball-by-ball situational context reconstruction
├── scripts/
│   └── run_eda.py           # Automated diagnostic and EDA runner
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup & Reproduction
1. Activate virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run EDA & Situational Reconstruction:
   ```bash
   python scripts/run_eda.py
   ```

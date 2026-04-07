import os
import pandas as pd
import numpy as np
import datetime
import subprocess
import sys
import argparse

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.pipeline import SportsDataPipeline, FeatureEngineer
from src.models import ModelSimulator
import scripts.generate_assets as generate_assets

def update_system_reports(v1, v2, v3):
    # Ensure we are in the project root
    if os.path.basename(os.getcwd()) == 'scripts':
        os.chdir('..')
        
    print("📝 Updating System Reports (README.md & LATEST_ACTION.md)...")
    
    # --- HELPER FUNCTIONS ---
    def get_stats(d):
        if d.empty: return 0.0, 0.0, 0.0
        # Filter for settled bets (outcome 0 or 1)
        settled = d[d['outcome'].isin([0.0, 1.0])]
        if settled.empty: return 0.0, 0.0, 0.0
        
        p = settled['profit_actual'].sum()
        r = settled['wager_unit'].sum()
        roi = p/r if r>0 else 0.0
        wr = len(settled[settled['outcome']==1]) / len(settled)
        return p, roi, wr

    def get_volume_text(d):
        if d.empty: return "None (0 bets/day)"
        days = (d['pick_date'].max() - d['pick_date'].min()).days + 1
        avg = len(d) / max(days, 1)
        
        if avg > 50: cat = "Very High"
        elif avg > 20: cat = "High"
        elif avg > 10: cat = "Medium"
        elif avg > 5: cat = "Low"
        else: cat = "Very Low"
        
        return f"{cat} (~{int(avg)} bets/day)"

    # --- CALCULATE STATS ---
    p1, r1, w1 = get_stats(v1)
    p2, r2, w2 = get_stats(v2)
    p3, r3, w3 = get_stats(v3)
    
    vol_v1 = get_volume_text(v1)
    vol_v2 = get_volume_text(v2)
    vol_v3 = get_volume_text(v3)

    # --- 1. LATEST_ACTION.md ---
    dates = []
    if not v1.empty: dates.append(v1['pick_date'].max())
    if not v2.empty: dates.append(v2['pick_date'].max())
    if not v3.empty: dates.append(v3['pick_date'].max())
    
    last_date = max(dates) if dates else datetime.datetime.now()
    
    log_content = f"# 📝 Daily Action Log ({last_date.date()})\n\n"
    
    def make_table(df, title):
        if df.empty: return ""
        t = f"### {title}\n"
        t += "| LEAGUE | PICK | ODDS | UNIT | RES | PROFIT |\n"
        t += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        day_df = df[df['pick_date'] == last_date]
        if day_df.empty: return f"### {title}\n*No action for this date.*\n\n"
        
        for _, row in day_df.iterrows():
            res = "✅" if row['outcome'] == 1.0 else "❌" if row['outcome'] == 0.0 else "⏳"
            odds = f"+{int(row['odds_american'])}" if row['odds_american'] > 0 else f"{int(row['odds_american'])}"
            t += f"| {row['league_name']} | {row['pick_value']} | {odds} | {row['wager_unit']:.1f} | {res} | {row['profit_actual']:+.2f}u |\n"
        
        # Daily Sum
        daily_profit = day_df[day_df['outcome'].isin([0.0, 1.0])]['profit_actual'].sum()
        t += f"**Daily PnL (Settled): {daily_profit:+.2f} Units**\n\n"
        return t + "\n"

    log_content += make_table(v3, "V3 Obsidian Action")
    log_content += make_table(v2, "V2 Diamond Action")
    log_content += make_table(v1, "V1 Pyrite Action")
    
    with open("LATEST_ACTION.md", "w", encoding="utf-8") as f:
        f.write(log_content)

    # --- 2. README.md ---
    readme_text = f"""
<div align="center">
  <br />
  <h1>THE QUARRY <span style="color: #444; font-weight: normal;">// XGB-SNIPER</span></h1>
  <p style="font-family: monospace; letter-spacing: 2px; color: #888;">ADVANCED ALGORITHMIC ARBITRAGE SYSTEM</p>
  <br />

  [![Status](https://img.shields.io/badge/STATUS-OPERATIONAL-success?style=for-the-badge&logo=statuspage&logoColor=white)](https://ducky705.github.io/XGBoost-Sniper/selector.html)
  [![V2 ROI](https://img.shields.io/badge/V2_ROI-{p2:+.1f}u-00E0FF?style=for-the-badge)](https://ducky705.github.io/XGBoost-Sniper/diamond.html)
  [![V3 ROI](https://img.shields.io/badge/V3_ROI-{p3:+.1f}u-7c3aed?style=for-the-badge)](https://ducky705.github.io/XGBoost-Sniper/obsidian.html)

  <br />
  <br />
  <a href="https://ducky705.github.io/XGBoost-Sniper/selector.html"><strong>ENTER CONTROL CENTER</strong></a>
  <br />
  <br />
</div>

---

## ⚡ EXECUTIVE INTELLIGENCE

A multi-generational algorithmic trading system leveraging **Gradient Boosting Decision Trees (XGBoost)** and **Deep Neural Networks** to identify inefficiencies in sports betting markets.

| MODEL ARCHITECTURE | RELEASED | STRATEGY PROFILE | STATUS | VOLUME | TOTAL BETS | ROI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[V1 PYRITE](https://ducky705.github.io/XGBoost-Sniper/pyrite.html)** | `NOV 20, 2025` | `XGB-CLASSIC` <br> High-Frequency / Volatility Capture | 🟡 **LEGACY** | {vol_v1} | **{len(v1)}** | **{r1:+.1%}** |
| **[V2 DIAMOND](https://ducky705.github.io/XGBoost-Sniper/diamond.html)** | `NOV 30, 2025` | `XGB-HYBRID` <br> Precision / Regime Filtering | 🟢 **ACTIVE** | {vol_v2} | **{len(v2)}** | **{r2:+.1%}** |
| **[V3 OBSIDIAN](https://ducky705.github.io/XGBoost-Sniper/obsidian.html)** | `DEC 27, 2025` | `XGB-V3` <br> Non-Linear arbitrage | 🟣 **ALPHA** | {vol_v3} | **{len(v3)}** | **{r3:+.1%}** |

> [!IMPORTANT]
> **ACCESS PROTOCOL**: The primary interface for all models is the [**Model Selector**](https://ducky705.github.io/XGBoost-Sniper/selector.html).

---

## 🛰 SYSTEMS OVERVIEW

### V1 PYRITE // THE BRUTE FORCE
*The initial prototype.* Operated on raw probability differentials. While high-volume, it suffered from "false confidence" on heavy favorites.
*   **Verdict**: Profitable but volatile. Retired from primary rotation.

### V2 DIAMOND // THE SNIPER
*The current standard.* Introduces **Regime Filtering**—banning "toxic" low-predictability markets (NFL/MLB) and focusing on high-confidence setups (NBA/NCAAB).
*   **Mechanism**: Uses a Fade Score to identify public overexposure.
*   **Performance**: Consistent alpha generation with lower drawdown.

### V3 OBSIDIAN // THE ORACLE
*The next frontier.* An advanced ensemble hybrid designed to capture complex, non-linear dependencies that standard tree-based models miss.
*   **Status**: Currently ingesting data in shadow mode.

---

## 🛠 ARCHITECTURE

```mermaid
graph TD
    A[DATA LAKE] -->|Ingest| B(CORE ENGINE)
    B -->|Feature Engineering| C{{MODEL SELECTOR}}
    C -->|Legacy Track| D[V1 PYRITE]
    C -->|Regime Filter| E[V2 DIAMOND]
    C -->|Ensemble| F[V3 OBSIDIAN]
    D & E & F -->|Simulate| G[DECISION SUPPORT]
    G -->|Render| H[DASHBOARD SUITE]
```

### COMPONENTS
*   `monitor.py`: Central command. Fetches data, executes inference pipelines, and commits artifacts.
*   `models/`: Serialized XGBoost binaries and neural weights.
*   `docs/`: Static visualization layer hosted on GitHub Pages.

---

<div align="center">
    <p><em>© 2025 XGBOOST-SNIPER TECHNOLOGIES // PROPRIETARY RESEARCH</em></p>
</div>
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_text)
        
    print("✅ System reports updated.")

def run_monitor():
    parser = argparse.ArgumentParser(description='THE QUARRY // XGB-SNIPER Monitor')
    parser.add_argument('--full', action='store_true', help='Run full historical simulation (slow)')
    parser.add_argument('--days', type=int, default=30, help='Days of history to fetch in quick mode (default 30)')
    args = parser.parse_args()

    # Ensure we are in the project root
    if os.path.basename(os.getcwd()) == 'scripts':
        os.chdir('..')

    since = None if args.full else args.days
    mode_text = "FULL HISTORICAL" if args.full else f"QUICK INCREMENTAL ({args.days} days)"
    
    print(f"🛰️ THE QUARRY MONITOR STARTING [{mode_text}]...")
    
    # 1. Pipeline
    pipeline = SportsDataPipeline()
    raw = pipeline.fetch_data(since_days=since)
    
    if not raw.empty:
        eng = FeatureEngineer(raw)
        proc = eng.process()
        
        # === DIAGNOSTIC: Date range after feature engineering ===
        print(f"📐 AFTER FEATURES: {len(df)} picks")
        print(f"   Date range: {df['pick_date'].min().date()} to {df['pick_date'].max().date()}")
        
        # 2. Simulations
        sim = ModelSimulator(proc)
        v1 = sim.run_v1_pyrite()
        v2 = sim.run_v2_diamond()
        v3 = sim.run_v3_obsidian()
        
        # 3. Reports & Assets
        update_system_reports(v1, v2, v3)
        generate_assets.generate_live_assets(since_days=since)
        
        print("✨ Monitor Cycle Complete.")
    else:
        print("❌ No data fetched. Check Supabase connection or filters.")

if __name__ == "__main__":
    run_monitor()
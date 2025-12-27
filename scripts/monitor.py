import os
import pandas as pd
import numpy as np
import datetime
import subprocess
import sys

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.pipeline import SportsDataPipeline, FeatureEngineer
from src.models import ModelSimulator

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
        if avg > 15: cat = "High"
        elif avg > 5: cat = "Medium"
        else: cat = "Low"
        return f"{cat} (~{int(avg)} bets/day)"

    # --- CALCULATE STATS ---
    p1, r1, w1 = get_stats(v1)
    p2, r2, w2 = get_stats(v2)
    # p3, r3, w3 = get_stats(v3) # V3 might be empty or training
    
    vol_v1 = get_volume_text(v1)
    vol_v2 = get_volume_text(v2)

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
    readme_text = f"""# XGBoost-Sniper: Quantitative Sports Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Strict Mode](https://img.shields.io/badge/strict-myan-blueviolet)](https://mypy.readthedocs.io/en/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"The market is not efficient. It is just noisy."**

XGBoost-Sniper is an advanced, algorithmic trading system engineered to identify and exploit market inefficiencies in sports betting odds. By leveraging gradient boosting frameworks (XGBoost) and regime-based filtering, the system systematically generates alpha in high-volatility markets.

---

## 📊 Executive Performance

| Model | Strategy | Status | Daily Volume | ROI |
| :--- | :--- | :--- | :--- | :--- |
| **V1 Pyrite** | High-Variance / Volume | 🟡 Legacy | {{vol_v1}} | **{{r1:+.1%}}** |
| **V2 Diamond** | Precision / Sniper | 🟢 **Active** | {{vol_v2}} | **{{r2:+.1%}}** |
| **[V3 Obsidian](https://ducky705.github.io/XGBoost-Sniper/selector.html)** | Deep Learning / Hybrid | 🟣 **Alpha** | (Training) | **N/A** |

> [!IMPORTANT]
> **Live Dashboard**: View the real-time performance and active signals on the [Diamond Dashboard](https://ducky705.github.io/XGBoost-Sniper/diamond.html).
>
> **Model Selector**: Access all model dashboards via the [Selector Page](https://ducky705.github.io/XGBoost-Sniper/selector.html).

---

## 🚀 The Evolution of Alpha

This repository documents the transition from a raw statistical probability model to a sophisticated asset manager.

### Phase 1: Pyrite (The "Accuracy Fallacy")
Our initial prototype, **Pyrite**, operated on a simple premise: *bet on everything with >50% probability*.
*   **Result**: While it achieved a {{w1:.1%}} win rate, it lost money to the vigorish (fees) due to poor calibration on favorites.
*   **Lesson**: Accuracy ≠ Profitability.

### Phase 2: Diamond (The "Sniper" Approach)
**Diamond** introduced specific "Regime Filtering" and Kelly Criterion staking.
*   **Innovation**: It bans "toxic assets" (sports with low predictability like NFL/MLB) and only trades in high-confidence regimes (NBA/NCAAB).
*   **Mechanism**: Uses a Fade Score to identify when the public usage is dangerously high, effectively "sniping" lines before they move.

### Phase 3: Obsidian (The "Neural" Frontier)
**Obsidian** represents the next generation of predictive modeling, incorporating deep learning and hybrid architectures to capture non-linear relationships that decision trees might miss.
*   **Status**: Currently in Alpha testing.
*   **Access**: [Click here to view the Obsidian Dashboard](https://ducky705.github.io/XGBoost-Sniper/selector.html) (via Selector).

---

## 🛠 System Architecture

```mermaid
graph TD
    A[Supabase Data Lake] -->|Fetch Odds/Results| B(monitor.py)
    B -->|Feature Engineering| C{{Model Selector}}
    C -->|Legacy Logic| D[V1 Pyrite]
    C -->|Regime Logic| E[V2 Diamond]
    D -->|Simulate| F[Daily Report]
    E -->|Kelly Staking| F
    F -->|Generate Assets| G[Dashboard / README]
```

### Core Components
*   **`monitor.py`**: The central orchestration engine. Fetches data, runs inference, and commits results.
*   **`models/`**: Serialized XGBoost binaries (tracked via LFS or ignored for security).
*   **`docs/`**: Production-grade dashboards for visualizing model output.

---

## 🔒 Security & Privacy

This repository enforces strict security protocols:
*   **Credential Isolation**: All API keys are managed via `.env` and strictly excluded from version control.
*   **Model IP Protection**: Trained model artifacts (`.pkl`, `.model`) are strictly git-ignored.

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/Ducky705/XGBoost-Sniper.git
cd XGBoost-Sniper
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file with your credentials.

### 3. Run Inference
```bash
python monitor.py
```

---

*© 2025 XGBoost-Sniper Technologies. All rights reserved.*
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_text)
        
    print("✅ System reports updated.")

def main():
    # Ensure we are in the project root
    if os.path.basename(os.getcwd()) == 'scripts':
        os.chdir('..')

    print("🛰️ THE QUARRY MONITOR STARTING...")
    
    # 1. Pipeline
    pipeline = SportsDataPipeline()
    raw = pipeline.fetch_data()
    if raw.empty: return
    
    eng = FeatureEngineer(raw)
    df = eng.process()
    
    # 2. Simulations
    sim = ModelSimulator(df)
    v1 = sim.run_v1_pyrite()
    v2 = sim.run_v2_diamond()
    v3 = sim.run_v3_obsidian()
    
    # 3. Reports
    update_system_reports(v1, v2, v3)
    
    # 4. Refresh Assets
    print("🎨 Refreshing Web Assets...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["python", os.path.join(script_dir, "generate_assets.py")])
    
    print("✨ Monitor Cycle Complete.")

if __name__ == "__main__":
    main()
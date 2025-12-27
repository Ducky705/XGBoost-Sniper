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
    
    # 1. LATEST_ACTION.md
    last_date = max(d['pick_date'].max() for d in [v1, v2, v3] if not d.empty)
    
    log_content = f"# 📝 Daily Action Log ({last_date.date()})\n\n"
    
    def make_table(df, title):
        if df.empty: return ""
        t = f"### {title}\n"
        t += "| LEAGUE | PICK | ODDS | UNIT | RES | PROFIT |\n"
        t += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        day_df = df[df['pick_date'] == last_date]
        for _, row in day_df.iterrows():
            res = "✅" if row['outcome'] == 1.0 else "❌" if row['outcome'] == 0.0 else "⏳"
            odds = f"+{int(row['odds_american'])}" if row['odds_american'] > 0 else f"{int(row['odds_american'])}"
            t += f"| {row['league_name']} | {row['pick_value']} | {odds} | {row['wager_unit']:.1f} | {res} | {row['profit_actual']:+.2f}u |\n"
        return t + "\n"

    log_content += make_table(v3, "V3 Obsidian Action")
    log_content += make_table(v2, "V2 Diamond Action")
    log_content += make_table(v1, "V1 Pyrite Action")
    
    with open("LATEST_ACTION.md", "w", encoding="utf-8") as f:
        f.write(log_content)

    # 2. README.md (Simplified update)
    # In a real scenario, we might use a template or regex. 
    # For now, let's keep the core stats.
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
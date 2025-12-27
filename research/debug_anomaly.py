import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.pipeline import SportsDataPipeline, FeatureEngineer
from src.models import ModelSimulator

def main():
    print("🕵️ Debugging Obsidian V3 Anomaly...")
    
    pipeline = SportsDataPipeline()
    raw = pipeline.fetch_data()
    eng = FeatureEngineer(raw)
    df = eng.process()
    sim = ModelSimulator(df)
    
    v3 = sim.run_v3_obsidian()
    
    if v3.empty:
        print("V3 is empty.")
        return

    print(f"Total V3 Bets: {len(v3)}")
    print(f"Total Profit: {v3['profit_actual'].sum():.2f}")
    
    print("\n🚨 SUPER OUTLIERS (Profit > 10u or Loss < -10u):")
    outliers = v3[(v3['profit_actual'] > 10) | (v3['profit_actual'] < -10)]
    if not outliers.empty:
        print(outliers[['pick_date', 'league_name', 'pick_value', 'odds_american', 'wager_unit', 'outcome', 'profit_actual']])
    else:
        print("No obvious single bet limit outliers (checked >10u).")
        
    print("\n🔝 Top 10 Most Profitable Bets:")
    top = v3.sort_values('profit_actual', ascending=False).head(10)
    print(top[['pick_date', 'league_name', 'pick_value', 'odds_american', 'wager_unit', 'outcome', 'profit_actual', 'decimal_odds']])
    
    print("\n📉 Top 10 Worst Losses:")
    worst = v3.sort_values('profit_actual', ascending=True).head(10)
    print(worst[['pick_date', 'league_name', 'pick_value', 'odds_american', 'wager_unit', 'outcome', 'profit_actual']])

if __name__ == "__main__":
    main()

import os
import sys
import pandas as pd
import numpy as np
from tabulate import tabulate

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.pipeline import SportsDataPipeline, FeatureEngineer
from src.models import ModelSimulator

def main():
    print("📊 Running Head-to-Head Model Comparison...")
    
    # 1. Fetch Data
    pipeline = SportsDataPipeline()
    raw = pipeline.fetch_data()
    if raw.empty:
        print("No data.")
        return
        
    eng = FeatureEngineer(raw)
    df = eng.process()
    
    # 2. Run Simulations
    sim = ModelSimulator(df)
    
    print("\n🟢 Simulating V1 Pyrite...")
    v1 = sim.run_v1_pyrite()
    
    print("🔵 Simulating V2 Diamond...")
    v2 = sim.run_v2_diamond()
    
    print("🟣 Simulating V3 Obsidian (New)...")
    v3 = sim.run_v3_obsidian()
    
    # 3. Calculate Stats
    results = []
    
    for name, data in [('V1 Pyrite', v1), ('V2 Diamond', v2), ('V3 Obsidian', v3)]:
        if data.empty:
            results.append([name, 0, 0, "0%", 0, "0%"])
            continue
            
        # Filter for settled bets for stats
        settled = data[data['outcome'].notna()].copy()
        
        total_bets = len(settled)
        wins = settled['outcome'].sum()
        wr = (wins / total_bets * 100) if total_bets > 0 else 0
        
        # PnL & ROI
        profit = settled['profit_actual'].sum()
        risk = settled['wager_unit'].sum()
        roi = (profit / risk * 100) if risk > 0 else 0
        
        # Daily Volume
        days = settled['pick_date'].nunique()
        vol = total_bets / days if days > 0 else 0
        
        results.append([
            name, 
            total_bets, 
            f"{vol:.1f}", 
            f"{wr:.1f}%", 
            f"{profit:+.2f}u", 
            f"{roi:+.1f}%"
        ])
        
    print("\n🏆 Head-to-Head Results (All-Time Available Data):")
    headers = ["Model", "Total Bets", "Bets/Day", "Win Rate", "Total Profit", "ROI"]
    print(tabulate(results, headers=headers, tablefmt="github"))
    
    # Recent Performance (Last 30 Days)
    print("\n📅 Recent Performance (Last 30 Days):")
    recent_results = []
    cutoff = df['pick_date'].max() - pd.Timedelta(days=30)
    
    for name, data in [('V1 Pyrite', v1), ('V2 Diamond', v2), ('V3 Obsidian', v3)]:
        if data.empty: continue
        recent = data[data['pick_date'] >= cutoff].copy()
        settled = recent[recent['outcome'].notna()]
        
        total_bets = len(settled)
        wins = settled['outcome'].sum()
        wr = (wins / total_bets * 100) if total_bets > 0 else 0
        profit = settled['profit_actual'].sum()
        risk = settled['wager_unit'].sum()
        roi = (profit / risk * 100) if risk > 0 else 0
        
        recent_results.append([
            name, total_bets, f"{wr:.1f}%", f"{profit:+.2f}u", f"{roi:+.1f}%"
        ])
        
    print(tabulate(recent_results, headers=["Model", "Bets", "Win Rate", "Profit", "ROI"], tablefmt="github"))

if __name__ == "__main__":
    main()

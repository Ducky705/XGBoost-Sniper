import os
import json
import pandas as pd
from datetime import datetime
import sys

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from pipeline import SportsDataPipeline, FeatureEngineer
from models import ModelSimulator

def run_daily_update():
    print(f"🕒 Starting Daily Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Fetch & Hydrate Data
    pipeline = SportsDataPipeline()
    raw_df = pipeline.fetch_data_cached() # Incremental update
    
    fe = FeatureEngineer(raw_df)
    df = fe.process()
    
    ms = ModelSimulator(df)
    
    # 2. Run Simulations
    models = {
        "pyrite": ms.run_v1_pyrite(),
        "diamond": ms.run_v2_diamond(),
        "obsidian": ms.run_v3_obsidian(),
        "quartz": ms.run_v4_quartz()
    }
    
    # 3. Generate Stats
    stats = {
        "meta": {
            "last_update": datetime.now(pd.Timestamp.now(tz='UTC').tz).strftime('%Y-%m-%d %H:%M UTC'),
            "status": "NOMINAL"
        },
        "models": {}
    }
    
    for name, res in models.items():
        if res.empty:
            stats["models"][name] = {"roi": 0, "net": 0, "record": "0-0-0", "win_rate": 0}
            continue
            
        roi = (res['profit_actual'].sum() / res['wager_unit'].sum() * 100) if res['wager_unit'].sum() > 0 else 0
        net = res['profit_actual'].sum()
        wins = len(res[res['outcome'] == 1])
        losses = len(res[res['outcome'] == 0])
        pushes = len(res[res['outcome'] == 0.5]) # Handle pushes if any
        
        stats["models"][name] = {
            "roi": round(roi, 1),
            "net": round(net, 1),
            "record": f"{wins}-{losses}-{pushes}",
            "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0
        }

    # 4. Save Stats
    stats_path = os.path.join(BASE_DIR, 'docs', 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"📊 Stats saved to {stats_path}")

    # 5. Generate Comparison Graphics
    # We reuse the existing logic in research/generate_comparison.py but as a module
    sys.path.append(os.path.join(BASE_DIR, 'research'))
    import generate_comparison
    generate_comparison.generate_comparison_chart()
    
    print("✅ Daily Update Complete.")

if __name__ == "__main__":
    run_daily_update()

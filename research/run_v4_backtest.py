import pandas as pd
import numpy as np
from pipeline import SportsDataPipeline, FeatureEngineer
from models import ModelSimulator
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def run_backtest():
    print("🚀 Starting Comparative Backtest: v3 Obsidian vs v4 Quartz...")
    
    # 1. Fetch Data
    pipeline = SportsDataPipeline()
    raw_df = pipeline.fetch_data_cached()
    engineer = FeatureEngineer(raw_df)
    df = engineer.process()
    
    # 2. Run Simulations
    sim = ModelSimulator(df)
    
    print("💎 Running v3 Obsidian Simulation...")
    v3_results = sim.run_v3_obsidian()
    
    print("🌌 Running v4 Quartz Simulation...")
    v4_results = sim.run_v4_quartz()
    
    # 3. Calculate Metrics
    def get_stats(results, name):
        if results.empty:
            return {'Model': name, 'Profit': 0, 'ROI': 0, 'Bets': 0, 'Win%': 0, 'Sharpe': 0}
            
        profit = results['profit_actual'].sum()
        bets = len(results)
        roi = (profit / results['wager_unit'].sum()) * 100 if bets > 0 else 0
        win_rate = (results['outcome'] == 1).mean() * 100
        daily_profit = results.groupby('pick_date')['profit_actual'].sum()
        sharpe = daily_profit.mean() / (daily_profit.std() + 1e-6) * np.sqrt(365)
        
        # New Metric: Avg Bets/Day
        avg_bets_day = results.groupby('pick_date').size().mean()
        
        return {
            'Model': name,
            'Units Profit': f"{profit:.2f}u",
            'ROI': f"{roi:.2f}%",
            'Total Bets': bets,
            'Avg Bets/Day': f"{avg_bets_day:.1f}",
            'Win Rate': f"{win_rate:.1f}%",
            'Sharpe Ratio': f"{sharpe:.2f}"
        }

    stats = [
        get_stats(v3_results, 'v3 Obsidian'),
        get_stats(v4_results, 'v4 Quartz')
    ]
    
    stats_df = pd.DataFrame(stats)
    print("\n📈 BACKTEST RESULTS:")
    print(stats_df.to_string(index=False))
    
    # 4. Save results to markdown for the walkthrough
    with open('research/backtest_results.md', 'w') as f:
        f.write("# Comparative Backtest Results\n\n")
        f.write(stats_df.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"*Backtest run on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

if __name__ == "__main__":
    run_backtest()

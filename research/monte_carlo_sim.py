import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pipeline import SportsDataPipeline, FeatureEngineer
from models import ModelSimulator

def run_monte_carlo(iterations=1000):
    print(f"🎲 Beginning v5 Monte Carlo Stress-Test ({iterations} iterations)...")
    
    # 1. Get Base v5 Results
    pipeline = SportsDataPipeline()
    raw_df = pipeline.fetch_data_cached()
    engineer = FeatureEngineer(raw_df)
    processed = engineer.process()
    simulator = ModelSimulator(processed)
    
    v5_results = simulator.run_v4_quantum_sniper()
    if v5_results.empty:
        print("❌ Error: No v5 results to simulate.")
        return

    # 2. Extract Outcomes (P&L per bet)
    pnl = v5_results['profit_actual'].values
    initial_bankroll = 100.0 # Starting units
    
    all_curves = []
    max_drawdowns = []
    
    for i in range(iterations):
        # Shuffle outcomes (Random Walk)
        sim_pnl = np.random.choice(pnl, size=len(pnl), replace=True)
        curve = initial_bankroll + np.cumsum(sim_pnl)
        all_curves.append(curve)
        
        # Calculate Drawdown
        rolling_max = np.maximum.accumulate(curve)
        drawdown = (rolling_max - curve)
        max_drawdowns.append(np.max(drawdown))
        
        if (i+1) % 250 == 0: print(f"  Processed {i+1} sims...")

    # 3. Statistical Analysis
    curves_df = pd.DataFrame(all_curves).T
    mean_curve = curves_df.mean(axis=1)
    p5 = curves_df.quantile(0.05, axis=1) # 5th Percentile (Pessimistic)
    p95 = curves_df.quantile(0.95, axis=1) # 95th Percentile (Optimistic)
    
    avg_mdd = np.mean(max_drawdowns)
    p95_mdd = np.percentile(max_drawdowns, 95) # "Worst Case" MDD
    
    print("\n📊 STRESS-TEST RESULTS:")
    print(f"Total Bets Simulated: {len(pnl)}")
    print(f"Mean Final Bankroll: {mean_curve.iloc[-1]:.2f}u")
    print(f"P5 (Worst Case) Final: {p5.iloc[-1]:.2f}u")
    print(f"Average Max Drawdown: {avg_mdd:.2f}u")
    print(f"95% Confidence Max Drawdown: {p95_mdd:.2f}u")
    
    # 4. Visualization
    plt.figure(figsize=(12, 7))
    plt.plot(mean_curve, label='Mean Expectancy', color='blue', lw=2)
    plt.fill_between(range(len(pnl)), p5, p95, color='blue', alpha=0.1, label='90% Confidence Interval')
    
    # Plot a few random paths
    for i in range(5):
        plt.plot(all_curves[i], alpha=0.3, lw=0.5)
        
    plt.title(f"v5 Quantum Strategic: Monte Carlo Equity Stress-Test ({iterations} paths)")
    plt.xlabel("Number of Bets")
    plt.ylabel("Bankroll (Units)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'research/v5_monte_carlo.png'
    plt.savefig(save_path)
    print(f"\n📈 Chart saved to: {save_path}")

if __name__ == "__main__":
    run_monte_carlo()

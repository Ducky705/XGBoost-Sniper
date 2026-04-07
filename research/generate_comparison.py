import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pipeline import SportsDataPipeline, FeatureEngineer
from models import ModelSimulator

def generate_comparison_chart():
    print("🚀 Generating Comparative Alpha Graphs...")
    
    # Initialize Pipeline & Hydrate Features
    pipeline = SportsDataPipeline()
    raw_df = pipeline.fetch_data_cached() # Uses cache
    
    fe = FeatureEngineer(raw_df)
    df = fe.process() # Fully hydrated with v4 (shifted) and v3 (leaked) features
    
    simulator = ModelSimulator(df)

    # Run Simulations
    print("🌌 Simulating Quartz (Backtest Alpha)...")
    quartz_res = simulator.run_backtest_all() # Use full history for graph
    
    print("🌋 Simulating Obsidian...")
    obsidian_res = simulator.run_v3_obsidian()
    
    # For Pyrite and Diamond, if simulator doesn't have specific methods, 
    # we'll simulate them via edge filters (as they are usually based on Min_Edge)
    print("💎 Simulating Diamond (Surgical)...")
    diamond_res = simulator.run_v2_diamond() # Assuming it exists or fallback
    
    print("☄️ Simulating Pyrite (Aggressive)...")
    pyrite_res = simulator.run_v1_pyrite() # Assuming it exists or fallback

    # Extract Cumulative Series
    def get_series(res_df, label):
        if res_df.empty: return pd.Series()
        series = res_df.groupby('pick_date')['profit_actual'].sum().cumsum()
        series.name = label
        return series

    q_series = get_series(quartz_res, 'v4 Quartz')
    o_series = get_series(obsidian_res, 'v3 Obsidian')
    d_series = get_series(diamond_res, 'v2 Diamond')
    p_series = get_series(pyrite_res, 'v1 Pyrite')

    # Merge and FIX: Explicitly name and SORT
    bench = pd.concat([q_series, o_series, d_series, p_series], axis=1)
    bench.columns = ['v4 Quartz', 'v3 Obsidian', 'v2 Diamond', 'v1 Pyrite']
    bench = bench.ffill().fillna(0).sort_index()

    print("\n📈 CURRENT PERFORMANCE BENCHMARKS:")
    for col in bench.columns:
        print(f"  {col}: {bench[col].iloc[-1]:.2f}u")
    
    colors = {
        'v1 Pyrite': '#ffdd00',      # Gold/Pyrite
        'v2 Diamond': '#00f0ff',     # Ice/Diamond
        'v3 Obsidian': '#7c3aed',    # Purple/Obsidian
        'v4 Quartz': '#f8fafc'       # White/Quartz
    }

    plt.style.use('dark_background')

    # --- GENERATE 4 FOCUSED VERSIONS ---
    focus_models = [
        ('pyrite', 'v1 Pyrite'),
        ('diamond', 'v2 Diamond'),
        ('obsidian', 'v3 Obsidian'),
        ('quartz', 'v4 Quartz')
    ]

    for page_id, focus_label in focus_models:
        # --- PANAVISION RATIO: Ultra-slim 16x4 to eliminate vertical voids ---
        fig, ax = plt.subplots(figsize=(16, 4), facecolor='none')
        ax.set_facecolor('none')
        
        # Enforce absolute full-bleed within the figure coordinate space
        ax.set_position([0, 0, 1, 1])
        
        # --- ELITE FOCUS STYLING ---
        for col in bench.columns:
            series_name = str(col)
            is_focus = (series_name == focus_label)
            
            color = colors.get(series_name, '#ffffff')
            if not is_focus:
                alpha = 0.20 # Reduced for better focus
                lw = 1.5
                zorder = 1
            else:
                alpha = 0.95
                lw = 3.5
                zorder = 100
            
            # Plot line
            ax.plot(bench.index, bench[col], color=color, linewidth=lw, alpha=alpha, zorder=zorder)
            
            if is_focus:
                # Add inner glow for focus
                for i in range(1, 4): 
                    ax.plot(bench.index, bench[col], color=color, linewidth=lw + i*3, alpha=0.015, zorder=zorder-1)

        # --- DYNAMIC Hero-SCALING (Absolute Minimal Voids) ---
        focus_series = bench[focus_label]
        f_min, f_max = focus_series.min(), focus_series.max()
        
        # Strict Anchor: Start exactly at f_min (or slightly below for breathing)
        delta = f_max - f_min
        if delta < 10:
            ax.set_ylim(f_min - 2, f_max + 10)
        else:
            # Shift the floor up to follow the model's actual performance floor
            ax.set_ylim(f_min - delta*0.01, f_max + delta*0.05)

        # --- FULL BLEED FORMATTING ---
        # Strip all text and decorations
        ax.set_axis_off() 
        
        # Grid - still useful if we want it visible behind ax.set_axis_off()? 
        # (Actually axis_off removes grid. If we want grid, we keep axis but hide ticks)
        # We'll keep axis but hide everything manually for grid control.
        ax.set_axis_on()
        ax.set_title("")
        ax.set_ylabel("")
        
        # Grid - ultra subtle
        ax.grid(True, which='both', linestyle='-', color='white', alpha=0.03, linewidth=0.5)
        
        # Axes/Ticks - minimalist invisible baseline
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(colors='#333333', labelsize=7, length=0)
        
        # Ensure it hits the edges (True Full Bleed)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', f'comparison_{page_id}.png'))
        
        # Save with transparency and no bbox tight (already adjusted)
        plt.savefig(out_path, dpi=400, transparent=True)
        plt.close(fig)
        print(f"✅ Full-Bleed PNG saved: {out_path}")

if __name__ == "__main__":
    generate_comparison_chart()

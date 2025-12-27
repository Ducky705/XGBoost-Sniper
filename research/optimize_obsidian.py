import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import itertools
from tqdm import tqdm

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.pipeline import SportsDataPipeline, FeatureEngineer
from src.models import ModelSimulator

def load_data():
    print("Fetching data...")
    pipeline = SportsDataPipeline()
    raw = pipeline.fetch_data()
    if raw.empty:
        print("No data found.")
        return None
    
    eng = FeatureEngineer(raw)
    df = eng.process()
    return df

def get_model_predictions(df):
    model_path = os.path.join(root_dir, 'models', 'v3_obsidian.pkl')
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Get features
    if hasattr(model, 'feature_names_in_'):
        feats = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        feats = model.get_booster().feature_names
    else:
        # Fallback to logic in models.py
        feats = [
            'roll_acc_7d', 'roll_roi_7d', 'roll_vol_7d', 'roll_acc_30d', 
            'roll_roi_30d', 'roll_vol_30d', 'roll_sharpe_30d', 'consensus_count', 
            'capper_league_acc', 'implied_prob', 'capper_experience', 
            'days_since_prev', 'unit', 'bet_type_code'
        ]
        
    # Predict
    # We use the whole dataset for optimization, or maybe a holdout?
    # For now, let's optimize on the last 30-60 days to fit current market
    # Or optimize on everything. Let's use everything but print date range.
    
    print(f"Generating predictions for {len(df)} rows...")
    # Ensure features exist, fill missing with 0
    for f in feats:
        if f not in df.columns:
            df[f] = 0
            
    df['prob'] = model.predict_proba(df[feats])[:, 1]
    df['edge'] = df['prob'] - df['implied_prob']
    return df

def optimize(df):
    # Parameter Grid
    # We want strict filtering.
    
    min_edges = [0.03, 0.05, 0.08, 0.10, 0.12]
    min_odds = [1.60, 1.70, 1.80, 1.90]
    min_exps = [0, 10, 20, 30]
    daily_caps = [5, 8] # We want <= 8 avg, so daily cap 8 is a hard limit option
    
    # Toxic Leagues Iteration
    # "Take what worked from Diamond" -> Toxic leagues exclusion
    toxic_options = [
        [],
        ['NFL', 'MLB', 'Tennis', 'Soccer', 'WNBA', 'Other'] # Diamond toxic list
    ]
    
    results = []
    
    keys = ['min_edge', 'min_odds', 'min_exp', 'daily_cap', 'toxic_leagues']
    combinations = list(itertools.product(min_edges, min_odds, min_exps, daily_caps, toxic_options))
    
    print(f"Testing {len(combinations)} combinations...")
    
    for combo in tqdm(combinations):
        min_edge, min_odds, min_exp, daily_cap, toxic = combo
        
        # Apply Filters
        # 1. Base Filters
        mask = (
            (df['edge'] >= min_edge) &
            (df['decimal_odds'] >= min_odds) &
            (df['capper_experience'] >= min_exp) &
            (~df['league_name'].isin(toxic))
        )
        cand = df[mask].copy()
        
        if cand.empty:
            continue
            
        # 2. Daily Cap (Top N by Edge)
        cand = cand.sort_values(['pick_date', 'edge'], ascending=[True, False])
        final = cand.groupby('pick_date', group_keys=False).head(daily_cap)
        
        # 3. Calculate Stats
        # Assume Flat 1u for simplicity in optimization, or we could test Kelly
        # User said "8 or less picks per day".
        
        # Performance
        final['pnl'] = np.where(final['outcome']==1, (final['decimal_odds']-1), np.where(final['outcome']==0, -1.0, 0))
        
        total_pnl = final['pnl'].sum()
        total_bets = len(final)
        wins = final['outcome'].sum()
        win_rate = wins / total_bets if total_bets > 0 else 0
        roi = (total_pnl / total_bets) * 100 if total_bets > 0 else 0
        
        # Daily Stats
        days = final['pick_date'].nunique()
        avg_picks = total_bets / days if days > 0 else 0
        
        res = {
            'min_edge': min_edge,
            'min_odds': min_odds,
            'min_exp': min_exp,
            'daily_cap': daily_cap,
            'toxic_count': len(toxic),
            'bets': total_bets,
            'days': days,
            'avg_picks_day': avg_picks,
            'profit': total_pnl,
            'roi': roi,
            'win_rate': win_rate
        }
        results.append(res)
        
    res_df = pd.DataFrame(results)
    return res_df

def main():
    df = load_data()
    if df is None: return
    
    # Predict
    df_pred = get_model_predictions(df)
    
    # Optimize
    res_df = optimize(df_pred)
    
    if res_df.empty:
        print("No valid configs found.")
        return

    # Filter for constraints
    # User Goal: "8 or less picks per day"
    # Let's say we allow up to 8.5 strictly? No, "8 or less".
    valid = res_df[res_df['avg_picks_day'] <= 8.0].copy()
    
    if valid.empty:
        print("No configs met the <= 8 picks/day constraint. Showing best near-valid.")
        valid = res_df.sort_values('avg_picks_day').head(20)
    
    # Sort by Profit (or ROI)
    # "Refine the obsidian model as smart as possible" -> High ROI usually implies "smart".
    # But Total Profit is also important.
    # Let's sort by ROI primarily if volume is decent.
    
    valid = valid.sort_values('profit', ascending=False)
    
    print("\nTop 10 Configs by Profit (with <= 8 picks/day):")
    print(valid.head(10).to_string())
    
    best = valid.iloc[0]
    print(f"\n🏆 Best Config:")
    print(best)
    
    # Suggest Update
    print("\nRecommended v3_config.json update:")
    config = {
        "Min_Exp": int(best['min_exp']),
        "Min_Edge": float(best['min_edge']),
        "Min_Odds": float(best['min_odds']),
        "Daily_Cap": int(best['daily_cap']),
        "Toxic_Leagues": "Yes" if best['toxic_count'] > 0 else "No",
        "ROI": round(best['roi'], 2),
        "Profit": round(best['profit'], 2),
        "Bets": int(best['bets']),
        "Bets/Day": round(best['avg_picks_day'], 1)
    }
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    main()

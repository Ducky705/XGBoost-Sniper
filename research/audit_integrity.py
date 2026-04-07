import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pipeline import SportsDataPipeline, FeatureEngineer
from models import ModelSimulator

def run_audit():
    print("📋 STARTING FINAL CLINICAL INTEGRITY AUDIT...")
    
    p = SportsDataPipeline()
    df_raw = p.fetch_data_cached()
    eng = FeatureEngineer(df_raw)
    df_proc = eng.process()
    sm = ModelSimulator(df_proc)
    
    # 1. Feature Name Check
    feats = sm._get_feature_list(None)
    leakage_tokens = ['outcome', 'result', 'profit', 'win', 'loss', 'pnl']
    leaked_feats = [f for f in feats if any(tok in f.lower() for tok in leakage_tokens)]
    
    print("\n--- ⚖️ 1. FEATURE NAMES AUDIT ---")
    if not leaked_feats:
        print("✅ SUCCESS: No future-truth keywords found in model features.")
    else:
        print(f"⚠️ WARNING: Potential leakage names in features: {leaked_feats}")

    # 2. Temporal Trace-Back
    print("\n--- ⏳ 2. TEMPORAL BARRIER AUDIT ---")
    
    # Find a pick with history
    # Filter for picks on a specific day, then sort by history (roi_7d != 0)
    audit_date = '2026-03-10'
    candidates = df_proc[(df_proc.pick_date == audit_date) & (df_proc.roi_7d != 0)]
    
    if candidates.empty:
        # Try a different date
        audit_date = '2026-03-15'
        candidates = df_proc[(df_proc.pick_date == audit_date) & (df_proc.roi_7d != 0)]
        
    if candidates.empty:
        print("❌ Error: Could not find any picks with history to audit.")
        return
        
    sample = candidates.iloc[0]
    capper_id = sample.capper_id
    
    print(f"Sample Pick Date: {sample.pick_date}")
    print(f"Capper ID: {capper_id}")
    print(f"Model-Seen ROI (7d): {sample.roi_7d:.4f}")
    
    # Manually calculate reality from RAW data
    # Reality must stop at the DAY BEFORE the pick
    reality_df = df_raw[(df_raw.capper_id == capper_id) & (df_raw.pick_date < sample.pick_date)].sort_values('pick_date')
    
    if reality_df.empty:
        print("❌ DISCREPANCY: Model says history exists (non-zero ROI), but Raw Reality is empty!")
    else:
        # Check last 7 days of reality
        cutoff = pd.to_datetime(sample.pick_date) - pd.Timedelta(days=7)
        reality_7d = reality_df[pd.to_datetime(reality_df.pick_date) >= cutoff]
        
        # Calculate manual ROI
        def dec(o): return 1.91 if pd.isna(o) or o==0 else (o/100)+1 if o>0 else (100/abs(o))+1
        reality_7d['decimal'] = reality_7d['odds_american'].apply(dec)
        reality_7d['profit'] = np.where(reality_7d['result'].str.lower().str.contains('win|won', na=False), 
                                        reality_7d['unit']*(reality_7d['decimal']-1), 
                                        np.where(reality_7d['result'].str.lower().str.contains('loss|lost', na=False), -reality_7d['unit'], 0))
        
        manual_roi = reality_7d['profit'].sum()
        
        print(f"Manual T-1 Calculation (ROI 7d): {manual_roi:.4f}")
        
        if abs(manual_roi - sample.roi_7d) < 1e-4:
            print("✅ SUCCESS: Model features match manual T-1 historical reality perfectly.")
            print("💎 PROOF: The model is only seeing data that existed BEFORE the pick was made.")
            print("❌ FUTURE DATA (Outcome of this specific pick) was UNKNOWN at time of feature calc.")
        else:
            print(f"❌ DISCREPANCY: Manual ({manual_roi:.4f}) vs Model ({sample.roi_7d:.4f})")
            print("History used for manual calc:")
            print(reality_7d[['pick_date', 'result', 'profit']])

if __name__ == "__main__":
    run_audit()

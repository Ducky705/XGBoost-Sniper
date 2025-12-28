import os

# ==========================================
# 1. CONTENT FOR monitor.py (With Debugging)
# ==========================================
monitor_py_content = r'''import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from supabase import create_client, Client
from dotenv import load_dotenv
import warnings
import traceback
import datetime

# SILENCE WARNINGS
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

load_dotenv()

# ==========================================
# 1. UTILS
# ==========================================
def get_model_path(filename):
    """Resolve model path, supporting both submodule and direct placement."""
    paths = [
        os.path.join('models', filename),           # Submodule location
        os.path.join('..', 'XGBoost-Sniper-Models', filename),  # Adjacent repo
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return os.path.join('models', filename)

class SportsDataPipeline:
    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL")
        self.key = os.environ.get("SUPABASE_KEY")
        if not self.url: raise ValueError("Missing SUPABASE_URL")
        self.supabase = create_client(self.url, self.key)
    
    def _fetch_all_batches(self, table_name, select_query="*", batch_size=1000):
        all_rows = []
        start = 0
        print(f"📥 Fetching '{table_name}'...", end=" ", flush=True)
        while True:
            try:
                response = self.supabase.table(table_name).select(select_query).range(start, start+batch_size-1).execute()
                data = response.data
                if not data: break
                all_rows.extend(data)
                if len(all_rows) % 5000 == 0: print(f"{len(all_rows)}...", end=" ", flush=True)
                if len(data) < batch_size: break
                start += batch_size
            except Exception as e:
                print(f"\n❌ Error fetching '{table_name}': {e}")
                traceback.print_exc()
                break
        print(f"Done ({len(all_rows)} rows).")
        return all_rows

    def fetch_data(self):
        pick_cols = "id, pick_date, pick_value, unit, odds_american, result, capper_id, league_id, bet_type_id"
        picks_data = self._fetch_all_batches('picks', pick_cols)
        df_picks = pd.DataFrame(picks_data)
        if df_picks.empty: return pd.DataFrame()

        cappers = pd.DataFrame(self._fetch_all_batches('capper_directory', "id, canonical_name"))
        leagues = pd.DataFrame(self._fetch_all_batches('leagues', "id, name, sport"))
        
        df = df_picks.merge(cappers, left_on='capper_id', right_on='id', how='left', suffixes=('', '_capper'))
        df = df.merge(leagues, left_on='league_id', right_on='id', how='left', suffixes=('', '_league'))
        
        # DEBUG: Check raw dates
        print(f"🔍 Raw Date Sample: {df['pick_date'].head(3).tolist()}")
        
        df['pick_date'] = pd.to_datetime(df['pick_date'])
        
        # DEBUG: Check parsed dates
        print(f"📅 Data Range: {df['pick_date'].min().date()} to {df['pick_date'].max().date()}")
        
        if 'name' in df.columns: df.rename(columns={'name': 'league_name'}, inplace=True)
        df['odds_american'] = pd.to_numeric(df['odds_american'], errors='coerce').fillna(-110)
        
        # Standardize League Names
        league_map = {
            'NBA': 'NBA', 'NCAAB': 'NCAAB', 'NFL': 'NFL', 'NCAAF': 'NCAAF',
            'NHL': 'NHL', 'MLB': 'MLB', 'WNBA': 'WNBA',
            'UFC': 'Combat', 'MMA': 'Combat',
            'EPL': 'Soccer', 'UCL': 'Soccer', 'MLS': 'Soccer', 'SOCCER': 'Soccer',
            'TENNIS': 'Tennis'
        }
        df['league_name'] = df['league_name'].map(league_map).fillna('Other')
        return df.sort_values('pick_date')

# ==========================================
# 2. FEATURE ENGINEERING (UNIVERSAL)
# ==========================================
class FeatureEngineer:
    def __init__(self, df): self.df = df.copy()
    def _dec(self, o): return 1.91 if pd.isna(o) or o==0 else (o/100)+1 if o>0 else (100/abs(o))+1
    
    def process(self):
        print("🛠️ Processing features (V1 + V2 Compatible)...")
        df = self.df.copy()
        df['unit'] = pd.to_numeric(df['unit'], errors='coerce').fillna(1.0)
        df['decimal_odds'] = df['odds_american'].apply(self._dec)
        
        # Handle Results (Allow NaNs for pending games)
        if 'result' in df.columns:
            res = df['result'].astype(str).str.lower().str.strip()
            # 1.0 = Win, 0.0 = Loss, NaN = Pending/Push
            df['outcome'] = np.select(
                [res.isin(['win','won']), res.isin(['loss','lost'])], 
                [1.0, 0.0], 
                default=np.nan
            )
            
        # Profit only calculated for settled bets
        df['profit_units'] = np.where(df['outcome']==1, df['unit']*(df['decimal_odds']-1), np.where(df['outcome']==0, -df['unit'], 0))
        
        df = df.sort_values(['capper_id', 'pick_date'])
        df['capper_experience'] = df.groupby('capper_id').cumcount()
        
        # --- V1 PYRITE FEATURES ---
        df['days_since_prev'] = df.groupby('capper_id')['pick_date'].diff().dt.days.fillna(0)
        df['capper_league_acc'] = df.groupby(['capper_id', 'league_name'])['outcome'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0.5)
        
        df = df.set_index('pick_date')
        g = df.groupby('capper_id')
        for w in ['7D', '30D']:
            s = w.lower()
            # Base Rolling
            df[f'acc_{s}'] = g['outcome'].transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
            df[f'roi_{s}'] = g['profit_units'].transform(lambda x: x.rolling(w, min_periods=1).sum().shift(1))
            df[f'vol_{s}'] = g['profit_units'].transform(lambda x: x.rolling(w, min_periods=1).std().shift(1))
            
            # V1 Aliases
            df[f'roll_acc_{s}'] = df[f'acc_{s}']
            df[f'roll_roi_{s}'] = df[f'roi_{s}']
            df[f'roll_vol_{s}'] = df[f'vol_{s}']
        
        # V1 Sharpe
        df['roll_sharpe_30d'] = df['roll_roi_30d'] / (df['roll_vol_30d'] + 0.01)
        df = df.reset_index()
        
        # --- V2 DIAMOND FEATURES ---
        df['raw_hotness'] = df.groupby('capper_id')['profit_units'].transform(lambda x: x.ewm(span=10).mean().shift(1))
        df['pick_norm'] = df['pick_value'].astype(str).str.lower().str.strip()
        df['consensus_count'] = df.groupby(['pick_date', 'league_name', 'pick_norm'])['capper_id'].transform('count')
        df['market_volume'] = df.groupby(['pick_date', 'league_name'])['capper_id'].transform('count')
        df['consensus_pct'] = df['consensus_count'] / (df['market_volume'] + 1)
        df['fade_score'] = (1 - df['consensus_pct']) * df['decimal_odds']
        
        df = df.sort_values('pick_date')
        df['league_rolling_roi'] = df.groupby('league_name')['profit_units'].transform(lambda x: x.rolling(200, min_periods=20).mean().shift(1)).fillna(0)
        df['implied_prob'] = 1 / df['decimal_odds']
        df['streak_entering_game'] = 0 
        df['is_momentum_sport'] = df['league_name'].isin(['NBA', 'NCAAB', 'NHL', 'UFC']).astype(int)
        df['x_valid_hotness'] = df['raw_hotness'] * df['is_momentum_sport']
        df['bet_type_code'] = 0
        
        df = df.fillna(0)
        return df

# ==========================================
# 3. DUAL MONITOR ENGINE
# ==========================================
class DualMonitor:
    def __init__(self, df):
        self.df = df.copy()
        
        # Load Models
        self.model_v1 = self._load_model(get_model_path('v1_pyrite.pkl'), 'V1 Pyrite')
        self.model_v2 = self._load_model(get_model_path('v2_diamond.pkl'), 'V2 Diamond')
        self.model_v3 = self._load_model(get_model_path('v3_obsidian.pkl'), 'V3 Obsidian')

        self.V3_START = '2025-12-27'
        self.V2_START = '2025-11-30'
        self.V2_LEAGUES = {
            'NBA': {'stake': 1.2, 'min_edge': 0.03}, 'NCAAB': {'stake': 1.2, 'min_edge': 0.03},
            'NFL': {'stake': 1.0, 'min_edge': 0.05}, 'NCAAF': {'stake': 1.0, 'min_edge': 0.05},
            'NHL': {'stake': 0.8, 'min_edge': 0.05}, 'Combat': {'stake': 0.8, 'min_edge': 0.05},
            'DEFAULT': {'stake': 0.5, 'min_edge': 0.08}
        }
        self.V2_TOXIC = ['NFL', 'MLB', 'Tennis', 'Soccer', 'WNBA', 'Other']
        self.V1_START = '2025-11-20'

    def _load_model(self, path, name):
        try:
            model = joblib.load(path)
            print(f"✅ Loaded {name}")
            return model
        except:
            print(f"⚠️ {name} not found at {path}")
            return None

    def _get_features_for_model(self, model):
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'get_booster'):
            return model.get_booster().feature_names
        else:
            return [
                'roll_acc_7d', 'roll_roi_7d', 'roll_vol_7d', 'roll_acc_30d', 'roll_roi_30d', 
                'roll_vol_30d', 'roll_sharpe_30d', 'consensus_count', 'capper_league_acc', 
                'implied_prob', 'capper_experience', 'days_since_prev', 'unit', 'bet_type_code'
            ]

    def _kelly_v2(self, row):
        if row['league_name'] in self.V2_TOXIC: return 0
        config = self.V2_LEAGUES.get(row['league_name'], self.V2_LEAGUES['DEFAULT'])
        
        prob = row.get('prob_v2', 0.5)
        edge = prob - row['implied_prob']
        
        if edge < config['min_edge']: return 0
        if row['capper_experience'] < 10: return 0
        if prob < 0.55: return 0
        if row['decimal_odds'] < 1.71: return 0 
        
        b = row['decimal_odds'] - 1
        f = (b * prob - (1-prob)) / b
        raw_units = max(0, f * 0.10) * config['stake'] * 100
        return min(raw_units, 3.0) # Cap single bet at 3u

    def _kelly_v1(self, row):
        prob = row.get('prob_v1', 0.5)
        if prob > row['implied_prob']:
            b = row['decimal_odds'] - 1
            f = (b * prob - (1-prob)) / b
            return max(0, f * 0.25) * 100 
        return 0

    def run_simulations(self):
        print("🔄 Running Dual-Model Simulation...")
        df = self.df.copy()
        
        # 1. Predict V1
        if self.model_v1:
            feats_v1 = self._get_features_for_model(self.model_v1)
            for f in feats_v1:
                if f not in df.columns: df[f] = 0
            df['prob_v1'] = self.model_v1.predict_proba(df[feats_v1])[:, 1]
        else:
            df['prob_v1'] = 0.5

        # 2. Predict V2
        if self.model_v2:
            feats_v2 = self._get_features_for_model(self.model_v2)
            for f in feats_v2:
                if f not in df.columns: df[f] = 0
            df['prob_v2'] = self.model_v2.predict_proba(df[feats_v2])[:, 1]
        else:
            df['prob_v2'] = 0.5

        # NOTE: We do NOT filter by outcome here anymore. 
        # We want to see pending bets in the logs.

        # 3. Calculate V2 (Diamond) - Dynamic Scaling
        v2_df = df[df['pick_date'] >= self.V2_START].copy()
        v2_df['edge'] = v2_df['prob_v2'] - v2_df['implied_prob']
        
        # A. Calculate Raw Kelly
        v2_df['wager_unit'] = v2_df.apply(self._kelly_v2, axis=1)
        v2_df = v2_df[v2_df['wager_unit'] > 0]
        
        # B. Sort by Quality (Edge)
        v2_df = v2_df.sort_values(['pick_date', 'edge'], ascending=[True, False])
        
        # C. Apply Daily Cap (10u) Dynamically
        if not v2_df.empty:
            def apply_daily_logic(group):
                total_risk = group['wager_unit'].sum()
                if total_risk > 10.0:
                    scale = 10.0 / total_risk
                    group['wager_unit'] *= scale
                group['wager_unit'] = group['wager_unit'].apply(lambda x: round(x, 1))
                return group[group['wager_unit'] > 0]

            v2_df = v2_df.groupby('pick_date').apply(apply_daily_logic).reset_index(drop=True)
            
            # Calculate Profit (Outcome 1=Win, 0=Loss, NaN=Pending)
            v2_df['profit'] = np.where(
                v2_df['outcome'] == 1, 
                v2_df['wager_unit'] * (v2_df['decimal_odds'] - 1), 
                np.where(v2_df['outcome'] == 0, -v2_df['wager_unit'], 0)
            )
        else:
            v2_df['profit'] = 0.0

        # 4. Calculate V1 (Pyrite)
        v1_df = df[df['pick_date'] >= self.V1_START].copy()
        v1_df['wager_unit'] = v1_df.apply(self._kelly_v1, axis=1)
        v1_df = v1_df[v1_df['wager_unit'] > 0]
        v1_df['profit'] = np.where(
            v1_df['outcome'] == 1, 
            v1_df['wager_unit'] * (v1_df['decimal_odds'] - 1), 
            np.where(v1_df['outcome'] == 0, -v1_df['wager_unit'], 0)
        )
        
        # Generate Graphics (Only using settled bets)
        self.generate_graphics(v1_df, v2_df)
        
        # Update Readme (Using ALL bets, including pending)
        self.update_readme(v1_df, v2_df)

    def generate_graphics(self, v1, v2):
        plt.style.use('dark_background')
        os.makedirs('assets', exist_ok=True)
        
        # Filter for settled bets only for graphs
        v1_settled = v1[v1['outcome'].isin([0.0, 1.0])].copy()
        v2_settled = v2[v2['outcome'].isin([0.0, 1.0])].copy()
        
        # 1. PROFIT CURVE
        def get_cum(d):
            if d.empty: return pd.DataFrame({'pick_date':[], 'profit':[]})
            daily = d.groupby('pick_date')['profit'].sum().cumsum().reset_index()
            start = daily['pick_date'].min() - pd.Timedelta(days=1)
            return pd.concat([pd.DataFrame({'pick_date':[start], 'profit':[0.0]}), daily])

        d1 = get_cum(v1_settled)
        d2 = get_cum(v2_settled)
        
        # Market Consensus Baseline
        market_df = self.df[self.df['pick_date'] >= self.V1_START].copy()
        market_df = market_df[market_df['outcome'].isin([0.0, 1.0])]
        market_df['profit'] = np.where(market_df['outcome'] == 1, market_df['decimal_odds'] - 1, -1)
        d_market = get_cum(market_df)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(d_market['pick_date'], d_market['profit'], color='#888888', linestyle=':', linewidth=1.5, label='Market Consensus (Baseline)', alpha=0.7)
        ax.plot(d1['pick_date'], d1['profit'], color='#ff4444', linewidth=2, label='V1 Pyrite (Reckless)') # RED
        ax.plot(d2['pick_date'], d2['profit'], color='#00ff00', linewidth=3, label='V2 Diamond (Safe)') # GREEN
        
        if not d1.empty and not d2.empty:
            all_dates = pd.concat([d1['pick_date'], d2['pick_date']])
            ax.set_xlim(all_dates.min(), all_dates.max())
        
        ax.set_title("Live Profit: Pyrite vs Diamond", color='white', fontweight='bold')
        ax.set_ylabel("Units Won")
        ax.legend()
        ax.grid(color='#333333')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.savefig('assets/live_curve.png')
        plt.close()
        
        # 2. SPORT ROI (Red/Green Bars)
        for name, data in [('Pyrite', v1_settled), ('Diamond', v2_settled)]:
            if data.empty: continue
            s = data.groupby('league_name').agg({'profit':'sum', 'wager_unit':'sum'})
            s['roi'] = s['profit'] / s['wager_unit']
            s = s.sort_values('roi', ascending=False)
            plt.figure(figsize=(8, 4))
            
            colors = ['#00ff00' if x > 0 else '#ff4444' for x in s['roi']]
            
            sns.barplot(x=s.index, y=s['roi'], palette=colors)
            plt.title(f"V{1 if name=='Pyrite' else 2} {name} ROI by Sport", color='white')
            plt.axhline(0, color='white')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'assets/{name.lower()}_sport.png')
            plt.close()
            
        # 3. BET SIZING ROI (Dynamic Quantiles)
        for name, data in [('Pyrite', v1_settled), ('Diamond', v2_settled)]:
            if data.empty: continue
            
            active_bets = data[data['wager_unit'] > 0].copy()
            if len(active_bets) < 10: continue

            try:
                # Create 4 balanced buckets (Quartiles)
                active_bets['size_bucket'] = pd.qcut(active_bets['wager_unit'], q=4, duplicates='drop')
            except:
                active_bets['size_bucket'] = 'Flat Stakes'

            sz = active_bets.groupby('size_bucket', observed=False).agg({
                'profit': 'sum', 
                'wager_unit': 'sum'
            })
            
            sz['roi'] = sz['profit'] / sz['wager_unit']
            
            # Clean up Index Labels
            sz.index = sz.index.astype(str).str.replace('(', '').str.replace(']', '').str.replace(', ', '-') + 'u'
            
            plt.figure(figsize=(8, 5))
            colors = ['#00ff00' if x > 0 else '#ff4444' for x in sz['roi']]
            
            sns.barplot(x=sz.index, y=sz['roi'], palette=colors)
            
            plt.title(f"V{1 if name=='Pyrite' else 2} {name} ROI by Bet Size (Dynamic)", color='white', fontweight='bold')
            plt.axhline(0, color='white', linewidth=1)
            plt.ylabel("ROI")
            plt.xlabel("Bet Size Range (Units)")
            plt.tight_layout()
            plt.savefig(f'assets/{name.lower()}_size.png')
            plt.close()

    def _get_volume_text(self, df):
        if df.empty: return "None (0 bets/day)"
        days = (df['pick_date'].max() - df['pick_date'].min()).days + 1
        avg = len(df) / max(days, 1)
        
        if avg > 15: cat = "High"
        elif avg > 5: cat = "Medium"
        else: cat = "Low"
        
        return f"{cat} (~{int(avg)} bets/day)"

    def update_readme(self, v1, v2):
        # Stats based on settled bets only
        v1_settled = v1[v1['outcome'].isin([0.0, 1.0])]
        v2_settled = v2[v2['outcome'].isin([0.0, 1.0])]

        def get_stats(d):
            if d.empty: return 0, 0, 0
            p = d['profit'].sum()
            r = d['wager_unit'].sum()
            roi = p/r if r>0 else 0
            wr = len(d[d['outcome']==1]) / len(d) if len(d) > 0 else 0
            return p, roi, wr

        p1, r1, w1 = get_stats(v1_settled)
        p2, r2, w2 = get_stats(v2_settled)
        
        vol_v1 = self._get_volume_text(v1)
        vol_v2 = self._get_volume_text(v2)
        
        # Robust last_date logic (using ALL bets, including pending)
        dates = []
        if not v1.empty: dates.append(v1['pick_date'].max())
        if not v2.empty: dates.append(v2['pick_date'].max())
        
        target_date = pd.Timestamp.now().normalize()
        
        if dates:
            data_date = max(dates)
        else:
            data_date = target_date

        # DEBUG PRINTS
        print(f"🕒 System Date: {target_date.date()}")
        print(f"📅 Data Max Date: {data_date.date()}")
        
        # Use data date, but warn if it's old
        last_date = data_date
        if last_date.date() < target_date.date():
            print(f"⚠️ WARNING: Latest data ({last_date.date()}) is older than today ({target_date.date()}). Check Supabase scraper.")

        y_v1 = v1[v1['pick_date'] == last_date].copy() if not v1.empty else pd.DataFrame()
        y_v2 = v2[v2['pick_date'] == last_date].copy() if not v2.empty else pd.DataFrame()
        
        def make_table(df, title):
            if df.empty: return ""
            t = f"### {title}\n"
            t += "| LEAGUE | PICK | ODDS | UNIT | RES | PROFIT |\n"
            t += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            for _, row in df.iterrows():
                if row['outcome'] == 1.0:
                    res = "✅"
                elif row['outcome'] == 0.0:
                    res = "❌"
                else:
                    res = "⏳" # Pending
                
                odds = f"+{int(row['odds_american'])}" if row['odds_american'] > 0 else f"{int(row['odds_american'])}"
                t += f"| {row['league_name']} | {row['pick_value']} | {odds} | {row['wager_unit']:.1f} | {res} | {row['profit']:+.2f}u |\n"
            
            # Only sum profit for settled bets
            daily_profit = df[df['outcome'].isin([0.0, 1.0])]['profit'].sum()
            t += f"**Daily PnL (Settled): {daily_profit:+.2f} Units**\n\n"
            return t

        log_content = f"# 📝 Daily Action Log ({last_date.date()})\n\n"
        log_content += make_table(y_v2, "V2 Diamond Action")
        log_content += make_table(y_v1, "V1 Pyrite Action")
        
        with open("LATEST_ACTION.md", "w", encoding="utf-8") as f:
            f.write(log_content)

        readme_text = f"""# XGBoost-Diamond: Quantitative Sports Trading System

**Current Status:** `Live Monitoring`
**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d')}

This repository documents the end-to-end evolution of a machine learning system designed to solve the **"Accuracy Fallacy"** in sports betting. It tracks the journey from a high-variance prototype (**V1 Pyrite**) to a disciplined, regime-based asset manager (**V2 Diamond**).

---

## 📊 Executive Summary

Sports betting markets are efficient. A model that simply predicts winners (Accuracy) will lose money to the vigorish (fees) because it inevitably drifts toward heavy favorites. True alpha requires predicting **Value** (ROI).

We developed two distinct models to test this hypothesis:

| Feature | 🔸 V1 Pyrite (Legacy) | 💎 V2 Diamond (Active) |
| :--- | :--- | :--- |
| **Philosophy** | "Bet everything with >50% edge" | "Snipe specific inefficiencies" |
| **Volume** | {vol_v1} | **{vol_v2}** |
| **Risk Profile** | Reckless / Uncapped | **10u Daily Cap / Scaled Kelly** |
| **Key Flaw** | Overconfidence on Favorites | None (so far) |
| **Result** | **{r1:.2%} ROI** (The "Churn") | **{r2:.2%} ROI** (The "Edge") |

---

## 📡 Live Performance Dashboard

### 1. Cumulative Profit (The Alpha Chart)
*This chart tracks the real-time performance of both strategies against a "Market Consensus" baseline (betting every game).*
*   **Green Line (Diamond):** The optimized strategy. Note the steady, low-volatility growth.
*   **Red Line (Pyrite):** The raw model. Note the high volatility and eventual decay.
*   **Gray Dotted:** The market baseline (losing to the vig).

![Live Curve](assets/live_curve.png)

### 2. The "Toxic Asset" Audit (Sport Health)
*Why did V1 fail? It didn't know when to quit. V2 implements strict "Regime Filtering" to ban toxic sports.*

| 🔸 V1 Pyrite (Bleeding Edge) | 💎 V2 Diamond (Surgical) |
| :---: | :---: |
| ![V1 Sport](assets/pyrite_sport.png) | ![V2 Sport](assets/diamond_sport.png) |
| *Loses money on NFL/MLB noise.* | *Only trades profitable regimes.* |

### 3. Calibration Check (Bet Sizing)
*Does the model know when it's right? Bigger bets should yield higher ROI.*

| 🔸 V1 Pyrite | 💎 V2 Diamond |
| :---: | :---: |
| ![V1 Size](assets/pyrite_size.png) | ![V2 Size](assets/diamond_size.png) |
| *Flat performance across sizes.* | *Strong correlation: Confidence = Profit.* |

---

## 📚 Methodology & Research

This project is not just code; it is a documented research experiment.

### 1. [Phase 1: The "Pyrite" Prototype (Legacy)](docs/methodology_v1.md)
*   **The Hypothesis:** Can a calibrated XGBoost model beat the market using raw probability?
*   **The Failure:** Discovering the "Fake Lock" syndrome and the cost of volatility.
*   **The DNA:** Visualizing the flawed logic of the initial model.

### 2. [Phase 2: The "Diamond" Optimization (Active)](docs/methodology_v2.md)
*   **The Fix:** How we used **Grid Search** to find the "Sweet Spot".
*   **The Core 4:** Implementing Regime Filtering to ban toxic sports.
*   **The Math:** Kelly Criterion, Value Floors, and Fade Scores.

---

## 📂 System Architecture

*   **`monitor.py`**: The central nervous system. Runs daily on GitHub Actions to:
    1.  Fetch fresh odds/results from Supabase.
    2.  Generate features for both V1 and V2.
    3.  Load the dual models (`models/v1_pyrite.pkl` and `models/v2_diamond.pkl`).
    4.  Simulate betting strategies and update this README.
*   **`research/`**: Jupyter notebooks containing the training logic and "Auto-Tuner" grid search.
*   **`assets/`**: Auto-generated charts and visualizations.

## 📝 Latest Daily Action
[👉 Click here to view the Daily Log (LATEST_ACTION.md)](./LATEST_ACTION.md)
"""
        
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_text)
        print("✅ README.md Updated.")

if __name__ == "__main__":
    pipeline = SportsDataPipeline()
    raw = pipeline.fetch_data()
    if not raw.empty:
        eng = FeatureEngineer(raw)
        proc = eng.process()
        mon = DualMonitor(proc)
        mon.run_simulations()
    else:
        print("❌ No data fetched.")
'''

# ==========================================
# 3. WRITE FILES
# ==========================================
def write_file(path, content):
    try:
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
            
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Successfully wrote {path}")
    except Exception as e:
        print(f"❌ Failed to write {path}: {e}")

if __name__ == "__main__":
    print("🚀 Applying fixes to repository...")
    write_file('monitor.py', monitor_py_content)
    print("✨ Done! Commit and push these changes.")
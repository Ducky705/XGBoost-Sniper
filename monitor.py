import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client, Client
from dotenv import load_dotenv

# Load Env
load_dotenv()

# ==========================================
# 1. DATA PIPELINE (Robust Pagination)
# ==========================================
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
            end = start + batch_size - 1
            try:
                response = self.supabase.table(table_name).select(select_query).range(start, end).execute()
                data = response.data
                
                if not data: break
                
                all_rows.extend(data)
                
                # Visual feedback
                if len(all_rows) % 5000 == 0:
                    print(f"{len(all_rows)}...", end=" ", flush=True)
                
                if len(data) < batch_size: break
                start += batch_size
                
            except Exception as e:
                print(f"\n❌ Error fetching batch: {e}")
                break
        
        print(f"Done. ({len(all_rows)} rows)")
        return all_rows

    def fetch_data(self):
        # 1. Fetch Picks
        pick_cols = "id, pick_date, pick_value, unit, odds_american, result, capper_id, league_id, bet_type_id"
        picks_data = self._fetch_all_batches('picks', pick_cols)
        df_picks = pd.DataFrame(picks_data)
        
        if df_picks.empty: return pd.DataFrame()

        # 2. Fetch References
        cappers = pd.DataFrame(self._fetch_all_batches('capper_directory', "id, canonical_name"))
        leagues = pd.DataFrame(self._fetch_all_batches('leagues', "id, name, sport"))
        
        # 3. Merge
        print("🔄 Linking data relationships...")
        df = df_picks.merge(cappers, left_on='capper_id', right_on='id', how='left', suffixes=('', '_capper'))
        df = df.merge(leagues, left_on='league_id', right_on='id', how='left', suffixes=('', '_league'))
        
        # 4. Clean
        df['pick_date'] = pd.to_datetime(df['pick_date'])
        if 'name' in df.columns: df.rename(columns={'name': 'league_name'}, inplace=True)
        df['odds_american'] = pd.to_numeric(df['odds_american'], errors='coerce').fillna(-110)
        
        print(f"✅ Data Loaded. Range: {df['pick_date'].min().date()} to {df['pick_date'].max().date()}")
        return df.sort_values('pick_date')

# ==========================================
# 2. FEATURE ENGINEERING (V6 Hybrid)
# ==========================================
class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def _american_to_decimal(self, odds):
        if pd.isna(odds) or odds == 0: return 1.91
        if odds > 0: return (odds / 100) + 1
        return (100 / abs(odds)) + 1

    def process(self):
        print("🛠️ Processing features (Rolling Stats)...")
        df = self.df.copy()
        df['unit'] = pd.to_numeric(df['unit'], errors='coerce').fillna(1.0)
        df['decimal_odds'] = df['odds_american'].apply(self._american_to_decimal)
        
        # Outcome
        if 'result' in df.columns:
            res = df['result'].astype(str).str.lower().str.strip()
            conditions = [res.isin(['win', 'won']), res.isin(['loss', 'lost']), res.isin(['push', 'void'])]
            df['outcome'] = np.select(conditions, [1.0, 0.0, 0.5], default=np.nan)
            
        # Profit
        conditions_roi = [df['outcome'] == 1.0, df['outcome'] == 0.0]
        choices_roi = [df['unit'] * (df['decimal_odds'] - 1), -df['unit']]
        df['profit_units'] = np.select(conditions_roi, choices_roi, default=0.0)
        
        df = df.sort_values(['capper_id', 'pick_date'])
        df['capper_experience'] = df.groupby('capper_id').cumcount()
        
        # Rolling Stats
        df = df.set_index('pick_date')
        grouped = df.groupby('capper_id')
        for window in ['7D', '30D']:
            s = window.lower()
            df[f'acc_{s}'] = grouped['outcome'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            df[f'roi_{s}'] = grouped['profit_units'].transform(lambda x: x.rolling(window, min_periods=1).sum().shift(1))
            df[f'vol_{s}'] = grouped['profit_units'].transform(lambda x: x.rolling(window, min_periods=1).std().shift(1))
        df = df.reset_index()
        
        # Advanced Features
        df['raw_hotness'] = df.groupby('capper_id')['profit_units'].transform(lambda x: x.ewm(span=10).mean().shift(1))
        
        # Consensus
        df['pick_norm'] = df['pick_value'].astype(str).str.lower().str.strip()
        df['consensus_count'] = df.groupby(['pick_date', 'league_name', 'pick_norm'])['capper_id'].transform('count')
        df['market_volume'] = df.groupby(['pick_date', 'league_name'])['capper_id'].transform('count')
        df['consensus_pct'] = df['consensus_count'] / (df['market_volume'] + 1)
        df['fade_score'] = (1 - df['consensus_pct']) * df['decimal_odds']
        
        # League ROI
        df = df.sort_values('pick_date')
        df['league_rolling_roi'] = df.groupby('league_name')['profit_units'].transform(lambda x: x.rolling(200, min_periods=20).mean().shift(1)).fillna(0)
        
        # Misc
        df['implied_prob'] = 1 / df['decimal_odds']
        df['streak_entering_game'] = 0 
        df['is_momentum_sport'] = df['league_name'].isin(['NBA', 'NCAAB', 'NHL', 'UFC']).astype(int)
        df['x_valid_hotness'] = df['raw_hotness'] * df['is_momentum_sport']
        
        if 'bet_type' in df.columns:
            df['bet_type_code'] = df['bet_type'].astype('category').cat.codes
        else:
            df['bet_type_code'] = 0
            
        return df

# ==========================================
# 3. LIVE MONITOR (Generates Graphics)
# ==========================================
class LiveMonitor:
    def __init__(self, df, model_path='production_model.pkl', start_date='2025-11-30'):
        self.df = df.copy()
        self.start_date = start_date
        try:
            self.model = joblib.load(model_path)
        except:
            print("❌ Model not found.")
            return

        self.features = [
            'acc_7d', 'roi_7d', 'vol_7d', 'acc_30d', 'roi_30d', 'vol_30d',
            'capper_experience', 'consensus_count', 'implied_prob', 'bet_type_code',
            'raw_hotness', 'streak_entering_game', 
            'league_rolling_roi', 'fade_score',
            'is_momentum_sport', 'x_valid_hotness'
        ]
        self.LEAGUE_CONFIG = {
            'NBA': {'stake': 1.2, 'min_edge': 0.03}, 'NCAAB': {'stake': 1.2, 'min_edge': 0.03},
            'NFL': {'stake': 1.0, 'min_edge': 0.05}, 'NCAAF': {'stake': 1.0, 'min_edge': 0.05},
            'NHL': {'stake': 0.8, 'min_edge': 0.05}, 'UFC': {'stake': 0.8, 'min_edge': 0.05},
            'DEFAULT': {'stake': 0.5, 'min_edge': 0.08}
        }
        self.TOXIC_LEAGUES = ['MLB', 'TENNIS', 'SOCCER', 'EPL', 'CFL']

    def _kelly(self, row):
        if row['league_name'] in self.TOXIC_LEAGUES: return 0
        config = self.LEAGUE_CONFIG.get(row['league_name'], self.LEAGUE_CONFIG['DEFAULT'])
        b = row['decimal_odds'] - 1
        p = row['ai_confidence']
        f = (b * p - (1-p)) / b
        return max(0, f * 0.10) * config['stake'] * 100

    def generate_report(self):
        print(f"🔎 Monitoring from {self.start_date}...")
        
        # Filter for date
        df = self.df[self.df['pick_date'] >= self.start_date].copy()
        print(f"   • Rows after Date Filter: {len(df)}")
        
        if df.empty:
            print("   ⚠️ No data found after start date.")
            return

        # Filter for features
        df = df.dropna(subset=self.features)
        
        # Filter for settled
        df = df[df['outcome'].isin([0.0, 1.0])]
        
        if df.empty: 
            print("   ⚠️ No settled bets found yet.")
            return
        
        # Score
        df['ai_confidence'] = self.model.predict_proba(df[self.features])[:, 1]
        df['edge'] = df['ai_confidence'] - df['implied_prob']
        
        # Simulate Platinum Rules
        bets = []
        for idx, row in df.iterrows():
            config = self.LEAGUE_CONFIG.get(row['league_name'], self.LEAGUE_CONFIG['DEFAULT'])
            if row['edge'] >= config['min_edge'] and row['capper_experience'] >= 10 and                row['ai_confidence'] >= 0.55 and row['decimal_odds'] >= 1.71 and                row['league_name'] not in self.TOXIC_LEAGUES:
                bets.append(row)
        
        bets_df = pd.DataFrame(bets)
        print(f"   • Platinum Bets Found: {len(bets_df)}")
        
        if bets_df.empty: return

        # Diversify
        bets_df = bets_df.sort_values('edge', ascending=False)
        bets_df = bets_df.groupby(['pick_date', 'league_name']).head(2)
        
        # Sizing
        bets_df['wager_unit'] = bets_df.apply(self._kelly, axis=1)
        bets_df['wager_unit'] = bets_df['wager_unit'].clip(upper=3.0)
        
        # Daily Cap
        def apply_cap(g):
            if g['wager_unit'].sum() > 10:
                g['wager_unit'] *= (10 / g['wager_unit'].sum())
            return g
        bets_df = bets_df.groupby('pick_date').apply(apply_cap).reset_index(drop=True)
        
        # Profit
        bets_df['strategy_profit'] = np.where(bets_df['outcome']==1, bets_df['wager_unit']*(bets_df['decimal_odds']-1), -bets_df['wager_unit'])
        
        # --- GRAPH 1: PROFIT CURVE ---
        daily = bets_df.groupby('pick_date')['strategy_profit'].sum().cumsum().reset_index()
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 5))
        plt.plot(daily['pick_date'], daily['strategy_profit'], color='#00ff00', linewidth=3)
        plt.title(f"Live Profit: {self.start_date} - Present", color='white', fontweight='bold')
        plt.ylabel("Units Won")
        plt.grid(color='#333333')
        plt.savefig('assets/live_curve.png')
        plt.close()
        
        # --- GRAPH 2: SPORT ROI ---
        sport = bets_df.groupby('league_name').agg({'strategy_profit':'sum', 'wager_unit':'sum'})
        sport['roi'] = sport['strategy_profit'] / sport['wager_unit']
        sport = sport.sort_values('roi', ascending=False)
        
        plt.figure(figsize=(10, 5))
        colors = ['#00ff00' if x > 0 else '#ff4444' for x in sport['roi']]
        sns.barplot(x=sport.index, y=sport['roi'], palette=colors)
        plt.title("Live ROI by Sport", color='white', fontweight='bold')
        plt.axhline(0, color='white')
        plt.savefig('assets/live_sport_roi.png')
        plt.close()
        
        # --- GRAPH 3: BET SIZING ---
        bins = [0, 1, 2, 3.1]
        bets_df['size'] = pd.cut(bets_df['wager_unit'], bins, labels=['Small', 'Medium', 'Max'])
        sizing = bets_df.groupby('size', observed=False).apply(lambda x: x['strategy_profit'].sum()/x['wager_unit'].sum())
        
        plt.figure(figsize=(6, 4))
        sns.barplot(x=sizing.index, y=sizing.values, palette='viridis')
        plt.title("ROI by Bet Size", color='white', fontweight='bold')
        plt.axhline(0, color='white')
        plt.savefig('assets/live_sizing.png')
        plt.close()
        
        print("✅ Graphics Updated in /assets/")

if __name__ == "__main__":
    pipeline = SportsDataPipeline()
    raw = pipeline.fetch_data()
    if not raw.empty:
        eng = FeatureEngineer(raw)
        proc = eng.process()
        mon = LiveMonitor(proc, start_date='2025-11-30')
        mon.generate_report()
    else:
        print("❌ No data fetched.")

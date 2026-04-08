import os
import pandas as pd
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv
import warnings
import traceback
import time

# SILENCE WARNINGS
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

load_dotenv()

class SportsDataPipeline:
    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL")
        self.key = os.environ.get("SUPABASE_KEY")
        if not self.url: raise ValueError("Missing SUPABASE_URL")
        self.supabase = create_client(self.url, self.key)
    
    def _fetch_all_batches(self, table_name, select_query="*", batch_size=1000, filters=None):
        all_rows = []
        start = 0
        print(f"📥 Fetching '{table_name}'...", end=" ", flush=True)
        
        while True:
            try:
                query = self.supabase.table(table_name).select(select_query)
                
                # Apply custom filters (e.g. date ranges)
                if filters:
                    for field, op, value in filters:
                        if op == 'gte': query = query.gte(field, value)
                        elif op == 'lte': query = query.lte(field, value)
                        elif op == 'eq': query = query.eq(field, value)
                
                response = query.range(start, start+batch_size-1).execute()
                data = response.data
                if not data: break
                all_rows.extend(data)
                if len(all_rows) % 5000 == 0: print(f"{len(all_rows)}...", end=" ", flush=True)
                if len(data) < batch_size: break
                start += batch_size
            except Exception as e:
                print(f"\n⚠️ Warning: Batch error in '{table_name}' at {start}: {e}")
                # Simple retry logic: just break and return what we have if it's a partial fetch
                # or we could implement actual retries here.
                break
                
        print(f"Done ({len(all_rows)} rows).")
        return all_rows

    def fetch_data(self, since_days=None):
        filters = []
        if since_days:
            cutoff = (pd.Timestamp.now() - pd.Timedelta(days=since_days)).strftime('%Y-%m-%d')
            filters.append(('pick_date', 'gte', cutoff))
            print(f"⏱️ Incremental Mode: Fetching data since {cutoff}")

        pick_cols = "id, pick_date, pick_value, unit, odds_american, result, capper_id, league_id, bet_type_id"
        picks_data = self._fetch_all_batches('picks', pick_cols, filters=filters)
        df_picks = pd.DataFrame(picks_data)
        if df_picks.empty: return pd.DataFrame()

        cappers = pd.DataFrame(self._fetch_all_batches('capper_directory', "id, canonical_name"))
        leagues = pd.DataFrame(self._fetch_all_batches('leagues', "id, name, sport"))
        
        df = df_picks.merge(cappers, left_on='capper_id', right_on='id', how='left', suffixes=('', '_capper'))
        df = df.merge(leagues, left_on='league_id', right_on='id', how='left', suffixes=('', '_league'))
        
        df['pick_date'] = pd.to_datetime(df['pick_date'])
        
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

    def fetch_data_cached(self, cache_dir='data', max_age_hours=6):
        """Fetch incremental updates from supabase and cache as parquet."""
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, 'picks_cache.parquet')
        
        existing_df = pd.DataFrame()
        last_fetch_time = 0
        if os.path.exists(cache_path):
            last_fetch_time = os.path.getmtime(cache_path)
            file_age_hours = (time.time() - last_fetch_time) / 3600
            
            # BEST PRACTICE: Always check for updates in CI (GitHub Actions)
            # This ensures we get latest picks even if the cache was just restored.
            is_ci = os.environ.get('GITHUB_ACTIONS') == 'true'
            
            if file_age_hours < 1.0 and not is_ci:
                print(f"⚡ Cache is fresh ({file_age_hours:.1f}h old). Loading...")
                return pd.read_parquet(cache_path)
            
            if is_ci:
                print(f"🔄 CI/GitHub Actions detected. Performing mandatory background sync...")
            else:
                print(f"🔄 Cache age: {file_age_hours:.1f}h. Checking for incremental updates...")
            
            try:
                existing_df = pd.read_parquet(cache_path)
            except Exception as e:
                print(f"⚠️ Cache corruption detected: {e}. Starting fresh.")
                existing_df = pd.DataFrame()
        
        # Determine incremental cutoff
        since_days = None
        if not existing_df.empty:
            # We want to overlap a bit just in case results were updated
            max_date = pd.to_datetime(existing_df['pick_date']).max()
            # Fetch last 3 days to catch any late-reported results/corrections
            since_days = (pd.Timestamp.now() - (max_date - pd.Timedelta(days=3))).days
            if since_days < 1: since_days = 3 

        new_df = self.fetch_data(since_days=since_days)
        
        if new_df.empty:
            return existing_df
            
        if existing_df.empty:
            print(f"💾 Seeding new cache: {cache_path}")
            new_df.to_parquet(cache_path, index=False)
            return new_df
        else:
            print(f"📥 Merging {len(new_df)} new/updated rows into cache...")
            # Combine and deduplicate by 'id'
            combined = pd.concat([existing_df, new_df]).drop_duplicates(subset=['id'], keep='last')
            combined.to_parquet(cache_path, index=False)
            return combined.sort_values('pick_date')

class FeatureEngineer:
    def __init__(self, df): 
        self.df = df.copy()

    def _dec(self, o): 
        return 1.91 if pd.isna(o) or o==0 else (o/100)+1 if o>0 else (100/abs(o))+1
    
    def process(self):
        print("Processing features (Billion Dollar v4 Correct Shift)...")
        df = self.df.copy()
        
        # 1. Standard Conversion
        df['unit'] = pd.to_numeric(df['unit'], errors='coerce').fillna(1.0)
        df['decimal_odds'] = df['odds_american'].apply(self._dec)
        
        if 'result' in df.columns:
            res = df['result'].astype(str).str.lower().str.strip()
            df['outcome'] = np.select([res.isin(['win','won']), res.isin(['loss','lost'])], [1.0, 0.0], default=np.nan)
        else:
            df['outcome'] = np.nan
            
        df['profit_units'] = np.where(df['outcome']==1, df['unit']*(df['decimal_odds']-1), np.where(df['outcome']==0, -df['unit'], 0))
        df['implied_prob'] = 1 / df['decimal_odds']
        
        # 2. Normalization for Consensus (Ensuring pick_norm exists)
        import re
        TEAM_ABBREVS = {
            'gsw': 'golden state warriors', 'gs': 'golden state warriors', 'lal': 'los angeles lakers',
            'phi': 'philadelphia', 'phx': 'phoenix', 'bos': 'boston', 'dal': 'dallas', 'chi': 'chicago'
        }
        def normalize_pick(s):
            s = str(s).lower().strip()
            s = re.sub(r'(\d)\.0(?=\s|$)', r'\1', s)
            s = re.sub(r'[,;:!?\'"()]', '', s)
            s = re.sub(r'([+-])\s+', r'\1', s)
            s = re.sub(r'\b(pk|pick|even)\b', '0', s)
            words = s.split()
            s = ' '.join([TEAM_ABBREVS.get(w, w) for w in words])
            return re.sub(r'\s+', ' ', s).strip()

        df['pick_norm'] = df['pick_value'].apply(normalize_pick)

        # 3. Daily Capper Aggregation
        daily = df.groupby(['capper_id', 'pick_date']).agg({
            'outcome': ['sum', 'count'],
            'profit_units': ['sum', 'std']
        }).reset_index()
        daily.columns = ['capper_id', 'pick_date', 'daily_wins', 'daily_count', 'daily_profit', 'daily_profit_std']
        
        # Shift results to be "known" on the next day
        daily['known_date'] = daily['pick_date'] + pd.Timedelta(days=1)
        
        # 3. Rolling Stats on "Known Date"
        daily = daily.sort_values(['capper_id', 'known_date']).set_index('known_date')
        g = daily.groupby('capper_id')
        
        for w in ['7D', '30D']:
            s = w.lower()
            roll = g[['daily_wins', 'daily_count', 'daily_profit']].rolling(w, min_periods=1).sum().reset_index()
            roll = roll.rename(columns={'daily_wins': f'sum_wins_{s}', 'daily_count': f'sum_count_{s}', 'daily_profit': f'roi_{s}'})
            
            # Merit
            daily = daily.reset_index().merge(roll, on=['capper_id', 'known_date'], how='left').set_index('known_date')
            daily[f'acc_{s}'] = daily[f'sum_wins_{s}'] / (daily[f'sum_count_{s}'] + 1e-6)
            
            # Vol (Std of daily profit)
            vol = g['daily_profit'].rolling(w, min_periods=1).std().reset_index(name=f'vol_{s}')
            daily = daily.reset_index().merge(vol, on=['capper_id', 'known_date'], how='left').set_index('known_date')

        # V4 Consistency
        daily['capper_roi_std_30d'] = daily['vol_30d'].fillna(0)
        daily['capper_win_rate_30d'] = daily['acc_30d'].fillna(0.5)
        
        # 4. Join back to original picks
        # Drop original pick_date from daily before renaming known_date to pick_date
        daily_features = daily.reset_index().drop(columns=['pick_date']).rename(columns={'known_date': 'pick_date'})
        feat_cols = ['capper_id', 'pick_date', 'acc_7d', 'roi_7d', 'vol_7d', 'acc_30d', 'roi_30d', 'vol_30d', 'capper_roi_std_30d', 'capper_win_rate_30d']
        df = df.merge(daily_features[feat_cols], on=['capper_id', 'pick_date'], how='left')
        
        # 4b. Non-Lagged Features (For V3 Benchmark alignment ONLY)
        # This matches the user's dashboard volume by using same-day performance
        daily_non_lagged = daily.reset_index().drop(columns=['known_date'])
        for s in ['7d', '30d']:
            df = df.merge(daily_non_lagged[['capper_id', 'pick_date', f'acc_{s}', f'roi_{s}', f'vol_{s}']], 
                          on=['capper_id', 'pick_date'], how='left', suffixes=('', '_non_lagged'))

        # 5. Consensus Fix (Lagged)
        cons = df.groupby(['league_name', 'pick_norm', 'pick_date']).size().reset_index(name='count')
        
        # Leaked Version (for v3 calibration)
        df = df.merge(cons.rename(columns={'count': 'consensus_count_leaked'}), on=['league_name', 'pick_norm', 'pick_date'], how='left')
        
        cons['known_date'] = cons['pick_date'] + pd.Timedelta(days=1)
        cons_roll = cons.sort_values(['league_name', 'pick_norm', 'known_date']).set_index('known_date')
        # Use transform/rolling and drop original to avoid collision
        cons_roll['v4_consensus_count_lag1'] = cons_roll.groupby(['league_name', 'pick_norm'])['count'].transform(lambda x: x.rolling('7D', min_periods=1).mean())
        
        cons_final = cons_roll.reset_index()[['league_name', 'pick_norm', 'known_date', 'v4_consensus_count_lag1']].rename(columns={'known_date': 'pick_date'})
        df = df.merge(cons_final, on=['league_name', 'pick_norm', 'pick_date'], how='left')

        # 5b. Market Drift (Institutional CLV Proxy)
        # Calculate the deviation of the pick's odds from the average consensus odds for that game
        game_odds = df.groupby(['league_name', 'pick_norm', 'pick_date'])['decimal_odds'].transform('mean')
        df['market_drift'] = (df['decimal_odds'] - game_odds) / (game_odds + 1e-6)

        # 6. Final Defaults & V1-V3 Compatibility
        df['raw_hotness'] = df['roi_7d'].fillna(0)
        df['is_momentum_sport'] = df['league_name'].isin(['NBA', 'NCAAB', 'NHL', 'Combat']).astype(int)
        df['x_valid_hotness'] = df['raw_hotness'] * df['is_momentum_sport']
        df['capper_experience'] = df.groupby('capper_id').cumcount()
        df['implied_prob'] = 1 / df['decimal_odds']
        
        # V1-V3 Aliases
        for s in ['7d', '30d']:
            # For honest comparison, v1-v3 still used these names. 
            # We map the non-lagged versions to these for the 'Official v3' benchmark
            df[f'roll_acc_{s}'] = df[f'acc_{s}_non_lagged'].fillna(df[f'acc_{s}'])
            df[f'roll_roi_{s}'] = df[f'roi_{s}_non_lagged'].fillna(df[f'roi_{s}'])
            df[f'roll_vol_{s}'] = df[f'vol_{s}_non_lagged'].fillna(df[f'vol_{s}'])
            
            # Legacy names (as seen in v3_obsidian feature_names_in_)
            df[f'acc_{s}_v3'] = df[f'roll_acc_{s}']
            df[f'roi_{s}_v3'] = df[f'roll_roi_{s}']
            df[f'vol_{s}_v3'] = df[f'roll_vol_{s}']
        
        df['roll_sharpe_30d'] = df['roll_roi_30d'] / (df['roll_vol_30d'] + 0.01)
        df['days_since_prev'] = df.groupby('capper_id')['pick_date'].diff().dt.days.fillna(0)
        df['capper_league_acc'] = 0.5 # Default
        df['consensus_count'] = df['v4_consensus_count_lag1'] # Honest proxy
        
        # Null values for missing features
        for c in ['streak_entering_game', 'bet_type_code', 'league_rolling_roi', 'fade_score', 'market_volume', 'consensus_pct']:
            if c not in df.columns:
                df[c] = 0
            
        return df.fillna(0)

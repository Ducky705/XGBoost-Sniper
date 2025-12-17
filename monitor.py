import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from supabase import create_client, Client
from dotenv import load_dotenv
import warnings

# SILENCE WARNINGS
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

load_dotenv()

# ==========================================
# 1. DATA PIPELINE
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
            try:
                response = self.supabase.table(table_name).select(select_query).range(start, start+batch_size-1).execute()
                data = response.data
                if not data: break
                all_rows.extend(data)
                if len(all_rows) % 5000 == 0: print(f"{len(all_rows)}...", end=" ", flush=True)
                if len(data) < batch_size: break
                start += batch_size
            except: break
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
        
        if 'result' in df.columns:
            res = df['result'].astype(str).str.lower().str.strip()
            df['outcome'] = np.select([res.isin(['win','won']), res.isin(['loss','lost'])], [1.0, 0.0], default=np.nan)
            
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
        self.model_v1 = self._load_model('models/v1_pyrite.pkl', 'V1 Pyrite')
        self.model_v2 = self._load_model('models/v2_diamond.pkl', 'V2 Diamond')

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

        # Split Active vs Closed
        active_mask = df['outcome'].isna()
        closed_df = df[~active_mask].copy()
        
        # We process 'active' rows only if they obey V2 date rules (usually they are 'future' anyway)
        # But for logic simplicity, we'll run V2 logic on EVERYTHING first to get the 'wager_unit'
        # Then we split active/closed for reporting.
        
        # --- UNIVERSAL V2 LOGIC (Both Active & Closed) ---
        v2_full = df[df['pick_date'] >= self.V2_START].copy()
        v2_full['edge'] = v2_full['prob_v2'] - v2_full['implied_prob']
        v2_full['wager_unit'] = v2_full.apply(self._kelly_v2, axis=1)
        v2_full = v2_full[v2_full['wager_unit'] > 0]
        
        # Sort & Cap
        v2_full = v2_full.sort_values(['pick_date', 'edge'], ascending=[True, False])
        
        if not v2_full.empty:
            def apply_daily_logic(group):
                total_risk = group['wager_unit'].sum()
                if total_risk > 10.0:
                    scale = 10.0 / total_risk
                    group['wager_unit'] *= scale
                group['wager_unit'] = group['wager_unit'].apply(lambda x: round(x, 1))
                return group[group['wager_unit'] > 0]

            v2_full = v2_full.groupby('pick_date').apply(apply_daily_logic).reset_index(drop=True)
            
            # Now calculate profit ONLY for closed
            v2_full['profit'] = np.where(
                v2_full['outcome'].isin([1.0]), 
                v2_full['wager_unit'] * (v2_full['decimal_odds'] - 1), 
                np.where(v2_full['outcome'].isin([0.0]), -v2_full['wager_unit'], 0.0)
            )
        else:
            v2_full['profit'] = 0.0

        # Split back into V2 Active and V2 Closed
        v2_closed = v2_full[v2_full['outcome'].isin([0.0, 1.0])].copy()
        v2_active = v2_full[~v2_full['outcome'].isin([0.0, 1.0])].copy()

        # --- V1 LOGIC (Closed Only for Charts) ---
        # V1 is legacy, we only care about historical performance/readiness
        v1_df = closed_df[closed_df['pick_date'] >= self.V1_START].copy()
        v1_df['wager_unit'] = v1_df.apply(self._kelly_v1, axis=1)
        v1_df = v1_df[v1_df['wager_unit'] > 0]
        v1_df['profit'] = np.where(v1_df['outcome']==1, v1_df['wager_unit']*(v1_df['decimal_odds']-1), -v1_df['wager_unit'])
        
        self.generate_graphics(v1_df, v2_closed)
        self.generate_diamond_page(v2_active, v2_closed)
        self.update_readme(v1_df, v2_closed)

    def generate_diamond_page(self, active, closed):
        # 1. HEADLINE STATS
        total_profit = closed['profit'].sum()
        total_wager = closed['wager_unit'].sum()
        roi = (total_profit / total_wager) * 100 if total_wager > 0 else 0.0
        
        wins = len(closed[closed['profit'] > 0])
        losses = len(closed[closed['profit'] < 0])
        pushes = len(closed[closed['profit'] == 0])
        
        # 2. ACTIVE SIGNALS
        signals = []
        for _, row in active.iterrows():
            if row['wager_unit'] >= 2.0: strength = "STRONG"
            elif row['wager_unit'] >= 1.0: strength = "SOLID"
            else: strength = "LEAN"
            
            signals.append({
                "date": str(row['pick_date'].date()),
                "league": row['league_name'],
                "pick": row['pick_value'],
                "odds": float(row['odds_american']),
                "model_prob": round(float(row['prob_v2']), 3),
                "edge": round(float(row['edge']) * 100, 1),
                "units": float(row['wager_unit']),
                "strength": strength
            })

        # 3. HISTORY (Yesterday's Picks)
        history = []
        yesterday_stats = {"record": "0-0", "win_pct": 0, "roi": 0, "net": 0, "date": ""}
        if not closed.empty:
            # Get the most recent date with data (which is effectively "yesterday" since today's games aren't graded yet)
            latest_date = closed['pick_date'].max()
            yesterdays_picks = closed[closed['pick_date'] == latest_date].sort_values('profit', ascending=False)
            
            # Calculate yesterday's stats
            y_wins = len(yesterdays_picks[yesterdays_picks['profit'] > 0])
            y_losses = len(yesterdays_picks[yesterdays_picks['profit'] < 0])
            y_pushes = len(yesterdays_picks[yesterdays_picks['profit'] == 0])
            y_total_wager = yesterdays_picks['wager_unit'].sum()
            y_total_profit = yesterdays_picks['profit'].sum()
            y_roi = (y_total_profit / y_total_wager) * 100 if y_total_wager > 0 else 0
            y_win_pct = (y_wins / (y_wins + y_losses)) * 100 if (y_wins + y_losses) > 0 else 0
            
            yesterday_stats = {
                "record": f"{y_wins}-{y_losses}-{y_pushes}",
                "win_pct": round(y_win_pct, 1),
                "roi": round(y_roi, 1),
                "net": round(y_total_profit, 2),
                "date": str(latest_date.date())
            }
            
            for _, row in yesterdays_picks.iterrows():
                res = "WIN" if row['profit'] > 0 else "LOSS" if row['profit'] < 0 else "PUSH"
                history.append({
                    "date": str(row['pick_date'].date()),
                    "match": f"{row['league_name']} - {row['pick_value']}",
                    "odds": float(row['odds_american']),
                    "wager": float(row['wager_unit']),
                    "result": res,
                    "profit": round(float(row['profit']), 2)
                })
            
        data = {
            "meta": {
                "last_update": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC'),
                "status": "NOMINAL" if roi > -5 else "VOLATILITY DETECTED"
            },
            "stats": {
                "roi": round(roi, 1),
                "net_units": round(total_profit, 2),
                "record": f"{wins}-{losses}-{pushes}",
                "win_rate": round((wins / (wins+losses))*100, 1) if (wins+losses) > 0 else 0
            },
            "yesterday": yesterday_stats,
            "active_signals": signals,
            "history": history
        }
        
        # Also save JSON for API use
        with open('assets/diamond_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        # Generate complete HTML with embedded data
        html_content = self._get_diamond_html_template(data)
        with open('docs/diamond.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        print("✅ docs/diamond.html Updated.")
    
    def _get_diamond_html_template(self, data):
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THE QUARRY // DIAMOND MODEL</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;800&family=Space+Grotesk:wght@300;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        'void': '#050505',
                        'panel': '#0A0A0A',
                        'border-dim': '#1F1F1F',
                        'acid-lime': '#CCFF00',
                        'warning-orange': '#FF4D00',
                        'ghost': '#444444'
                    }},
                    fontFamily: {{
                        'sans': ['"Space Grotesk"', 'sans-serif'],
                        'mono': ['"JetBrains Mono"', 'monospace'],
                    }},
                    animation: {{
                        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        'ticker': 'ticker 30s linear infinite',
                        'slide-up-fade': 'slide-up-fade 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards',
                    }},
                    keyframes: {{
                        ticker: {{
                            '0%': {{ transform: 'translateX(0)' }},
                            '100%': {{ transform: 'translateX(-100%)' }},
                        }},
                        'slide-up-fade': {{
                            '0%': {{ opacity: '0', transform: 'translateY(10px)' }},
                            '100%': {{ opacity: '1', transform: 'translateY(0)' }},
                        }}
                    }}
                }}
            }}
        }}
    </script>
    <style>
        body {{ background-color: #050505; color: #E2E2E2; -webkit-font-smoothing: antialiased; }}
        .noise {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.03'/%3E%3C/svg%3E"); pointer-events: none; z-index: 50; }}
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: #0A0A0A; }}
        ::-webkit-scrollbar-thumb {{ background: #333; }}
        .glow-text {{ text-shadow: 0 0 10px rgba(204, 255, 0, 0.3); }}
        .chart-container {{ background: #0A0A0A; border: 1px solid #1F1F1F; border-radius: 4px; padding: 1rem; }}
        .chart-container img {{ width: 100%; height: auto; border-radius: 4px; }}
    </style>
</head>
<body class="min-h-screen relative overflow-x-hidden">
    <div class="noise"></div>
    <div class="fixed top-0 w-full bg-acid-lime text-black font-mono text-xs font-bold py-1 z-40 overflow-hidden whitespace-nowrap border-b border-acid-lime">
        <div class="inline-block animate-ticker">
            THE QUARRY DIAMOND ONLINE // LAST UPDATE: {data["meta"]["last_update"]} // ROI: {data["stats"]["roi"]}% // RECORD: {data["stats"]["record"]} // ALPHA: {data["stats"]["net_units"]}u // STATUS: {data["meta"]["status"]} // 
            THE QUARRY DIAMOND ONLINE // LAST UPDATE: {data["meta"]["last_update"]} // ROI: {data["stats"]["roi"]}% // RECORD: {data["stats"]["record"]} // ALPHA: {data["stats"]["net_units"]}u // STATUS: {data["meta"]["status"]} // 
        </div>
    </div>
    <div class="max-w-7xl mx-auto pt-16 pb-20 px-4 sm:px-6">
        <header class="mb-12 flex flex-col md:flex-row justify-between items-end border-b border-border-dim pb-6 opacity-0 animate-slide-up-fade" style="animation-delay: 0ms;">
            <div>
                <div class="flex items-center gap-2 mb-2">
                    <div class="w-3 h-3 bg-acid-lime rounded-full animate-pulse-slow"></div>
                    <span class="font-mono text-xs text-acid-lime tracking-widest">LIVE MONITORING // DIAMOND</span>
                </div>
                <h1 class="text-5xl md:text-7xl font-bold tracking-tighter text-white">THE <span class="text-ghost">QUARRY</span></h1>
                <p class="font-mono text-zinc-500 mt-2 text-sm uppercase tracking-wide">XGBoost Algorithmic Infrastructure // V2 DIAMOND</p>
            </div>
            <div class="text-right mt-6 md:mt-0">
                <div class="font-mono text-4xl font-bold {"text-acid-lime" if data["stats"]["net_units"] >= 0 else "text-warning-orange"} glow-text">{'+' if data["stats"]["net_units"] >= 0 else ''}{data["stats"]["net_units"]}u</div>
                <div class="font-mono text-xs text-zinc-500 uppercase tracking-widest">Net Alpha Generated</div>
            </div>
        </header>

        <!-- BENTO GRID STATS -->
        <section class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-16">
            <!-- ROI Card -->
            <div class="bg-panel border border-border-dim p-6 flex flex-col justify-between h-32 hover:border-acid-lime transition-all duration-300 group cursor-default hover:translate-y-[-2px] hover:shadow-[0_4px_20px_rgba(204,255,0,0.1)] opacity-0 animate-slide-up-fade" style="animation-delay: 100ms;">
                <div class="flex justify-between items-start">
                    <span class="font-mono text-xs text-zinc-500 uppercase group-hover:text-zinc-400 transition-colors">Return on Investment</span>
                    <svg class="w-4 h-4 text-acid-lime opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                </div>
                <div class="text-3xl font-bold {"text-acid-lime" if data["stats"]["roi"] >= 0 else "text-warning-orange"} group-hover:text-acid-lime transition-colors">{data["stats"]["roi"]}%</div>
            </div>
            <!-- Win Rate Card -->
            <div class="bg-panel border border-border-dim p-6 flex flex-col justify-between h-32 hover:border-acid-lime transition-all duration-300 group cursor-default hover:translate-y-[-2px] hover:shadow-[0_4px_20px_rgba(204,255,0,0.1)] opacity-0 animate-slide-up-fade" style="animation-delay: 200ms;">
                <div class="flex justify-between items-start">
                    <span class="font-mono text-xs text-zinc-500 uppercase group-hover:text-zinc-400 transition-colors">Win Rate</span>
                    <svg class="w-4 h-4 text-acid-lime opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </div>
                <div class="flex items-end gap-3">
                    <div class="text-3xl font-bold text-white group-hover:text-acid-lime transition-colors">{data["stats"]["win_rate"]}%</div>
                    <div class="h-1 flex-1 bg-zinc-800 mb-2 rounded-full overflow-hidden">
                        <div class="h-full bg-white group-hover:bg-acid-lime transition-colors" style="width: {data["stats"]["win_rate"]}%"></div>
                    </div>
                </div>
            </div>
            <!-- Record Card -->
            <div class="bg-panel border border-border-dim p-6 flex flex-col justify-between h-32 hover:border-acid-lime transition-all duration-300 group cursor-default hover:translate-y-[-2px] hover:shadow-[0_4px_20px_rgba(204,255,0,0.1)] opacity-0 animate-slide-up-fade" style="animation-delay: 300ms;">
                <div class="flex justify-between items-start">
                    <span class="font-mono text-xs text-zinc-500 uppercase group-hover:text-zinc-400 transition-colors">W - L - T</span>
                    <svg class="w-4 h-4 text-acid-lime opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                </div>
                <div class="text-3xl font-mono text-zinc-300 group-hover:text-white transition-colors">{data["stats"]["record"].replace("-", '<span class="text-zinc-600 mx-1">/</span>')}</div>
            </div>
            <!-- System Status Card -->
            <div class="bg-panel border border-border-dim p-6 flex flex-col justify-between h-32 relative overflow-hidden hover:border-acid-lime transition-all duration-300 group cursor-default hover:translate-y-[-2px] hover:shadow-[0_4px_20px_rgba(204,255,0,0.1)] opacity-0 animate-slide-up-fade" style="animation-delay: 400ms;">
                <div class="flex justify-between items-start z-10 relative">
                    <span class="font-mono text-xs text-zinc-500 uppercase group-hover:text-zinc-400 transition-colors">System Status</span>
                    <svg class="w-4 h-4 text-acid-lime opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"></path></svg>
                </div>
                <div class="text-xl font-bold {"text-acid-lime" if data["meta"]["status"] == "NOMINAL" else "text-warning-orange"} z-10 relative group-hover:text-acid-lime transition-colors">{data["meta"]["status"]}</div>
                <svg class="absolute bottom-0 left-0 w-full h-16 text-zinc-900 group-hover:text-zinc-800 transition-colors z-0" fill="currentColor" viewBox="0 0 1440 320" preserveAspectRatio="none"><path fill-opacity="1" d="M0,224L48,213.3C96,203,192,181,288,181.3C384,181,480,203,576,224C672,245,768,267,864,250.7C960,235,1056,181,1152,165.3C1248,149,1344,171,1392,181.3L1440,192L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path></svg>
            </div>
        </section>

        <!-- LIVE PROFIT CURVE -->
        <section class="mb-16">
            <h2 class="font-sans text-2xl font-bold text-white tracking-tight mb-2">CUMULATIVE PROFIT CURVE</h2>
            <p class="font-mono text-zinc-500 text-sm mb-6">Real-time performance tracking: Diamond (Green) vs Pyrite (Red) vs Market Baseline (Gray)</p>
            <div class="chart-container group hover:border-acid-lime/50 transition-all duration-300 hover:translate-y-[-2px] hover:shadow-[0_4px_30px_rgba(204,255,0,0.08)] opacity-0 animate-slide-up-fade" style="animation-delay: 500ms;">
                <img src="../assets/live_curve.png" alt="Live Profit Curve" class="group-hover:brightness-110 transition-all duration-300" />
            </div>
        </section>

        <!-- MODEL COMPARISON -->
        <section class="mb-16">
            <h2 class="font-sans text-2xl font-bold text-white tracking-tight mb-2">MODEL COMPARISON</h2>
            <p class="font-mono text-zinc-500 text-sm mb-6">Head-to-head analysis of the two algorithmic strategies</p>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <!-- PYRITE CARD -->
                <div class="bg-panel border border-warning-orange/30 p-6 rounded group hover:border-warning-orange transition-all duration-300 cursor-default hover:translate-y-[-2px] hover:shadow-[0_4px_30px_rgba(255,77,0,0.15)] opacity-0 animate-slide-up-fade" style="animation-delay: 600ms;">
                    <div class="flex items-center gap-3 mb-4">
                        <div class="w-10 h-10 bg-warning-orange/20 rounded flex items-center justify-center group-hover:bg-warning-orange/30 transition-colors">
                            <span class="text-xl group-hover:scale-110 transition-transform inline-block">🔸</span>
                        </div>
                        <div>
                            <h3 class="text-xl font-bold text-warning-orange">V1 PYRITE</h3>
                            <span class="font-mono text-xs text-zinc-500 group-hover:text-zinc-400 transition-colors">LEGACY // HIGH VARIANCE</span>
                        </div>
                    </div>
                    <ul class="space-y-2 font-mono text-sm text-zinc-400">
                        <li class="flex justify-between group-hover:text-zinc-300 transition-colors"><span>Philosophy</span><span class="text-zinc-300">"Bet everything with &gt;50% edge"</span></li>
                        <li class="flex justify-between group-hover:text-zinc-300 transition-colors"><span>Volume</span><span class="text-zinc-300">High (~15+ bets/day)</span></li>
                        <li class="flex justify-between group-hover:text-zinc-300 transition-colors"><span>Risk Profile</span><span class="text-warning-orange">Reckless / Uncapped</span></li>
                        <li class="flex justify-between group-hover:text-zinc-300 transition-colors"><span>Key Flaw</span><span class="text-warning-orange">Overconfidence on Favorites</span></li>
                    </ul>
                </div>
                
                <!-- DIAMOND CARD -->
                <div class="bg-panel border border-acid-lime/30 p-6 rounded group hover:border-acid-lime transition-all duration-300 cursor-default hover:translate-y-[-2px] hover:shadow-[0_4px_30px_rgba(204,255,0,0.15)] opacity-0 animate-slide-up-fade" style="animation-delay: 700ms;">
                    <div class="flex items-center gap-3 mb-4">
                        <div class="w-10 h-10 bg-acid-lime/20 rounded flex items-center justify-center group-hover:bg-acid-lime/30 transition-colors">
                            <span class="text-xl group-hover:scale-110 transition-transform inline-block">💎</span>
                        </div>
                        <div>
                            <h3 class="text-xl font-bold text-acid-lime">V2 DIAMOND</h3>
                            <span class="font-mono text-xs text-zinc-500 group-hover:text-zinc-400 transition-colors">ACTIVE // LOW VARIANCE</span>
                        </div>
                    </div>
                    <ul class="space-y-2 font-mono text-sm text-zinc-400">
                        <li class="flex justify-between group-hover:text-zinc-300 transition-colors"><span>Philosophy</span><span class="text-zinc-300">"Snipe specific inefficiencies"</span></li>
                        <li class="flex justify-between group-hover:text-zinc-300 transition-colors"><span>Volume</span><span class="text-zinc-300">Low (~5 bets/day)</span></li>
                        <li class="flex justify-between group-hover:text-zinc-300 transition-colors"><span>Risk Profile</span><span class="text-acid-lime">10u Daily Cap / Scaled Kelly</span></li>
                        <li class="flex justify-between group-hover:text-zinc-300 transition-colors"><span>Key Edge</span><span class="text-acid-lime">Regime Filtering + Value Floor</span></li>
                    </ul>
                </div>
            </div>
        </section>

        <!-- SPORT ROI ANALYSIS -->
        <section class="mb-16">
            <h2 class="font-sans text-2xl font-bold text-white tracking-tight mb-2">SPORT ROI ANALYSIS</h2>
            <p class="font-mono text-zinc-500 text-sm mb-6">Identifying toxic assets vs profitable regimes. Diamond filters out high-variance losing sports.</p>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="chart-container group hover:border-warning-orange/50 transition-all duration-300 hover:translate-y-[-2px] hover:shadow-[0_4px_30px_rgba(255,77,0,0.08)] opacity-0 animate-slide-up-fade" style="animation-delay: 800ms;">
                    <div class="flex items-center gap-2 mb-3">
                        <span class="text-warning-orange text-lg group-hover:scale-110 transition-transform inline-block">🔸</span>
                        <span class="font-mono text-sm text-zinc-400 group-hover:text-zinc-300 transition-colors">V1 Pyrite - Bleeds on NFL/MLB</span>
                    </div>
                    <img src="../assets/pyrite_sport.png" alt="Pyrite Sport ROI" class="group-hover:brightness-110 transition-all duration-300" />
                </div>
                <div class="chart-container group hover:border-acid-lime/50 transition-all duration-300 hover:translate-y-[-2px] hover:shadow-[0_4px_30px_rgba(204,255,0,0.08)] opacity-0 animate-slide-up-fade" style="animation-delay: 900ms;">
                    <div class="flex items-center gap-2 mb-3">
                        <span class="text-acid-lime text-lg group-hover:scale-110 transition-transform inline-block">💎</span>
                        <span class="font-mono text-sm text-zinc-400 group-hover:text-zinc-300 transition-colors">V2 Diamond - Surgical Regime Selection</span>
                    </div>
                    <img src="../assets/diamond_sport.png" alt="Diamond Sport ROI" class="group-hover:brightness-110 transition-all duration-300" />
                </div>
            </div>
        </section>

        <!-- BET SIZING CALIBRATION -->
        <section class="mb-16">
            <h2 class="font-sans text-2xl font-bold text-white tracking-tight mb-2">BET SIZING CALIBRATION</h2>
            <p class="font-mono text-zinc-500 text-sm mb-6">Does the model know when it's confident? Bigger bets should yield higher ROI.</p>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="chart-container group hover:border-warning-orange/50 transition-all duration-300 hover:translate-y-[-2px] hover:shadow-[0_4px_30px_rgba(255,77,0,0.08)] opacity-0 animate-slide-up-fade" style="animation-delay: 1000ms;">
                    <div class="flex items-center gap-2 mb-3">
                        <span class="text-warning-orange text-lg group-hover:scale-110 transition-transform inline-block">🔸</span>
                        <span class="font-mono text-sm text-zinc-400 group-hover:text-zinc-300 transition-colors">V1 Pyrite - Flat performance across sizes</span>
                    </div>
                    <img src="../assets/pyrite_size.png" alt="Pyrite Bet Sizing" class="group-hover:brightness-110 transition-all duration-300" />
                </div>
                <div class="chart-container group hover:border-acid-lime/50 transition-all duration-300 hover:translate-y-[-2px] hover:shadow-[0_4px_30px_rgba(204,255,0,0.08)] opacity-0 animate-slide-up-fade" style="animation-delay: 1100ms;">
                    <div class="flex items-center gap-2 mb-3">
                        <span class="text-acid-lime text-lg group-hover:scale-110 transition-transform inline-block">💎</span>
                        <span class="font-mono text-sm text-zinc-400 group-hover:text-zinc-300 transition-colors">V2 Diamond - Confidence = Profit</span>
                    </div>
                    <img src="../assets/diamond_size.png" alt="Diamond Bet Sizing" class="group-hover:brightness-110 transition-all duration-300" />
                </div>
            </div>
        </section>

        <!-- METHODOLOGY -->
        <section class="mb-16 border border-border-dim p-8 bg-panel hover:border-acid-lime/30 transition-all duration-300 hover:shadow-[0_4px_30px_rgba(204,255,0,0.05)]">
            <h2 class="font-sans text-2xl font-bold text-white tracking-tight mb-4 opacity-0 animate-slide-up-fade" style="animation-delay: 1200ms;">DIAMOND METHODOLOGY</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="border-l-2 border-acid-lime pl-4 group hover:border-l-4 transition-all duration-300 cursor-default opacity-0 animate-slide-up-fade" style="animation-delay: 1300ms;">
                    <h3 class="font-mono text-acid-lime text-sm uppercase mb-2 group-hover:text-white transition-colors">The Value Floor</h3>
                    <p class="text-zinc-400 text-sm group-hover:text-zinc-300 transition-colors">Hard rejection of odds worse than <span class="text-white font-bold">-140</span>. Forces the model to find true underdogs, not expensive favorites.</p>
                </div>
                <div class="border-l-2 border-acid-lime pl-4 group hover:border-l-4 transition-all duration-300 cursor-default opacity-0 animate-slide-up-fade" style="animation-delay: 1400ms;">
                    <h3 class="font-mono text-acid-lime text-sm uppercase mb-2 group-hover:text-white transition-colors">Regime Filtering</h3>
                    <p class="text-zinc-400 text-sm group-hover:text-zinc-300 transition-colors">The "Core 4" (NBA, NCAAB, NHL, UFC) are allowed. NFL, MLB, Tennis, Soccer are <span class="text-warning-orange font-bold">blacklisted</span>.</p>
                </div>
                <div class="border-l-2 border-acid-lime pl-4 group hover:border-l-4 transition-all duration-300 cursor-default opacity-0 animate-slide-up-fade" style="animation-delay: 1500ms;">
                    <h3 class="font-mono text-acid-lime text-sm uppercase mb-2 group-hover:text-white transition-colors">Bankroll Governor</h3>
                    <p class="text-zinc-400 text-sm group-hover:text-zinc-300 transition-colors">Max <span class="text-white font-bold">3u</span> per bet. Max <span class="text-white font-bold">10u</span> daily exposure. Proportional scaling on busy days.</p>
                </div>
            </div>
        </section>

        <!-- GRADED EXECUTION LOG -->
        <section class="border-t border-border-dim pt-12">
            <h2 class="font-sans text-xl font-bold text-zinc-400 mb-6 tracking-tight opacity-0 animate-slide-up-fade" style="animation-delay: 1600ms;">YESTERDAY'S ACTION</h2>
            <div class="overflow-x-auto opacity-0 animate-slide-up-fade" style="animation-delay: 1700ms;">
                <table class="w-full font-mono text-sm text-left">
                    <thead class="text-xs text-zinc-600 uppercase border-b border-border-dim">
                        <tr>
                            <th class="py-3 pl-4">Date</th>
                            <th class="py-3">Matchup</th>
                            <th class="py-3 text-right">Odds</th>
                            <th class="py-3 pl-8">Wager</th>
                            <th class="py-3 text-right">Result</th>
                            <th class="py-3 text-right pr-4">P&amp;L</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-border-dim text-zinc-400" id="ledger-body"></tbody>
                </table>
            </div>
            
            <!-- YESTERDAY SUMMARY -->
            <div class="mt-8 mb-4 flex items-center justify-between opacity-0 animate-slide-up-fade" style="animation-delay: 1800ms;">
                <h3 class="font-mono text-sm text-zinc-500 uppercase tracking-wide">📊 Daily Summary</h3>
                <span class="font-mono text-xs text-acid-lime" id="y-date-label">--</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="yesterday-summary">
                <div class="bg-panel border border-border-dim p-4 group hover:border-acid-lime/50 transition-all duration-300 opacity-0 animate-slide-up-fade" style="animation-delay: 1900ms;">
                    <div class="font-mono text-xs text-zinc-500 uppercase mb-1">Record</div>
                    <div class="text-2xl font-bold text-white" id="y-record">--</div>
                </div>
                <div class="bg-panel border border-border-dim p-4 group hover:border-acid-lime/50 transition-all duration-300 opacity-0 animate-slide-up-fade" style="animation-delay: 2000ms;">
                    <div class="font-mono text-xs text-zinc-500 uppercase mb-1">Win Rate</div>
                    <div class="text-2xl font-bold text-white" id="y-winpct">--%</div>
                </div>
                <div class="bg-panel border border-border-dim p-4 group hover:border-acid-lime/50 transition-all duration-300 opacity-0 animate-slide-up-fade" style="animation-delay: 2100ms;">
                    <div class="font-mono text-xs text-zinc-500 uppercase mb-1">ROI</div>
                    <div class="text-2xl font-bold" id="y-roi">--%</div>
                </div>
                <div class="bg-panel border border-border-dim p-4 group hover:border-acid-lime/50 transition-all duration-300 opacity-0 animate-slide-up-fade" style="animation-delay: 2200ms;">
                    <div class="font-mono text-xs text-zinc-500 uppercase mb-1">Net Profit</div>
                    <div class="text-2xl font-bold" id="y-net">--u</div>
                </div>
            </div>
        </section>

        <footer class="mt-20 border-t border-border-dim pt-8 text-center md:text-left flex flex-col md:flex-row justify-between items-center text-zinc-600 text-xs font-mono">
            <p>THE QUARRY © 2025. DESIGNED BY THE VOID.</p>
            <p class="mt-2 md:mt-0">PAST PERFORMANCE IS NOT INDICATIVE OF FUTURE ALPHA.</p>
        </footer>
    </div>
    <script>
        const DATA = {json.dumps(data)};
        
        function renderLedger() {{
            const ledgerBody = document.getElementById('ledger-body');
            DATA.history.forEach(row => {{
                const isWin = row.result === 'WIN';
                const profitClass = row.profit > 0 ? 'text-acid-lime' : (row.profit < 0 ? 'text-warning-orange' : 'text-zinc-500');
                const html = `
                    <tr class="hover:bg-zinc-900/70 transition-all duration-200 cursor-default group ${{isWin ? 'hover:border-l-2 hover:border-l-acid-lime' : 'hover:border-l-2 hover:border-l-warning-orange'}}">
                        <td class="py-3 pl-4 border-b border-zinc-900 group-hover:text-white transition-colors">${{row.date}}</td>
                        <td class="py-3 border-b border-zinc-900 font-bold text-zinc-300 group-hover:text-white transition-colors">${{row.match}}</td>
                        <td class="py-3 text-right border-b border-zinc-900 text-zinc-500 group-hover:text-zinc-300 transition-colors">${{row.odds}}</td>
                        <td class="py-3 pl-8 border-b border-zinc-900"><span class="bg-zinc-800 text-white px-2 py-0.5 text-xs font-bold group-hover:bg-zinc-700 transition-colors">${{row.wager}}u</span></td>
                        <td class="py-3 text-right border-b border-zinc-900">
                            <span class="${{isWin ? 'text-acid-lime' : 'text-warning-orange'}} font-bold">${{row.result}}</span>
                        </td>
                        <td class="py-3 text-right pr-4 border-b border-zinc-900 ${{profitClass}} font-bold">${{row.profit > 0 ? '+' : ''}}${{row.profit}}u</td>
                    </tr>
                `;
                ledgerBody.innerHTML += html;
            }});
            
            // Render yesterday summary
            const y = DATA.yesterday;
            document.getElementById('y-date-label').textContent = y.date ? 'Results for ' + y.date : '';
            document.getElementById('y-record').textContent = y.record;
            document.getElementById('y-winpct').textContent = y.win_pct + '%';
            
            const roiEl = document.getElementById('y-roi');
            roiEl.textContent = (y.roi >= 0 ? '+' : '') + y.roi + '%';
            roiEl.className = 'text-2xl font-bold ' + (y.roi >= 0 ? 'text-acid-lime' : 'text-warning-orange');
            
            const netEl = document.getElementById('y-net');
            netEl.textContent = (y.net >= 0 ? '+' : '') + y.net + 'u';
            netEl.className = 'text-2xl font-bold ' + (y.net >= 0 ? 'text-acid-lime' : 'text-warning-orange');
        }}
        renderLedger();
    </script>
</body>
</html>'''

    def generate_graphics(self, v1, v2):
        # Protocol 705 Color Palette
        VOID = '#050505'
        PANEL = '#0A0A0A'
        BORDER = '#1F1F1F'
        ACID_LIME = '#CCFF00'
        WARNING_ORANGE = '#FF4D00'
        GHOST = '#444444'
        TEXT = '#E2E2E2'
        
        # Set matplotlib style
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': VOID,
            'axes.facecolor': PANEL,
            'axes.edgecolor': BORDER,
            'axes.labelcolor': TEXT,
            'axes.titlecolor': TEXT,
            'xtick.color': GHOST,
            'ytick.color': GHOST,
            'text.color': TEXT,
            'legend.facecolor': PANEL,
            'legend.edgecolor': BORDER,
            'grid.color': BORDER,
            'grid.alpha': 0.3,
            'font.family': 'monospace',
            'font.size': 10,
        })
        
        # 1. PROFIT CURVE
        def get_cum(d):
            if d.empty: return pd.DataFrame({'pick_date':[], 'profit':[]})
            daily = d.groupby('pick_date')['profit'].sum().cumsum().reset_index()
            start = daily['pick_date'].min() - pd.Timedelta(days=1)
            return pd.concat([pd.DataFrame({'pick_date':[start], 'profit':[0.0]}), daily])

        d1 = get_cum(v1)
        d2 = get_cum(v2)
        
        # Market Consensus Baseline
        market_df = self.df[self.df['pick_date'] >= self.V1_START].copy()
        market_df = market_df[market_df['outcome'].isin([0.0, 1.0])]
        market_df['profit'] = np.where(market_df['outcome'] == 1, market_df['decimal_odds'] - 1, -1)
        d_market = get_cum(market_df)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor(VOID)
        ax.set_facecolor(PANEL)
        
        # Plot lines
        ax.plot(d_market['pick_date'], d_market['profit'], color=GHOST, linestyle=':', linewidth=1.5, label='Market Baseline', alpha=0.5)
        ax.plot(d1['pick_date'], d1['profit'], color=WARNING_ORANGE, linewidth=2, label='🔸 V1 Pyrite', alpha=0.8)
        ax.plot(d2['pick_date'], d2['profit'], color=ACID_LIME, linewidth=3, label='💎 V2 Diamond')
        
        # Fill under Diamond curve for emphasis
        if not d2.empty:
            ax.fill_between(d2['pick_date'], 0, d2['profit'], color=ACID_LIME, alpha=0.1)
        
        # Zero line
        ax.axhline(0, color=GHOST, linewidth=0.5, linestyle='--')
        
        if not d1.empty and not d2.empty:
            all_dates = pd.concat([d1['pick_date'], d2['pick_date']])
            ax.set_xlim(all_dates.min(), all_dates.max())
        
        ax.set_title("CUMULATIVE PROFIT // PYRITE vs DIAMOND", fontweight='bold', fontsize=14, pad=15)
        ax.set_ylabel("UNITS", fontsize=10)
        ax.set_xlabel("")
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('assets/live_curve.png', dpi=150, facecolor=VOID, edgecolor='none')
        plt.close()
        
        # 2. SPORT ROI (Horizontal Bars)
        for name, data, accent in [('pyrite', v1, WARNING_ORANGE), ('diamond', v2, ACID_LIME)]:
            if data.empty: continue
            s = data.groupby('league_name').agg({'profit':'sum', 'wager_unit':'sum'})
            s['roi'] = (s['profit'] / s['wager_unit']) * 100  # Convert to percentage
            s = s.sort_values('roi', ascending=True)  # Ascending for horizontal bars
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(VOID)
            ax.set_facecolor(PANEL)
            
            colors = [ACID_LIME if x > 0 else WARNING_ORANGE for x in s['roi']]
            
            bars = ax.barh(s.index, s['roi'], color=colors, edgecolor=BORDER, linewidth=0.5)
            
            # Add value labels
            for bar, val in zip(bars, s['roi']):
                x_pos = bar.get_width() + (2 if val >= 0 else -2)
                ha = 'left' if val >= 0 else 'right'
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                       va='center', ha=ha, fontsize=9, color=TEXT, fontweight='bold')
            
            ax.axvline(0, color=GHOST, linewidth=1)
            ax.set_title(f"{'🔸 V1 PYRITE' if name=='pyrite' else '💎 V2 DIAMOND'} // ROI BY SPORT", 
                        fontweight='bold', fontsize=12, pad=15)
            ax.set_xlabel("ROI %", fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(BORDER)
            ax.spines['left'].set_color(BORDER)
            
            plt.tight_layout()
            plt.savefig(f'assets/{name}_sport.png', dpi=150, facecolor=VOID, edgecolor='none')
            plt.close()
            
        # 3. BET SIZING ROI
        for name, data, accent in [('pyrite', v1, WARNING_ORANGE), ('diamond', v2, ACID_LIME)]:
            if data.empty: continue
            
            active_bets = data[data['wager_unit'] > 0].copy()
            if len(active_bets) < 10: continue

            try:
                active_bets['size_bucket'] = pd.qcut(active_bets['wager_unit'], q=4, duplicates='drop')
            except:
                active_bets['size_bucket'] = 'Flat'

            sz = active_bets.groupby('size_bucket', observed=False).agg({
                'profit': 'sum', 
                'wager_unit': 'sum'
            })
            
            sz['roi'] = (sz['profit'] / sz['wager_unit']) * 100
            sz.index = sz.index.astype(str).str.replace('(', '').str.replace(']', '').str.replace(', ', '-') + 'u'
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(VOID)
            ax.set_facecolor(PANEL)
            
            colors = [ACID_LIME if x > 0 else WARNING_ORANGE for x in sz['roi']]
            
            bars = ax.bar(sz.index, sz['roi'], color=colors, edgecolor=BORDER, linewidth=0.5, width=0.6)
            
            # Add value labels on top
            for bar, val in zip(bars, sz['roi']):
                y_pos = bar.get_height() + (2 if val >= 0 else -5)
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.1f}%', 
                       ha='center', fontsize=10, color=TEXT, fontweight='bold')
            
            ax.axhline(0, color=GHOST, linewidth=1)
            ax.set_title(f"{'🔸 V1 PYRITE' if name=='pyrite' else '💎 V2 DIAMOND'} // ROI BY BET SIZE", 
                        fontweight='bold', fontsize=12, pad=15)
            ax.set_ylabel("ROI %", fontsize=10)
            ax.set_xlabel("BET SIZE RANGE", fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(BORDER)
            ax.spines['left'].set_color(BORDER)
            
            plt.tight_layout()
            plt.savefig(f'assets/{name}_size.png', dpi=150, facecolor=VOID, edgecolor='none')
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
        def get_stats(d):
            if d.empty: return 0, 0, 0
            p = d['profit'].sum()
            r = d['wager_unit'].sum()
            roi = p/r if r>0 else 0
            wr = len(d[d['outcome']==1]) / len(d) if len(d) > 0 else 0
            return p, roi, wr

        p1, r1, w1 = get_stats(v1)
        p2, r2, w2 = get_stats(v2)
        
        vol_v1 = self._get_volume_text(v1)
        vol_v2 = self._get_volume_text(v2)
        
        last_date = max(v1['pick_date'].max(), v2['pick_date'].max()) if not v1.empty and not v2.empty else pd.Timestamp.now()
        y_v1 = v1[v1['pick_date'] == last_date].copy() if not v1.empty else pd.DataFrame()
        y_v2 = v2[v2['pick_date'] == last_date].copy() if not v2.empty else pd.DataFrame()
        
        def make_table(df, title):
            if df.empty: return ""
            t = f"### {title}\n"
            t += "| LEAGUE | PICK | ODDS | UNIT | RES | PROFIT |\n"
            t += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            for _, row in df.iterrows():
                res = "✅" if row['outcome']==1 else "❌"
                odds = f"+{int(row['odds_american'])}" if row['odds_american'] > 0 else f"{int(row['odds_american'])}"
                t += f"| {row['league_name']} | {row['pick_value']} | {odds} | {row['wager_unit']:.1f} | {res} | {row['profit']:+.2f}u |\n"
            t += f"**Daily PnL: {df['profit'].sum():+.2f} Units**\n\n"
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
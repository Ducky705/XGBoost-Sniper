import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import re
import matplotlib.dates as mdates
import sys

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.pipeline import SportsDataPipeline, FeatureEngineer
from src.models import ModelSimulator

# ==========================================
# CONFIGURATION
# ==========================================
# Ensure we are in the project root
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir('..')

plt.style.use('dark_background')
os.makedirs('assets', exist_ok=True)
os.makedirs('docs/assets', exist_ok=True)

COLORS = {
    'void': '#050505',
    'obsidian': '#7c3aed',
    'diamond': '#00E0FF',
    'pyrite': '#FFC125', # Fool's Gold
    'ghost': '#444444',
    'text': '#E5E7EB',
    'grid': '#1A1A1A',
    'loss': '#FF4D00' # Safety Orange for losses
}

# ==========================================
# I. SYNTHETIC ASSETS (Methodology Docs)
# ==========================================
def generate_synthetic_assets():
    print("🧪 Generating Synthetic Assets for Methodology...")
    
    def generate_synthetic_data(n_rows=1000):
        np.random.seed(42)
        dates = pd.date_range(start='2025-11-01', periods=n_rows, freq='H')
        leagues = ['NBA', 'NCAAB', 'NFL', 'NCAAF', 'NHL', 'UFC', 'MLB', 'TENNIS']
        league_choices = np.random.choice(leagues, n_rows)
        base_probs = {'NBA': 0.55, 'NCAAB': 0.54, 'NFL': 0.48, 'NCAAF': 0.52, 'NHL': 0.53, 'UFC': 0.60, 'MLB': 0.45, 'TENNIS': 0.45}
        
        outcomes, odds, confidences = [], [], []
        for lg in league_choices:
            win_prob = base_probs.get(lg, 0.50)
            outcomes.append(1.0 if np.random.random() < win_prob else 0.0)
            odds.append(np.random.choice([-110, -120, -130, -140, 100, 110, 120]))
            conf = 0.50 + (np.random.random() * 0.15)
            if outcomes[-1] == 1.0: conf += 0.02
            confidences.append(conf)
            
        df = pd.DataFrame({'pick_date': dates, 'league_name': league_choices, 'outcome': outcomes, 'odds_american': odds, 'ai_confidence': confidences, 'capper_experience': np.random.randint(0, 50, n_rows)})
        df['decimal_odds'] = df['odds_american'].apply(lambda o: (o/100)+1 if o>0 else (100/abs(o))+1)
        df['implied_prob'] = 1 / df['decimal_odds']
        df['edge'] = df['ai_confidence'] - df['implied_prob']
        return df

    df = generate_synthetic_data(1000)
    df['cum_market'] = np.where(df['outcome'] == 1, df['decimal_odds'] - 1, -1).cumsum()
    df['v1_profit'] = np.where(df['edge'] > 0, np.where(df['outcome']==1, 2.0*(df['decimal_odds']-1), -2.0), 0)
    df['cum_v1'] = df['v1_profit'].cumsum()
    
    # Fig 1: Initial Failure
    plt.figure(figsize=(10, 5))
    plt.plot(df['pick_date'], df['cum_market'], color='gray', linestyle='--', label='Market Baseline')
    plt.plot(df['pick_date'], df['cum_v1'], color=COLORS['pyrite'], label='Pyrite Model')
    plt.title("Figure 1: The Initial Failure (October Crash)", color='white')
    plt.legend()
    plt.savefig('assets/figure_1_initial_failure.png')
    plt.close()

    # Fig 2: Calibration Failure
    plt.figure(figsize=(8, 5))
    x = ['50-55%', '55-60%', '60-65%', '65%+']
    y = [0.52, 0.56, 0.45, 0.30]
    sns.barplot(x=x, y=y, palette='magma')
    plt.title("Figure 2: The 'Fake Lock' Syndrome", color='white')
    plt.ylim(0, 0.7)
    plt.axhline(0.5, color='white', linestyle='--')
    plt.savefig('assets/figure_2_calibration_failure.png')
    plt.close()

    # Fig 3: Feature Importance
    feats = ['Consensus', 'Volatility', 'ROI (7D)', 'Implied Prob', 'Experience']
    imps = [0.35, 0.25, 0.20, 0.15, 0.05]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=imps, y=feats, palette='cool')
    plt.title("Figure 3: Feature Importance", color='white')
    plt.savefig('assets/figure_3_feature_importance.png')
    plt.close()

    # Fig 4: Winning Formula DNA
    categories = ['Stability (Low Vol)', 'Consensus', 'Value (Odds)', 'Experience', 'Recent ROI']
    values = [0.9, 0.8, 0.4, 0.2, 0.1]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', color=COLORS['pyrite'])
    ax.fill(angles, values, COLORS['pyrite'], alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', fontsize=10, fontweight='bold')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([])
    ax.set_title("Figure 4: The Pyrite DNA (Stability + Consensus)", color='white', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('assets/figure_4_winning_formula_dna.png')
    plt.close()
    
    # Strat Heatmap (Synthetic)
    plt.figure(figsize=(8, 6))
    data = np.array([[0.05, 0.10, 0.15], [0.02, 0.08, 0.25], [-0.05, 0.01, 0.12]])
    sns.heatmap(data, annot=True, fmt='.0%', cmap='RdYlGn', xticklabels=['3%', '5%', '7%'], yticklabels=['0', '10', '20'])
    plt.title("Figure 2: Strategy Heatmap (Exp vs Edge)", color='white')
    plt.xlabel("Min Edge")
    plt.ylabel("Min Experience")
    plt.savefig('assets/v2_fig2_heatmap.png')
    plt.close()

# ==========================================
# II. LIVE ASSETS (Dashboards)
# ==========================================
def generate_live_assets():
    print("🚀 Generating Live Assets from Supabase...")
    
    pipeline = SportsDataPipeline()
    raw_df = pipeline.fetch_data()
    if raw_df.empty:
        print("❌ No data found.")
        return

    eng = FeatureEngineer(raw_df)
    df = eng.process()
    sim = ModelSimulator(df)
    
    v1 = sim.run_v1_pyrite()
    if not v1.empty: v1['edge'] = v1['prob'] - v1['implied_prob']
    v2 = sim.run_v2_diamond()
    v3 = sim.run_v3_obsidian()
    
    # --- 1. PLOTS ---
    # cumulative profit
    def get_cum(d):
        if d.empty: return pd.DataFrame({'pick_date':[], 'profit':[]})
        daily = d.groupby('pick_date')['profit_actual'].sum().cumsum().reset_index()
        daily.columns = ['pick_date', 'profit']
        start_node = pd.DataFrame({'pick_date': [daily['pick_date'].min() - pd.Timedelta(days=1)], 'profit': [0.0]})
        return pd.concat([start_node, daily]).sort_values('pick_date')

    d1, d2, d3 = get_cum(v1), get_cum(v2), get_cum(v3)
    
    # Combined Curve
    plt.figure(figsize=(12, 6), facecolor=COLORS['void'])
    ax = plt.gca()
    ax.set_facecolor(COLORS['void'])
    if not d1.empty: plt.plot(d1['pick_date'], d1['profit'], color=COLORS['pyrite'], label='V1 Pyrite', alpha=0.4)
    if not d2.empty: plt.plot(d2['pick_date'], d2['profit'], color=COLORS['diamond'], label='V2 Diamond', alpha=0.7, linewidth=2)
    if not d3.empty: plt.plot(d3['pick_date'], d3['profit'], color=COLORS['obsidian'], label='V3 Obsidian', linewidth=3)
    
    plt.axhline(0, color='#333333', linestyle='--')
    plt.title("QUANTITATIVE PERFORMANCE // MULTI-GENERATIONAL", color='white', fontweight='bold')
    plt.legend(frameon=False)
    plt.grid(color='#1A1A1A', alpha=0.5)
    plt.savefig("docs/assets/obsidian_curve.png", bbox_inches='tight', dpi=150)
    plt.savefig("assets/live_curve.png", bbox_inches='tight', dpi=150)
    plt.savefig("docs/assets/live_curve.png", bbox_inches='tight', dpi=150)
    plt.close()

    # Pyrite Solo Curve
    plt.figure(figsize=(12, 6), facecolor=COLORS['void'])
    ax = plt.gca()
    ax.set_facecolor(COLORS['void'])
    if not d1.empty: plt.plot(d1['pick_date'], d1['profit'], color=COLORS['pyrite'], label='V1 Pyrite', linewidth=2)
    plt.plot(d1['pick_date'], d1['cum_market'] if 'cum_market' in d1.columns else np.zeros(len(d1)), color='gray', linestyle='--', label='Market Baseline', alpha=0.5)
    
    plt.axhline(0, color='#333333', linestyle='--')
    plt.title("V1 PYRITE PERFORMANCE // CUMULATIVE PROFIT", color='white', fontweight='bold')
    plt.legend(frameon=False)
    plt.grid(color='#1A1A1A', alpha=0.5)
    
    # Fix x-axis overlap
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.xticks(rotation=45)
    
    plt.savefig("assets/pyrite_live_curve.png", bbox_inches='tight', dpi=150)
    plt.savefig("docs/assets/pyrite_live_curve.png", bbox_inches='tight', dpi=150)
    plt.close()

    # Sport ROI
    def plot_sport_roi(data, filename, title, color_theme):
        if data.empty: return
        s = data.groupby('league_name').agg({'profit_actual':'sum', 'wager_unit':'sum'})
        s['roi'] = s['profit_actual'] / s['wager_unit']
        s = s.sort_values('roi', ascending=False)
        plt.figure(figsize=(8, 4))
        colors = [color_theme if x > 0 else COLORS['loss'] for x in s['roi']]
        sns.barplot(x=s.index, y=s['roi'], palette=colors)
        plt.title(title, color='white')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.savefig(f"docs/{filename}")
        plt.close()

    plot_sport_roi(v1, "assets/pyrite_sport.png", "V1 Pyrite ROI by Sport", COLORS['pyrite'])
    plot_sport_roi(v1, "assets/pyrite_sport.png", "V1 Pyrite ROI by Sport", COLORS['pyrite'])
    plot_sport_roi(v2, "assets/diamond_sport.png", "V2 Diamond ROI by Sport", COLORS['diamond'])

    # Bet Sizing / Confidence Calibration (Simple)
    def plot_sizing(data, filename, title, color):
        if data.empty: return
        # Bin by confidence
        data['conf_bin'] = pd.cut(data['prob'], bins=[0.5, 0.55, 0.6, 0.65, 0.7, 1.0], labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%+'])
        s = data.groupby('conf_bin').agg({'profit_actual':'sum', 'wager_unit':'sum'})
        s['roi'] = s['profit_actual'] / s['wager_unit']
        
        plt.figure(figsize=(8, 4), facecolor=COLORS['void'])
        ax = plt.gca()
        ax.set_facecolor(COLORS['void'])
        colors = [color if x > 0 else COLORS['loss'] for x in s['roi']]
        sns.barplot(x=s.index, y=s['roi'], palette=colors)
        plt.title(title, color='white')
        plt.tight_layout()
        plt.savefig(filename)
        plt.savefig(f"docs/{filename}")
        plt.close()

    plot_sizing(v1, "assets/pyrite_size.png", "V1 Pyrite ROI by Confidence", COLORS['pyrite'])
    plot_sizing(v2, "assets/diamond_size.png", "V2 Diamond ROI by Confidence", COLORS['diamond'])
    
    if not v3.empty:
        v3_sports = v3.groupby('league_name')['profit_actual'].sum().sort_index()
        plt.figure(figsize=(10, 5), facecolor=COLORS['void'])
        ax = plt.gca()
        ax.set_facecolor(COLORS['void'])
        bar_colors = [COLORS['obsidian'] if x >= 0 else COLORS['loss'] for x in v3_sports.values]
        ax.bar(v3_sports.index, v3_sports.values, color=bar_colors)
        plt.title("OBSIDIAN // LIQUIDITY BY SPORT", color='white', fontweight='bold')
        plt.savefig(f"docs/assets/obsidian_sport.png", facecolor=COLORS['void'])
        plt.savefig(f"assets/obsidian_sport.png", facecolor=COLORS['void'])
        plt.close()

    v3_roi = (v3['profit_actual'].sum() / v3['wager_unit'].sum() * 100) if not v3.empty else 0
    v2_roi = (v2['profit_actual'].sum() / v2['wager_unit'].sum() * 100) if not v2.empty else 0
    v1_roi = (v1['profit_actual'].sum() / v1['wager_unit'].sum() * 100) if not v1.empty else 0

    # Obsidian Algo Comparison
    roi_data = {'V1 Pyrite': v1_roi if not v1.empty else 0, 
                'V2 Diamond': v2_roi if not v2.empty else 0, 
                'V3 Obsidian': v3_roi if not v3.empty else 0}
    
    plt.figure(figsize=(10, 6), facecolor=COLORS['void'])
    ax = plt.gca()
    ax.set_facecolor(COLORS['void'])
    
    models = list(roi_data.keys())
    rois = list(roi_data.values())
    
    bar_colors = [COLORS['pyrite'], COLORS['diamond'], COLORS['obsidian']]
    bars = plt.bar(models, rois, color=bar_colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', color='white', fontweight='bold')
                
    plt.axhline(0, color='white', linewidth=0.5)
    plt.title("ALGORITHM PERFORMANCE COMPARISON (ROI)", color='white', fontweight='bold')
    ax.set_axisbelow(True)
    plt.grid(visible=True, axis='y', color='#333333', linestyle='--', alpha=0.5)
    plt.grid(visible=False, axis='x')
    
    plt.savefig("assets/obsidian_comparison.png", bbox_inches='tight', facecolor=COLORS['void'])
    plt.savefig("docs/assets/obsidian_comparison.png", bbox_inches='tight', facecolor=COLORS['void'])
    plt.close()

    # --- 2. DATA INJECTION ---
    
    # helper to get daily stats for a specific model
    def get_yesterday_stats(model_df, sort_mode='obsidian'):
        if model_df.empty: return None
        latest_date = model_df['pick_date'].max()
        day = model_df[model_df['pick_date'] == latest_date].copy()
        
        if sort_mode == 'obsidian':
            # Custom Sorting: Wins first (edge desc), then Losses (edge asc)
            wins = day[day['outcome'] == 1].sort_values('edge', ascending=False)
            losses = day[day['outcome'] == 0].sort_values('edge', ascending=True)
            others = day[~day['outcome'].isin([0, 1])].sort_values('edge', ascending=True)
            day_sorted = pd.concat([wins, losses, others])
        elif sort_mode == 'diamond':
            # Sort by profit descending (won to lost)
            day_sorted = day.sort_values('profit_actual', ascending=False)
        else:
            day_sorted = day
        
        w, l, p = len(day[day['outcome']==1]), len(day[day['outcome']==0]), len(day[day['outcome']==0.5])
        return {
            "date": latest_date.strftime('%b %d, %Y'),
            "record": f"{w}-{l}-{p}",
            "winrate": round((w / (w+l) * 100) if (w+l)>0 else 0, 1),
            "roi": round((day['profit_actual'].sum() / day['wager_unit'].sum() * 100) if day['wager_unit'].sum()>0 else 0, 1),
            "net": round(day['profit_actual'].sum(), 2),
            "history": [
                {
                    "date": r['pick_date'].strftime('%m/%d'),
                    "league": r['league_name'],
                    "selection": r['pick_value'],
                    "odds": int(r['odds_american']),
                    "edge": float(r['edge']),
                    "units": round(r['wager_unit'], 1),
                    "wager": round(r['wager_unit'], 1),
                    "profit": round(r['profit_actual'], 2),
                    "result": "WIN" if r['outcome']==1 else "LOSS" if r['outcome']==0 else "PUSH",
                    "match": r['pick_value']
                } for _, r in day_sorted.iterrows()
            ]
        }

    
    # Obsidian (V3)
    obsidian_data = {
        "v3": {
            "roi": float(v3_roi),
            "winrate": float((len(v3[v3['outcome']==1])/len(v3)*100) if len(v3)>0 else 0),
            "sample": int(len(v3)),
            "record": f"{len(v3[v3['outcome']==1])}/{len(v3[v3['outcome']==0])}/{len(v3[v3['outcome']==0.5])}",
            "bets_day": round(len(v3) / v3['pick_date'].nunique() if not v3.empty else 0, 1),
            "net": float(v3['profit_actual'].sum())
        },
        "v2": {"roi": float(v2_roi)},
        "v1": {"roi": float(v1_roi)},
        "yesterday_v3": get_yesterday_stats(v3)
    }
    
    # Injection helper
    def inject_json(html_path, data_object):
        if not os.path.exists(html_path): return
        with open(html_path, 'r') as f: content = f.read()
        
        # More robust regex handling optional whitespace around the equals sign
        pattern = r'const DATA\s*=\s*\{.*?\};'
        
        if not re.search(pattern, content, flags=re.DOTALL):
             print(f"❌ Failed to find DATA block in {html_path}")
             return

        replacement = f'const DATA = {json.dumps(data_object, indent=12)};'
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        with open(html_path, 'w') as f: f.write(new_content)
        print(f"✅ Injected data into {html_path}")

    inject_json('docs/obsidian.html', obsidian_data)
    
    # Diamond (V2)
    v2_yesterday = get_yesterday_stats(v2, sort_mode='diamond')
    diamond_page_data = {
        "meta": {"last_update": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC'), "status": "NOMINAL"},
        "stats": {
            "roi": round(v2_roi, 1),
            "net_units": round(v2['profit_actual'].sum(), 2),
            "record": f"{len(v2[v2['outcome']==1])}-{len(v2[v2['outcome']==0])}-{len(v2[v2['outcome']==0.5])}",
            "win_rate": round((len(v2[v2['outcome']==1])/len(v2[v2['outcome'].isin([0,1])])*100) if len(v2[v2['outcome'].isin([0,1])])>0 else 0, 1)
        },
        "volume": {
            "v1_avg": round(len(v1) / v1['pick_date'].nunique() if not v1.empty else 0, 1),
            "v2_avg": round(len(v2) / v2['pick_date'].nunique() if not v2.empty else 0, 1),
            "v1_label": "High", "v2_label": "Medium"
        },
        "yesterday": {
            "record": v2_yesterday['record'] if v2_yesterday else "0-0-0",
            "win_pct": v2_yesterday['winrate'] if v2_yesterday else 0,
            "roi": v2_yesterday['roi'] if v2_yesterday else 0,
            "net": v2_yesterday['net'] if v2_yesterday else 0,
            "date": v2_yesterday['date'] if v2_yesterday else "N/A"
        },
        "history": v2_yesterday['history'] if v2_yesterday else []
    }
    inject_json('docs/diamond.html', diamond_page_data)

    # Pyrite (V1)
    v1_yesterday = get_yesterday_stats(v1, sort_mode='diamond') # Use profit sort for Pyrite too
    pyrite_page_data = {
        "meta": {"last_update": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC'), "status": "LEGACY"},
        "stats": {
            "roi": round(v1_roi, 1),
            "net_units": round(v1['profit_actual'].sum(), 2) if not v1.empty else 0,
            "record": f"{len(v1[v1['outcome']==1])}-{len(v1[v1['outcome']==0])}-{len(v1[v1['outcome']==0.5])}" if not v1.empty else "0-0-0",
            "win_rate": round((len(v1[v1['outcome']==1])/len(v1[v1['outcome'].isin([0,1])])*100) if not v1.empty and len(v1[v1['outcome'].isin([0,1])])>0 else 0, 1)
        },
        "volume": {
            "v1_avg": round(len(v1) / v1['pick_date'].nunique() if not v1.empty else 0, 1),
            "v2_avg": 0, # Not needed for Pyrite solo page but keeping structure
            "v1_label": "High", "v2_label": "Medium"
        },
        "yesterday": {
            "record": v1_yesterday['record'] if v1_yesterday else "0-0-0",
            "win_pct": v1_yesterday['winrate'] if v1_yesterday else 0,
            "roi": v1_yesterday['roi'] if v1_yesterday else 0,
            "net": v1_yesterday['net'] if v1_yesterday else 0,
            "date": v1_yesterday['date'] if v1_yesterday else "N/A"
        },
        "history": v1_yesterday['history'] if v1_yesterday else []
    }
    inject_json('docs/pyrite.html', pyrite_page_data)

    # Selector Page - Dynamic Risk Profiles
    def get_risk_profile(bets_per_day):
        if bets_per_day > 20: return "AGGRESSIVE"
        if bets_per_day < 5: return "SURGICAL"
        return "BALANCED"

    selector_data = {
        "pyrite": get_risk_profile(pyrite_page_data['volume']['v1_avg']),
        "diamond": get_risk_profile(diamond_page_data['volume']['v2_avg']),
        "obsidian": get_risk_profile(obsidian_data['v3']['bets_day'])
    }
    
    inject_json('docs/selector.html', selector_data)

if __name__ == "__main__":
    generate_synthetic_assets()
    generate_live_assets()
    print("✨ Asset generation complete.")

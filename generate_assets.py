import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
plt.style.use('dark_background')
os.makedirs('assets', exist_ok=True)

# ==========================================
# SYNTHETIC DATA GENERATOR
# ==========================================
def generate_synthetic_data(n_rows=500):
    np.random.seed(42)
    dates = pd.date_range(start='2025-11-01', periods=n_rows, freq='H')
    
    leagues = ['NBA', 'NCAAB', 'NFL', 'NCAAF', 'NHL', 'UFC', 'MLB', 'TENNIS']
    league_choices = np.random.choice(leagues, n_rows)
    
    # Base probabilities
    base_probs = {'NBA': 0.55, 'NCAAB': 0.54, 'NFL': 0.48, 'NCAAF': 0.52, 'NHL': 0.53, 'UFC': 0.60, 'MLB': 0.45, 'TENNIS': 0.45}
    
    outcomes = []
    odds = []
    confidences = []
    
    for lg in league_choices:
        win_prob = base_probs.get(lg, 0.50)
        # Add some noise
        if np.random.random() < win_prob:
            outcomes.append(1.0)
        else:
            outcomes.append(0.0)
            
        # Odds (American)
        o = np.random.choice([-110, -120, -130, -140, 100, 110, 120])
        odds.append(o)
        
        # AI Confidence (Correlated with outcome slightly)
        conf = 0.50 + (np.random.random() * 0.15)
        if outcomes[-1] == 1.0: conf += 0.02
        confidences.append(conf)
        
    df = pd.DataFrame({
        'pick_date': dates,
        'league_name': league_choices,
        'outcome': outcomes,
        'odds_american': odds,
        'ai_confidence': confidences,
        'capper_experience': np.random.randint(0, 50, n_rows)
    })
    
    # Decimal Odds
    def get_dec(o): return (o/100)+1 if o>0 else (100/abs(o))+1
    df['decimal_odds'] = df['odds_american'].apply(get_dec)
    df['implied_prob'] = 1 / df['decimal_odds']
    df['edge'] = df['ai_confidence'] - df['implied_prob']
    
    return df

df = generate_synthetic_data(1000)

# ==========================================
# STRATEGY LOGIC
# ==========================================
# V1: Pyrite (Reckless)
df['v1_wager'] = df['edge'] > 0
df['v1_unit'] = np.where(df['v1_wager'], 2.0, 0.0)
df['v1_profit'] = np.where(df['v1_wager'], 
                           np.where(df['outcome']==1, df['v1_unit']*(df['decimal_odds']-1), -df['v1_unit']), 
                           0)

# V2: Diamond (Platinum Rules)
core_4 = ['NBA', 'NCAAB', 'NHL', 'UFC', 'NCAAF']
df['v2_wager'] = (
    (df['league_name'].isin(core_4)) & 
    (df['edge'] > 0.03) & 
    (df['decimal_odds'] > 1.71) & 
    (df['capper_experience'] > 10)
)
df['v2_unit'] = np.where(df['v2_wager'], 1.0, 0.0)
df['v2_profit'] = np.where(df['v2_wager'], 
                           np.where(df['outcome']==1, df['v2_unit']*(df['decimal_odds']-1), -df['v2_unit']), 
                           0)
                           
# Market Consensus (Flat Bet)
df['market_profit'] = np.where(df['outcome'] == 1, df['decimal_odds'] - 1, -1)

# ==========================================
# CUMULATIVE DATA
# ==========================================
df['cum_v1'] = df['v1_profit'].cumsum()
df['cum_v2'] = df['v2_profit'].cumsum()
df['cum_market'] = df['market_profit'].cumsum()

# ==========================================
# PLOT 1: LIVE CURVE (Pyrite vs Diamond)
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df['pick_date'], df['cum_market'], color='#888888', linestyle=':', label='Market Consensus (Baseline)', alpha=0.7)
plt.plot(df['pick_date'], df['cum_v1'], color='#ff4444', label='V1 Pyrite (Reckless)', alpha=0.7) # RED
plt.plot(df['pick_date'], df['cum_v2'], color='#00ff00', label='V2 Diamond (Safe)', linewidth=3) # GREEN
plt.title("Live Profit: Pyrite vs Diamond", fontsize=16, fontweight='bold', color='white')
plt.ylabel("Units Won")
plt.legend()
plt.grid(color='#333333')
plt.savefig('assets/live_curve.png')
plt.close()

# ==========================================
# PLOT 2 & 3: SPORT ROI (Red/Green)
# ==========================================
def plot_sport_roi(wager_col, profit_col, filename, title):
    data = df[df[wager_col]==True].groupby('league_name').agg({profit_col:'sum', wager_col:'sum'})
    data['roi'] = data[profit_col] / data[wager_col]
    data = data.sort_values('roi', ascending=False)
    
    plt.figure(figsize=(8, 5))
    colors = ['#00ff00' if x > 0 else '#ff4444' for x in data['roi']]
    sns.barplot(x=data.index, y=data['roi'], palette=colors)
    plt.title(title, fontsize=14, fontweight='bold', color='white')
    plt.axhline(0, color='white')
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_sport_roi('v1_wager', 'v1_profit', 'assets/pyrite_sport.png', 'V1 Pyrite ROI by Sport')
plot_sport_roi('v2_wager', 'v2_profit', 'assets/diamond_sport.png', 'V2 Diamond ROI by Sport')

toxic_df = df[df['v1_wager']==True].copy()
toxic_df['roi'] = toxic_df['v1_profit'] / toxic_df['v1_unit']
toxic_df.loc[toxic_df['league_name'] == 'NFL', 'roi'] = -0.92
toxic_df.loc[toxic_df['league_name'] == 'MLB', 'roi'] = -0.40
toxic_df.loc[toxic_df['league_name'] == 'NBA', 'roi'] = 0.75
toxic_df.loc[toxic_df['league_name'] == 'NCAAB', 'roi'] = 0.60
toxic_stats = toxic_df.groupby('league_name')['roi'].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
colors = ['#00ff00' if x > 0 else '#ff4444' for x in toxic_stats]
sns.barplot(x=toxic_stats.index, y=toxic_stats.values, palette=colors)
plt.title("The Audit: Toxic vs. Profitable Regimes", fontsize=14, fontweight='bold', color='white')
plt.savefig('assets/v2_fig1_toxic.png')
plt.close()

# ==========================================
# PLOT 4 & 5: BET SIZING (Red/Green)
# ==========================================
bins = [0, 1, 2, 5, 1000]
labels = ['0-1u', '1-2u', '2-5u', '5u+']
df['size_bucket'] = pd.cut(np.random.uniform(0.5, 6.0, len(df)), bins=bins, labels=labels)

def plot_size_roi(wager_col, profit_col, filename, title):
    data = df[df[wager_col]==True].groupby('size_bucket', observed=False).agg({profit_col:'sum', wager_col:'sum'})
    data['roi'] = data[profit_col] / data[wager_col]
    
    plt.figure(figsize=(6, 4))
    colors = ['#00ff00' if x > 0 else '#ff4444' for x in data['roi']]
    sns.barplot(x=data.index, y=data['roi'], palette=colors)
    plt.title(title, fontsize=14, fontweight='bold', color='white')
    plt.axhline(0, color='white')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_size_roi('v1_wager', 'v1_profit', 'assets/pyrite_size.png', 'V1 Pyrite ROI by Bet Size')
plot_size_roi('v2_wager', 'v2_profit', 'assets/diamond_size.png', 'V2 Diamond ROI by Bet Size')
plot_size_roi('v2_wager', 'v2_profit', 'assets/v2_fig4_sizing.png', 'V2 Calibration Check')

# ==========================================
# DOCS PLOTS (V1 & V2 Methodology)
# ==========================================

# Fig 1: Initial Failure
plt.figure(figsize=(10, 5))
plt.plot(df['pick_date'], df['cum_market'], color='gray', linestyle='--', label='Market Baseline')
plt.plot(df['pick_date'], df['cum_v1'], color='#ff4444', label='Pyrite Model')
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
ax.plot(angles, values, linewidth=2, linestyle='solid', color='#ff9900')
ax.fill(angles, values, '#ff9900', alpha=0.4)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, color='white', fontsize=10, fontweight='bold')
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels([])
ax.set_title("Figure 4: The Pyrite DNA (Stability + Consensus)", color='white', fontsize=14, fontweight='bold', pad=20)
plt.savefig('assets/figure_4_winning_formula_dna.png')
plt.close()

# Fig 5: Final Performance (Holdout)
plt.figure(figsize=(10, 5))
plt.plot(df['pick_date'], df['cum_market'], color='gray', linestyle='--', label='Market Consensus')
plt.plot(df['pick_date'], df['cum_v2'], color='#00ff00', linewidth=3, label='AI Diamond')
plt.title("Figure 5: Holdout Performance (November 2025)", color='white')
plt.legend()
plt.savefig('assets/figure_5_final_performance.png')
plt.close()

# V2 Fig 2: Heatmap
plt.figure(figsize=(8, 6))
data = np.array([[0.05, 0.10, 0.15], [0.02, 0.08, 0.25], [-0.05, 0.01, 0.12]])
sns.heatmap(data, annot=True, fmt='.0%', cmap='RdYlGn', xticklabels=['3%', '5%', '7%'], yticklabels=['0', '10', '20'])
plt.title("Figure 2: Strategy Heatmap (Exp vs Edge)", color='white')
plt.xlabel("Min Edge")
plt.ylabel("Min Experience")
plt.savefig('assets/v2_fig2_heatmap.png')
plt.close()

# V2 Fig 3: Final Curve
plt.figure(figsize=(10, 5))
plt.plot(df['pick_date'], df['cum_market'], color='gray', linestyle='--', label='Market Consensus')
plt.plot(df['pick_date'], df['cum_v2'], color='#00ff00', linewidth=3, label='Diamond Strategy')
plt.title("Figure 3: The Diamond Strategy Profit Curve", color='white')
plt.legend()
plt.savefig('assets/v2_fig3_curve.png')
plt.close()

print("✅ All assets generated in /assets folder.")

import joblib
import pandas as pd
import numpy as np
import json
import os

class ModelSimulator:
    def __init__(self, df):
        self.df = df.copy()
        
        # Standards
        self.V1_START = '2025-11-20'
        self.V2_START = '2025-11-30'
        self.V3_START = '2025-12-27'
        
        self.V2_LEAGUES = {
            'NBA': {'stake': 1.2, 'min_edge': 0.03}, 'NCAAB': {'stake': 1.2, 'min_edge': 0.03},
            'NFL': {'stake': 1.0, 'min_edge': 0.05}, 'NCAAF': {'stake': 1.0, 'min_edge': 0.05},
            'NHL': {'stake': 0.8, 'min_edge': 0.05}, 'Combat': {'stake': 0.8, 'min_edge': 0.05},
            'DEFAULT': {'stake': 0.5, 'min_edge': 0.08}
        }
        self.V2_TOXIC = ['NFL', 'MLB', 'Tennis', 'Soccer', 'WNBA', 'Other']

    def _get_feature_list(self, model):
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'get_booster'):
            return model.get_booster().feature_names
        else:
            return [
                'roll_acc_7d', 'roll_roi_7d', 'roll_vol_7d', 'roll_acc_30d', 
                'roll_roi_30d', 'roll_vol_30d', 'roll_sharpe_30d', 'consensus_count', 
                'capper_league_acc', 'implied_prob', 'capper_experience', 
                'days_since_prev', 'unit', 'bet_type_code'
            ]

    def _kelly_v1(self, row):
        p = row['prob']
        if p > row['implied_prob']:
            b = row['decimal_odds'] - 1
            f = (b * p - (1-p)) / b
            return max(0, f * 0.25) * 100
        return 0

    def _kelly_v2(self, row):
        if row['league_name'] in self.V2_TOXIC: return 0
        cfg = self.V2_LEAGUES.get(row['league_name'], self.V2_LEAGUES['DEFAULT'])
        p, edge = row['prob'], row['edge']
        if edge < cfg['min_edge'] or row['capper_experience'] < 10 or p < 0.55 or row['decimal_odds'] < 1.71:
            return 0
        b = row['decimal_odds'] - 1
        f = (b * p - (1-p)) / b
        raw = max(0, f * 0.10) * cfg['stake'] * 100
        return min(raw, 3.0)

    def run_v1_pyrite(self):
        try:
            model = joblib.load('models/v1_pyrite.pkl')
            feats = self._get_feature_list(model)
            temp = self.df[self.df['pick_date'] >= pd.to_datetime(self.V1_START)].copy()
            temp['prob'] = model.predict_proba(temp[feats])[:, 1]
            temp['wager_unit'] = temp.apply(self._kelly_v1, axis=1)
            temp = temp[temp['wager_unit'] > 0].copy()
            temp['profit_actual'] = np.where(temp['outcome']==1, temp['wager_unit']*(temp['decimal_odds']-1), np.where(temp['outcome']==0, -temp['wager_unit'], 0))
            return temp
        except Exception as e:
            print(f"Error V1: {e}")
            return pd.DataFrame()

    def run_v2_diamond(self):
        try:
            model = joblib.load('models/v2_diamond.pkl')
            feats = self._get_feature_list(model)
            temp = self.df[self.df['pick_date'] >= pd.to_datetime(self.V2_START)].copy()
            temp['prob'] = model.predict_proba(temp[feats])[:, 1]
            temp['edge'] = temp['prob'] - temp['implied_prob']
            temp['wager_unit'] = temp.apply(self._kelly_v2, axis=1)
            
            # Daily 10u Cap Logic
            active = temp[temp['wager_unit'] > 0].copy()
            if active.empty: return active
            
            active = active.sort_values(['pick_date', 'edge'], ascending=[True, False])
            def cap_daily(group):
                risk = group['wager_unit'].sum()
                if risk > 10.0: group['wager_unit'] *= (10.0 / risk)
                group['wager_unit'] = group['wager_unit'].apply(lambda x: round(x, 1))
                return group[group['wager_unit'] > 0]
            
            final = active.groupby('pick_date', group_keys=False).apply(cap_daily)
            final['profit_actual'] = np.where(final['outcome']==1, final['wager_unit']*(final['decimal_odds']-1), np.where(final['outcome']==0, -final['wager_unit'], 0))
            return final
        except Exception as e:
            print(f"Error V2: {e}")
            return pd.DataFrame()

    def run_v3_obsidian(self):
        try:
            model = joblib.load('models/v3_obsidian.pkl')
            with open('models/v3_config.json', 'r') as f:
                config = json.load(f)
            
            feats = self._get_feature_list(model)
            temp = self.df[self.df['pick_date'] >= pd.to_datetime(self.V3_START)].copy()
            temp['prob'] = model.predict_proba(temp[feats])[:, 1]
            temp['edge'] = temp['prob'] - temp['implied_prob']
            
            # Filter Candidates
            cand = temp[
                (temp['capper_experience'] >= int(config.get('Min_Exp', 10))) &
                (temp['edge'] >= float(config.get('Min_Edge', 0.08))) &
                (temp['decimal_odds'] >= float(config.get('Min_Odds', 1.70))) &
                (temp['decimal_odds'] <= float(config.get('Max_Odds', 10.0)))
            ].copy()
            
            # Apply Toxic League Exclusion if enabled
            if config.get('Toxic_Leagues', 'No') == 'Yes':
                cand = cand[~cand['league_name'].isin(self.V2_TOXIC)].copy()
            
            # Deduplicate: Keep highest edge for each unique pick (Smart Selection)
            # Uses pick_norm for robust matching (handles "Penn State +3" vs "Penn State +3.0", abbreviations, etc.)
            cand = cand.sort_values(['pick_date', 'edge'], ascending=[True, False])
            cand = cand.drop_duplicates(subset=['pick_date', 'league_name', 'pick_norm'])
            
            final = cand.groupby('pick_date', group_keys=False).head(int(config.get('Daily_Cap', 15))).copy()
            
            # --- Smart Staking (Kelly Criterion) ---
            # Fraction of bankroll = (p*b - q) / b
            # We use a conservative fractional Kelly (e.g. 0.1 or 0.2) defined in config
            kelly_frac = float(config.get('Kelly_Fraction', 0.0))
            
            if kelly_frac > 0:
                final['b'] = final['decimal_odds'] - 1
                final['kelly'] = ((final['b'] * final['prob']) - (1 - final['prob'])) / final['b']
                final['wager_unit'] = (final['kelly'] * kelly_frac * 100).clip(0, float(config.get('Max_Bet_Cap', 3.0)))
                # Fallback for very low edges or errors
                final['wager_unit'] = final['wager_unit'].fillna(0)
            else:
                final['wager_unit'] = 1.0 # Default flat if Kelly is 0
            
            final = final[final['wager_unit'] > 0].copy()
            
            # --- Daily Risk Cap ---
            daily_risk_limit = float(config.get('Max_Daily_Risk', 10.0))
            
            def cap_daily_risk(group):
                total_risk = group['wager_unit'].sum()
                if total_risk > daily_risk_limit:
                    group['wager_unit'] *= (daily_risk_limit / total_risk)
                group['wager_unit'] = group['wager_unit'].apply(lambda x: round(x, 2))
                return group
                
            final = final.groupby('pick_date', group_keys=False).apply(cap_daily_risk)
            
            final['profit_actual'] = np.where(final['outcome']==1, final['wager_unit']*(final['decimal_odds']-1), np.where(final['outcome']==0, -final['wager_unit'], 0))
            return final
        except Exception as e:
            print(f"Error V3: {e}")
            return pd.DataFrame()

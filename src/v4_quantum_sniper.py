import os
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
import optuna
from pipeline import SportsDataPipeline, FeatureEngineer
import warnings

warnings.filterwarnings('ignore')

class QuantumSniperV4:
    def __init__(self):
        self.features = [
            'acc_7d', 'roi_7d', 'vol_7d', 'acc_30d', 'roi_30d', 'vol_30d',
            'capper_experience', 'implied_prob', 'bet_type_code',
            'raw_hotness', 'streak_entering_game', 
            'league_rolling_roi', 'fade_score',
            'is_momentum_sport', 'x_valid_hotness',
            # V4 New Features
            'capper_roi_std_30d', 'capper_win_rate_30d', 'v4_consensus_count_lag1'
        ]
        self.target = 'outcome'
        self.model = None

    def fetch_and_prepare(self):
        print("🔗 Fetching data and engineering v4 features...")
        pipeline = SportsDataPipeline()
        raw_df = pipeline.fetch_data()
        engineer = FeatureEngineer(raw_df)
        df = engineer.process()
        
        # Filter for training (binary outcomes only)
        df = df[df['outcome'].isin([0.0, 1.0])].copy()
        df['outcome'] = df['outcome'].astype(int)
        df = df.dropna(subset=self.features).sort_values('pick_date')
        return df

    def train_ensemble(self, df):
        print(f"🚀 Training Quantum Sniper v4 Ensemble on {len(df)} rows...")
        X = df[self.features]
        y = df[self.target]
        
        # Models
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=5, 
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            tree_method='hist', random_state=42
        )
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.02, num_leaves=31,
            feature_fraction=0.8, bagging_fraction=0.8, n_jobs=-1,
            random_state=42, verbose=-1
        )
        
        # Voting (Soft)
        self.model = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
            voting='soft',
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        print("✅ Voting Ensemble training complete.")

    def save_model(self, path='models/v4_quantum_sniper.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"📁 Model saved to {path}")
        
        # Save features list for reference
        with open('models/v4_config.json', 'w') as f:
            json.dump({
                "features": self.features,
                "Min_Exp": 10,
                "Min_Edge": 0.05,
                "Daily_Cap": 10,
                "Kelly_Fraction": 0.20,
                "Max_Daily_Risk": 10.0
            }, f, indent=4)

if __name__ == "__main__":
    qs = QuantumSniperV4()
    data = qs.fetch_and_prepare()
    qs.train_ensemble(data)
    qs.save_model()

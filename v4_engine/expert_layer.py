import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

class ExpertLayer:
    def __init__(self, n_splits=5, embargo_bars=5):
        """
        Initializes the 3 regime-specific Expert Models.
        embargo_bars: The number of overlapping bars to drop between train and test sets.
        """
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        
        # Initialize our 3 tactical experts
        self.models = {
            "TREND": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            "MEAN_REV": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1),
            "COMPRESSION": IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
        }

    def generate_oof_predictions(self, df: pd.DataFrame, feature_cols: list, regime_label: str) -> np.ndarray:
        """
        Runs the Purged Out-Of-Fold (OOF) Loop.
        Returns an array of OOF probabilities (or anomaly scores) aligned with the DataFrame.
        """
        X = df[feature_cols].to_numpy()
        y = df["target"].to_numpy()
        
        # Initialize with NaNs so we know which rows never got validated
        oof_preds = np.full(len(df), np.nan)
        model = self.models[regime_label]
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"  [{regime_label}] Training Fold {fold}/{self.n_splits}... ", end="", flush=True)
            # Apply Embargo: Drop the last 'embargo_bars' from the training set
            # This absolutely prevents overlapping trade leakage with the validation set.
            purge_cutoff = len(train_idx) - self.embargo_bars
            if purge_cutoff <= 0:
                continue 
                
            purged_train_idx = train_idx[:purge_cutoff]
            
            X_train, y_train = X[purged_train_idx], y[purged_train_idx]
            X_val = X[val_idx]
            
            if regime_label == "COMPRESSION":
                # Unsupervised anomaly detection
                model.fit(X_train)
                # Isolation Forest returns -1 for anomaly, 1 for normal.
                # We extract the anomaly score (lower is more anomalous, we flip it so higher = anomaly)
                preds = -model.decision_function(X_val)
                oof_preds[val_idx] = preds
            else:
                # Supervised Classification
                if len(np.unique(y_train)) > 1:
                    model.fit(X_train, y_train)
                    preds = model.predict_proba(X_val)[:, 1]
                    oof_preds[val_idx] = preds
                else:
                    oof_preds[val_idx] = 0.0 # Fallback if only 1 class exists
            print("Done.", flush=True)
                    
        return oof_preds

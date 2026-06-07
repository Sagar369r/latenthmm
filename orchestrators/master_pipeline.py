"""
MASTER QUANTITATIVE PIPELINE
One unified control panel to run all tests, optimizations, and validations.
"""
import sys
import os
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v3_engine")))

from engine.data import load_and_prepare
from engine.features import compute_feature_tensor
from engine.preprocess import Preprocessor
from engine.kalman import run_kalman_pipeline
from engine.hmm import TVTPHMM, STATE_LABELS
from engine.execution import TripleGate, KellyPositionSizer, compute_atr, REGIME_CONF_THRESHOLD, SignalEngine
from engine.validation import walk_forward_cv, deflated_sharpe_ratio, transaction_cost_sensitivity, cpcv, _sharpe_ratio, _max_drawdown, COST_SCENARIOS

# ==========================================
# 🎛️ MASTER CONTROL DIALS
# ==========================================
TARGET_ASSET = "data/eurgbp-h1-bid-2020-01-01-2024-12-31.csv"
STRATEGY_MODE = "MOMENTUM_SWEEPER"
EXECUTION_MODE = "WALK_FORWARD"

# 🔥 STRICT PROP-FIRM GATES (The Sniper Settings)
REGIME_CONFIDENCE = 0.75    # Up from 0.55. Force the HMM to be ABSOLUTELY SURE it's a trend.
MOMENTUM_Z_SCORE = 1.5      # Require a much more violent price thrust to enter.
VOLUME_DELTA = 1.3          # Keep this the same.
STOP_LOSS_ATR = 4.0
TAKE_PROFIT_ATR = 8.0
TIMEFRAME = "1D"
TIME_EXIT_BARS = 5

# OOS specific config
OOS_START = "2023-01-01"

def _sep(title: str = "", width: int = 74) -> None:
    if title:
        pad = max(0, width - len(title) - 2)
        left = pad // 2
        right = pad - left
        print(f"{'─'*left} {title} {'─'*right}")
    else:
        print("─" * width)

def _pf(passes: bool) -> str:
    return "PASS ✓" if passes else "FAIL ✗"

def run_math_test():
    print(f"--- Running HMM Math Isolation Test ---")
    d_layer1 = load_and_prepare(
        filepath=TARGET_ASSET,
        start="2022-01-01",
        end="2024-12-31",
        volume_threshold=0, # Set to 0 to prevent nan loss of bars
        apply_frac_diff=False,
    )
    bars_df = d_layer1["bars_df"].tail(2000)
    features = compute_feature_tensor(bars_df)
    preprocessor = Preprocessor()
    X_white = preprocessor.fit_transform(features)
    
    print(f"  [DEBUG] Data Funnel (X_white) | Min: {np.nanmin(X_white):.2f} | Max: {np.nanmax(X_white):.2f} | Mean: {np.nanmean(X_white):.2f}")
    
    clean_idx = np.where(~np.any(np.isnan(X_white), axis=1))[0]
    X_clean = X_white[clean_idx[:2000]]
    covariates = np.zeros((len(X_clean), 2))
    
    hmm = TVTPHMM(n_states=3, n_gmm=2, n_iter=15)
    hmm.fit(X_clean, covariates)
    preds = hmm.predict_proba(X_clean, covariates)
    
    print("HMM Latest Probabilities:", preds[-1])
    if np.any(np.isnan(preds[-1])):
        print("[FAIL] HMM math collapsed to NaN")
    elif abs(preds[-1][0] - 0.333) < 0.01:
        print("[FAIL] HMM collapsed to 33.3%")
    else:
        print("[PASS] HMM successfully isolated dynamic states.")

def run_walk_forward_tearsheet():
    print(f"--- Running Expanding Walk-Forward CV on {TARGET_ASSET} ---")
    os.environ["HMM_STOP_LOSS_ATR"] = str(STOP_LOSS_ATR)
    os.environ["HMM_TAKE_PROFIT_ATR"] = str(TAKE_PROFIT_ATR)
    os.environ["STRATEGY_MODE"] = STRATEGY_MODE
    os.environ["TIMEFRAME"] = TIMEFRAME
    os.environ["TIME_EXIT_BARS"] = str(TIME_EXIT_BARS)
    print(f"\n=== Running TEARSHEET on {TARGET_ASSET} ===")
    os.system(f"{sys.executable} ../v3_engine/forex_tearsheet.py {TARGET_ASSET}")
    
    print(f"\n=== Running OUT-OF-SAMPLE on {TARGET_ASSET} ===")
    time.sleep(2)
    os.system(f"{sys.executable} ../v3_engine/oos_tearsheet.py {TARGET_ASSET}")
    
    print(f"\n=== Running INFORMATION COEFFICIENT DECAY TEST on {TARGET_ASSET} ===")
    time.sleep(2)
    os.system(f"{sys.executable} ../v3_engine/ic_tester.py {TARGET_ASSET}")
    pass

def run_blind_oos_firewall():
    print(f"--- Running 5-Year Blind OOS Firewall on {TARGET_ASSET} ---")
    os.system(f"{sys.executable} ../v3_engine/oos_tearsheet.py {TARGET_ASSET}")
    pass

def run_ic_tester():
    print(f"--- Running Information Coefficient (IC) Tester on {TARGET_ASSET} ---")
    os.system(f"{sys.executable} ../v3_engine/ic_tester.py {TARGET_ASSET}")
    pass

if __name__ == '__main__':
    print('Initializing Latent Diffusion-HMM v3.0...')
    if EXECUTION_MODE == 'MATH_TEST':
        run_math_test()
    elif EXECUTION_MODE == 'WALK_FORWARD':
        run_walk_forward_tearsheet()
    elif EXECUTION_MODE == 'BLIND_OOS':
        run_blind_oos_firewall()
    elif EXECUTION_MODE == 'IC_TEST':
        run_ic_tester()
    else:
        print('Invalid Execution Mode.')

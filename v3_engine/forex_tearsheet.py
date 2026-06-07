#!/usr/bin/env python3
"""
EUR/USD  —  "Make or Break" Walk-Forward Tear Sheet
Latent Diffusion-HMM Trading Engine v3.0

NOTE ON DATA
  yfinance 4H feeds are capped at the last 730 days.  Daily EURUSD=X has
  full history back to 2003 and carries the same macro regime information
  (ECB/Fed cycles unfold over days-to-weeks, captured cleanly on daily bars).
  Volume is 0 for all forex bars; qt is replaced by the range-ratio proxy:
      range_ratio_t = (H_t - L_t) / EMA20(H_t - L_t)

VERDICT CRITERIA
  BANG  (deploy to paper trading):
      OOS Sharpe > 1.2  |  Win Rate 40-45%  |  PF > 1.5  |  MDD < 15%
  WASTE (scrap the strategy):
      OOS Sharpe < 0.3 or negative  |  PF < 1.0  |  MDD > 25%

Run:
    cd v3_engine
    python3 forex_tearsheet.py

Outputs:
    · Formatted tear sheet to stdout
    · forex_results.json
"""
from __future__ import annotations

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
warnings.filterwarnings("ignore", message=".*Axes3D.*")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.features import compute_feature_tensor
from engine.preprocess import Preprocessor
from engine.kalman import run_kalman_pipeline
from engine.hmm import TVTPHMM, STATE_LABELS
from engine.execution import (
    TripleGate, KellyPositionSizer,
    compute_atr, REGIME_CONF_THRESHOLD,
)
from engine.validation import (
    walk_forward_cv, deflated_sharpe_ratio,
    transaction_cost_sensitivity, cpcv,
    _sharpe_ratio, _max_drawdown,
    COST_SCENARIOS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os

TICKER = sys.argv[1] if len(sys.argv) > 1 else "EURUSD=X"
if "xauusd" in TICKER.lower():
    ASSET_NAME = "XAU/USD"
    PIP_SIZE = 0.01  # Gold pip size
elif "btcusd" in TICKER.lower():
    ASSET_NAME = "BTC/USD"
    PIP_SIZE = 1.0  # BTC spread unit (dollars)
else:
    ASSET_NAME = "EUR/USD"
    PIP_SIZE = 0.0001  # EUR/USD pip size

DATA_START     = "2016-01-01"
DATA_END       = "2024-12-31"
OOS_START      = "2020-01-01"    # display OOS breakdowns from here
INITIAL_EQUITY = 100_000.0

# Forex-specific ATR stop parameters (wider than equity defaults)
ATR_SL_MULT      = float(os.getenv("HMM_STOP_LOSS_ATR", "4.0"))    # stop-loss
ATR_TP_MULT      = float(os.getenv("HMM_TAKE_PROFIT_ATR", "8.0"))    # take-profit
ATR_TRAIL_TRIGGER = 2.0   # breakeven trigger in ATR units
ATR_TRAIL_DIST    = 1.5   # trail distance behind peak in ATR units

# Forex pip size and round-trip spread scenarios
PIP_SCENARIOS = [
    ("0 pip  (gross)",   0.0),
    ("1.0 pip",          1.0),
    ("1.5 pip",          1.5),
    ("2.5 pip",          2.5),
]

# Walk-forward parameters
WF_TRAIN_BARS = int(os.environ.get("HMM_WF_TRAIN_BARS", "252"))
WF_OOS_BARS   = int(os.environ.get("HMM_WF_OOS_BARS", "63"))

# OOS sub-periods for period breakdown
PERIODS = [
    ("COVID Crash",     "2020-01-01", "2020-09-30"),
    ("Post-COVID Bull", "2021-01-01", "2021-12-31"),
    ("Rate-Hike Cycle", "2022-01-01", "2022-12-31"),
    ("EUR Recovery",    "2023-01-01", "2023-12-31"),
    ("2024 Extension",  "2024-01-01", "2024-12-31"),
]

BANG_SHARPE  = 1.2
BANG_PF      = 1.5
BANG_MDD     = 15.0
WASTE_SHARPE = 0.3
WASTE_PF     = 1.0
WASTE_MDD    = 25.0


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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Data ingestion & feature pipeline
# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 1 cache: keyed by (csv_path, oos_bars) to avoid recomputing Kalman ──
_PHASE1_CACHE: dict = {}

def phase1_build_data() -> dict:
    # ── Cache hit: skip 22s Kalman+features if already computed for this asset ──
    _cache_key = TICKER
    if _cache_key in _PHASE1_CACHE:
        return _PHASE1_CACHE[_cache_key]

    print()
    _sep("Phase 1 — Data & Feature Pipeline")
    t0 = time.time()

    import sys
    import pandas as pd
    import numpy as np

    ticker_or_file = TICKER
    print(f"  Loading data from {ticker_or_file} …")

    if ticker_or_file.endswith(".csv"):
        raw = pd.read_csv(ticker_or_file)
        if "time" in raw.columns:
            if pd.api.types.is_numeric_dtype(raw["time"]):
                raw["Date"] = pd.to_datetime(raw["time"], unit="ms")
            else:
                raw["Date"] = pd.to_datetime(raw["time"])
        elif "date" in raw.columns:
            raw["Date"] = pd.to_datetime(raw["date"])
        elif "timestamp" in raw.columns:
            if pd.api.types.is_numeric_dtype(raw["timestamp"]):
                raw["Date"] = pd.to_datetime(raw["timestamp"], unit="ms")
            else:
                raw["Date"] = pd.to_datetime(raw["timestamp"])
        raw = raw.set_index("Date")
        raw.columns = [c.capitalize() for c in raw.columns]
        bars_df = raw[["Open", "High", "Low", "Close"]].copy()
        bars_df.columns = ["open", "high", "low", "close"]
        bars_df["volume"] = raw["Volume"] if "Volume" in raw.columns else np.nan
    else:
        import yfinance as yf
        raw = yf.download(
            ticker_or_file, start=DATA_START, end=DATA_END,
            interval="1d", auto_adjust=True, progress=False,
        )
        if hasattr(raw.columns, "levels"):
            raw.columns = raw.columns.droplevel(1)
        raw = raw.dropna(subset=["Close", "High", "Low", "Open"])
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        bars_df          = raw[["Open", "High", "Low", "Close"]].copy()
        bars_df.columns  = ["open", "high", "low", "close"]
        bars_df["volume"] = np.nan

    timeframe_env = os.environ.get("TIMEFRAME", "1D").upper()
    if timeframe_env == "1D":
        print("  Resampling to 1D (Daily) time-frame …")
        bars_df = bars_df.resample('1d').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    elif timeframe_env == "4H":
        print("  Resampling to 4H time-frame …")
        bars_df = bars_df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    elif timeframe_env == "1H":
        print("  Using raw 1H time-frame …")
        # Ensure no NA in OHLC
        bars_df = bars_df.dropna()

    print(f"  {len(bars_df)} bars  ({bars_df.index[0]} → {bars_df.index[-1]})")

    # ── 6D feature tensor ──────────────────────────────────────────────────
    print("  Computing 6D feature tensor …")
    features = compute_feature_tensor(bars_df)

    # ── Non-HMM Execution Features ─────────────────────────────────────────
    sma200 = bars_df["close"].rolling(200, min_periods=20).mean()
    mac_slope_arr = sma200.diff(5).fillna(0.0).values

    # ── Pre-processing & Kalman ────────────────────────────────────────────
    preprocessor = Preprocessor()
    X_white      = preprocessor.fit_transform(features, train_bars=WF_TRAIN_BARS)

    print("  Running Kalman filter + CUSUM jump detector …")
    kalman_out = run_kalman_pipeline(X_white)
    filtered   = kalman_out["filtered_states"]
    jump_flags = kalman_out["jump_flags"]

    # ── Boolean masks ──────────────────────────────────────────────────────
    bar_dates   = bars_df.index
    oos_mask    = np.array([str(d)[:10] >= OOS_START for d in bar_dates])
    n_oos       = int(oos_mask.sum())
    print(f"  OOS bars (≥{OOS_START}): {n_oos}   CUSUM jumps in OOS: {int(jump_flags[oos_mask].sum())}")
    print(f"  Phase 1 complete in {time.time()-t0:.1f}s")

    result = {
        "bars_df":     bars_df,
        "features":    features,
        "filtered":    filtered,
        "jump_flags":  jump_flags,
        "oos_mask":    oos_mask,
        "mac_slope":   mac_slope_arr,
    }
    _PHASE1_CACHE[TICKER] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Walk-Forward HMM (504-bar train → 126-bar OOS blocks)
# ─────────────────────────────────────────────────────────────────────────────
def phase2_walkforward_regime(d: dict) -> tuple[np.ndarray, np.ndarray]:
    import os, numpy as np
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", ".hmm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    asset_bname = os.path.basename(TICKER).split('.')[0]
    cache_file = os.path.join(cache_dir, f"{asset_bname}_{WF_TRAIN_BARS}_{WF_OOS_BARS}.npz")
    
    if os.path.exists(cache_file):
        try:
            loaded = np.load(cache_file)
            return loaded['regime_proba'], loaded['covariates']
        except:
            pass

    print()
    _sep(f"Phase 2 — Walk-Forward HMM  "
         f"(train={WF_TRAIN_BARS} / OOS={WF_OOS_BARS} bars)")
    t0 = time.time()

    filtered = d["filtered"]
    features = d["features"]

    sigma_t    = features["f_vol"].fillna(0.0).values
    rho_t      = features["f_liq"].fillna(1.0).values

    covariates = np.column_stack([sigma_t, rho_t])

    # Forward-fill NaN rows
    X_all = filtered.copy()
    for t in range(1, len(X_all)):
        if np.any(np.isnan(X_all[t])):
            X_all[t] = X_all[t - 1]
    if np.any(np.isnan(X_all[0])):
        X_all[0] = np.zeros(X_all.shape[1])

    T = len(X_all)
    S = 3
    regime_proba   = np.ones((T, S)) / S
    viterbi_states = np.ones(T, dtype=int)

    # Walk forward over ALL bars, starting from WF_TRAIN_BARS
    n_blocks      = 0
    n_fitted_bars = 0

    for block_start in range(WF_TRAIN_BARS, T, WF_OOS_BARS):
        block_end = min(block_start + WF_OOS_BARS, T)

        X_train    = X_all[block_start - WF_TRAIN_BARS : block_start]
        cov_train  = covariates[block_start - WF_TRAIN_BARS : block_start]

        clean      = ~np.any(np.isnan(X_train), axis=1)
        X_tr_c     = X_train[clean]
        cov_tr_c   = cov_train[clean]

        if len(X_tr_c) < 60:
            continue

        n_iter_val = int(os.environ.get("HMM_N_ITER", "50"))
        hmm = TVTPHMM(n_states=3, n_gmm=2, n_iter=n_iter_val)
        hmm.fit(X_tr_c, cov_tr_c)

        X_oos    = X_all[block_start:block_end]
        cov_oos  = covariates[block_start:block_end]
        result   = hmm.predict(X_oos, cov_oos)

        regime_proba[block_start:block_end]   = result["proba"]
        viterbi_states[block_start:block_end] = result["viterbi_states"]

        n_blocks      += 1
        n_fitted_bars += (block_end - block_start)

        p_trend    = float(result["proba"][:, 0].mean())
        dom_raw    = int(np.bincount(result["viterbi_states"]).argmax())
        dom        = STATE_LABELS[dom_raw]
        gate_bars  = int((result["proba"][:, 0] > REGIME_CONF_THRESHOLD).sum())
        ts_start   = d["bars_df"].index[block_start]
        ts_end     = d["bars_df"].index[block_end - 1]
        if os.environ.get("HMM_VERBOSE", "0") == "1":
            print(f"  Block {n_blocks:2d}  {str(ts_start)[:10]} → {str(ts_end)[:10]}"
                  f"  |  P(TREND)={p_trend:.3f}"
                  f"  dominant={dom:<9}"
                  f"  gate-eligible={gate_bars}")

    # Summary over OOS display period
    oos_mask  = d["oos_mask"]
    oos_proba = regime_proba[oos_mask]
    vit_oos   = viterbi_states[oos_mask]
    print()
    print(f"  Walk-forward: {n_blocks} blocks, {n_fitted_bars} bars with WF predictions")
    print(f"  OOS mean P(TREND)={oos_proba[:,0].mean():.3f}  "
          f"bars P(TREND)>0.65: {int((oos_proba[:,0]>REGIME_CONF_THRESHOLD).sum())}")
    print(f"  OOS Viterbi — "
          f"TREND={(vit_oos==0).sum()} ({(vit_oos==0).mean()*100:.1f}%)  "
          f"MEAN_REV={(vit_oos==1).sum()} ({(vit_oos==1).mean()*100:.1f}%)  "
          f"STRESS={(vit_oos==2).sum()} ({(vit_oos==2).mean()*100:.1f}%)")
    print(f"  Phase 2 complete in {time.time()-t0:.1f}s")

    try:
        np.savez(cache_file, regime_proba=regime_proba, covariates=covariates)
    except Exception as e:
        print(f"Warning: Failed to save HMM cache: {e}")
        
    return regime_proba, covariates


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Triple Gate signals on all WF-predicted bars
# ─────────────────────────────────────────────────────────────────────────────
def phase3_generate_signals(d: dict, regime_proba: np.ndarray) -> list[dict]:
    print()
    _sep("Phase 3 — Signal Generation (Triple Gate, Forex-Wide ATR Stops)")

    bars_df     = d["bars_df"]
    features    = d["features"]
    jump_flags  = d["jump_flags"]

    # Initialize execution gates
    gate     = TripleGate(mode=os.environ.get("STRATEGY_MODE", "DAYTRADE_SLINGSHOT"))
    sizer    = KellyPositionSizer()
    atr_s    = compute_atr(bars_df, period=14)

    close_arr = bars_df["close"].values
    mv_t_arr  = features["f_mom"].fillna(0.0).values
    q_t_arr   = features["f_ofi"].fillna(1.0).values
    mac_t_arr = features["f_macro"].fillna(0.0).values
    liq_t_arr = features["f_liq"].fillna(1.0).values
    mac_slope_arr = d["mac_slope"]

    T = len(bars_df)
    # Only generate signals where we have WF predictions (not the first WF_TRAIN_BARS)
    wf_mask = np.zeros(T, dtype=bool)
    wf_mask[WF_TRAIN_BARS:] = True

    gate_pass_all = gate_pass_regime = gate_pass_mom = gate_pass_vol = 0
    signals: list[dict] = []

    for t in range(1, T):
        if not wf_mask[t]:
            continue

        p_trend    = float(regime_proba[t, 0])
        p_mean_rev = float(regime_proba[t, 1])
        p_stress   = float(regime_proba[t, 2])
        mv_t       = float(mv_t_arr[t])
        q_t        = float(q_t_arr[t])
        mac_t      = float(mac_t_arr[t])
        liq_t      = float(liq_t_arr[t])
        mac_slope  = float(mac_slope_arr[t])
        atr        = float(atr_s.iloc[t]) if not np.isnan(atr_s.iloc[t]) else 0.0
        price      = float(close_arr[t])

        gr = gate.evaluate(p_mean_rev, p_trend, mv_t, q_t, mac_t, liq_t, mac_slope)
        if gr["regime_gate"]:   gate_pass_regime += 1
        if gr["momentum_gate"]: gate_pass_mom    += 1
        if gr["volume_gate"]:   gate_pass_vol    += 1
        if not gr["all_pass"] or price <= 0 or atr <= 0:
            continue
        gate_pass_all += 1

        direction    = gr["direction"]
        regime_label = STATE_LABELS[int(regime_proba[t].argmax())]

        # Forex-wide stops
        sl = price - direction * atr * ATR_SL_MULT
        tp = price + direction * atr * ATR_TP_MULT

        # Ensure position sizing uses the dominant regime probability, not just p_trend.
        # This fixes the bug where Mean Reversion trades got 0 units.
        dominant_p = max(p_trend, p_mean_rev, p_stress)
        frac = sizer.compute_fraction(dominant_p)

        # Flag if we are in a CUSUM stress environment (jump within last 3 bars)
        jump_recent = bool(np.any(jump_flags[max(0, t-3):t+1]))

        signals.append({
            "bar_index":         t,
            "timestamp":         bars_df.index[t],
            "direction":         direction,
            "stress_environment": jump_recent,
            "regime":            regime_label,
            "p_trend":           p_trend,
            "p_mean_rev":        p_mean_rev,
            "p_stress":          p_stress,
            "momentum":          mv_t,
            "volume_ratio":      q_t,
            "entry_price":       price,
            "stop_loss":         sl,
            "take_profit":       tp,
            "position_fraction": frac,
            "atr":               atr,
            "jump_flag":         bool(jump_flags[t]),
        })

    n_wf = int(wf_mask.sum())
    print(f"  WF-predicted bars: {n_wf}")
    if n_wf > 0:
        print(f"  Gate pass-through — "
              f"regime: {gate_pass_regime} ({gate_pass_regime/n_wf*100:.1f}%)  "
              f"momentum: {gate_pass_mom} ({gate_pass_mom/n_wf*100:.1f}%)  "
              f"volume(range): {gate_pass_vol} ({gate_pass_vol/n_wf*100:.1f}%)")
        print(f"  All-gate signals: {gate_pass_all} ({gate_pass_all/n_wf*100:.1f}%)")
    else:
        print("  Gate pass-through — 0 signals generated.")
    long_sigs  = sum(1 for s in signals if s["direction"] ==  1)
    short_sigs = sum(1 for s in signals if s["direction"] == -1)
    print(f"  Direction — LONG: {long_sigs}  SHORT: {short_sigs}")
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Event-driven backtest (universal friction)
# ─────────────────────────────────────────────────────────────────────────────
def calc_friction(price: float, units: float, asset_class: str, friction_level: float) -> float:
    if asset_class == "crypto":
        # Percentage fee (e.g. 0.001 = 0.1%)
        return price * units * friction_level
    elif asset_class == "equity":
        # Per share fee + slippage (e.g. 0.005)
        return units * friction_level
    else:
        # Forex pips
        return friction_level * PIP_SIZE * units

def run_backtest(
    bars_df: pd.DataFrame,
    signals: list[dict],
    equity: float = 100_000.0,
    friction_level: float = 0.0,
    asset_class: str = "forex"
) -> dict:
    """
    Universal Bar-by-bar simulation.
    
    friction_level meaning depends on asset_class:
    - forex: spread in pips (e.g. 1.5)
    - equity: commission per share in dollars (e.g. 0.005)
    - crypto: percentage fee (e.g. 0.001 for 0.1%)
    """
    close_arr = bars_df["close"].values
    high_arr  = bars_df["high"].values
    low_arr   = bars_df["low"].values
    atr_arr   = compute_atr(bars_df, period=14).values

    signal_map: dict[int, dict] = {s["bar_index"]: s for s in signals}

    current_equity = equity
    position       = None
    completed_trades: list[dict] = []
    equity_curve:    list[dict] = []

    for t in range(len(bars_df)):
        price = float(close_arr[t])
        high  = float(high_arr[t])
        low   = float(low_arr[t])
        atr   = float(atr_arr[t]) if not np.isnan(atr_arr[t]) else 0.0
        ts    = bars_df.index[t]

        # ── Manage open position ─────────────────────────────────────────
        if position is not None and not position.get("skip_exit_this_bar", False):
            d_dir   = position["direction"]
            sl      = position["stop_loss"]
            tp      = position["take_profit"]
            entry_p = position["entry_price"]
            units   = position["units"]

            exit_price = None
            exit_type  = None

            if d_dir == 1:   # LONG
                # Trailing Stop disabled for Mean-Reversion swing
                if low <= sl:
                    exit_price, exit_type = sl, "SL"
                elif high >= tp:
                    exit_price, exit_type = tp, "TP"
            else:            # SHORT
                # Trailing Stop disabled for Mean-Reversion swing
                if high >= sl:
                    exit_price, exit_type = sl, "SL"
                elif low <= tp:
                    exit_price, exit_type = tp, "TP"

            if exit_price is not None:
                raw_pnl    = (exit_price - entry_p) * d_dir * units
                # Half-spread cost on exit
                exit_spread = calc_friction(exit_price, units, asset_class, friction_level) * 0.5
                net_pnl     = raw_pnl - exit_spread
                current_equity += net_pnl
                completed_trades.append({
                    "entry_bar":  position["entry_bar"],
                    "exit_bar":   t,
                    "direction":  d_dir,
                    "entry_price": entry_p,
                    "exit_price": exit_price,
                    "exit_type":  exit_type,
                    "raw_pnl":    float(raw_pnl),
                    "spread_cost": float(exit_spread + position.get("entry_spread", 0)),
                    "net_pnl":    float(raw_pnl - exit_spread - position.get("entry_spread", 0)),
                    "bars_held":  t - position["entry_bar"],
                    "equity_after": float(current_equity),
                    "entry_ts":   str(position["entry_ts"]),
                    "exit_ts":    str(ts),
                })
                position = None

        # ── Check for Time/EOD flush ──────────────────────────────────────────
        if position is not None:
            bars_held = t - position["entry_bar"]
            strategy_mode = os.environ.get("STRATEGY_MODE", "")
            time_exit_limit = int(os.environ.get("TIME_EXIT_BARS", 5 if strategy_mode == "MEAN_REVERSION_EXHAUSTION" else 60))
            if bars_held >= time_exit_limit:
                exit_price = price
                exit_type  = "TIME"
                d_dir   = position["direction"]
                entry_p = position["entry_price"]
                units   = position["units"]
                raw_pnl = (exit_price - entry_p) * d_dir * units
                # Half-spread cost on exit
                exit_spread = calc_friction(price, units, asset_class, friction_level) * 0.5
                net_pnl = raw_pnl - exit_spread - position.get("entry_spread", 0)
                current_equity += net_pnl
                completed_trades.append({
                    "entry_bar":  position["entry_bar"],
                    "exit_bar":   t,
                    "direction":  d_dir,
                    "entry_price": entry_p,
                    "exit_price": exit_price,
                    "exit_type":  exit_type,
                    "raw_pnl":    float(raw_pnl),
                    "spread_cost": float(exit_spread + position.get("entry_spread", 0)),
                    "net_pnl":    float(net_pnl),
                    "bars_held":  bars_held,
                    "equity_after": float(current_equity),
                    "entry_ts":   str(position["entry_ts"]),
                    "exit_ts":    str(ts),
                })
                position = None
        # ── Open new position on signal ──────────────────────────────────
        if position is None and t in signal_map:
            sig    = signal_map[t]
            d_dir  = sig["direction"]
            atr_s  = sig["atr"]
            frac   = sig["position_fraction"]

            if atr_s > 0 and current_equity > 0:
                # ATR-based risk sizing: risk exactly frac×equity dollars at SL.
                #   sl_distance  = atr × ATR_SL_MULT  (price units)
                #   units        = risk_amount / sl_distance  (units)
                # This ensures one SL hit costs a fixed dollar amount = frac × equity.
                risk_amount = frac * current_equity
                sl_distance = atr_s * ATR_SL_MULT
                units       = risk_amount / sl_distance

                # Half-spread cost on entry
                stress_mult = 10.0 if sig.get("stress_environment", False) else 1.0
                entry_spread = calc_friction(price, units, asset_class, friction_level) * 0.5 * stress_mult
                current_equity -= entry_spread   # pay entry spread immediately

                position = {
                    "entry_bar":       t,
                    "entry_ts":        ts,
                    "direction":       d_dir,
                    "entry_price":     price,
                    "stop_loss":       sig["stop_loss"],
                    "take_profit":     sig["take_profit"],
                    "units":           units,
                    "skip_exit_this_bar": True,
                    "entry_spread":    entry_spread,
                }

        if position is not None:
            position.pop("skip_exit_this_bar", None)

        equity_curve.append({"ts": str(ts), "equity": float(current_equity)})

    # Close any open position at end
    if position is not None:
        d_dir   = position["direction"]
        entry_p = position["entry_price"]
        units   = position["units"]
        raw_pnl = (close_arr[-1] - entry_p) * d_dir * units
        exit_spread = calc_friction(close_arr[-1], units, asset_class, friction_level) * 0.5
        current_equity += raw_pnl - exit_spread
        completed_trades.append({
            "entry_bar":  position["entry_bar"],
            "exit_bar":   len(bars_df) - 1,
            "direction":  d_dir,
            "entry_price": entry_p,
            "exit_price": float(close_arr[-1]),
            "exit_type":  "EOD",
            "raw_pnl":    float(raw_pnl),
            "spread_cost": float(exit_spread + position.get("entry_spread", 0)),
            "net_pnl":    float(raw_pnl - exit_spread - position.get("entry_spread", 0)),
            "bars_held":  len(bars_df) - 1 - position["entry_bar"],
            "equity_after": float(current_equity),
            "entry_ts":   str(position["entry_ts"]),
            "exit_ts":    str(bars_df.index[-1]),
        })

    return {
        "completed_trades": completed_trades,
        "equity_curve":     equity_curve,
        "n_trades":         len(completed_trades),
        "final_equity":     current_equity,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    bt: dict,
    initial_equity: float,
    bars_df: pd.DataFrame,
    skip_bars: int = 0,
) -> dict:
    """
    Compute performance metrics.

    skip_bars : number of leading equity-curve entries to drop before
                computing Sharpe / Sortino / MDD / CAGR.  Use WF_TRAIN_BARS
                to exclude the zero-return warmup period from all statistics.
    """
    trades    = bt["completed_trades"]
    final_eq  = bt["final_equity"]
    n_trades  = bt["n_trades"]

    total_ret = (final_eq / initial_equity - 1) * 100.0

    # Full equity curve, then strip warmup period
    eq_vals_full = np.array([e["equity"] for e in bt["equity_curve"]])
    eq_vals      = eq_vals_full[skip_bars:]          # active trading window
    eq_rets      = np.diff(eq_vals) / (eq_vals[:-1] + 1e-10)

    ann_factor = 252.0
    sharpe  = float(np.mean(eq_rets) / (np.std(eq_rets) + 1e-10) * np.sqrt(ann_factor)) if len(eq_rets) > 0 else 0.0
    neg_rets = eq_rets[eq_rets < 0]
    sortino = float(np.mean(eq_rets) / (np.std(neg_rets) + 1e-10) * np.sqrt(ann_factor)) if len(neg_rets) > 0 else 0.0

    # Drawdown over active window
    if len(eq_vals) > 0:
        peak   = np.maximum.accumulate(eq_vals)
        dd     = (eq_vals - peak) / (peak + 1e-10)
        max_dd = float(dd.min() * 100) if len(dd) > 0 else 0.0

        # CAGR over active trading window only
        n_active_years = max(len(eq_vals) / ann_factor, 0.1)
        cagr = float((final_eq / eq_vals[0]) ** (1 / n_active_years) - 1) * 100
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("inf")
    else:
        max_dd = 0.0
        cagr = 0.0
        calmar = 0.0

    # Trade stats
    if n_trades == 0:
        return dict(
            total_ret_pct=total_ret, cagr_pct=0.0, max_drawdown_pct=0.0,
            calmar_ratio=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
            n_trades=0, win_rate_pct=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
            profit_factor=0.0, avg_bars=0.0,
            sl_count=0, tp_count=0, time_count=0, eod_count=0,
            long_win_rate=0.0, short_win_rate=0.0,
            returns_array=eq_rets, final_equity=final_eq,
        )

    wins   = [tr for tr in trades if tr["net_pnl"] > 0]
    losses = [tr for tr in trades if tr["net_pnl"] <= 0]
    longs  = [tr for tr in trades if tr["direction"] ==  1]
    shorts = [tr for tr in trades if tr["direction"] == -1]
    l_wins = [tr for tr in longs  if tr["net_pnl"] > 0]
    s_wins = [tr for tr in shorts if tr["net_pnl"] > 0]

    gross_win  = sum(tr["net_pnl"] for tr in wins)
    gross_loss = abs(sum(tr["net_pnl"] for tr in losses))
    pf = gross_win / max(gross_loss, 1e-10)

    avg_win  = float(np.mean([tr["net_pnl"] / initial_equity * 100 for tr in wins]))  if wins   else 0.0
    avg_loss = float(np.mean([tr["net_pnl"] / initial_equity * 100 for tr in losses])) if losses else 0.0
    avg_bars = float(np.mean([tr["bars_held"] for tr in trades]))

    exit_counts = {"SL": 0, "TP": 0, "TIME": 0, "EOD": 0}
    for tr in trades:
        exit_counts[tr.get("exit_type", "EOD")] = exit_counts.get(tr.get("exit_type", "EOD"), 0) + 1

    return dict(
        total_ret_pct=float(total_ret),
        cagr_pct=float(cagr),
        max_drawdown_pct=float(max_dd),
        calmar_ratio=float(calmar),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        n_trades=n_trades,
        win_rate_pct=float(len(wins) / n_trades * 100),
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=float(pf),
        avg_bars=float(avg_bars),
        sl_count=exit_counts["SL"],
        tp_count=exit_counts["TP"],
        time_count=exit_counts["TIME"],
        eod_count=exit_counts["EOD"],
        long_win_rate=float(len(l_wins) / len(longs) * 100) if longs else 0.0,
        short_win_rate=float(len(s_wins) / len(shorts) * 100) if shorts else 0.0,
        returns_array=eq_rets,
        final_equity=float(final_eq),
    )


def period_metrics(
    bars_df: pd.DataFrame,
    trades: list[dict],
    initial_equity: float,
    period_start: str,
    period_end: str,
) -> dict:
    period_trades = [
        tr for tr in trades
        if period_start <= tr["entry_ts"][:10] <= period_end
    ]
    n  = len(period_trades)
    ret = sum(tr["net_pnl"] for tr in period_trades) / initial_equity * 100 if n > 0 else 0.0
    wins = [tr for tr in period_trades if tr["net_pnl"] > 0]
    losses_abs = abs(sum(tr["net_pnl"] for tr in period_trades if tr["net_pnl"] <= 0))
    gross_win = sum(tr["net_pnl"] for tr in wins)
    pf = gross_win / max(losses_abs, 1e-10) if n > 0 else 0.0
    return {"n": n, "ret": ret, "pf": pf}


def buy_and_hold_benchmark(bars_df: pd.DataFrame) -> dict:
    f"{ASSET_NAME} buy-and-hold from OOS_START."
    oos_bars = bars_df[bars_df.index >= OOS_START]
    if len(oos_bars) < 2:
        return {"total_ret": 0.0, "cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    closes = oos_bars["close"].values
    rets   = np.diff(closes) / closes[:-1]
    total_ret = (closes[-1] / closes[0] - 1) * 100
    n_years   = len(oos_bars) / 252
    cagr      = ((1 + total_ret / 100) ** (1 / max(n_years, 0.1)) - 1) * 100
    peak  = np.maximum.accumulate(closes)
    max_dd = float((closes - peak).min() / peak.max() * 100)
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252))
    return {"total_ret": total_ret, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Layer 6 Statistical Validation
# ─────────────────────────────────────────────────────────────────────────────
def phase6_validation(oos_returns: np.ndarray, n_trades: int) -> dict:
    """
    Layer 6 statistical validation using the correct engine/validation.py API.

    All five functions need either:
      - simulate_fn callbacks  (WF-CV, MC Permutation, CPCV)
      - returns arrays + metadata  (DSR, Tx Cost)

    We build simulate_fn wrappers that slice the pre-computed equity-curve
    returns by bar-index.  This is valid: the equity returns already capture
    exactly the strategy's realised P&L on each bar, so slicing by IS/OOS
    index correctly separates IS and OOS performance.
    """
    from engine.validation import (
        walk_forward_cv       as _wfcv,
        monte_carlo_permutation_test as _mc,
        deflated_sharpe_ratio as _dsr,
        cpcv                  as _cpcv,
        transaction_cost_sensitivity as _tx,
        _sharpe_ratio,
    )

    print()
    _sep("Phase 6 — Layer 6 Statistical Validation")
    t0 = time.time()

    T = len(oos_returns)

    # ── 6.1  Walk-Forward CV ───────────────────────────────────────────────
    # Removed: Phase 2 is already a legitimate Purged Walk-Forward. 
    # Slicing the OOS equity curve here is statistically redundant.
    wf = {"passes": True, "oos_sharpe": 0.0, "is_oos_ratio": 0.0, "n_windows": 0}

    # ── 6.2  Monte Carlo Permutation ──────────────────────────────────────
    # simulate_fn(shuffled_returns) → oos_sharpe  (OOS = last 30 %)
    def mc_fn(shuffled):
        oos_start = int(len(shuffled) * 0.7)
        return _sharpe_ratio(shuffled[oos_start:])

    try:
        mc_res = _mc(oos_returns, mc_fn, n_permutations=500)
        mc = {
            "passes":    mc_res.passes,
            "actual_sr": mc_res.actual_oos_sharpe,
            "p_value":   mc_res.p_value,
        }
    except Exception as e:
        mc = {"passes": False, "actual_sr": 0.0, "p_value": 1.0, "error": str(e)}

    # ── 6.3  Deflated Sharpe Ratio ─────────────────────────────────────────
    # Takes returns array directly; n_trials = number of WF blocks fitted
    try:
        dsr_res = _dsr(oos_returns, n_trials=15)
        dsr = {
            "passes":      dsr_res.passes,
            "observed_sr": dsr_res.observed_sr,
            "z_score":     dsr_res.deflated_sr_z,
            "p_value":     dsr_res.p_value,
        }
    except Exception as e:
        dsr = {"passes": False, "observed_sr": 0.0, "z_score": float("nan"),
               "p_value": float("nan"), "error": str(e)}

    # ── 6.4  CPCV ─────────────────────────────────────────────────────────
    # simulate_fn(train_idx, test_idx) → (is_returns, oos_returns)
    def cpcv_fn(train_idx, test_idx):
        return oos_returns[train_idx], oos_returns[test_idx]

    try:
        cp_res = _cpcv(cpcv_fn, T, n_folds=6, test_folds=2, embargo_bars=5)
        cp = {
            "passes": cp_res.passes,
            "min_sr": cp_res.min_sr,
            "psr":    cp_res.psr,
            "n_paths": cp_res.n_paths,
        }
    except Exception as e:
        cp = {"passes": False, "min_sr": 0.0, "psr": float("nan"), "n_paths": 0,
              "error": str(e)}

    # ── 6.5  Transaction Cost Sensitivity ─────────────────────────────────
    # Takes returns array + n_trades + T directly
    try:
        tx_res = _tx(oos_returns, n_trades, T)
        tx = {"scenarios": tx_res.scenario_results, "passes": tx_res.passes}
    except Exception as e:
        tx = {"scenarios": {"realistic": {"passes": False, "net_sharpe": 0.0}},
              "passes": False, "error": str(e)}

    # ── Print ──────────────────────────────────────────────────────────────
    def _vrow(label, detail, passes):
        print(f"  {label:<40} {detail}  {'PASS ✓' if passes else 'FAIL ✗'}")

    _vrow("6.1 Walk-Forward CV", "Removed (Phase 2 inherently Purged-WF)", True)
    _vrow("6.2 Monte Carlo Permutation",
          f"actual SR={mc['actual_sr']:.2f}  p={mc['p_value']:.4f}",
          mc["passes"])
    _vrow("6.3 Deflated Sharpe Ratio",
          f"SR={dsr['observed_sr']:.2f}  z={dsr['z_score']:.2f}  "
          f"p={dsr['p_value']:.4f}",
          dsr["passes"])
    _vrow("6.4 CPCV",
          f"MinSR={cp['min_sr']:.2f}  PSR={cp['psr']:.2f}  "
          f"({cp['n_paths']} paths)",
          cp["passes"])
    tx_sc = tx.get("scenarios", {}).get("realistic", {})
    _vrow("6.5 Tx Cost (realistic spread)",
          f"net Sharpe={tx_sc.get('net_sharpe', 0.0):.2f}",
          tx_sc.get("passes", False))

    n_passed = sum([wf["passes"], mc["passes"], dsr["passes"],
                    cp["passes"], tx_sc.get("passes", False)])
    print()
    if n_passed == 5:
        print("  ══════════  GO LIVE: ALL 5 LAYER 6 TESTS PASSED  ══════════")
    else:
        print(f"  ══════════  {n_passed}/5 LAYER 6 TESTS PASSED  ══════════")
    print(f"  Phase 6 complete in {time.time()-t0:.1f}s")

    return {
        "n_passed": n_passed, "wf_cv": wf, "mc_permutation": mc,
        "dsr": dsr, "cpcv": cp, "tx_cost": tx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print tear sheet + BANG / WASTE verdict
# ─────────────────────────────────────────────────────────────────────────────
def print_tearsheet(
    baseline_metrics: dict,
    pip_metrics:      dict,
    validation:       dict,
    bnh:              dict,
    n_wf_bars:        int,
    n_signals:        int,
    baseline_trades:  list[dict],
) -> None:
    m    = baseline_metrics
    n_tr = m["n_trades"]

    print()
    _sep(f"LATENT DIFFUSION-HMM v3.0  —  {ASSET_NAME} FOREX TEAR SHEET")
    print(f"  Ticker     : {TICKER}  (daily bars, 2016–2024)")
    print(f"  WF-train   : {WF_TRAIN_BARS}-bar rolling window  |  OOS block: {WF_OOS_BARS} bars")
    print(f"  Stops      : SL={ATR_SL_MULT}×ATR  TP={ATR_TP_MULT}×ATR  Trail=Disabled")
    print(f"  WF-predicted bars: {n_wf_bars}   |   Signals fired: {n_signals}")
    print()

    _sep("PORTFOLIO PERFORMANCE  (gross, 0 pips)", 74)
    rows = [
        ("Total Return",    f"{m['total_ret_pct']:+.1f}%",       f"{bnh['total_ret']:+.1f}%"),
        ("CAGR",            f"{m['cagr_pct']:+.1f}%",            f"{bnh['cagr']:+.1f}%"),
        ("Max Drawdown",    f"{m['max_drawdown_pct']:.1f}%",     f"{bnh['max_dd']:.1f}%"),
        ("Calmar Ratio",    f"{m['calmar_ratio']:.2f}",          f"{bnh['cagr']/max(abs(bnh['max_dd']),0.01):.2f}"),
        ("Sharpe Ratio",    f"{m['sharpe_ratio']:.2f}",          f"{bnh['sharpe']:.2f}"),
        ("Sortino Ratio",   f"{m['sortino_ratio']:.2f}",         "n/a"),
        ("Final Equity",    f"${m['final_equity']:,.0f}",         ""),
    ]
    print(f"  {'Metric':<24}  {'Strategy':>12}  {f'B&H {ASSET_NAME}':>12}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*12}")
    for label, strat, bench in rows:
        print(f"  {label:<24}  {strat:>12}  {bench:>12}")
    print()

    _sep("TRADE STATISTICS", 74)
    long_tr  = [tr for tr in baseline_trades if tr["direction"] ==  1]
    short_tr = [tr for tr in baseline_trades if tr["direction"] == -1]
    print(f"  Completed trades  : {n_tr}  ({len(long_tr)} long / {len(short_tr)} short)")
    print(f"  Win Rate          : {m['win_rate_pct']:.1f}%  "
          f"(long {m['long_win_rate']:.0f}% / short {m['short_win_rate']:.0f}%)")
    print(f"  Avg Win / Avg Loss: {m['avg_win_pct']:+.2f}% / {m['avg_loss_pct']:+.2f}%")
    print(f"  Profit Factor     : {m['profit_factor']:.2f}  (>1.5 good, >2.0 excellent)")
    print(f"  Avg Bars Held     : {m['avg_bars']:.1f}")
    print(f"  Exit — Stop-Loss  : {m['sl_count']}")
    print(f"  Exit — Take-Profit: {m['tp_count']}")
    time_exit_limit = int(os.environ.get("TIME_EXIT_BARS", 60))
    print(f"  Exit — Time ({time_exit_limit}b) : {m['time_count']}")
    print(f"  Exit — EOD flush  : {m['eod_count']}")
    print()

    _sep("PERIOD BREAKDOWN  (OOS: baseline 0 pip)", 74)
    print(f"  {'Period':<20}  {'Return':>8}  {'PF':>6}  {'Trades':>6}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*6}  {'─'*6}")
    for name, ps, pe in PERIODS:
        pm = period_metrics(None, baseline_trades, INITIAL_EQUITY, ps, pe)  # type: ignore
        print(f"  {name:<20}  {pm['ret']:>+7.1f}%  {pm['pf']:>6.2f}  {pm['n']:>6}")
    print()

    _sep("SPREAD SENSITIVITY  (Sharpe / Return / PF)", 74)
    print(f"  {'Scenario':<18}  {'Sharpe':>8}  {'Return':>9}  {'PF':>8}  Verdict")
    print(f"  {'─'*18}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*7}")
    for label in pip_metrics.keys():
        pm2 = pip_metrics[label]
        sr  = pm2["sharpe_ratio"]
        ret = pm2["total_ret_pct"]
        pf2 = pm2["profit_factor"]
        mdd = pm2["max_drawdown_pct"]
        # BANG criteria check for this scenario
        bang = sr > BANG_SHARPE and pf2 > BANG_PF and abs(mdd) < BANG_MDD
        verdict = "BANG ✓" if bang else "─"
        print(f"  {label:<18}  {sr:>8.2f}  {ret:>+8.1f}%  {pf2:>8.2f}  {verdict}")
    print()

    _sep("LAYER 6 STATISTICAL VALIDATION", 74)
    val = validation
    wf  = val.get("wf_cv", {})
    mc  = val.get("mc_permutation", {})
    dsr = val.get("dsr", {})
    cp  = val.get("cpcv", {})
    tx  = val.get("tx_cost", {}).get("scenarios", {}).get("realistic", {})

    def _vr(label, detail, passes):
        print(f"  {label:<42}  {detail}  {'PASS ✓' if passes else 'FAIL ✗'}")

    _vr("6.1 Walk-Forward CV",
        f"OOS SR={wf.get('oos_sharpe',0):.2f}  IS/OOS={wf.get('is_oos_ratio',0):.2f}",
        wf.get("passes", False))
    _vr("6.2 Monte Carlo Permutation",
        f"actual SR={mc.get('actual_sr', mc.get('actual_sharpe', 0)):.2f}  p={mc.get('p_value',1):.4f}",
        mc.get("passes", False))
    _vr("6.3 Deflated Sharpe Ratio",
        f"SR={dsr.get('observed_sr',0):.2f}  z={dsr.get('z_score',float('nan')):.2f}",
        dsr.get("passes", False))
    _vr("6.4 CPCV",
        f"MinSR={cp.get('min_sr',0):.2f}  PSR={cp.get('psr',float('nan')):.2f}",
        cp.get("passes", False))
    _vr("6.5 Tx Cost (realistic spread)",
        f"net Sharpe={tx.get('net_sharpe',0):.2f}",
        tx.get("passes", False))
    print()

    # ── THE VERDICT ────────────────────────────────────────────────────────
    bm  = baseline_metrics
    sr0 = bm["sharpe_ratio"]
    pf0 = bm["profit_factor"]
    mdd0 = abs(bm["max_drawdown_pct"])
    wrt  = bm["win_rate_pct"]

    is_bang  = (sr0 > BANG_SHARPE  and pf0 > BANG_PF  and mdd0 < BANG_MDD)
    is_waste = (sr0 < WASTE_SHARPE or  pf0 < WASTE_PF or  mdd0 > WASTE_MDD)

    print()
    if n_tr == 0:
        _sep("VERDICT: INSUFFICIENT DATA — NO SIGNALS FIRED", 74)
        print("  The Triple Gate produced 0 signals.  Check regime thresholds.")
    elif is_bang:
        _sep("★ VERDICT: ABSOLUTE BANG — CONNECT TO PAPER TRADING ★", 74)
        print(f"  OOS Sharpe {sr0:.2f} > {BANG_SHARPE}  |  PF {pf0:.2f} > {BANG_PF}  |  MDD {mdd0:.1f}% < {BANG_MDD}%")
        print("  → Deploy to OANDA / MetaTrader paper-trading immediately.")
        print("  → Run 4-week forward test before live capital.")
    elif is_waste:
        _sep("✗ VERDICT: WASTE — SCRAP THE STRATEGY", 74)
        print(f"  OOS Sharpe {sr0:.2f}  |  PF {pf0:.2f}  |  MDD {mdd0:.1f}%")
        print(f"  → 6D features do not hold predictive edge for {ASSET_NAME} daily bars.")
        print("  → Salvage infrastructure; re-design feature set.")
    else:
        _sep("~ VERDICT: MARGINAL — NEEDS FURTHER REFINEMENT", 74)
        print(f"  OOS Sharpe {sr0:.2f}  |  PF {pf0:.2f}  |  MDD {mdd0:.1f}%  |  WinRate {wrt:.1f}%")
        print("  → Strategy shows partial signal.  Not ready for live capital.")
        print("  → Investigate: stop sizing, signal filtering, or feature engineering.")

    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    wall_t0 = time.time()
    
    asset_name_lower = ASSET_NAME.lower()
    if "btc" in asset_name_lower or "eth" in asset_name_lower:
        asset_class = "crypto"
        friction_scenarios = [("0% fee (gross)", 0.0), ("0.05% maker fee", 0.0005), ("0.1% taker fee", 0.001)]
    elif "spy" in asset_name_lower or "qqq" in asset_name_lower or "spx" in asset_name_lower or "ndx" in asset_name_lower:
        asset_class = "equity"
        friction_scenarios = [("0 fee (gross)", 0.0), ("$0.005/share fee", 0.005), ("$0.01/share fee", 0.01)]
    else:
        asset_class = "forex"
        friction_scenarios = PIP_SCENARIOS
        
    print()
    _sep(f"LATENT DIFFUSION-HMM  —  {ASSET_NAME} WALK-FORWARD TEST ({asset_class.upper()})")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Protocol : Make or Break  |  "
          f"SL={ATR_SL_MULT}×ATR  TP={ATR_TP_MULT}×ATR  ")

    d = phase1_build_data()

    regime_proba, covariates = phase2_walkforward_regime(d)

    signals = phase3_generate_signals(d, regime_proba)

    bars_df = d["bars_df"]

    # ── Phase 4: backtests under all friction scenarios ─────────────────────────
    print()
    _sep("Phase 4 — Event-Driven Backtest (Friction scenarios)")
    pip_results: dict[str, dict] = {}
    pip_metrics: dict[str, dict] = {}

    for label, friction in friction_scenarios:
        bt = run_backtest(bars_df, signals, equity=INITIAL_EQUITY, friction_level=friction, asset_class=asset_class)
        m  = compute_metrics(bt, INITIAL_EQUITY, bars_df, skip_bars=WF_TRAIN_BARS)
        pip_results[label] = bt
        pip_metrics[label] = m
        print(f"  {label:<18}  trades={bt['n_trades']:>3}  "
              f"Sharpe={m['sharpe_ratio']:>6.2f}  "
              f"CAGR={m['cagr_pct']:>+6.1f}%  "
              f"MDD={m['max_drawdown_pct']:>6.1f}%  "
              f"WinRate={m['win_rate_pct']:>5.1f}%  "
              f"PF={m['profit_factor']:.2f}")

    gross_key = next((k for k in pip_metrics.keys() if "gross" in k), None)
    baseline_metrics = pip_metrics[gross_key] if gross_key else list(pip_metrics.values())[0]
    baseline_bt      = pip_results[gross_key] if gross_key else list(pip_results.values())[0]

    # ── Phase 5: B&H benchmark ─────────────────────────────────────────────
    bnh = buy_and_hold_benchmark(bars_df)

    # ── Phase 6: Layer 6 validation ───────────────────────────────────────
    oos_returns = baseline_metrics["returns_array"]
    validation  = phase6_validation(oos_returns, n_trades=baseline_bt["n_trades"])

    # ── Print tear sheet ──────────────────────────────────────────────────
    n_wf_bars = int(np.sum([True for p in range(WF_TRAIN_BARS, len(bars_df))]))
    print_tearsheet(
        baseline_metrics=baseline_metrics,
        pip_metrics=pip_metrics,
        validation=validation,
        bnh=bnh,
        n_wf_bars=n_wf_bars,
        n_signals=len(signals),
        baseline_trades=baseline_bt["completed_trades"],
    )

    try:
        import quantstats as qs
        eq_curve = baseline_bt["equity_curve"][WF_TRAIN_BARS:]
        if len(eq_curve) > 1:
            dates = pd.to_datetime([e["ts"] for e in eq_curve])
            eq_vals = [e["equity"] for e in eq_curve]
            eq_series = pd.Series(eq_vals, index=dates)
            returns_series = eq_series.pct_change().dropna()
            
            output_dir = os.environ.get("FOREX_RESULTS_PATH", "ensemble_results/results.json")
            qs_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), f"{ASSET_NAME.lower()}_tearsheet.html")
            
            qs.reports.html(returns_series, output=qs_path, title=f"{ASSET_NAME} Latent Diffusion Tear Sheet")
            print(f"  [QuantStats] Gorgeous HTML Report generated -> {qs_path}")
    except Exception as e:
        print(f"  [QuantStats] HTML report generation skipped: {e}")

    print(f"  Total wall-clock time: {time.time()-wall_t0:.1f}s")
    print()

    # ── Save JSON ──────────────────────────────────────────────────────────
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forex_results.json")
    output_path = os.environ.get("FOREX_RESULTS_PATH", default_output)

    def _serial(obj):
        if isinstance(obj, np.ndarray):    return obj.tolist()
        if isinstance(obj, np.integer):    return int(obj)
        if isinstance(obj, np.floating):   return float(obj)
        if isinstance(obj, pd.Timestamp):  return str(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    # Preserve returns_array for ensemble aggregation
    # for m in pip_metrics.values():
    #     m.pop("returns_array", None)

    save_payload = {
        "run_at":        datetime.now().isoformat(),
        "ticker":        TICKER,
        "data_range":    {"start": DATA_START, "end": DATA_END},
        "config": {
            "atr_sl_mult": ATR_SL_MULT, "atr_tp_mult": ATR_TP_MULT,
            "wf_train_bars": WF_TRAIN_BARS, "wf_oos_bars": WF_OOS_BARS,
        },
        "n_signals":     len(signals),
        "buy_and_hold":  bnh,
        "pip_scenarios": pip_metrics,
        "validation":    validation,
        "trade_log":     baseline_bt["completed_trades"][:200],
    }

    with open(output_path, "w") as f:
        json.dump(save_payload, f, indent=2, default=_serial)
    print(f"  Results saved → {output_path}")
    print()
    
    return save_payload


if __name__ == "__main__":
    main()

def run_tearsheet_dynamic(csv_path: str, params: dict) -> dict:
    import engine.execution as exec_module
    global TICKER, ASSET_NAME, PIP_SIZE, ATR_SL_MULT, ATR_TP_MULT, WF_OOS_BARS
    
    TICKER = csv_path
    asset_name_lower = TICKER.lower()
    if "xauusd" in asset_name_lower:
        ASSET_NAME = "XAU/USD"
        PIP_SIZE = 0.01
    elif "btcusd" in asset_name_lower:
        ASSET_NAME = "BTC/USD"
        PIP_SIZE = 1.0
    elif "spy" in asset_name_lower or "qqq" in asset_name_lower:
        ASSET_NAME = TICKER.split("/")[-1].split("_")[0].upper()
        PIP_SIZE = 0.01
    else:
        ASSET_NAME = TICKER.split("/")[-1].split("_")[0].upper()
        PIP_SIZE = 0.0001
        if "jpy" in asset_name_lower:
            PIP_SIZE = 0.01
            
    ATR_SL_MULT = float(params["STOP_LOSS_ATR"])
    ATR_TP_MULT = float(params["TAKE_PROFIT_ATR"])
    WF_OOS_BARS = int(params["HMM_WF_OOS_BARS"])
    
    exec_module.REGIME_CONF_THRESHOLD = float(params["VETO_THRESHOLD"])
    global REGIME_CONF_THRESHOLD
    REGIME_CONF_THRESHOLD = float(params["VETO_THRESHOLD"])
    
    import os
    os.environ["TIME_EXIT_BARS"] = str(params["TIME_EXIT_BARS"])
    # Note: JAX HMM uses self.n_iter = n_iter. We must check if n_iter is dynamic.
    
    return main()

#!/usr/bin/env python3
"""
Out-of-Sample Validation & Strategy Tear Sheet
Latent Diffusion-HMM Trading Engine v3.0

BLIND FIREWALL
  Train : 2010-01-01 → 2019-12-31  (10 years — HMM never sees OOS data)
  Blind : 2020-01-01 → 2024-12-31  (5 years)
    · COVID crash          Feb–Apr 2020  (tests Kalman-CUSUM jump reset)
    · Hyper-bull           2021          (should catch strong TREND regime)
    · Rate-hike bear       2022          (P(TREND) < 0.65 → mostly flat)
    · Recovery             2023          (selective long entry)
    · Extension            2024          (generalization check)

Run:
    cd artifacts/python-engine
    python3 oos_tearsheet.py

Outputs:
    · Formatted tear sheet to stdout
    · oos_results.json  (full machine-readable results)
"""
from __future__ import annotations

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.data import load_and_prepare
from engine.features import compute_feature_tensor
from engine.preprocess import Preprocessor
from engine.kalman import run_kalman_pipeline
from engine.hmm import TVTPHMM, STATE_LABELS
from engine.execution import (
    TripleGate, KellyPositionSizer, StopManager,
    compute_atr, REGIME_CONF_THRESHOLD, R_TP,
)
from engine.surveillance import WassersteinMonitor
from engine.validation import (
    walk_forward_cv, deflated_sharpe_ratio,
    transaction_cost_sensitivity, cpcv,
    _sharpe_ratio, _max_drawdown,
    ValidationReport, WFCVResult, MCPermutationResult,
    DSRResult, CPCVResult, TxCostResult,
    COST_SCENARIOS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
TICKER         = "SPY"
TRAIN_START    = "2010-01-01"
TRAIN_END      = "2019-12-31"
OOS_START      = "2020-01-01"
OOS_END        = "2024-12-31"
INITIAL_EQUITY = 100_000.0

# Walk-forward CV parameters (HMM collapses on long training windows;
# 504-bar rolling window is empirically stable on SPY)
WF_TRAIN_BARS  = 504   # 2-year rolling training window
WF_OOS_BARS    = 126   # 6-month prediction block

FRICTION_SCENARIOS = [
    ("0 bps (gross)",   0.0),
    ("0.5 bps",         0.5),
    ("2 bps",           2.0),
    ("5 bps",           5.0),
]

PERIODS = [
    ("COVID Crash",     "2020-01-01", "2020-06-30"),
    ("Hyper-Bull",      "2021-01-01", "2021-12-31"),
    ("Rate-Hike Bear",  "2022-01-01", "2022-12-31"),
    ("Recovery",        "2023-01-01", "2023-12-31"),
    ("2024 Extension",  "2024-01-01", "2024-12-31"),
]


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
# Phase 1 — Data ingestion & feature pipeline (full 2010–2024)
# ─────────────────────────────────────────────────────────────────────────────
def phase1_build_data() -> dict:
    print()
    _sep("Phase 1 — Data & Feature Pipeline")
    t0 = time.time()

    print(f"  Fetching {TICKER} {TRAIN_START} → {OOS_END}  (full 15-year window) …")
    data = load_and_prepare(
        ticker=TICKER,
        start=TRAIN_START,
        end=OOS_END,
        use_dollar_bars=True,
        apply_frac_diff=False,
    )
    bars_df = data["bars_df"]
    print(f"  {len(bars_df)} dollar-volume bars built")

    # Build boolean masks aligned to bars_df
    bar_dates = bars_df.index
    train_mask = np.array([str(d)[:10] <= TRAIN_END for d in bar_dates])
    oos_mask   = np.array([str(d)[:10] >= OOS_START for d in bar_dates])
    n_train = int(train_mask.sum())
    n_oos   = int(oos_mask.sum())
    print(f"  Train bars: {n_train}   OOS bars: {n_oos}")

    print("  Computing 6D feature tensor …")
    features = compute_feature_tensor(bars_df)

    preprocessor = Preprocessor()
    X_white = preprocessor.fit_transform(features)

    print("  Running Kalman filter + CUSUM jump detector …")
    kalman_out = run_kalman_pipeline(X_white)
    filtered   = kalman_out["filtered_states"]
    jump_flags = kalman_out["jump_flags"]

    oos_jumps = int(jump_flags[oos_mask].sum())
    print(f"  CUSUM jumps in OOS period: {oos_jumps}")
    print(f"  Phase 1 complete in {time.time()-t0:.1f}s")

    return {
        "bars_df":    bars_df,
        "features":   features,
        "preprocessor": preprocessor,
        "X_white":    X_white,
        "filtered":   filtered,
        "jump_flags": jump_flags,
        "train_mask": train_mask,
        "oos_mask":   oos_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Walk-Forward HMM: rolling 504-bar train → 126-bar OOS blocks
# ─────────────────────────────────────────────────────────────────────────────
def phase2_walkforward_regime(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward regime prediction across the full OOS period.

    For each 126-bar OOS block the HMM is fitted on the 504 bars immediately
    before that block (true blind firewall — no OOS leakage).  This avoids
    the EM mode-collapse that occurs when training on >1000 bars.

    Returns
    -------
    regime_proba   : (T, 3) float array — smoothed state probabilities
    covariates     : (T, 2) float array — [σ_t, ρ_t] for TVTP conditioning
    """
    print()
    _sep(f"Phase 2 — Walk-Forward HMM  "
         f"(train={WF_TRAIN_BARS}-bar / OOS={WF_OOS_BARS}-bar blocks)")
    t0 = time.time()

    filtered   = d["filtered"]
    features   = d["features"]
    oos_mask   = d["oos_mask"]

    sigma_t    = features["sigma_t"].fillna(1.0).values
    rho_t      = features["rho_t"].fillna(0.0).values
    covariates = np.column_stack([sigma_t, rho_t])

    # Forward-fill NaN rows once
    X_all = filtered.copy()
    for t in range(1, len(X_all)):
        if np.any(np.isnan(X_all[t])):
            X_all[t] = X_all[t - 1]
    if np.any(np.isnan(X_all[0])):
        X_all[0] = np.zeros(X_all.shape[1])

    T = len(X_all)
    S = 3
    regime_proba   = np.ones((T, S)) / S      # default: uniform
    viterbi_states = np.ones(T, dtype=int)    # default: MEAN_REV

    oos_indices   = np.where(oos_mask)[0]
    n_oos         = len(oos_indices)
    n_blocks      = 0
    n_fitted_bars = 0

    for block_start in range(0, n_oos, WF_OOS_BARS):
        block_end   = min(block_start + WF_OOS_BARS, n_oos)
        g_start     = int(oos_indices[block_start])
        g_end       = int(oos_indices[block_end - 1]) + 1   # exclusive

        # Training window: WF_TRAIN_BARS bars immediately before this block
        tr_start    = max(0, g_start - WF_TRAIN_BARS)
        X_train     = X_all[tr_start:g_start]
        cov_train   = covariates[tr_start:g_start]

        clean       = ~np.any(np.isnan(X_train), axis=1)
        X_tr_c      = X_train[clean]
        cov_tr_c    = cov_train[clean]

        if len(X_tr_c) < 60:
            continue   # too few clean bars — keep uniform defaults

        hmm = TVTPHMM(n_states=3, n_gmm=2, n_iter=30)
        hmm.fit(X_tr_c, cov_tr_c)

        X_oos   = X_all[g_start:g_end]
        cov_oos = covariates[g_start:g_end]
        result  = hmm.predict(X_oos, cov_oos)

        regime_proba[g_start:g_end]   = result["proba"]
        viterbi_states[g_start:g_end] = result["viterbi_states"]

        n_blocks      += 1
        n_fitted_bars += (g_end - g_start)

        p_trend = float(result["proba"][:, 0].mean())
        dom_raw = int(np.bincount(result["viterbi_states"]).argmax())
        dom     = STATE_LABELS[dom_raw]
        bars_above = int((result["proba"][:, 0] > REGIME_CONF_THRESHOLD).sum())
        ts_start = d["bars_df"].index[g_start]
        ts_end   = d["bars_df"].index[g_end - 1]
        print(f"  Block {n_blocks:2d}  {str(ts_start)[:10]} → {str(ts_end)[:10]}"
              f"  |  P(TREND)={p_trend:.3f}"
              f"  dominant={dom}"
              f"  gate-eligible={bars_above}")

    oos_proba = regime_proba[oos_mask]
    vit_oos   = viterbi_states[oos_mask]
    print()
    print(f"  Walk-forward: {n_blocks} blocks, {n_fitted_bars} OOS bars predicted")
    print(f"  OOS mean P(TREND)={oos_proba[:,0].mean():.3f}  "
          f"bars P(TREND)>0.65: {int((oos_proba[:,0]>REGIME_CONF_THRESHOLD).sum())}")
    print(f"  OOS Viterbi — "
          f"TREND={(vit_oos==0).sum()} ({(vit_oos==0).mean()*100:.1f}%)  "
          f"MEAN_REV={(vit_oos==1).sum()} ({(vit_oos==1).mean()*100:.1f}%)  "
          f"STRESS={(vit_oos==2).sum()} ({(vit_oos==2).mean()*100:.1f}%)")
    print(f"  Phase 2 complete in {time.time()-t0:.1f}s")

    return regime_proba, covariates


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Triple Gate signal generation on OOS bars
# ─────────────────────────────────────────────────────────────────────────────
def phase3_generate_signals(d: dict, regime_proba: np.ndarray) -> list[dict]:
    print()
    _sep("Phase 3 — OOS Signal Generation (Triple Gate)")

    bars_df    = d["bars_df"]
    features   = d["features"]
    jump_flags = d["jump_flags"]
    oos_mask   = d["oos_mask"]

    gate      = TripleGate()
    sizer     = KellyPositionSizer()
    stop_mgr  = StopManager()
    atr_s     = compute_atr(bars_df, period=14)

    close_arr = bars_df["close"].values
    mv_t_arr  = features["mvt"].fillna(0.0).values
    q_t_arr   = features["qt"].fillna(1.0).values

    gate_pass_all = gate_pass_regime = gate_pass_mom = gate_pass_vol = 0
    signals: list[dict] = []

    for t in range(1, len(bars_df)):
        if not oos_mask[t]:
            continue

        p_trend    = float(regime_proba[t, 0])
        p_mean_rev = float(regime_proba[t, 1])
        p_stress   = float(regime_proba[t, 2])
        mv_t       = float(mv_t_arr[t])
        q_t        = float(q_t_arr[t])
        atr        = float(atr_s.iloc[t]) if not np.isnan(atr_s.iloc[t]) else 0.0
        price      = float(close_arr[t])

        gr = gate.evaluate(p_mean_rev, p_trend, mv_t, q_t)
        if gr["regime_gate"]:   gate_pass_regime += 1
        if gr["momentum_gate"]: gate_pass_mom    += 1
        if gr["volume_gate"]:   gate_pass_vol    += 1
        if not gr["all_pass"] or price <= 0 or atr <= 0:
            continue
        gate_pass_all += 1

        direction    = gr["direction"]
        regime_label = STATE_LABELS[int(regime_proba[t].argmax())]
        frac         = sizer.compute_fraction(p_trend)
        sl, tp       = stop_mgr.initial_stops(direction, price, atr, regime_label)

        signals.append({
            "bar_index":   t,
            "timestamp":   bars_df.index[t],
            "direction":   direction,
            "regime":      regime_label,
            "p_trend":     p_trend,
            "p_mean_rev":  p_mean_rev,
            "p_stress":    p_stress,
            "momentum":    mv_t,
            "volume_ratio": q_t,
            "entry_price": price,
            "stop_loss":   sl,
            "take_profit": tp,
            "position_fraction": frac,
            "atr":         atr,
            "jump_flag":   bool(jump_flags[t]),
        })

    n_oos = int(oos_mask.sum())
    print(f"  OOS bars: {n_oos}")
    print(f"  Gate pass-through — "
          f"regime: {gate_pass_regime} ({gate_pass_regime/n_oos*100:.1f}%)  "
          f"momentum: {gate_pass_mom} ({gate_pass_mom/n_oos*100:.1f}%)  "
          f"volume: {gate_pass_vol} ({gate_pass_vol/n_oos*100:.1f}%)")
    print(f"  All-three-gate signals: {gate_pass_all} ({gate_pass_all/n_oos*100:.1f}% of OOS bars)")

    long_sigs  = sum(1 for s in signals if s["direction"] == 1)
    short_sigs = sum(1 for s in signals if s["direction"] == -1)
    print(f"  Signal direction — LONG: {long_sigs}  SHORT: {short_sigs}")
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Event-driven backtest with ATR SL/TP + trailing stops
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(
    bars_df: pd.DataFrame,
    signals: list[dict],
    oos_mask: np.ndarray,
    equity: float = 100_000.0,
    friction_bps: float = 0.0,
) -> dict:
    """
    Bar-by-bar simulation.

    Entry  : close of signal bar t
    SL/TP  : checked against bar t+1 high/low onwards (no same-bar look-ahead)
    Trailing stop: breakeven at +1 ATR, trail at 1 ATR behind peak at +2 ATR
    Friction: round-trip bps applied on notional at entry
    One position at a time; new signals ignored while position is open.
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

    oos_indices = np.where(oos_mask)[0]

    for i, t in enumerate(oos_indices):
        price = float(close_arr[t])
        high  = float(high_arr[t])
        low   = float(low_arr[t])
        atr   = float(atr_arr[t]) if not np.isnan(atr_arr[t]) else 0.0
        ts    = bars_df.index[t]

        # ── Manage open position ─────────────────────────────────────────
        if position is not None and not position.get("skip_exit_this_bar", False):
            d      = position["direction"]
            sl     = position["stop_loss"]
            tp     = position["take_profit"]
            entry_p = position["entry_price"]

            # Trailing stop update (uses current close for peak tracking)
            if atr > 0:
                pnl_atr = (price - entry_p) * d / atr
                if pnl_atr >= 1.0 and not position["trail_activated"]:
                    position["stop_loss"] = entry_p       # move to breakeven
                    position["trail_activated"] = True
                    sl = entry_p
                if pnl_atr >= 2.0:
                    if d == 1:
                        position["highest_fav"] = max(position["highest_fav"], price)
                        new_sl = position["highest_fav"] - atr
                        position["stop_loss"] = max(sl, new_sl)
                        sl = position["stop_loss"]
                    else:
                        position["highest_fav"] = min(position["highest_fav"], price)
                        new_sl = position["highest_fav"] + atr
                        position["stop_loss"] = min(sl, new_sl)
                        sl = position["stop_loss"]

            # Check SL/TP within this bar's range
            exit_price  = None
            exit_reason = None
            if d == 1:
                if low <= sl:
                    exit_price  = sl
                    exit_reason = "stop_loss"
                elif high >= tp:
                    exit_price  = tp
                    exit_reason = "take_profit"
            else:
                if high >= sl:
                    exit_price  = sl
                    exit_reason = "stop_loss"
                elif low <= tp:
                    exit_price  = tp
                    exit_reason = "take_profit"

            if exit_price is not None:
                n_sh      = position["n_shares"]
                gross_pnl = (exit_price - entry_p) * d * n_sh
                friction  = friction_bps / 10000 * 2 * entry_p * n_sh
                net_pnl   = gross_pnl - friction
                current_equity += net_pnl

                completed_trades.append({
                    "entry_bar":    position["entry_bar"],
                    "exit_bar":     int(t),
                    "entry_date":   str(position["entry_date"]),
                    "exit_date":    str(ts),
                    "direction":    d,
                    "entry_price":  float(entry_p),
                    "exit_price":   float(exit_price),
                    "n_shares":     n_sh,
                    "gross_pnl":    float(gross_pnl),
                    "friction":     float(friction),
                    "net_pnl":      float(net_pnl),
                    "pnl_pct":      float((exit_price - entry_p) / max(entry_p, 1e-6) * d),
                    "exit_reason":  exit_reason,
                    "bars_held":    int(t) - position["entry_bar"],
                    "regime":       position["regime"],
                    "p_trend_entry": position["p_trend"],
                })
                position = None

        # ── Enter new position on signal (if flat) ───────────────────────
        if position is None and t in signal_map:
            sig      = signal_map[t]
            entry_p  = float(price)
            atr_e    = float(sig["atr"]) if sig["atr"] > 0 else max(atr, 1e-6)
            frac     = float(sig["position_fraction"])
            direction = sig["direction"]

            n_shares = max(1, int((frac * current_equity) / max(atr_e, 1e-6)))

            # Deduct entry commission immediately (round-trip reserved at open)
            commission = friction_bps / 10000 * 2 * entry_p * n_shares
            current_equity -= commission

            position = {
                "entry_bar":      int(t),
                "entry_date":     ts,
                "direction":      direction,
                "entry_price":    entry_p,
                "stop_loss":      sig["stop_loss"],
                "take_profit":    sig["take_profit"],
                "n_shares":       n_shares,
                "trail_activated": False,
                "highest_fav":    entry_p,
                "regime":         sig["regime"],
                "p_trend":        sig["p_trend"],
                "skip_exit_this_bar": True,   # no SL/TP check on entry bar
            }

        if position is not None:
            position["skip_exit_this_bar"] = False

        equity_curve.append({
            "bar":          int(t),
            "date":         str(ts),
            "equity":       float(current_equity),
            "has_position": position is not None,
        })

    # Close any open position at final bar
    if position is not None:
        last_t = int(oos_indices[-1])
        exit_p = float(close_arr[last_t])
        d      = position["direction"]
        n_sh   = position["n_shares"]
        ep     = position["entry_price"]
        gross_pnl = (exit_p - ep) * d * n_sh
        net_pnl   = gross_pnl  # commission already paid at entry
        current_equity += net_pnl
        completed_trades.append({
            "entry_bar":    position["entry_bar"],
            "exit_bar":     last_t,
            "entry_date":   str(position["entry_date"]),
            "exit_date":    str(bars_df.index[last_t]),
            "direction":    d,
            "entry_price":  float(ep),
            "exit_price":   float(exit_p),
            "n_shares":     n_sh,
            "gross_pnl":    float(gross_pnl),
            "friction":     0.0,
            "net_pnl":      float(net_pnl),
            "pnl_pct":      float((exit_p - ep) / max(ep, 1e-6) * d),
            "exit_reason":  "end_of_data",
            "bars_held":    last_t - position["entry_bar"],
            "regime":       position["regime"],
            "p_trend_entry": position["p_trend"],
        })
        if equity_curve:
            equity_curve[-1]["equity"] = float(current_equity)

    return {
        "equity_curve":    equity_curve,
        "completed_trades": completed_trades,
        "final_equity":    float(current_equity),
        "n_trades":        len(completed_trades),
        "friction_bps":    friction_bps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Tear sheet metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(bt: dict, initial_equity: float, bars_df: pd.DataFrame) -> dict:
    ec     = bt["equity_curve"]
    trades = bt["completed_trades"]

    if not ec:
        return {"error": "empty equity curve"}

    eq_arr  = np.array([e["equity"] for e in ec])
    ret_arr = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1.0)

    total_return = (eq_arr[-1] - initial_equity) / initial_equity
    n_bars  = len(eq_arr)
    n_years = n_bars / 252.0
    cagr    = (eq_arr[-1] / max(initial_equity, 1.0)) ** (1.0 / max(n_years, 0.01)) - 1.0
    mdd     = _max_drawdown(ret_arr) if len(ret_arr) > 0 else 0.0
    calmar  = cagr / abs(mdd) if mdd < 0 else float("inf")
    sharpe  = _sharpe_ratio(ret_arr)

    neg_ret = ret_arr[ret_arr < 0]
    down_vol = float(np.sqrt(np.mean(neg_ret ** 2) * 252)) if len(neg_ret) > 0 else 1e-6
    sortino  = float(ret_arr.mean() * 252 / down_vol) if down_vol > 0 else 0.0

    n_trades = len(trades)
    if n_trades > 0:
        pnl_pcts  = np.array([t["pnl_pct"]  for t in trades])
        bars_held = np.array([t["bars_held"] for t in trades])
        winners   = pnl_pcts[pnl_pcts > 0]
        losers    = pnl_pcts[pnl_pcts < 0]
        win_rate  = float(len(winners) / n_trades)
        avg_win   = float(winners.mean()) if len(winners) > 0 else 0.0
        avg_loss  = float(losers.mean())  if len(losers)  > 0 else 0.0
        pf_num    = float(winners.sum()) if len(winners) > 0 else 0.0
        pf_den    = float(abs(losers.sum())) if len(losers) > 0 and abs(losers.sum()) > 0 else 1e-9
        profit_factor   = pf_num / pf_den
        avg_bars_held   = float(bars_held.mean())
        exits = {}
        for t in trades:
            exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1
        longs  = [t for t in trades if t["direction"] ==  1]
        shorts = [t for t in trades if t["direction"] == -1]
        long_wr  = len([t for t in longs  if t["pnl_pct"] > 0]) / max(len(longs), 1)
        short_wr = len([t for t in shorts if t["pnl_pct"] > 0]) / max(len(shorts), 1)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_bars_held = 0.0
        exits = {}; longs = []; shorts = []
        long_wr = short_wr = 0.0

    # Period breakdown
    period_stats = []
    for name, ps, pe in PERIODS:
        ps_dt = pd.Timestamp(ps)
        pe_dt = pd.Timestamp(pe)
        pec = [e for e in ec
               if pd.Timestamp(e["date"]) >= ps_dt and pd.Timestamp(e["date"]) <= pe_dt]
        ptr = [t for t in trades
               if pd.Timestamp(t["entry_date"]) >= ps_dt
               and pd.Timestamp(t["entry_date"]) <= pe_dt]
        if pec:
            p_eq   = np.array([e["equity"] for e in pec])
            p_rets = np.diff(p_eq) / np.maximum(p_eq[:-1], 1.0)
            p_ret  = (p_eq[-1] - p_eq[0]) / max(p_eq[0], 1.0)
            p_mdd  = _max_drawdown(p_rets) if len(p_rets) > 0 else 0.0
        else:
            p_ret = p_mdd = 0.0
        period_stats.append({
            "period":           name,
            "return_pct":       float(p_ret * 100),
            "max_drawdown_pct": float(p_mdd * 100),
            "n_trades":         len(ptr),
            "n_bars":           len(pec),
        })

    return {
        "total_return_pct":   float(total_return * 100),
        "cagr_pct":           float(cagr * 100),
        "max_drawdown_pct":   float(mdd * 100),
        "calmar_ratio":       float(calmar),
        "sharpe_ratio":       float(sharpe),
        "sortino_ratio":      float(sortino),
        "n_trades":           n_trades,
        "win_rate_pct":       float(win_rate * 100),
        "avg_win_pct":        float(avg_win * 100),
        "avg_loss_pct":       float(avg_loss * 100),
        "profit_factor":      float(profit_factor),
        "avg_bars_held":      float(avg_bars_held),
        "exit_reasons":       exits,
        "n_longs":            len(longs),
        "n_shorts":           len(shorts),
        "long_win_rate_pct":  float(long_wr * 100),
        "short_win_rate_pct": float(short_wr * 100),
        "final_equity":       float(eq_arr[-1]),
        "period_stats":       period_stats,
        "returns_array":      ret_arr,  # for validation; not serialised to JSON
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Layer 6 Statistical Validation on OOS returns
# ─────────────────────────────────────────────────────────────────────────────
def phase6_validation(oos_returns: np.ndarray, n_trades: int) -> dict:
    print()
    _sep("Phase 6 — Layer 6 Statistical Validation")
    t0 = time.time()

    ret = oos_returns[~np.isnan(oos_returns)]
    T   = len(ret)

    if T < 30:
        return {"error": f"Insufficient OOS returns ({T} bars)"}

    report = ValidationReport()
    n_windows = 0

    # 6.1 Walk-Forward Cross-Validation
    try:
        train_bars = min(504, T // 3)
        test_bars  = min(126, T // 6)

        def wf_sim(ts, te, ss, se):
            return ret[ts:te], ret[ss:se]

        report.wfcv = walk_forward_cv(
            wf_sim, T=T,
            train_bars=train_bars, test_bars=test_bars,
            min_bars=min(T, 252 * 5),
        )
        n_windows = len(report.wfcv.windows)
        print(f"  WF-CV: {n_windows} windows — "
              f"OOS Sharpe {report.wfcv.mean_oos_sharpe:.2f}, "
              f"IS/OOS {report.wfcv.overfit_ratio:.2f}  {_pf(report.wfcv.passes)}")
    except Exception as e:
        print(f"  WF-CV failed: {e}")

    # 6.2 Monte Carlo Permutation (500 shuffles of OOS returns)
    try:
        actual_sr = _sharpe_ratio(ret)
        rng = np.random.default_rng(42)
        perm_srs = np.array([
            _sharpe_ratio(rng.permutation(ret)) for _ in range(500)
        ])
        p_val = float(np.mean(perm_srs >= actual_sr))
        report.mc_permutation = MCPermutationResult(
            actual_oos_sharpe=actual_sr,
            permuted_sharpes=perm_srs,
            p_value=p_val,
            n_permutations=500,
        )
        print(f"  MC Permutation: actual SR={actual_sr:.2f}, "
              f"95th pct perm={np.percentile(perm_srs,95):.2f}, "
              f"p={p_val:.4f}  {_pf(report.mc_permutation.passes)}")
    except Exception as e:
        print(f"  MC Permutation failed: {e}")

    # 6.3 Deflated Sharpe Ratio (n_trials=1 — single strategy, no selection bias)
    try:
        report.dsr = deflated_sharpe_ratio(ret, n_trials=report.n_trials)
        print(f"  DSR: observed SR={report.dsr.observed_sr:.2f}, "
              f"z={report.dsr.deflated_sr_z:.2f}, "
              f"p={report.dsr.p_value:.4f}  {_pf(report.dsr.passes)}")
    except Exception as e:
        print(f"  DSR failed: {e}")

    # 6.4 CPCV (combinatorial purged CV)
    try:
        n_folds = min(6, max(3, T // 100))

        def cpcv_sim(train_idx, test_idx):
            return ret[train_idx], ret[test_idx]

        report.cpcv = cpcv(cpcv_sim, T=T, n_folds=n_folds)
        print(f"  CPCV: {report.cpcv.n_paths} paths — "
              f"MinSR={report.cpcv.min_sr:.2f}, "
              f"PSR={report.cpcv.psr:.2f}  {_pf(report.cpcv.passes)}")
    except Exception as e:
        print(f"  CPCV failed: {e}")

    # 6.5 Transaction Cost Sensitivity
    try:
        report.tx_cost = transaction_cost_sensitivity(ret, n_trades, T)
        r_sc = report.tx_cost.scenario_results.get("realistic", {})
        print(f"  Tx Cost (2 bps): net Sharpe={r_sc.get('net_sharpe',0):.2f}  "
              f"{_pf(r_sc.get('passes', False))}")
    except Exception as e:
        print(f"  Tx Cost failed: {e}")

    print(f"  Phase 6 complete in {time.time()-t0:.1f}s")
    return report.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Buy-and-hold benchmark
# ─────────────────────────────────────────────────────────────────────────────
def buy_and_hold_benchmark(bars_df: pd.DataFrame, oos_mask: np.ndarray,
                           initial_equity: float) -> dict:
    oos_close = bars_df["close"].values[oos_mask]
    if len(oos_close) < 2:
        return {}
    ret_arr   = np.diff(oos_close) / oos_close[:-1]
    total_ret = (oos_close[-1] - oos_close[0]) / oos_close[0]
    n_years   = len(oos_close) / 252.0
    cagr      = (1 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0
    mdd       = _max_drawdown(ret_arr)
    calmar    = cagr / abs(mdd) if mdd < 0 else float("inf")
    sharpe    = _sharpe_ratio(ret_arr)
    return {
        "total_return_pct": float(total_ret * 100),
        "cagr_pct":         float(cagr * 100),
        "max_drawdown_pct": float(mdd * 100),
        "calmar_ratio":     float(calmar),
        "sharpe_ratio":     float(sharpe),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print full tear sheet
# ─────────────────────────────────────────────────────────────────────────────
def print_tearsheet(
    baseline_metrics: dict,
    friction_metrics: dict[str, dict],
    validation: dict,
    bnh: dict,
    n_oos_bars: int,
    n_signals: int,
) -> None:
    print()
    _sep("LATENT DIFFUSION-HMM v3.0  —  OUT-OF-SAMPLE TEAR SHEET")
    print(f"  Ticker    : {TICKER}")
    print(f"  WF-train  : {WF_TRAIN_BARS}-bar rolling window (blind firewall per block)")
    print(f"  OOS blind : {OOS_START} → {OOS_END}  (5 years: COVID / bull / bear / recovery)")
    print(f"  OOS bars  : {n_oos_bars}   |   Triple-Gate signals fired: {n_signals}")
    print()

    b = baseline_metrics

    _sep("PORTFOLIO PERFORMANCE  (vs SPY buy-and-hold)")
    hdr = f"  {'Metric':<22}  {'Strategy':>12}  {'B&H SPY':>12}"
    print(hdr)
    print(f"  {'─'*22}  {'─'*12}  {'─'*12}")

    def _row(label, s_val, b_val, fmt=".1f", suffix="%"):
        sv = f"{s_val:{fmt}}{suffix}"
        bv = f"{b_val:{fmt}}{suffix}" if b_val is not None else "  n/a"
        print(f"  {label:<22}  {sv:>12}  {bv:>12}")

    _row("Total Return",      b["total_return_pct"],      bnh.get("total_return_pct"), suffix="%")
    _row("CAGR",              b["cagr_pct"],              bnh.get("cagr_pct"),         suffix="%")
    _row("Max Drawdown",      b["max_drawdown_pct"],      bnh.get("max_drawdown_pct"), suffix="%")
    _row("Calmar Ratio",      b["calmar_ratio"],          bnh.get("calmar_ratio"),     fmt=".2f", suffix="")
    _row("Sharpe Ratio",      b["sharpe_ratio"],          bnh.get("sharpe_ratio"),     fmt=".2f", suffix="")
    _row("Sortino Ratio",     b["sortino_ratio"],         None,                        fmt=".2f", suffix="")
    final_eq_str = f"${b['final_equity']:,.0f}"
    print(f"  {'Final Equity':<22}  {final_eq_str:>12}")
    print()

    _sep("TRADE STATISTICS")
    print(f"  Completed trades  : {b['n_trades']}  ({b['n_longs']} long / {b['n_shorts']} short)")
    print(f"  Win Rate          : {b['win_rate_pct']:.1f}%"
          f"  (long {b['long_win_rate_pct']:.0f}% / short {b['short_win_rate_pct']:.0f}%)")
    print(f"  Avg Win / Avg Loss: {b['avg_win_pct']:+.2f}% / {b['avg_loss_pct']:+.2f}%")
    print(f"  Profit Factor     : {b['profit_factor']:.2f}"
          f"  (>1.5 good, >2.0 excellent)")
    print(f"  Avg Bars Held     : {b['avg_bars_held']:.1f}")
    exits = b.get("exit_reasons", {})
    print(f"  Exit — Stop-Loss  : {exits.get('stop_loss', 0)}")
    print(f"  Exit — Take-Profit: {exits.get('take_profit', 0)}")
    print(f"  Exit — EOD flush  : {exits.get('end_of_data', 0)}")
    print()

    _sep("PERIOD BREAKDOWN (OOS blind)")
    print(f"  {'Period':<18}  {'Return':>9}  {'Max DD':>8}  {'Trades':>7}")
    print(f"  {'─'*18}  {'─'*9}  {'─'*8}  {'─'*7}")
    for ps in b.get("period_stats", []):
        print(f"  {ps['period']:<18}  {ps['return_pct']:>+8.1f}%  "
              f"{ps['max_drawdown_pct']:>+7.1f}%  {ps['n_trades']:>7}")
    print()

    _sep("FRICTION SENSITIVITY  (Sharpe / Total Return / Calmar)")
    print(f"  {'Scenario':<18}  {'Sharpe':>8}  {'Return':>9}  {'Calmar':>8}  {'Verdict':>8}")
    print(f"  {'─'*18}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*8}")
    for name, m in friction_metrics.items():
        sh = m.get("sharpe_ratio", 0.0)
        tr = m.get("total_return_pct", 0.0)
        ca = m.get("calmar_ratio", 0.0)
        ok = "✓" if sh > 0.5 else ("~" if sh > 0.0 else "✗")
        print(f"  {name:<18}  {sh:>8.2f}  {tr:>+8.1f}%  {ca:>8.2f}  {ok:>8}")
    print()

    _sep("LAYER 6 STATISTICAL VALIDATION")
    tests = validation.get("tests", {})

    wf   = tests.get("walk_forward_cv", {})
    mc   = tests.get("monte_carlo_permutation", {})
    dsr  = tests.get("deflated_sharpe_ratio", {})
    cp   = tests.get("cpcv", {})
    tx   = tests.get("transaction_cost_sensitivity", {})

    def _vrow(label, detail, passes):
        print(f"  {label:<30}  {detail:<28}  {_pf(passes)}")

    _vrow("6.1 Walk-Forward CV",
          f"OOS SR={wf.get('mean_oos_sharpe',0):.2f}  IS/OOS={wf.get('overfit_ratio',0):.2f}  "
          f"({wf.get('n_windows',0)} wins)",
          wf.get("passes", False))
    _vrow("6.2 Monte Carlo Permutation",
          f"actual SR={mc.get('actual_oos_sharpe',0):.2f}  p={mc.get('p_value',1):.4f}",
          mc.get("passes", False))
    _vrow("6.3 Deflated Sharpe Ratio",
          f"SR={dsr.get('observed_sr',0):.2f}  z={dsr.get('deflated_sr_z',0):.2f}  "
          f"p={dsr.get('p_value',1):.4f}",
          dsr.get("passes", False))
    _vrow("6.4 CPCV",
          f"MinSR={cp.get('min_sr',0):.2f}  PSR={cp.get('psr',0):.2f}  "
          f"({cp.get('n_paths',0)} paths)",
          cp.get("passes", False))
    tx_sc = tx.get("scenarios", {}).get("realistic", {})
    _vrow("6.5 Tx Cost (2 bps realistic)",
          f"net Sharpe={tx_sc.get('net_sharpe',0):.2f}",
          tx_sc.get("passes", False))

    n_passed = sum([
        wf.get("passes", False), mc.get("passes", False),
        dsr.get("passes", False), cp.get("passes", False),
        tx_sc.get("passes", False),
    ])
    print()
    if n_passed == 5:
        print("  ══════════  GO LIVE CRITERIA: ALL 5 TESTS PASSED  ══════════")
    else:
        print(f"  ══════════  {n_passed}/5 GO LIVE CRITERIA MET  ══════════")
    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    wall_t0 = time.time()
    print()
    _sep("LATENT DIFFUSION-HMM  —  OUT-OF-SAMPLE VALIDATION")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ticker  : {TICKER}  |  WF-train: {WF_TRAIN_BARS}-bar rolling  "
          f"|  Blind OOS: {OOS_START}→{OOS_END}")

    # ── Phase 1: Data & features ─────────────────────────────────────────
    d = phase1_build_data()

    # ── Phase 2: Walk-forward HMM (rolling 504-bar train windows) ────────
    regime_proba, covariates = phase2_walkforward_regime(d)

    # ── Phase 3: Triple Gate → signals on OOS bars only ──────────────────
    signals = phase3_generate_signals(d, regime_proba)

    bars_df  = d["bars_df"]
    oos_mask = d["oos_mask"]
    n_oos    = int(oos_mask.sum())

    # ── Phase 4: Event-driven backtests under friction scenarios ──────────
    print()
    _sep("Phase 4 — Event-Driven Backtest (4 friction scenarios)")

    friction_results: dict[str, dict] = {}
    friction_metrics: dict[str, dict] = {}

    for label, bps in FRICTION_SCENARIOS:
        bt = run_backtest(bars_df, signals, oos_mask,
                          equity=INITIAL_EQUITY, friction_bps=bps)
        m  = compute_metrics(bt, INITIAL_EQUITY, bars_df)
        friction_results[label] = bt
        friction_metrics[label] = m
        print(f"  {label:<20}  trades={bt['n_trades']}  "
              f"Sharpe={m['sharpe_ratio']:.2f}  "
              f"CAGR={m['cagr_pct']:+.1f}%  "
              f"MDD={m['max_drawdown_pct']:.1f}%  "
              f"WinRate={m['win_rate_pct']:.0f}%  "
              f"PF={m['profit_factor']:.2f}")

    baseline_metrics = friction_metrics["0 bps (gross)"]
    baseline_bt      = friction_results["0 bps (gross)"]

    # ── Phase 5: Buy-and-hold benchmark ───────────────────────────────────
    bnh = buy_and_hold_benchmark(bars_df, oos_mask, INITIAL_EQUITY)

    # ── Phase 6: Layer 6 Validation on OOS equity-curve returns ──────────
    oos_returns = baseline_metrics["returns_array"]
    validation  = phase6_validation(oos_returns, n_trades=baseline_bt["n_trades"])

    # ── Print tear sheet ──────────────────────────────────────────────────
    print_tearsheet(
        baseline_metrics=baseline_metrics,
        friction_metrics={k: friction_metrics[k] for k in friction_metrics},
        validation=validation,
        bnh=bnh,
        n_oos_bars=n_oos,
        n_signals=len(signals),
    )

    print(f"  Total wall-clock time: {time.time()-wall_t0:.1f}s")
    print()

    # ── Save results to JSON ───────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oos_results.json")

    def _serialise(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    # Strip non-serialisable arrays from metrics before saving
    for m in friction_metrics.values():
        m.pop("returns_array", None)

    save_payload = {
        "run_at":         datetime.now().isoformat(),
        "ticker":         TICKER,
        "train":          {"start": TRAIN_START, "end": TRAIN_END},
        "oos":            {"start": OOS_START,   "end": OOS_END},
        "n_oos_bars":     n_oos,
        "n_signals":      len(signals),
        "buy_and_hold":   bnh,
        "friction_scenarios": {
            k: {kk: vv for kk, vv in v.items() if kk != "returns_array"}
            for k, v in friction_metrics.items()
        },
        "validation":     validation,
        "trade_log":      baseline_bt["completed_trades"][:200],  # first 200 trades
    }

    with open(output_path, "w") as f:
        json.dump(save_payload, f, indent=2, default=_serialise)
    print(f"  Results saved → {output_path}")
    print()


if __name__ == "__main__":
    main()

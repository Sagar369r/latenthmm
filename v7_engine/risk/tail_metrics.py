"""
Tail Metrics — Realised risk statistics for prop firm evaluation.
All computed from actual P&L streams, not synthetic data.
"""
from __future__ import annotations
import numpy as np
from v7_engine.config import (
    RISK_TAIL_TRADING_DAYS, RISK_TAIL_VAR_99, RISK_TAIL_VAR_95, 
    RISK_TAIL_VAR_05, RISK_TAIL_MIN_STD
)


def sortino_ratio(returns: np.ndarray, target: float = 0.0) -> float:
    """Sortino ratio: mean return / downside deviation."""
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) <= 1:
        return np.inf
    dd_std = float(np.std(downside, ddof=1))
    if dd_std < RISK_TAIL_MIN_STD:
        return np.inf
    return float(np.mean(excess)) / dd_std


def sharpe_ratio(
    returns:  np.ndarray,
    risk_free: float = 0.0,
    annualise: bool  = True,
    freq:      int   = 252,
) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free
    if len(excess) <= 1:
        return 0.0
    std_   = float(np.std(excess, ddof=1))
    if std_ < RISK_TAIL_MIN_STD:
        return 0.0
    sr = float(np.mean(excess)) / std_
    return sr * np.sqrt(freq) if annualise else sr


def calmar_ratio(returns: np.ndarray) -> float:
    """Calmar = annualised return / max drawdown."""
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd   = (peak - cum) / (peak + RISK_TAIL_MIN_STD)
    max_dd = float(dd.max())
    if max_dd < RISK_TAIL_MIN_STD:
        return np.inf
    ann_ret = float(np.mean(returns)) * 252
    return ann_ret / max_dd


def historical_var(returns: np.ndarray, confidence: float = RISK_TAIL_VAR_99) -> float:
    """Historical VaR at given confidence level (positive number = loss)."""
    return float(-np.quantile(returns, 1 - confidence))


def conditional_var(returns: np.ndarray, confidence: float = RISK_TAIL_VAR_99) -> float:
    """Conditional VaR (Expected Shortfall) — mean of worst (1-conf) tail."""
    var  = historical_var(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-np.mean(tail))


def max_drawdown_usd(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown in USD."""
    peak = np.maximum.accumulate(equity_curve)
    dd   = peak - equity_curve
    return float(dd.max())


def profit_factor(returns: np.ndarray) -> float:
    """Gross profits / gross losses."""
    pos = returns[returns > 0].sum()
    neg = abs(returns[returns < 0].sum())
    if neg < RISK_TAIL_MIN_STD:
        return np.inf
    return float(pos / neg)

def calculate_sortino(returns: np.ndarray, target: float = 0.0) -> float:
    return sortino_ratio(returns, target)

def downside_std(returns: np.ndarray, target: float = 0.0) -> float:
    excess = returns - target
    neg = excess[excess < 0]
    return float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0

def calculate_var(returns: np.ndarray, confidence: float = RISK_TAIL_VAR_95) -> float:
    return historical_var(returns, confidence)

def calculate_cvar(returns: np.ndarray, confidence: float = RISK_TAIL_VAR_95) -> float:
    return conditional_var(returns, confidence)

def tail_ratio(returns: np.ndarray, confidence: float = RISK_TAIL_VAR_05) -> float:
    """Ratio of right tail to left tail at given percentile."""
    right = float(np.percentile(returns, 100 * (1 - confidence)))
    left  = float(abs(np.percentile(returns, 100 * confidence)))
    return right / (left + RISK_TAIL_MIN_STD)

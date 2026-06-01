"""
Layer 1: Data Ingestion
- Dollar-Volume bar construction
- Fractional Differentiation with MMMS (Maximum-Memory Minimal-Stationarity)
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


def fetch_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with auto-adjustment for splits/dividends."""
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} from {start} to {end}")
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def build_dollar_volume_bars(df: pd.DataFrame, target_bars_per_day: int = 20) -> pd.DataFrame:
    """
    Construct Dollar-Volume bars from OHLCV data.

    Bar triggers when: Σ(P_i × V_i) ≥ D*
    D* = EMA(daily_dollar_volume) / target_bars_per_day

    For daily input data each original bar contributes its own dollar volume.
    When the accumulated DV crosses D*, a new bar closes.
    """
    df = df.copy()
    df["dollar_volume"] = df["close"] * df["volume"]

    dv_ema = df["dollar_volume"].ewm(span=20, min_periods=5).mean()
    threshold = (dv_ema / target_bars_per_day).clip(lower=1.0)

    bars: list[dict] = []
    cumulative_dv = 0.0
    open_price = float(df["open"].iloc[0])
    high_price = float(df["high"].iloc[0])
    low_price = float(df["low"].iloc[0])
    bar_volume = 0.0
    bar_dv = 0.0

    for i, (idx, row) in enumerate(df.iterrows()):
        thresh = float(threshold.iloc[i])
        dv = float(row["dollar_volume"])
        cumulative_dv += dv
        bar_volume += float(row["volume"])
        bar_dv += dv
        high_price = max(high_price, float(row["high"]))
        low_price = min(low_price, float(row["low"]))

        if cumulative_dv >= thresh or i == len(df) - 1:
            bars.append({
                "date": idx,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": float(row["close"]),
                "volume": bar_volume,
                "dollar_volume": bar_dv,
            })
            cumulative_dv = 0.0
            open_price = float(row["close"])
            high_price = float(row["close"])
            low_price = float(row["close"])
            bar_volume = 0.0
            bar_dv = 0.0

    result = pd.DataFrame(bars).set_index("date")
    result.index = pd.to_datetime(result.index)
    return result


def _compute_frac_diff_weights(d: float, size: int) -> np.ndarray:
    """Compute binomial coefficients for fractional differentiation operator (1-B)^d."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def _apply_frac_diff(values: np.ndarray, d: float) -> np.ndarray:
    """Apply fractional differentiation with weight threshold for efficiency."""
    T = len(values)
    weights = _compute_frac_diff_weights(d, T)
    # Drop weights below threshold (memory cutoff)
    threshold = 1e-5
    active = np.abs(weights) > threshold
    max_lag = int(active.sum())

    result = np.full(T, np.nan)
    for t in range(max_lag, T):
        w_slice = weights[:t + 1][-max_lag:]
        x_slice = values[t - len(w_slice) + 1: t + 1][::-1]
        result[t] = float(np.dot(w_slice, x_slice))
    return result


def _adf_pvalue(series: np.ndarray) -> float:
    clean = series[~np.isnan(series)]
    if len(clean) < 20:
        return 1.0
    try:
        return float(adfuller(clean, maxlag=1, autolag=None)[1])
    except Exception:
        return 1.0


def _find_mmms_d(values: np.ndarray, max_d: float = 1.0, tol: float = 0.01,
                  pvalue_threshold: float = 0.05) -> float:
    """
    Binary search for d* = min{d : ADF p-value < 0.05}.
    Preserves maximum memory (minimum d) while achieving stationarity.
    """
    lo, hi = 0.0, max_d
    for _ in range(25):
        mid = (lo + hi) / 2.0
        if abs(hi - lo) < tol:
            break
        fd = _apply_frac_diff(values, mid)
        pval = _adf_pvalue(fd)
        if pval < pvalue_threshold:
            hi = mid
        else:
            lo = mid
    return float(hi)


def fractional_differentiation(
    series: pd.Series,
    min_bars: int = 500,
    reestimate_every: int = 252,
    pvalue_threshold: float = 0.05,
) -> tuple[pd.Series, list[float]]:
    """
    MMMS Fractional Differentiation.

    (1 - B)^(d*) x_t  where d* = min{d : ADF p-value < 0.05}

    Uses expanding window: re-estimates d* from bar `min_bars` onward
    every `reestimate_every` bars. Never looks ahead.
    Typical d* for equity prices: 0.35–0.45.
    """
    values = series.values.astype(float)
    T = len(values)
    result = np.full(T, np.nan)
    d_estimates: list[float] = []

    d_current = 0.40  # initial fallback

    for t in range(T):
        if t >= min_bars and (t - min_bars) % reestimate_every == 0:
            d_current = _find_mmms_d(values[: t + 1], pvalue_threshold=pvalue_threshold)

        fd = _apply_frac_diff(values[: t + 1], d_current)
        result[t] = fd[-1] if not np.isnan(fd[-1]) else np.nan
        d_estimates.append(d_current)

    return pd.Series(result, index=series.index, name="frac_diff"), d_estimates


def load_and_prepare(
    ticker: str,
    start: str,
    end: str,
    use_dollar_bars: bool = True,
    apply_frac_diff: bool = True,
) -> dict:
    """
    Full Layer 1 pipeline: fetch → dollar-volume bars → fractional differentiation.

    Returns a dict with keys:
        raw_df:        original OHLCV DataFrame
        bars_df:       dollar-volume bars DataFrame
        fd_close:      fractionally differentiated close series
        d_estimates:   list of d* values used per bar
    """
    raw_df = fetch_ohlcv(ticker, start, end)

    if use_dollar_bars:
        bars_df = build_dollar_volume_bars(raw_df)
    else:
        bars_df = raw_df.copy()

    fd_close: pd.Series | None = None
    d_estimates: list[float] = []

    if apply_frac_diff:
        fd_close, d_estimates = fractional_differentiation(bars_df["close"])

    return {
        "raw_df": raw_df,
        "bars_df": bars_df,
        "fd_close": fd_close,
        "d_estimates": d_estimates,
    }

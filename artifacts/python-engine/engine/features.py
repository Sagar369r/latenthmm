"""
Layer 2: 6-Dimensional Observation Tensor
Xt = [vt, mvt, qt, σt, ρt, Ht]^T ∈ ℝ^6

Feature definitions (spec-faithful):
  v_t  — Volatility Proximity
  mv_t — Momentum Velocity
  q_t  — Volume Delta Ratio
  σ_t  — Volatility Regime Ratio
  ρ_t  — Autocorrelation Signal
  H_t  — DFA Hurst Exponent
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from arch import arch_model

warnings.filterwarnings("ignore")


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def volatility_proximity(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    """
    v_t = (P_t - m_t) / ε_t
    m_t = Donchian midline = (rolling_high + rolling_low) / 2
    ε_t = 2.1 × ATR_14

    Bounded in [-1, +1]. |v_t| > 0.85 → breakout zone.
    """
    window = max(atr_period * 2, 20)
    rolling_high = df["close"].rolling(window).max()
    rolling_low = df["close"].rolling(window).min()
    m_t = (rolling_high + rolling_low) / 2.0
    atr = _atr(df, atr_period)
    epsilon = 2.1 * atr
    vt = (df["close"] - m_t) / epsilon.replace(0, np.nan)
    return vt.clip(-1.0, 1.0).rename("vt")


def _acf_peak_lag(returns: pd.Series, max_lag: int = 60) -> int:
    """Find the lag of maximum absolute autocorrelation in |returns|."""
    abs_ret = returns.abs().dropna()
    if len(abs_ret) < max_lag + 10:
        return 10
    acf_vals = [abs_ret.autocorr(lag=k) for k in range(1, max_lag + 1)]
    peak = int(np.nanargmax(np.abs(acf_vals))) + 1
    return max(peak, 5)


def momentum_velocity(df: pd.DataFrame, sigma_window: int = 50) -> pd.Series:
    """
    mv_t = log(P_t / P_{t-n}) / σ_{50}
    n chosen by autocorrelation peak of |returns| (re-estimated every 252 bars).
    σ-gating removes spurious momentum signals in low-vol environments.
    """
    close = df["close"]
    returns = close.pct_change()
    sigma_50 = returns.rolling(sigma_window).std()

    T = len(close)
    mvt = np.full(T, np.nan)
    n_current = 10

    reestimate_every = 252
    for t in range(sigma_window, T):
        if t % reestimate_every == 0:
            n_current = _acf_peak_lag(returns.iloc[:t])
        if t >= n_current:
            log_ret = np.log(close.iloc[t] / close.iloc[t - n_current])
            sig = sigma_50.iloc[t]
            if sig > 0:
                mvt[t] = log_ret / sig

    return pd.Series(mvt, index=df.index, name="mvt")


def volume_delta_ratio(df: pd.DataFrame, ema_span: int = 20) -> pd.Series:
    """
    q_t = V_t / EMA_20(V_t)
    Institutional footprint proxy.
    q > 2.0 on breakout bar → genuine demand.
    q < 0.5 on new high → distribution warning.
    """
    vol_ema = df["volume"].ewm(span=ema_span, adjust=False).mean()
    qt = df["volume"] / vol_ema.replace(0, np.nan)
    return qt.rename("qt")


def volatility_regime_ratio(df: pd.DataFrame, rv_window: int = 20) -> pd.Series:
    """
    σ_t = RV_t / GARCH(1,1)_t
    RV_t = √(Σ r²_{t-k} / k),  k = rv_window
    GARCH conditional volatility estimated on a rolling 500-bar window.

    σ > 1.5 → realized vol exceeds model (jump precursor)
    σ < 0.6 → compressed vol (coiling)
    """
    returns = df["close"].pct_change() * 100  # in percent for ARCH

    T = len(df)
    rv = (df["close"].pct_change() ** 2).rolling(rv_window).mean().apply(np.sqrt)

    garch_vol = pd.Series(np.nan, index=df.index)
    min_obs = 252
    reestimate_every = 126

    for t in range(min_obs, T):
        if (t - min_obs) % reestimate_every != 0 and not np.isnan(garch_vol.iloc[t - 1]):
            garch_vol.iloc[t] = garch_vol.iloc[t - 1]
            continue
        try:
            window_ret = returns.iloc[max(0, t - 500): t].dropna()
            if len(window_ret) < 50:
                continue
            am = arch_model(window_ret, vol="GARCH", p=1, q=1, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1, reindex=False)
            garch_vol.iloc[t] = float(np.sqrt(fc.variance.values[-1, 0])) / 100.0
        except Exception:
            pass

    sigma_t = rv / garch_vol.replace(0, np.nan)
    return sigma_t.rename("sigma_t")


def autocorrelation_signal(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    ρ_t = Corr(r_t, r_{t-1}) on a 30-bar rolling window.
    ρ > +0.15 → trending
    ρ < -0.15 → mean-reverting
    |ρ| < 0.15 → choppy
    """
    returns = df["close"].pct_change()
    rho = returns.rolling(window).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) >= window else np.nan,
        raw=False,
    )
    return rho.rename("rho_t")


def _dfa_hurst(series: np.ndarray, min_n: int = 10, max_n: int = 200, n_points: int = 20) -> float:
    """
    Detrended Fluctuation Analysis Hurst exponent.
    F(n) = √(1/N × Σ[y_k - ŷ_{k,n}]²) → F(n) ∝ n^H

    Unbiased for short windows (n = 50–200 bars), unlike R/S.
    H > 0.55 → trend persistence
    H < 0.45 → mean reversion
    """
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    N = len(series)
    if N < min_n * 2:
        return 0.5

    # Profile (cumulative sum of demeaned series)
    y = np.cumsum(series - series.mean())

    scales = np.unique(np.floor(
        np.logspace(np.log10(min_n), np.log10(min(max_n, N // 4)), n_points)
    ).astype(int))
    scales = scales[scales >= min_n]

    if len(scales) < 4:
        return 0.5

    fluctuations = []
    valid_scales = []
    for n in scales:
        n_segments = N // n
        if n_segments < 2:
            continue
        rms_list = []
        for seg in range(n_segments):
            segment = y[seg * n: (seg + 1) * n]
            x_local = np.arange(n, dtype=float)
            # Linear detrend
            coeffs = np.polyfit(x_local, segment, 1)
            trend = np.polyval(coeffs, x_local)
            rms_list.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms_list:
            fluctuations.append(np.mean(rms_list))
            valid_scales.append(n)

    if len(valid_scales) < 3:
        return 0.5

    log_scales = np.log(valid_scales)
    log_fluct = np.log(fluctuations)
    try:
        coeffs = np.polyfit(log_scales, log_fluct, 1)
        h = float(coeffs[0])
        return float(np.clip(h, 0.0, 1.0))
    except Exception:
        return 0.5


def dfa_hurst_series(df: pd.DataFrame, window: int = 100) -> pd.Series:
    """
    Rolling DFA Hurst exponent over a sliding window of `window` bars.
    """
    returns = df["close"].pct_change().values
    T = len(returns)
    ht = np.full(T, np.nan)

    for t in range(window, T):
        chunk = returns[t - window: t]
        ht[t] = _dfa_hurst(chunk)

    return pd.Series(ht, index=df.index, name="ht")


def compute_feature_tensor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full 6D observation tensor Xt = [vt, mvt, qt, σt, ρt, Ht].

    Returns a DataFrame with columns: vt, mvt, qt, sigma_t, rho_t, ht
    """
    vt = volatility_proximity(df)
    mvt = momentum_velocity(df)
    qt = volume_delta_ratio(df)
    sigma_t = volatility_regime_ratio(df)
    rho_t = autocorrelation_signal(df)
    ht = dfa_hurst_series(df)

    features = pd.DataFrame({
        "vt": vt,
        "mvt": mvt,
        "qt": qt,
        "sigma_t": sigma_t,
        "rho_t": rho_t,
        "ht": ht,
    }, index=df.index)

    return features

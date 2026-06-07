"""
Full Pipeline Orchestrator

Chains all 6 layers:
  L1: Data Ingestion (Dollar-Volume bars + Fractional Diff)
  L2: Feature Engineering (6D tensor + whitening)
  L3: Kalman Filter + CUSUM Jump Detection
  L4: TVTP-HMM (3-state GMM, Viterbi)
  L5: Execution (Triple Gate, Half-Kelly, ATR SL/TP)
  L5: Wasserstein Surveillance
  L6: Statistical Validation (WF-CV, MC Permutation, DSR, CPCV, Tx Cost)
"""
from __future__ import annotations

import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

from .data import load_and_prepare
from .features import compute_feature_tensor
from .preprocess import Preprocessor
from .kalman import run_kalman_pipeline
from .hmm import TVTPHMM, STATE_LABELS
from .execution import run_execution_layer, TradeSignal
from .surveillance import WassersteinMonitor
from .validation import (
    ValidationReport, walk_forward_cv, monte_carlo_permutation_test, 
    deflated_sharpe_ratio, cpcv, transaction_cost_sensitivity, _sharpe_ratio,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("pipeline")

@dataclass
class PipelineConfig:
    ticker: str
    start: str
    end: str
    volume_threshold: int = 10000
    apply_frac_diff: bool = True
    equity: float = 100_000.0
    run_validation: bool = False
    n_mc_permutations: int = 200
    train_fraction: float = 0.7

@dataclass
class PipelineResult:
    ticker: str
    start: str
    end: str
    timestamp: str
    n_bars: int

    bars_df: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    whitened: pd.DataFrame | None = None
    filtered_states: np.ndarray | None = None
    jump_flags: np.ndarray | None = None
    cusum_g: np.ndarray | None = None
    regime_proba: np.ndarray | None = None
    viterbi_states: np.ndarray | None = None
    regime_summary: dict = field(default_factory=dict)
    current_regime: str = "UNKNOWN"
    signals: list[TradeSignal] = field(default_factory=list)
    n_signals: int = 0
    surveillance: dict = field(default_factory=dict)
    validation: dict | None = None

    _hmm: Any = field(default=None, repr=False)
    _preprocessor: Any = field(default=None, repr=False)
    _kalman_filter: Any = field(default=None, repr=False)

    def to_api_dict(self) -> dict:
        signals_list = []
        for sig in self.signals:
            signals_list.append({
                "timestamp": str(sig.timestamp),
                "direction": sig.direction,
                "direction_label": "LONG" if sig.direction > 0 else "SHORT",
                "regime": sig.regime,
                "p_trend": round(sig.p_trend, 4),
                "p_mean_rev": round(sig.p_mean_rev, 4),
                "p_stress": round(sig.p_stress, 4),
                "entry_price": round(sig.entry_price, 4),
                "stop_loss": round(sig.stop_loss, 4),
                "take_profit": round(sig.take_profit, 4),
                "position_fraction": round(sig.position_fraction, 6),
                "momentum": round(sig.momentum, 4),
                "volume_ratio": round(sig.volume_ratio, 4),
                "atr": round(sig.atr, 4),
            })

        feature_ts = []
        if self.features is not None and self.regime_proba is not None:
            n = min(100, len(self.features))
            feat_tail = self.features.tail(n)
            prob_tail = self.regime_proba[-n:]
            vit_tail = self.viterbi_states[-n:] if self.viterbi_states is not None else None

            for i, (idx, row) in enumerate(feat_tail.iterrows()):
                entry = {
                    "date": str(idx.date() if hasattr(idx, "date") else idx),
                    "vt": _safe_float(row.get("vt")),
                    "mvt": _safe_float(row.get("mvt")),
                    "qt": _safe_float(row.get("qt")),
                    "sigma_t": _safe_float(row.get("sigma_t")),
                    "rho_t": _safe_float(row.get("rho_t")),
                    "ht": _safe_float(row.get("ht")),
                    "p_mean_rev": round(float(prob_tail[i, 0]), 4),
                    "p_trend": round(float(prob_tail[i, 1]), 4),
                    "p_stress": round(float(prob_tail[i, 2]), 4),
                    "regime": STATE_LABELS[int(prob_tail[i].argmax())],
                }
                if vit_tail is not None:
                    entry["viterbi_state"] = STATE_LABELS[int(vit_tail[i])]
                feature_ts.append(entry)

        result = {
            "ticker": self.ticker,
            "start": self.start,
            "end": self.end,
            "timestamp": self.timestamp,
            "n_bars": self.n_bars,
            "current_regime": self.current_regime,
            "regime_summary": self.regime_summary,
            "n_signals": self.n_signals,
            "signals": signals_list,
            "surveillance": self.surveillance,
            "feature_timeseries": feature_ts,
        }
        if self.validation:
            result["validation"] = self.validation
        return result

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except Exception:
        return None

class Pipeline:
    def run(self, config: PipelineConfig) -> PipelineResult:
        result = PipelineResult(
            ticker=config.ticker,
            start=config.start,
            end=config.end,
            timestamp=datetime.utcnow().isoformat(),
            n_bars=0,
        )

        logger.info(f"[L1] Fetching {config.ticker}")
        data = load_and_prepare(
            filepath=config.ticker,
            start=config.start,
            end=config.end,
            volume_threshold=config.volume_threshold,
            apply_frac_diff=config.apply_frac_diff,
        )
        bars_df = data["bars_df"]
        result.bars_df = bars_df
        result.n_bars = len(bars_df)

        if result.n_bars < 50:
            raise ValueError(f"Insufficient data: {result.n_bars} bars.")

        logger.info("[L2] Computing 6D feature tensor")
        features = compute_feature_tensor(bars_df)
        result.features = features

        preprocessor = Preprocessor()
        X_white = preprocessor.fit_transform(features)
        result._preprocessor = preprocessor
        result.whitened = pd.DataFrame(
            X_white, index=features.index,
            columns=[f"w_{c}" for c in features.columns]
        )

        logger.info("[L3] Running Kalman filter and CUSUM jump detector")
        kalman_out = run_kalman_pipeline(X_white)
        filtered = kalman_out["filtered_states"]
        result.filtered_states = filtered
        result.jump_flags = kalman_out["jump_flags"]
        result.cusum_g = kalman_out["cusum_g"]
        result._kalman_filter = kalman_out["kalman_filter"]

        logger.info("[L4] Fitting TVTP-HMM (3 states, K=2 GMM)")
        sigma_t = features["sigma_t"].fillna(1.0).values
        rho_t = features["rho_t"].fillna(0.0).values
        covariates = np.column_stack([sigma_t, rho_t])

        clean_mask = ~np.any(np.isnan(filtered), axis=1)
        train_end = int(result.n_bars * config.train_fraction)
        X_train = filtered[clean_mask & (np.arange(result.n_bars) < train_end)]
        cov_train = covariates[clean_mask & (np.arange(result.n_bars) < train_end)]

        hmm = TVTPHMM(n_states=3, n_gmm=2, n_iter=30)
        if len(X_train) >= 30:
            hmm.fit(X_train, cov_train)
        else:
            hmm.fit(filtered[clean_mask[:min(50, len(filtered))]][:20],
                    covariates[clean_mask[:min(50, len(filtered))]][:20])

        result._hmm = hmm

        X_full = filtered.copy()
        for t in range(1, len(X_full)):
            if np.any(np.isnan(X_full[t])):
                X_full[t] = X_full[t - 1]
        if np.any(np.isnan(X_full[0])):
            X_full[0] = np.zeros(X_full.shape[1])

        hmm_result = hmm.predict(X_full, covariates)
        result.regime_proba = hmm_result["proba"]
        result.viterbi_states = hmm_result["viterbi_states"]
        result.regime_summary = hmm_result["regime_summary"]
        result.current_regime = hmm_result["regime_summary"]["dominant_regime"]

        logger.info("[L5] Running execution layer")
        exec_out = run_execution_layer(
            bars_df, features, result.regime_proba, equity=config.equity
        )
        result.signals = exec_out["signals"]
        result.n_signals = exec_out["n_signals"]

        logger.info("[L5] Running Wasserstein distribution surveillance")
        monitor = WassersteinMonitor(window=min(50, result.n_bars // 4))
        live_window = X_white[-monitor.window:]
        clean_train = X_white[:train_end]
        clean_train = clean_train[~np.any(np.isnan(clean_train), axis=1)]
        
        if len(clean_train) >= 10:
            monitor.fit(clean_train)
            surv_result = monitor.check(live_window)
            if "position_scale" in surv_result:
                scale = surv_result["position_scale"]
                for sig in result.signals:
                    sig.position_fraction *= scale
        else:
            surv_result = {"status": "insufficient_training_data"}
        
        result.surveillance = surv_result

        if config.run_validation:
            logger.info("[L6] Running statistical validation suite")
            result.validation = self._run_validation(
                bars_df, features, result.regime_proba,
                config, result.signals
            )

        return result

    def _run_validation(self, bars_df, features, regime_proba, config, signals) -> dict:
        report = ValidationReport()
        daily_returns = self._compute_strategy_returns(bars_df, signals)

        try:
            def wf_simulate(ts, te, ss, se):
                return daily_returns[ts:te], daily_returns[ss:se]
            report.wfcv = walk_forward_cv(
                wf_simulate, T=len(daily_returns),
                train_bars=min(504, len(daily_returns) // 3),
                test_bars=min(126, len(daily_returns) // 6),
            )
        except Exception as e: logger.warning(f"WF-CV failed: {e}")

        try:
            def perm_simulate(perm_returns):
                return _sharpe_ratio(perm_returns[-len(daily_returns) // 3:])
            report.mc_permutation = monte_carlo_permutation_test(
                daily_returns, perm_simulate, n_permutations=config.n_mc_permutations
            )
        except Exception as e: logger.warning(f"MC permutation failed: {e}")

        try:
            oos_returns = daily_returns[int(len(daily_returns) * config.train_fraction):]
            report.dsr = deflated_sharpe_ratio(oos_returns, n_trials=report.n_trials)
        except Exception as e: logger.warning(f"DSR failed: {e}")

        try:
            n_folds = min(6, max(3, len(daily_returns) // 100))
            def cpcv_simulate(train_idx, test_idx):
                return daily_returns[train_idx], daily_returns[test_idx]
            report.cpcv = cpcv(cpcv_simulate, T=len(daily_returns), n_folds=n_folds)
        except Exception as e: logger.warning(f"CPCV failed: {e}")

        try:
            report.tx_cost = transaction_cost_sensitivity(
                daily_returns, len(signals), len(daily_returns)
            )
        except Exception as e: logger.warning(f"Tx cost failed: {e}")

        return report.to_dict()

    @staticmethod
    def _compute_strategy_returns(bars_df: pd.DataFrame, signals: list[TradeSignal]) -> np.ndarray:
        T = len(bars_df)
        strategy_returns = np.zeros(T - 1)
        
        close = bars_df["close"].values
        high = bars_df["high"].values
        low = bars_df["low"].values
        
        signal_map = {s.bar_index: s for s in signals}
        position = None
        
        for t in range(1, T):
            price = close[t]
            h = high[t]
            l = low[t]
            pnl_pct = 0.0
            
            if position is not None:
                d_dir = position["direction"]
                sl = position["stop_loss"]
                tp = position["take_profit"]
                frac = position["position_fraction"]
                
                exit_price = None
                if d_dir == 1:
                    if l <= sl: exit_price = sl
                    elif h >= tp: exit_price = tp
                else:
                    if h >= sl: exit_price = sl
                    elif l <= tp: exit_price = tp
                
                if exit_price is not None:
                    bar_ret = (exit_price - close[t-1]) / close[t-1] * d_dir
                    pnl_pct = bar_ret * frac
                    position = None
                else:
                    bar_ret = (price - close[t-1]) / close[t-1] * d_dir
                    pnl_pct = bar_ret * frac
            
            if position is None and t in signal_map:
                sig = signal_map[t]
                position = {
                    "direction": sig.direction,
                    "entry_price": sig.entry_price,
                    "stop_loss": sig.stop_loss,
                    "take_profit": sig.take_profit,
                    "position_fraction": sig.position_fraction,
                }
                
            strategy_returns[t - 1] = float(pnl_pct)
            
        return strategy_returns

_pipeline: Pipeline | None = None

def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline

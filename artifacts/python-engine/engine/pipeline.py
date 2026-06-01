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
from dataclasses import dataclass, field, asdict
from typing import Any

from .data import load_and_prepare
from .features import compute_feature_tensor
from .preprocess import Preprocessor
from .kalman import run_kalman_pipeline
from .hmm import TVTPHMM, STATE_LABELS
from .execution import run_execution_layer, compute_atr, TradeSignal
from .surveillance import WassersteinMonitor
from .validation import (
    ValidationReport, WFCVResult, MCPermutationResult, DSRResult,
    CPCVResult, TxCostResult,
    walk_forward_cv, monte_carlo_permutation_test, deflated_sharpe_ratio,
    cpcv, transaction_cost_sensitivity,
    _sharpe_ratio,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("pipeline")


@dataclass
class PipelineConfig:
    ticker: str
    start: str
    end: str
    use_dollar_bars: bool = True
    apply_frac_diff: bool = True
    equity: float = 100_000.0
    run_validation: bool = False
    n_mc_permutations: int = 200   # spec: 10k; use 200 for speed
    train_fraction: float = 0.7


@dataclass
class PipelineResult:
    ticker: str
    start: str
    end: str
    timestamp: str
    n_bars: int

    # Layer 1
    bars_df: pd.DataFrame | None = None

    # Layer 2
    features: pd.DataFrame | None = None
    whitened: pd.DataFrame | None = None

    # Layer 3
    filtered_states: np.ndarray | None = None
    jump_flags: np.ndarray | None = None
    cusum_g: np.ndarray | None = None

    # Layer 4
    regime_proba: np.ndarray | None = None
    viterbi_states: np.ndarray | None = None
    regime_summary: dict = field(default_factory=dict)
    current_regime: str = "UNKNOWN"

    # Layer 5 — Execution
    signals: list[TradeSignal] = field(default_factory=list)
    n_signals: int = 0

    # Layer 5 — Surveillance
    surveillance: dict = field(default_factory=dict)

    # Layer 6 — Validation
    validation: dict | None = None

    # Model objects (not serialised in API responses)
    _hmm: Any = field(default=None, repr=False)
    _preprocessor: Any = field(default=None, repr=False)
    _kalman_filter: Any = field(default=None, repr=False)

    def to_api_dict(self) -> dict:
        """Serialise for API response (exclude DataFrames and model objects)."""
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

        # Feature timeseries (last 100 bars)
        feature_ts: list[dict] = []
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
                    "p_trend": round(float(prob_tail[i, 0]), 4),
                    "p_mean_rev": round(float(prob_tail[i, 1]), 4),
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
    """
    Full Latent Diffusion-HMM pipeline.

    Usage:
        pipeline = Pipeline()
        result = pipeline.run(config)
    """

    def run(self, config: PipelineConfig) -> PipelineResult:
        result = PipelineResult(
            ticker=config.ticker,
            start=config.start,
            end=config.end,
            timestamp=datetime.utcnow().isoformat(),
            n_bars=0,
        )

        # ── Layer 1: Data Ingestion ──────────────────────────────────
        logger.info(f"[L1] Fetching {config.ticker} from {config.start} to {config.end}")
        data = load_and_prepare(
            ticker=config.ticker,
            start=config.start,
            end=config.end,
            use_dollar_bars=config.use_dollar_bars,
            apply_frac_diff=config.apply_frac_diff,
        )
        bars_df = data["bars_df"]
        result.bars_df = bars_df
        result.n_bars = len(bars_df)
        logger.info(f"[L1] {result.n_bars} dollar-volume bars constructed")

        if result.n_bars < 50:
            raise ValueError(
                f"Insufficient data: only {result.n_bars} bars. "
                "Need at least 50 bars to run the pipeline."
            )

        # ── Layer 2: Feature Engineering & Preprocessing ────────────
        logger.info("[L2] Computing 6D feature tensor")
        features = compute_feature_tensor(bars_df)
        result.features = features

        preprocessor = Preprocessor()
        X_white = preprocessor.fit_transform(features)
        result._preprocessor = preprocessor
        whitened_df = pd.DataFrame(
            X_white, index=features.index,
            columns=[f"w_{c}" for c in features.columns]
        )
        result.whitened = whitened_df
        logger.info("[L2] Feature tensor computed and whitened")

        # ── Layer 3: Kalman Filter + CUSUM ───────────────────────────
        logger.info("[L3] Running Kalman filter and CUSUM jump detector")
        kalman_out = run_kalman_pipeline(X_white)
        filtered = kalman_out["filtered_states"]
        result.filtered_states = filtered
        result.jump_flags = kalman_out["jump_flags"]
        result.cusum_g = kalman_out["cusum_g"]
        result._kalman_filter = kalman_out["kalman_filter"]
        n_jumps = int(result.jump_flags.sum())
        logger.info(f"[L3] Kalman done; {n_jumps} jumps detected")

        # ── Layer 4: TVTP-HMM ────────────────────────────────────────
        logger.info("[L4] Fitting TVTP-HMM (3 states, K=2 GMM)")
        sigma_t = features["sigma_t"].fillna(1.0).values
        rho_t = features["rho_t"].fillna(0.0).values
        covariates = np.column_stack([sigma_t, rho_t])

        # Use clean rows for HMM fitting
        clean_mask = ~np.any(np.isnan(filtered), axis=1)
        train_end = int(result.n_bars * config.train_fraction)
        X_train = filtered[clean_mask & (np.arange(result.n_bars) < train_end)]
        cov_train = covariates[clean_mask & (np.arange(result.n_bars) < train_end)]

        hmm = TVTPHMM(n_states=3, n_gmm=2, n_iter=30)
        if len(X_train) >= 30:
            hmm.fit(X_train, cov_train)
        else:
            # Minimal fallback
            hmm.fit(filtered[clean_mask[:min(50, len(filtered))]][:20],
                    covariates[clean_mask[:min(50, len(filtered))]][:20])

        result._hmm = hmm

        # Predict on full series (replace NaN rows with previous row)
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
        logger.info(f"[L4] HMM done; current regime: {result.current_regime}")

        # ── Layer 5: Execution ───────────────────────────────────────
        logger.info("[L5] Running execution layer (Triple Gate + Kelly sizing)")
        exec_out = run_execution_layer(
            bars_df, features, result.regime_proba, equity=config.equity
        )
        result.signals = exec_out["signals"]
        result.n_signals = exec_out["n_signals"]
        logger.info(f"[L5] {result.n_signals} trade signals generated")

        # ── Layer 5: Wasserstein Surveillance ────────────────────────
        logger.info("[L5] Running Wasserstein distribution surveillance")
        monitor = WassersteinMonitor(window=min(50, result.n_bars // 4))
        live_window = X_white[-monitor.window:]
        clean_train = X_white[:train_end]
        clean_train = clean_train[~np.any(np.isnan(clean_train), axis=1)]
        if len(clean_train) >= 10:
            monitor.fit(clean_train)
            surv_result = monitor.check(live_window)
        else:
            surv_result = {"status": "insufficient_training_data"}
        result.surveillance = surv_result
        logger.info(f"[L5] Surveillance status: {surv_result.get('status', 'unknown')}")

        # ── Layer 6: Validation (optional, slow) ─────────────────────
        if config.run_validation:
            logger.info("[L6] Running statistical validation suite")
            result.validation = self._run_validation(
                bars_df, features, result.regime_proba,
                config, result.signals
            )
            logger.info("[L6] Validation complete")

        return result

    def _run_validation(
        self,
        bars_df: pd.DataFrame,
        features: pd.DataFrame,
        regime_proba: np.ndarray,
        config: PipelineConfig,
        signals: list[TradeSignal],
    ) -> dict:
        """Run the full Layer 6 validation suite."""
        report = ValidationReport()

        # Build simple daily returns from signals for validation
        daily_returns = self._compute_strategy_returns(bars_df, regime_proba)

        # 6.1 Walk-Forward CV
        try:
            def wf_simulate(ts, te, ss, se):
                is_r = daily_returns[ts:te]
                oos_r = daily_returns[ss:se]
                return is_r, oos_r

            report.wfcv = walk_forward_cv(
                wf_simulate,
                T=len(daily_returns),
                train_bars=min(504, len(daily_returns) // 3),
                test_bars=min(126, len(daily_returns) // 6),
            )
        except Exception as e:
            logger.warning(f"WF-CV failed: {e}")

        # 6.2 Monte Carlo Permutation
        try:
            def perm_simulate(perm_returns):
                return _sharpe_ratio(perm_returns[-len(daily_returns) // 3:])

            report.mc_permutation = monte_carlo_permutation_test(
                daily_returns, perm_simulate,
                n_permutations=config.n_mc_permutations,
            )
        except Exception as e:
            logger.warning(f"MC permutation test failed: {e}")

        # 6.3 DSR
        try:
            oos_returns = daily_returns[int(len(daily_returns) * config.train_fraction):]
            report.dsr = deflated_sharpe_ratio(
                oos_returns, n_trials=report.n_trials
            )
        except Exception as e:
            logger.warning(f"DSR failed: {e}")

        # 6.4 CPCV
        try:
            n_folds = min(6, max(3, len(daily_returns) // 100))

            def cpcv_simulate(train_idx, test_idx):
                return daily_returns[train_idx], daily_returns[test_idx]

            report.cpcv = cpcv(
                cpcv_simulate, T=len(daily_returns), n_folds=n_folds
            )
        except Exception as e:
            logger.warning(f"CPCV failed: {e}")

        # 6.5 Transaction Cost Sensitivity
        try:
            n_trades = len(signals)
            report.tx_cost = transaction_cost_sensitivity(
                daily_returns, n_trades, len(daily_returns)
            )
        except Exception as e:
            logger.warning(f"Tx cost analysis failed: {e}")

        return report.to_dict()

    @staticmethod
    def _compute_strategy_returns(
        bars_df: pd.DataFrame,
        regime_proba: np.ndarray,
        regime_conf_threshold: float = 0.65,
    ) -> np.ndarray:
        """
        Simple regime-following strategy returns:
        +1 when P(TREND) > 0.65 and last return > 0, else flat.
        Used for validation only.
        """
        close = bars_df["close"].values
        raw_returns = np.diff(close) / close[:-1]
        T = len(raw_returns)

        proba_aligned = regime_proba[1: T + 1] if len(regime_proba) > T else regime_proba[:T]
        p_trend = proba_aligned[:, 0]

        momentum_sign = np.sign(raw_returns)
        position = np.where(p_trend > regime_conf_threshold, momentum_sign, 0.0)
        strategy_returns = position * raw_returns
        return strategy_returns.astype(float)


# ─────────────────────────────────────────────────────────────────── #
# Module-level singleton                                               #
# ─────────────────────────────────────────────────────────────────── #
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline

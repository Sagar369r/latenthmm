"""
FastAPI route handlers for the Latent Diffusion-HMM trading engine.

Endpoints:
  GET  /engine/healthz              Health check
  POST /engine/analyze              Full pipeline run
  GET  /engine/regime/{ticker}      Quick regime check (last 1y)
  POST /engine/signals              Get trade signals
  POST /engine/features             Get feature tensor
  POST /engine/validate             Full statistical validation suite
  GET  /engine/docs-summary         Architecture summary
"""
from __future__ import annotations

import logging
import traceback
from datetime import date, timedelta, datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, model_validator
import uuid
import os

from engine.pipeline import Pipeline, PipelineConfig, get_pipeline

logger = logging.getLogger("api")
router = APIRouter(prefix="/engine")


# ─────────────────────────────────────── #
# Request / Response models               #
# ─────────────────────────────────────── #

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. 'AAPL', 'SPY'")
    start: str = Field(
        default_factory=lambda: (date.today() - timedelta(days=365 * 3)).isoformat(),
        description="Start date ISO 8601, e.g. '2021-01-01'",
    )
    end: str = Field(
        default_factory=lambda: date.today().isoformat(),
        description="End date ISO 8601, e.g. '2024-01-01'",
    )
    equity: float = Field(default=100_000.0, ge=1.0, description="Portfolio equity in base currency")
    use_dollar_bars: bool = Field(default=True, description="Use dollar-volume bars (vs time bars)")
    apply_frac_diff: bool = Field(default=True, description="Apply fractional differentiation (MMMS)")
    run_validation: bool = Field(default=False, description="Run full statistical validation suite (slow)")
    n_mc_permutations: int = Field(default=200, ge=10, le=10000, description="Monte Carlo permutations")
    train_fraction: float = Field(default=0.7, ge=0.5, le=0.95)

    @model_validator(mode='after')
    def check_min_history(self) -> 'AnalyzeRequest':
        s = datetime.fromisoformat(self.start)
        e = datetime.fromisoformat(self.end)
        if (e - s).days < 730:
            raise ValueError("Engine mathematically requires at least 2 years (approx 504 bars) of history to fit Kalman window.")
        return self


class RegimeRequest(BaseModel):
    ticker: str
    lookback_days: int = Field(default=365, ge=30, le=365 * 10)


class SignalRequest(BaseModel):
    ticker: str
    start: str = Field(default_factory=lambda: (date.today() - timedelta(days=365 * 2)).isoformat())
    end: str = Field(default_factory=lambda: date.today().isoformat())
    equity: float = Field(default=100_000.0)


class FeatureRequest(BaseModel):
    ticker: str
    start: str = Field(default_factory=lambda: (date.today() - timedelta(days=365 * 2)).isoformat())
    end: str = Field(default_factory=lambda: date.today().isoformat())
    n_bars: int = Field(default=50, ge=1, le=500)


class ValidateRequest(BaseModel):
    ticker: str
    start: str = Field(default_factory=lambda: (date.today() - timedelta(days=365 * 5)).isoformat())
    end: str = Field(default_factory=lambda: date.today().isoformat())
    n_mc_permutations: int = Field(default=500, ge=50, le=10000)


# ─────────────────────────────────────── #
# Routes                                  #
# ─────────────────────────────────────── #

@router.get("/healthz")
async def health():
    csv_exists = os.path.exists("/home/suchith/Downloads/Latent-Diffusion-HMM/data/xauusd_4h_2014_2024.csv")
    if not csv_exists:
        raise HTTPException(status_code=503, detail="Golden CSV dataset missing from disk!")
    return {"status": "ok", "engine": "Latent Diffusion-HMM v3.0", "timestamp": datetime.utcnow().isoformat(), "csv_accessible": csv_exists}


def _run_pipeline_task(config: PipelineConfig, task_id: str):
    try:
        pipeline = get_pipeline()
        result = pipeline.run(config)
        logger.info(f"Task {task_id} completed successfully.")
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")


@router.post("/analyze", status_code=202)
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Run the full 6-layer pipeline on a ticker and date range.
    Dispatches to background to prevent HTTP timeouts.
    """
    try:
        config = PipelineConfig(
            ticker=req.ticker.upper(),
            start=req.start,
            end=req.end,
            equity=req.equity,
            use_dollar_bars=req.use_dollar_bars,
            apply_frac_diff=req.apply_frac_diff,
            run_validation=req.run_validation,
            n_mc_permutations=req.n_mc_permutations,
            train_fraction=req.train_fraction,
        )
        task_id = str(uuid.uuid4())
        background_tasks.add_task(_run_pipeline_task, config, task_id)
        return {"status": "ACCEPTED", "task_id": task_id, "message": "Pipeline Walk-Forward started in background"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.get("/regime/{ticker}")
async def get_regime(ticker: str, lookback_days: int = 365):
    """
    Quick regime classification for a ticker over the last N days.
    Returns current regime, probabilities, and recent Viterbi state sequence.
    """
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        config = PipelineConfig(
            ticker=ticker.upper(),
            start=start,
            end=end,
            run_validation=False,
        )
        pipeline = get_pipeline()
        result = pipeline.run(config)
        api = result.to_api_dict()
        return {
            "ticker": ticker.upper(),
            "current_regime": api["current_regime"],
            "regime_summary": api["regime_summary"],
            "surveillance": api["surveillance"],
            "n_bars": api["n_bars"],
            "recent_regimes": [
                {
                    "date": row["date"],
                    "regime": row["regime"],
                    "p_trend": row["p_trend"],
                    "p_mean_rev": row["p_mean_rev"],
                    "p_stress": row["p_stress"],
                }
                for row in api["feature_timeseries"][-20:]
            ],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals")
async def get_signals(req: SignalRequest):
    """
    Generate trade signals for a ticker over a date range.
    Returns all Triple Gate signals with entry, SL, TP, and position sizing.
    """
    try:
        config = PipelineConfig(
            ticker=req.ticker.upper(),
            start=req.start,
            end=req.end,
            equity=req.equity,
            run_validation=False,
        )
        pipeline = get_pipeline()
        result = pipeline.run(config)
        api = result.to_api_dict()
        return {
            "ticker": req.ticker.upper(),
            "period": {"start": req.start, "end": req.end},
            "n_signals": api["n_signals"],
            "signals": api["signals"],
            "current_regime": api["current_regime"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features")
async def get_features(req: FeatureRequest):
    """
    Compute and return the 6D feature tensor for a ticker.
    Returns last N bars of: vt, mvt, qt, sigma_t, rho_t, ht + regime probabilities.
    """
    try:
        config = PipelineConfig(
            ticker=req.ticker.upper(),
            start=req.start,
            end=req.end,
            run_validation=False,
        )
        pipeline = get_pipeline()
        result = pipeline.run(config)
        api = result.to_api_dict()
        tail = api["feature_timeseries"][-req.n_bars:]
        return {
            "ticker": req.ticker.upper(),
            "n_bars": len(tail),
            "features": tail,
            "feature_descriptions": {
                "vt": "Volatility Proximity [-1, +1]. |vt| > 0.85 = breakout zone",
                "mvt": "Momentum Velocity (σ-gated log-momentum). |mvt| > 1.0 = signal",
                "qt": "Volume Delta Ratio. qt > 2.0 = institutional demand; < 0.5 = distribution",
                "sigma_t": "Volatility Regime Ratio (RV / GARCH). > 1.5 = jump precursor; < 0.6 = coiling",
                "rho_t": "Autocorrelation Signal. > 0.15 = trending; < -0.15 = mean-reverting",
                "ht": "DFA Hurst Exponent. > 0.55 = trend persistence; < 0.45 = mean reversion",
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate(req: ValidateRequest):
    """
    Run the full Layer 6 statistical validation suite.
    This is slow (Monte Carlo) — recommend n_mc_permutations=200 for quick checks.

    Returns WF-CV, Monte Carlo permutation, DSR, CPCV, and Tx Cost sensitivity results.
    GO LIVE requires all 5 tests to pass simultaneously.
    """
    try:
        config = PipelineConfig(
            ticker=req.ticker.upper(),
            start=req.start,
            end=req.end,
            run_validation=True,
            n_mc_permutations=req.n_mc_permutations,
        )
        pipeline = get_pipeline()
        result = pipeline.run(config)
        api = result.to_api_dict()
        return {
            "ticker": req.ticker.upper(),
            "period": {"start": req.start, "end": req.end},
            "n_bars": api["n_bars"],
            "validation": api.get("validation", {}),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/docs-summary")
async def docs_summary():
    """Return a summary of the pipeline architecture."""
    return {
        "name": "Latent Diffusion-HMM Trading Engine",
        "version": "3.0",
        "layers": [
            {
                "layer": 1,
                "name": "Data Ingestion",
                "components": ["Dollar-Volume Bars", "MMMS Fractional Differentiation"],
                "key_params": {"target_bars_per_day": 20, "d_range": "0.35–0.45", "reestimate_every": 252},
            },
            {
                "layer": 2,
                "name": "6D Observation Tensor",
                "features": ["vt (Volatility Proximity)", "mvt (Momentum Velocity)", "qt (Volume Delta Ratio)",
                             "sigma_t (Volatility Regime Ratio)", "rho_t (Autocorrelation Signal)", "ht (DFA Hurst)"],
                "preprocessing": ["Expanding winsorize (1/99 pct)", "Expanding standardise", "Robust PCA whitening"],
            },
            {
                "layer": 3,
                "name": "Kalman Filter + CUSUM",
                "components": ["Linear Gaussian Kalman Filter (EM-fitted Q, R)", "CUSUM Jump Detector"],
                "key_params": {"kappa": 0.5, "threshold": 5.0, "fit_window": 504, "refit_every": 63},
            },
            {
                "layer": 4,
                "name": "TVTP-HMM",
                "states": {"0": "TREND", "1": "MEAN_REV", "2": "STRESS"},
                "key_params": {
                    "n_states": 3, "gmm_components": 2,
                    "tvtp_covariates": ["sigma_t", "rho_t"],
                    "beta_params": 18, "reg": {"lambda_A": 0.1, "lambda_mu": 0.01, "lambda_beta": 0.05},
                },
            },
            {
                "layer": 5,
                "name": "Execution + Surveillance",
                "components": [
                    "Triple Gate (Regime: P_TREND > 0.65, Momentum: |mvt| > 1.0, Volume: qt > 1.3)",
                    "Half-Kelly Position Sizing (max 2% equity cap)",
                    "ATR-based SL/TP with trailing stop",
                    "Wasserstein W1 Distribution Monitor",
                ],
            },
            {
                "layer": 6,
                "name": "Statistical Validation",
                "tests": [
                    "6.1 Walk-Forward Cross-Validation (IS/OOS Sharpe ratio < 2.0)",
                    "6.2 Monte Carlo Permutation Test (p < 0.05)",
                    "6.3 Deflated Sharpe Ratio (DSR > 0)",
                    "6.4 CPCV (MinSR > 0.3, PSR > 0.95)",
                    "6.5 Transaction Cost Sensitivity (Sharpe > 0.6 net at 2bps)",
                ],
                "go_live": "All 5 tests must pass simultaneously before live deployment",
            },
        ],
    }

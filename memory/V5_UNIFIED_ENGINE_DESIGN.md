# LATENT DIFFUSION-HMM: V5 UNIFIED ENGINE
## Complete Merger Design — V3 + V4 → V5
### Single Config · All-ONNX Inference · Risk Engine Module · Volume Fixed · Visualizer Layer
**Date:** 2026-06-11  
**Status:** Design specification for implementation

---

---

# PART 1 — WHY MERGE AND WHAT CHANGES

---

## 1.1 The problem with V3 and V4 as separate systems

V3 runs well on CPU but has no compiled inference — every live tick re-runs the full TVTP-HMM Baum-Welch EM loop (slow, non-deterministic). The manual walk-forward loop in `reports.py` takes 10–40 minutes per asset per grid-search run.

V4 has compiled ONNX inference that runs in microseconds but loses the mathematically rigorous statistical validation suite (DSR, CPCV, Monte Carlo), the Kalman filter, and the Wasserstein circuit-breaker from V3.

Neither engine has a dedicated risk management module — risk logic is scattered across `execution.py`, `reports.py`, `live_bridge.py`, and `config.py` with hardcoded values overriding each other.

Neither engine uses volume correctly. V3 uses `volume` as a raw ratio denominator in `qt` (Volume Delta Ratio). V4 generates 20 indicators from volume but then Z-scores them all flat — no distinction between tick volume, notional volume, and volume imbalance (order flow). Both miss the most important volume signal for FX: **bid/ask volume imbalance from tick data**, which is already available in the Dukascopy tick download.

There is no visualizer. You cannot see regime probabilities, signals, drawdowns, or Wasserstein distances live without building your own plots manually.

## 1.2 What the V5 merger produces

```
ONE DIRECTORY: v5_engine/
ONE CONFIG FILE: config.py (no hardcoding anywhere else)
ONE MODEL SPEC: model_spec.py
ONE TRAINING PIPELINE: train.py
ONE COMPILED INFERENCE: all models in bin/ as .onnx
ONE RISK ENGINE: risk.py (separate module, receives signals, returns sized positions)
ONE VISUALIZER: visualizer.py (standalone, connects to the API)
ONE LIVE BRIDGE: live_bridge_ctrader.py
ONE TEST SUITE: tests/ (all v3 + v4 tests merged and extended)
```

---

---

# PART 2 — FULL DIRECTORY STRUCTURE

---

```
project_root/
│
├── config.py                    # ALL parameters — no hardcoding anywhere else
├── model_spec.py                # model architecture only (dims, layers, dropout)
│
├── data_pipeline/
│   ├── data_manager.py          # Dukascopy tick download, OHLCV resampling (FIXED: mid-price not bid-only)
│   └── volume_processor.py      # NEW: bid/ask volume imbalance, OFI, tick volume normalisation
│
├── v5_engine/
│   ├── __init__.py
│   │
│   ├── features.py              # MERGED: V3 6D tensor + V4 20 z-score indicators = 26D unified tensor
│   ├── preprocess.py            # FIXED: proper fit() / transform() split, no state leak
│   ├── kalman.py                # V3 Kalman + CUSUM (unchanged but O(T) instead of O(T^2))
│   ├── hmm.py                   # V3 TVTP-HMM (FIXED: causal forward-only probabilities)
│   ├── vae.py                   # V4 VAE (from vae_model.py, using model_spec dims)
│   ├── router.py                # V4 Gumbel-Softmax router + LatentRouter (merged gmm_hmm.py)
│   ├── experts.py               # V4 expert layer (RF, XGBoost, IsolationForest per regime)
│   ├── meta.py                  # V4 ResidualMetaLearner (from meta_learner.py)
│   ├── execution.py             # V3 TripleGate (unchanged, no hardcoding)
│   ├── risk.py                  # NEW: dedicated risk management module
│   ├── surveillance.py          # V3 Wasserstein monitor (unchanged)
│   ├── validation.py            # V3 full statistical suite (unchanged)
│   │
│   ├── pipeline_train.py        # Training pipeline (V4 train_pipeline.py, refactored)
│   ├── pipeline_backtest.py     # V3 walk-forward backtest (reports.py, refactored stateless)
│   ├── pipeline_live.py         # Unified live inference (was live_bridge.py)
│   ├── compile.py               # ONNX export for all models (was compile_engine.py)
│   │
│   ├── bin/                     # Compiled ONNX models (all inference goes through here)
│   │   ├── vae.onnx
│   │   ├── router.onnx
│   │   ├── meta_judge.onnx
│   │   ├── expert_TREND.onnx
│   │   ├── expert_MEAN_REV.onnx
│   │   └── expert_COMPRESSION.onnx
│   │
│   └── models/                  # Trained weights (PyTorch .pth, sklearn .pkl)
│       ├── vae_weights.pth
│       ├── gumbel_router_weights.pth
│       ├── meta_judge.pth
│       ├── expert_TREND.pkl
│       ├── expert_MEAN_REV.pkl
│       └── expert_COMPRESSION.pkl
│
├── risk_engine/
│   ├── __init__.py
│   └── risk.py                  # Standalone risk module (Kelly, ATR sizing, position limits, drawdown kill)
│
├── visualizer/
│   ├── __init__.py
│   ├── app.py                   # FastAPI server for visualizer data endpoints
│   ├── dashboard.html           # Single-page dashboard (pure HTML+JS, no framework)
│   └── websocket.py             # Real-time WebSocket feed for live regime/signal data
│
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app (was v3_engine/main.py)
│   └── routes.py                # All endpoints (was v3_engine/api/routes.py, FIXED)
│
├── orchestrators/
│   ├── ensemble_pipeline.py     # Multi-asset backtest (FIXED: correct column names, stateless)
│   ├── grid_search.py           # Hyperparameter sweep
│   └── stage2_global_test.py    # Cross-asset stress test (FIXED: indentation bug)
│
├── live_trading/
│   ├── get_token.py             # FIXED: uses POST not GET
│   └── live_bridge_ctrader.py   # Unified live bridge (V4 version with V3 validation)
│
├── diagnostics/
│   └── audit_pipeline.py        # Updated audit script (was audit_real_data.py)
│
└── tests/
    ├── conftest.py
    ├── test_features.py          # V3 frac-diff + V4 z-score tensor tests
    ├── test_preprocess.py        # NEW: causality tests for fit/transform split
    ├── test_kalman_hmm.py        # V3 Kalman CUSUM + HMM Viterbi accuracy tests
    ├── test_execution.py         # V3 Triple gate tests
    ├── test_surveillance.py      # V3 Wasserstein circuit-breaker tests
    ├── test_validation.py        # V3 statistical validation suite tests
    ├── test_vae.py               # V4 VAE reconstruction + KL divergence tests
    ├── test_router.py            # V4 Gumbel router + LatentRouter tests
    ├── test_experts.py           # V4 triple barrier + expert OOF tests
    ├── test_meta.py              # V4 meta-learner tests
    ├── test_risk.py              # NEW: risk engine tests
    ├── test_compiler.py          # V4 ONNX vs PyTorch accuracy tests (FIXED: uses model_spec)
    └── test_integration.py       # NEW: full end-to-end pipeline smoke test
```

---

---

# PART 3 — THE UNIFIED FEATURE TENSOR (V3 + V4 MERGED)

---

## 3.1 Why V3 and V4 used different features

V3 computed a 6-dimensional handcrafted tensor (vt, mvt, qt, sigma_t, rho_t, ht) representing regime geometry. Each dimension had explicit financial meaning. The features were passed through Kalman filtering for noise reduction before HMM fitting.

V4 computed 20 z-scored indicators covering 5 families: trend (SMA deviations), momentum (ROC, MACD), volatility (ATR ratio, Bollinger width, log-return variance), volume surge, and order flow imbalance (OFI). These were fed raw into the VAE for latent encoding.

The V3 features are better for regime classification. The V4 features are better for trade direction confidence within a regime. The unified approach feeds both to the pipeline at the right stage.

## 3.2 Volume problem — what was wrong

V3 `qt` = `ask_volume / bid_volume` ratio from OHLCV. But after resampling from ticks, the OHLCV only has total volume (`ask_volume + bid_volume`). The bid/ask split is lost. So `qt` was computed as a pointless `total_volume / smoothed_volume` ratio — it measured nothing about order flow direction.

V4 used `ofi_ema / vol_ema` which is slightly better (uses bar direction as a proxy for buy/sell pressure) but still ignores the actual tick-level bid/ask imbalance.

The real volume signal for FX is:

```
TICK IMBALANCE = (bid_volume - ask_volume) / (bid_volume + ask_volume)
```

This is available directly from Dukascopy tick data before resampling. It measures net buying or selling pressure at the tick level. After resampling: sum the signed tick imbalance across the bar period, normalise by total volume.

## 3.3 The V5 unified 26-dimensional feature tensor

```
DIMENSION  NAME           SOURCE               MEANING
─────────────────────────────────────────────────────────────────────────────────
V3 REGIME GEOMETRY (6 dims — fed to Kalman → HMM)
  0        vt             features.py          Volatility Proximity [-1, +1]
  1        mvt            features.py          Momentum Velocity (σ-gated log-mom)
  2        sigma_t        features.py          Vol Regime Ratio (RV/GARCH)
  3        rho_t          features.py          Autocorrelation Signal
  4        ht             features.py          DFA Hurst Exponent
  5        tib            volume_processor.py  Tick Imbalance (bid-ask)/total [NEW]

V4 TREND FEATURES (5 dims — fed directly to VAE)
  6        trend_sma_10   feature_expansion.py (close - SMA10) / SMA10
  7        trend_sma_20   feature_expansion.py (close - SMA20) / SMA20
  8        trend_sma_50   feature_expansion.py (close - SMA50) / SMA50
  9        trend_sma_100  feature_expansion.py (close - SMA100) / SMA100
  10       trend_sma_200  feature_expansion.py (close - SMA200) / SMA200

V4 MOMENTUM FEATURES (5 dims — fed directly to VAE)
  11       mom_roc_5      feature_expansion.py 5-bar ROC
  12       mom_roc_10     feature_expansion.py 10-bar ROC
  13       mom_roc_20     feature_expansion.py 20-bar ROC
  14       mom_roc_50     feature_expansion.py 50-bar ROC
  15       mom_macd       feature_expansion.py (EMA12 - EMA26) / close

V4 VOLATILITY FEATURES (5 dims — fed directly to VAE)
  16       vol_atr_ratio  feature_expansion.py ATR5 / ATR20
  17       vol_bb_20      feature_expansion.py BB width 20
  18       vol_bb_50      feature_expansion.py BB width 50
  19       vol_var_10     feature_expansion.py log-return variance 10
  20       vol_var_20     feature_expansion.py log-return variance 20

V4 VOLUME FEATURES — FIXED (5 dims — fed directly to VAE)
  21       tib_z          volume_processor.py  Z-scored Tick Imbalance (90-bar window)
  22       tib_ema_z      volume_processor.py  Z-scored EMA of TIB (spans 20)
  23       vol_surge_z    feature_expansion.py Z-scored volume surge (5/20)
  24       ofi_10_z       feature_expansion.py Z-scored OFI EMA 10
  25       ofi_20_z       feature_expansion.py Z-scored OFI EMA 20
─────────────────────────────────────────────────────────────────────────────────

ROUTING:
  dims 0-5   → Preprocessor.fit/transform → Kalman → TVTP-HMM (regime probabilities)
  dims 6-25  → Z-score normalise → VAE (latent encoding) → Gumbel Router → Experts
  dims 0-25  → Combined → Expert input (latent + z-tensor concatenated)
```

## 3.4 Implementation of the tick imbalance fix

In `data_pipeline/data_manager.py`, the current code:
```python
df['volume'] = df['ask_volume'] + df['bid_volume']
ohlcv = df.resample(timeframe).agg({'bid': [...], 'volume': 'sum'})
```

Replace with:
```python
df['volume'] = df['ask_volume'] + df['bid_volume']
df['signed_volume'] = df['bid_volume'] - df['ask_volume']   # positive = net buying pressure
ohlcv = df.resample(timeframe).agg({
    'bid': ['first', 'max', 'min', 'last'],
    'ask': ['first', 'max', 'min', 'last'],   # keep ask for mid-price
    'volume': 'sum',
    'signed_volume': 'sum'
})
ohlcv.columns = ['open_bid', 'high_bid', 'low_bid', 'close_bid',
                 'open_ask', 'high_ask', 'low_ask', 'close_ask',
                 'volume', 'signed_volume']
# Compute mid-prices
ohlcv['open']  = (ohlcv['open_bid']  + ohlcv['open_ask'])  / 2
ohlcv['high']  = (ohlcv['high_bid']  + ohlcv['high_ask'])  / 2
ohlcv['low']   = (ohlcv['low_bid']   + ohlcv['low_ask'])   / 2
ohlcv['close'] = (ohlcv['close_bid'] + ohlcv['close_ask']) / 2
# Tick imbalance: net buying pressure per bar, normalised
ohlcv['tib'] = ohlcv['signed_volume'] / (ohlcv['volume'].clip(lower=1e-10))
# Final columns
ohlcv = ohlcv[['open', 'high', 'low', 'close', 'volume', 'tib']]
```

In `v5_engine/features.py`, the feature tensor function reads `tib` directly from the OHLCV and adds it as dimension 5 of the regime geometry tensor.

---

---

# PART 4 — THE MERGED INFERENCE PIPELINE (ALL ONNX)

---

## 4.1 The V3 training path (produces TVTP-HMM)

The TVTP-HMM cannot be compiled to ONNX because it uses a custom Numba/JAX Baum-Welch EM loop. However, the HMM only needs to run once per walk-forward fold during training. In live inference, only the forward pass (filtering step) is needed — and this CAN be expressed as a simple matrix loop that runs fast enough in Python for 1-second latency requirements.

The V5 strategy:
- HMM training: Python (JAX/Numba, same as V3) — runs offline
- HMM live inference: forward algorithm only, implemented in numpy, runs in ~1ms per bar
- Kalman filter live inference: numpy, ~0.5ms per bar
- All deep learning (VAE, router, meta-learner): compiled ONNX, runs in ~0.1ms per forward pass
- Expert trees: compiled to ONNX via onnxmltools/skl2onnx, ~0.2ms per forward pass

**Total live inference latency estimate: ~2ms per bar**

This is well within the 1-second requirement for 1-hour candle trading.

## 4.2 The V5 inference flow (live bar arrives)

```
[New OHLCV bar received]
        │
        ▼
[1. Feature Extraction]
   v5_engine/features.py → unified_features(bar, history)
   Output: 26-dim feature vector
        │
        ├─── dims 0-5 (regime geometry) ──────────────────────────────┐
        │                                                              │
        ▼                                                              │
[2. Preprocessor Transform]                                           │
   pp.transform(dims_0_5)  ← uses saved fit() state from training    │
   Output: 6-dim whitened vector                                      │
        │                                                              │
        ▼                                                              │
[3. Kalman Forward Filter]                                            │
   kf.filter_single_step(whitened_bar)  ← O(D^2) per bar, very fast  │
   Output: 6-dim filtered state vector + CUSUM g-value                │
        │                                                              │
        ▼                                                              │
[4. HMM Forward Pass (causal)]                                        │
   hmm.forward_step(filtered_bar, covariate)                          │
   Output: p_trend, p_mean_rev, p_stress (3 regime probabilities)    │
        │                                                              │
        └──────────────────────────────────────────────────────────────┘
        │
        ├─── dims 6-25 (deep features, z-scored) ─────────────────────┐
        │                                                              │
        ▼                                                              │
[5. VAE Latent Encoding]  ← ONNX: vae.onnx                           │
   Input: 20-dim z-score tensor (dims 6-25)                           │
   Output: mu (4-dim latent vector)                                   │
        │                                                              │
        ▼                                                              │
[6. Gumbel Router]  ← ONNX: router.onnx                              │
   Input: mu (4-dim latent)                                            │
   Output: regime_proba (3-dim softmax) — cross-check vs HMM         │
        │                                                              │
        └──────────────────────────────────────────────────────────────┘
        │
        ▼
[7. Regime Consensus Gate]
   v5_engine/execution.py
   HMM says TREND AND router says TREND → high confidence
   HMM says TREND, router says MEAN_REV → low confidence, reduce size
   Output: active_regime, regime_confidence (0–1)
        │
        ▼
[8. Triple Gate]
   regime_gate: regime_confidence > REGIME_CONF_THRESHOLD
   momentum_gate: |mvt| > MOMENTUM_THRESHOLD
   volume_gate: tib_z > VOLUME_THRESHOLD (FIXED — was always True)
   Output: direction (+1/-1/0), all_pass (bool)
        │
        ▼
[9. Expert Classifier]  ← ONNX: expert_{regime}.onnx
   Input: concat(mu, z_tensor) — 24-dim
   Output: raw_pred (0–1 confidence for direction)
        │
        ▼
[10. Meta-Learner Correction]  ← ONNX: meta_judge.onnx
   Input: concat(mu, regime_ohe, conf_ohe) — 17-dim
   Output: predicted_delta
   p_final = clip(raw_pred + predicted_delta, 0, 1)
        │
        ▼
[11. RISK ENGINE]  ← risk_engine/risk.py (NEW MODULE)
   Input: p_final, direction, atr, current_equity, open_positions
   Output: position_size, sl_price, tp_price, or VETO if risk limits breached
        │
        ▼
[12. Wasserstein Circuit Breaker]
   monitor.check(filtered_bar) → position_scale
   Final size = position_size × position_scale
        │
        ▼
[13. Order → cTrader]
   live_bridge_ctrader.py → ProtoOANewOrderReq
```

---

---

# PART 5 — THE RISK ENGINE MODULE

---

## 5.1 Why a separate risk module

Currently, risk logic is scattered in five places:
- `config.py`: HALF_KELLY_FRACTION, MAX_POSITION_FRACTION, ATR_SL_MULT, ATR_TP_MULT
- `execution.py`: KellyPositionSizer class, ATR computation, SL/TP placement
- `reports.py`: friction-adjusted position sizing, spread cost
- `live_bridge.py`: hardcoded `atr * 1.0` and `atr * 2.0`
- `ensemble_pipeline.py`: ENSEMBLE_HALF_KELLY_RISK scaling

None of these coordinate. If equity drops to $80,000, `KellyPositionSizer` does not know about it. If two positions are open simultaneously, neither position knows about the other's risk.

The Risk Engine is a single class that holds all state and enforces all risk rules before any order is sent.

## 5.2 risk_engine/risk.py — full specification

```python
# risk_engine/risk.py
# ALL parameters come from config — no defaults hardcoded here

class RiskEngine:
    """
    Centralised risk management. Receives signal intention, returns
    either a sized position specification or a VETO.

    Rules enforced (in order):
      1. Daily drawdown kill-switch  (breached → all trades vetoed for rest of day)
      2. Max total drawdown kill-switch  (breached → stop trading entirely)
      3. Max simultaneous positions limit
      4. Per-position ATR-based stop-loss sizing (Kelly-fractional)
      5. Wasserstein position scale adjustment
      6. Correlation veto: if two correlated positions already open, veto third
      7. p_final veto threshold
    """

    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.open_positions = {}        # symbol → PositionRecord
        self.daily_start_equity = initial_equity
        self.daily_reset_date = None
        self.trading_halted = False
        self.halt_reason = ""
        self._kelly_history = []        # rolling trade outcomes for Kelly estimation

    def update_equity(self, new_equity: float, current_date):
        """Call this after every closed trade."""
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        # Reset daily P&L tracker on new day
        if self.daily_reset_date != current_date:
            self.daily_start_equity = new_equity
            self.daily_reset_date = current_date
        # Check kill-switches
        daily_dd = (new_equity - self.daily_start_equity) / self.daily_start_equity
        total_dd = (new_equity - self.peak_equity) / self.peak_equity
        if daily_dd < -config.DAILY_DRAWDOWN_LIMIT:
            self.trading_halted = True
            self.halt_reason = f"Daily drawdown limit {config.DAILY_DRAWDOWN_LIMIT*100:.1f}% hit"
        if total_dd < -config.MAX_DRAWDOWN_LIMIT:
            self.trading_halted = True
            self.halt_reason = f"Max drawdown limit {config.MAX_DRAWDOWN_LIMIT*100:.1f}% hit"

    def evaluate(
        self,
        symbol: str,
        direction: int,               # +1 long, -1 short
        p_final: float,               # meta-learner final confidence
        atr: float,
        current_price: float,
        wasserstein_scale: float,     # from WassersteinMonitor (0.25 to 1.0)
        regime: str,
    ) -> dict:
        """
        Returns either:
          {"action": "VETO", "reason": str}
        or:
          {"action": "EXECUTE", "symbol": str, "direction": int,
           "units": float, "stop_loss": float, "take_profit": float,
           "position_fraction": float, "kelly_f": float}
        """
        # Rule 1: Trading halt check
        if self.trading_halted:
            return {"action": "VETO", "reason": self.halt_reason}

        # Rule 2: p_final below veto threshold
        if p_final < config.LIVE_P_FINAL_THRESHOLD:
            return {"action": "VETO", "reason": f"p_final {p_final:.3f} < threshold"}

        # Rule 3: max simultaneous positions
        if len(self.open_positions) >= config.MAX_SIMULTANEOUS_POSITIONS:
            return {"action": "VETO", "reason": f"Max {config.MAX_SIMULTANEOUS_POSITIONS} positions open"}

        # Rule 4: no re-entry on same symbol
        if symbol in self.open_positions:
            return {"action": "VETO", "reason": f"Position already open for {symbol}"}

        # Rule 5: Kelly sizing
        kelly_f = self._compute_kelly(p_final)
        kelly_f = min(kelly_f, config.MAX_POSITION_FRACTION)
        kelly_f = kelly_f * wasserstein_scale     # scale down on distribution shift

        # Rule 6: ATR-based position sizing
        if regime == "TREND":
            sl_mult = config.ATR_SL_MULT_TREND
            tp_mult = config.ATR_TP_MULT_TREND
        elif regime == "MEAN_REV":
            sl_mult = config.ATR_SL_MULT_MEAN_REV
            tp_mult = config.ATR_TP_MULT_MEAN_REV
        else:  # COMPRESSION
            sl_mult = config.ATR_SL_MULT_COMPRESSION
            tp_mult = config.ATR_TP_MULT_COMPRESSION

        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        risk_amount = kelly_f * self.current_equity
        units = risk_amount / max(sl_distance, 1e-10)

        if direction == 1:
            stop_loss   = current_price - sl_distance
            take_profit = current_price + tp_distance
        else:
            stop_loss   = current_price + sl_distance
            take_profit = current_price - tp_distance

        return {
            "action": "EXECUTE",
            "symbol": symbol,
            "direction": direction,
            "units": units,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_fraction": kelly_f,
            "kelly_f": kelly_f,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
        }

    def register_position(self, symbol: str, position: dict):
        self.open_positions[symbol] = position

    def close_position(self, symbol: str, exit_price: float, exit_time):
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return
        pnl = (exit_price - pos["entry_price"]) * pos["direction"] * pos["units"]
        self._kelly_history.append(pnl)
        if len(self._kelly_history) > config.KELLY_ESTIMATION_WINDOW:
            self._kelly_history.pop(0)
        self.update_equity(self.current_equity + pnl, exit_time.date())

    def _compute_kelly(self, p_final: float) -> float:
        """Half-Kelly using rolling win rate if available, else p_final directly."""
        if len(self._kelly_history) >= 30:
            wins = sum(1 for r in self._kelly_history if r > 0)
            win_rate = wins / len(self._kelly_history)
            avg_win = sum(r for r in self._kelly_history if r > 0) / max(wins, 1)
            avg_loss = abs(sum(r for r in self._kelly_history if r <= 0)) / max(len(self._kelly_history) - wins, 1)
            b = avg_win / max(avg_loss, 1e-10)
            full_kelly = (win_rate * b - (1 - win_rate)) / max(b, 1e-10)
            return max(0.0, full_kelly * config.HALF_KELLY_FRACTION)
        # Fallback: use p_final as win probability, assume 1:1 risk-reward
        return max(0.0, (p_final - (1 - p_final)) * config.HALF_KELLY_FRACTION)
```

## 5.3 New config.py parameters for the risk engine

```python
# --- RISK ENGINE ---
DAILY_DRAWDOWN_LIMIT      = 0.05      # 5% daily loss → halt all trading today
MAX_DRAWDOWN_LIMIT        = 0.10      # 10% total drawdown from peak → stop system
MAX_SIMULTANEOUS_POSITIONS = 3        # max open positions across all symbols at once
KELLY_ESTIMATION_WINDOW   = 100       # rolling trade count for Kelly win-rate estimation

# ATR multipliers per regime (replaces the single ATR_SL_MULT and ATR_TP_MULT)
ATR_SL_MULT_TREND         = 3.0       # tighter stop in trend — let winners run
ATR_TP_MULT_TREND         = 8.0
ATR_SL_MULT_MEAN_REV      = 1.5       # tighter stop in mean-rev — quick reversals
ATR_TP_MULT_MEAN_REV      = 3.0
ATR_SL_MULT_COMPRESSION   = 2.0       # breakout plays — medium stop
ATR_TP_MULT_COMPRESSION   = 5.0

# Keep these for compatibility
ATR_SL_MULT = ATR_SL_MULT_TREND       # default (used in validation)
ATR_TP_MULT = ATR_TP_MULT_TREND
```

---

---

# PART 6 — THE VISUALIZER

---

## 6.1 What it shows

The visualizer is a standalone FastAPI + HTML dashboard that polls the API and shows:

```
Panel 1: REGIME PROBABILITY CHART (real-time, last 200 bars)
  - Rolling p_trend, p_mean_rev, p_stress as stacked area chart
  - Current regime highlighted in header
  - Viterbi path overlaid as discrete colour bands

Panel 2: PRICE + SIGNALS (last 200 bars)
  - Candlestick chart (open, high, low, close)
  - BUY signals as green triangles, SELL signals as red triangles
  - SL/TP lines on active positions

Panel 3: WASSERSTEIN DISTANCE (last 200 bars)
  - W1 distance as line chart
  - Threshold line (red dashed)
  - Halt zones shaded red

Panel 4: RISK DASHBOARD (live numbers)
  - Current equity
  - Open positions table (symbol, direction, entry, SL, TP, unrealised P&L)
  - Daily P&L gauge
  - Total drawdown gauge
  - Kelly fraction current value

Panel 5: VOLUME (last 200 bars)
  - Raw volume as bar chart
  - TIB (tick imbalance) as line chart overlaid, colour-coded (positive = green, negative = red)
  - OFI EMA as secondary line

Panel 6: FEATURE TENSOR (last 50 bars)
  - Heatmap of all 6 V3 features normalised
  - Useful for diagnosing why signals fire or don't

Panel 7: BACKTEST TEARSHEET (on-demand)
  - Equity curve
  - Drawdown curve
  - Monthly returns heatmap
  - Phase 6 validation results table
```

## 6.2 visualizer/app.py — endpoint design

```python
# visualizer/app.py
# Adds to the main FastAPI app in api/main.py

# Static HTML dashboard
@app.get("/visualizer")
async def serve_dashboard():
    return FileResponse("visualizer/dashboard.html")

# JSON data feed for dashboard charts
@app.get("/visualizer/data/{ticker}")
async def get_dashboard_data(ticker: str, bars: int = 200):
    """Returns all panel data in one JSON call."""
    # Loads from cache (last pipeline result for this ticker)
    # Returns:
    return {
        "ticker": ticker,
        "bars": [...],            # OHLCV + tib for price chart
        "regime_proba": [...],    # p_trend, p_mean_rev, p_stress per bar
        "signals": [...],         # entry signals with direction, SL, TP
        "wasserstein": [...],     # W1 distance + threshold per bar
        "features": [...],        # 6D feature heatmap data
        "risk": {...},            # live risk dashboard numbers
        "volume": [...],          # volume + TIB per bar
    }

# WebSocket for live updates (pushes new data every bar or on signal)
@app.websocket("/visualizer/ws/{ticker}")
async def websocket_feed(websocket: WebSocket, ticker: str):
    await websocket.accept()
    while True:
        data = get_latest_bar_data(ticker)
        await websocket.send_json(data)
        await asyncio.sleep(1.0)  # check every second; push only on new data
```

## 6.3 visualizer/dashboard.html — chart library

Use Chart.js (loaded from CDN, no build step) for all panels. This keeps the visualizer fully self-contained in a single HTML file.

```html
<!-- In dashboard.html head -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js"></script>
```

The WebSocket connection in JavaScript:
```javascript
const ws = new WebSocket(`ws://${location.host}/visualizer/ws/${ticker}`);
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateAllCharts(data);   // pushes new bar to all chart instances
};
```

---

---

# PART 7 — THE COMPLETE config.py (SINGLE SOURCE OF TRUTH)

---

All parameters. No hardcoded values anywhere else in the codebase. Every module reads from here.

```python
# config.py — COMPLETE V5 CONFIGURATION
# All parameters live here. To override for an experiment, set environment variables.
# Example: ATR_SL_MULT_TREND=2.5 python v5_engine/pipeline_backtest.py

import os

# =============================================================================
# DATA
# =============================================================================
DATA_DIR                = "data"
RESULTS_DIR             = "results"
SYMBOLS                 = ["EURUSD", "XAUUSD", "GBPUSD", "CHFJPY", "EURNZD", "AUDCAD"]
DEFAULT_TIMEFRAME       = "1h"
BARS_PER_DAY            = 24              # 24 for FX 1h; 6.5 for equity 1h; 1 for daily
ANN_FACTOR              = int(252 * BARS_PER_DAY)   # = 6048 for FX 1h
OOS_START_DATE          = "2022-01-01"
PIPET_SCALES = {
    "AUDCHF":100000,"AUDCAD":100000,"AUDNZD":100000,"AUDUSD":100000,
    "CADCHF":100000,"EURAUD":100000,"EURCAD":100000,"EURCHF":100000,
    "EURGBP":100000,"EURNZD":100000,"EURNOK":100000,"EURSEK":100000,
    "EURUSD":100000,"GBPAUD":100000,"GBPCAD":100000,"GBPCHF":100000,
    "GBPNZD":100000,"NOKSEK":100000,"GBPUSD":100000,"NZDCAD":100000,
    "NZDCHF":100000,"NZDUSD":100000,"USDCAD":100000,"USDCHF":100000,
    "CADJPY":1000,"CHFJPY":1000,"USDJPY":1000,"BTCUSD":100,
}

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
Z_SCORE_WINDOW          = 90             # rolling window for V4 z-score normalisation
HURST_WINDOW            = 100            # DFA window for ht feature
GARCH_WINDOW            = 60             # GARCH estimation window for sigma_t
AUTOCORR_LAG            = 1              # lag for rho_t autocorrelation
FRAC_DIFF_D_MIN         = 0.35           # MMMS minimum d range
FRAC_DIFF_D_MAX         = 0.45

# =============================================================================
# PREPROCESSOR
# =============================================================================
WINSORIZE_LOWER         = 1              # percentile (1st)
WINSORIZE_UPPER         = 99             # percentile (99th)
WELFORD_WARMUP          = 100            # bars before standardisation activates

# =============================================================================
# KALMAN FILTER
# =============================================================================
KALMAN_FIT_WINDOW       = 504
KALMAN_REFIT_EVERY      = 63
CUSUM_KAPPA             = 0.5
CUSUM_THRESHOLD         = 5.0
CUSUM_WARMUP            = 30

# =============================================================================
# HMM
# =============================================================================
HMM_N_STATES            = 3
HMM_N_GMM               = 2
HMM_N_ITER              = 50
HMM_LAMBDA_A            = 0.1
HMM_LAMBDA_MU           = 0.01
HMM_LAMBDA_BETA         = 0.05
HMM_RIDGE_SIGMA         = 1e-5
HMM_REGIME_CONF_THRESHOLD = 0.51        # minimum p_trend to pass regime gate

# =============================================================================
# WALK-FORWARD BACKTEST
# =============================================================================
WF_TRAIN_BARS           = 504
WF_OOS_BARS             = 126
WF_TIMEFRAME            = "1h"
INITIAL_EQUITY          = 100_000.0
STRATEGY_MODE           = "MOMENTUM"    # MOMENTUM | MEAN_REV | MEAN_REVERSION_EXHAUSTION

# =============================================================================
# EXECUTION — TRIPLE GATE
# =============================================================================
REGIME_CONF_THRESHOLD   = 0.51          # same as HMM_REGIME_CONF_THRESHOLD
MOMENTUM_THRESHOLD      = 0.1           # |mvt| must exceed this to pass momentum gate
VOLUME_THRESHOLD        = 0.3           # tib_z must exceed this (FIXED: was always True)

# =============================================================================
# ATR — PER REGIME (replaces single ATR_SL_MULT, ATR_TP_MULT)
# =============================================================================
ATR_PERIOD              = 14
ATR_SL_MULT_TREND       = 3.0
ATR_TP_MULT_TREND       = 8.0
ATR_SL_MULT_MEAN_REV    = 1.5
ATR_TP_MULT_MEAN_REV    = 3.0
ATR_SL_MULT_COMPRESSION = 2.0
ATR_TP_MULT_COMPRESSION = 5.0
# Aliases for validation suite compatibility
ATR_SL_MULT             = ATR_SL_MULT_TREND
ATR_TP_MULT             = ATR_TP_MULT_TREND
ATR_TRAIL_TRIGGER       = 2.0
ATR_TRAIL_DIST          = 1.5
TIME_EXIT_BARS          = 5

# =============================================================================
# RISK ENGINE
# =============================================================================
HALF_KELLY_FRACTION         = 0.25
MAX_POSITION_FRACTION       = 0.02
KELLY_ESTIMATION_WINDOW     = 100
DAILY_DRAWDOWN_LIMIT        = 0.05      # 5% daily loss → halt
MAX_DRAWDOWN_LIMIT          = 0.10      # 10% total → halt
MAX_SIMULTANEOUS_POSITIONS  = 3
R_SL_TREND                  = 3.0
R_SL_MEAN_REV               = 1.0
R_TP                        = 5.0

# =============================================================================
# LIVE TRADING
# =============================================================================
LIVE_P_FINAL_THRESHOLD      = 0.65
LIVE_SL_MULT                = ATR_SL_MULT_TREND
LIVE_TP_MULT                = ATR_TP_MULT_TREND
LIVE_W1_THRESHOLD_MULTIPLIER = 0.3
LIVE_POSITION_SCALE_ON_HALT  = 0.25
POLL_INTERVAL_SEC           = 60
PORT                        = 8000

# cTrader credentials — always from environment, never hardcoded
CTRADER_CLIENT_ID           = os.getenv("CTRADER_CLIENT_ID", "")
CTRADER_CLIENT_SECRET       = os.getenv("CTRADER_CLIENT_SECRET", "")
AUTH_CODE                   = os.getenv("CTRADER_AUTH_CODE", "")
CTRADER_ACCOUNT_ID          = int(os.getenv("CTRADER_ACCOUNT_ID", "0"))
CTRADER_ACCESS_TOKEN        = os.getenv("CTRADER_ACCESS_TOKEN", "")
ENVIRONMENT_MODE            = os.getenv("ENVIRONMENT_MODE", "demo")

# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================
MC_PERMUTATIONS             = 500
MC_DEEP_SHUFFLE             = True
DSR_TRIALS                  = 200
CPCV_FOLDS                  = 6
CPCV_TEST_FOLDS             = 2
CPCV_EMBARGO_BARS           = 5
COST_SCENARIOS = {
    "optimistic":   {"bps": 0.5,  "min_sharpe": 0.8},
    "realistic":    {"bps": 2.0,  "min_sharpe": 0.6},
    "conservative": {"bps": 5.0,  "min_sharpe": 0.4},
    "stress":       {"bps": 10.0, "min_sharpe": 0.2},
}

# =============================================================================
# ENSEMBLE / PORTFOLIO
# =============================================================================
ENSEMBLE_SHARPE_FIREWALL    = 0.40
ENSEMBLE_HALF_KELLY_RISK    = 0.01
BANG_SHARPE                 = 1.2
BANG_PF                     = 1.5
BANG_MDD                    = 15.0
WASTE_SHARPE                = 0.3
WASTE_PF                    = 1.0
WASTE_MDD                   = 25.0

# =============================================================================
# GRID SEARCH
# =============================================================================
GRID_SEARCH_PARAMS = {
    "VETO_THRESHOLD":  [0.60, 0.65, 0.70, 0.75],
    "TIME_EXIT_BARS":  [5, 8, 12, 24, 48],
    "ATR_SL_MULT_TREND":  [2.0, 3.0, 4.0],
    "ATR_TP_MULT_TREND":  [6.0, 8.0, 10.0],
    "HMM_WF_OOS_BARS":    [120, 240, 504],
    "HMM_N_STATES":       [2, 3],
}
GRID_SLIPPAGE_KEY           = "1.5 pip"

# =============================================================================
# V4 / DEEP LEARNING TRAINING
# =============================================================================
VAE_BATCH_SIZE              = 256
VAE_EPOCHS                  = 30
VAE_LR                      = 1e-3
VAE_BETA                    = 0.1
VAE_VAL_SPLIT               = 0.2
VAE_CHRONO_CUTOFF           = 1672531200000   # 2023-01-01 in ms — IS/OOS split
META_BATCH_SIZE             = 256
META_EPOCHS                 = 20
META_LR                     = 0.001
META_WEIGHT_DECAY           = 1e-3
META_TRAIN_SPLIT            = 0.8
EXPERT_N_SPLITS             = 5
EXPERT_EMBARGO_BARS         = 5
EXPERT_BINS                 = 10
BARRIER_TP_MULT             = 2.0
BARRIER_SL_MULT             = 1.0
BARRIER_TIME_LIMIT          = 24

# =============================================================================
# PERIODS (for period breakdown in tear sheet)
# =============================================================================
PERIODS = [
    ("COVID Crash",     "2020-01-01", "2020-09-30"),
    ("Post-COVID Bull", "2021-01-01", "2021-12-31"),
    ("Rate-Hike Cycle", "2022-01-01", "2022-12-31"),
    ("EUR Recovery",    "2023-01-01", "2023-12-31"),
    ("2024 Extension",  "2024-01-01", "2024-12-31"),
]

# =============================================================================
# CORS (production — restrict to actual domain)
# =============================================================================
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# =============================================================================
# ENV OVERRIDE (applied last — env vars win over file values)
# =============================================================================
def _apply_env_overrides():
    import json
    import sys
    mod = sys.modules[__name__]
    for key in dir(mod):
        if key.isupper() and not key.startswith("_"):
            env_val = os.getenv(key)
            if env_val is not None:
                original = getattr(mod, key)
                try:
                    if isinstance(original, bool):
                        setattr(mod, key, env_val.lower() in ("true","1","yes"))
                    elif isinstance(original, int):
                        setattr(mod, key, int(env_val))
                    elif isinstance(original, float):
                        setattr(mod, key, float(env_val))
                    elif isinstance(original, str):
                        setattr(mod, key, env_val)
                    elif isinstance(original, (dict, list)):
                        setattr(mod, key, json.loads(env_val))
                except (ValueError, json.JSONDecodeError):
                    pass

_apply_env_overrides()
```

---

---

# PART 8 — MODEL SPEC (architecture only)

---

```python
# model_spec.py — MODEL ARCHITECTURE CONSTANTS ONLY
# These define the shape of the neural network.
# Training hyperparameters (lr, epochs, batch) are in config.py.
# If you change these, you must retrain all models and recompile to ONNX.

# VAE
VAE_INPUT_DIM       = 20              # 20 z-scored features (dims 6-25 of unified tensor)
VAE_LATENT_DIM      = 4               # latent space dimension
VAE_ENCODER_LAYERS  = [16, 8]
VAE_DECODER_LAYERS  = [8, 16]

# Gumbel Router
ROUTER_LATENT_DIM   = 4               # must match VAE_LATENT_DIM
ROUTER_N_REGIMES    = 3               # must match HMM_N_STATES

# Expert models — input dim = VAE_LATENT_DIM + VAE_INPUT_DIM
EXPERT_INPUT_DIM    = VAE_LATENT_DIM + VAE_INPUT_DIM   # = 24

# Meta-Learner
# Input: VAE_LATENT_DIM + ROUTER_N_REGIMES + EXPERT_BINS = 4 + 3 + 10 = 17
META_INPUT_DIM      = VAE_LATENT_DIM + ROUTER_N_REGIMES + 10  # = 17
META_HIDDEN_LAYERS  = [16, 8]
META_DROPOUT        = 0.5             # architecture constant — NOT in config.py
META_TANH_MULTIPLIER = 1.0
```

---

---

# PART 9 — MIGRATION PLAN FROM V3+V4 TO V5

---

## Phase A — Create the v5_engine directory (Week 1)

```
Day 1:  Create v5_engine/__init__.py and stub all module files
        Copy v3_engine/engine/*.py into v5_engine/ with new names
        Copy v4_engine/*.py into v5_engine/ with new names
        Create the new config.py (from Part 7 above)
        Create the new model_spec.py (from Part 8 above)

Day 2:  Fix Preprocessor (BUG-001 from audit):
          Add fit() method that updates Welford state without applying
          Modify transform() to use saved state without updating
          Write test: assert transform(oos_bar) does not change after fit()

Day 3:  Fix HMM (BUG-002 from audit):
          Add predict_proba_causal() using forward algorithm only
          Verify no backward pass in predict_proba
          Write test: assert bar T output does not change when bar T+1 is added

Day 4:  Implement volume_processor.py (tick imbalance):
          Add signed_volume computation in data_manager.py
          Add tib feature to features.py as dimension 5
          Verify tib is not NaN for any symbol with tick data

Day 5:  Create unified features.py:
          merge v3 compute_feature_tensor() (6 dims)
          merge v4 generate_20_indicators() (20 dims)
          return unified 26-dim tensor with correct routing labels
```

## Phase B — Risk engine and visualizer (Week 2)

```
Day 6:  Implement risk_engine/risk.py as specified in Part 5
        Write full test suite: test_risk.py covering all 7 rules
        Wire risk engine into pipeline_live.py (replace scattered risk logic)

Day 7:  Fix all BUGs from audit report Phases 1-3
        (BUG-003 reports.py stateless, BUG-005 stage2 indentation,
         BUG-006 column names, BUG-008 POST not GET, BUG-010 ANN_FACTOR,
         BUG-013 OOS_START, BUG-014 forex_tearsheet import, BUG-015 CORS,
         BUG-017 META_DROPOUT, BUG-019 prop firm simulator)

Day 8:  Implement visualizer/app.py and visualizer/dashboard.html
        Wire up all 7 panels with Chart.js
        Test WebSocket live feed on demo data

Day 9:  Fix Sharpe annualisation everywhere (BUG-010)
        Replace all 252 with config.ANN_FACTOR
        Re-run ensemble on EURUSD to verify Sharpe number changes correctly
        Update Sharpe firewall threshold accordingly

Day 10: Implement volume gate in Triple Gate execution
        VOLUME_THRESHOLD = 0.3 from config
        tib_z must exceed threshold (was always True before)
        Backtest EURUSD with and without volume gate to measure impact
```

## Phase C — ONNX compilation of entire pipeline (Week 3)

```
Day 11: Run v5_engine/train_pipeline.py on full dataset (2018-2023)
        Train: VAE, experts, meta-learner
        Compile: compile.py exports all to bin/

Day 12: Run full walk-forward backtest with v5 engine
        Compare V3 IS/OOS Sharpe vs V5 IS/OOS Sharpe
        If V5 IS/OOS ratio < 2.0 and V3 was > 2.0, the leakage fixes worked

Day 13: Run statistical validation suite (all 6 pytest tests)
        Run Phase 6 validation on backtest results
        Must pass: MC Permutation p < 0.05, DSR > 0, CPCV PSR > 0.95

Day 14: Run ensemble pipeline (all 6 symbols)
        Compare V3 ensemble vs V5 ensemble
        V5 should have lower absolute Sharpe (less inflated) but better IS/OOS ratio

Day 15: Start live demo bridge, monitor for 48 hours
        Check visualizer shows live regime probabilities, TIB, signals
        Verify risk engine halts trading if daily DD limit hit
```

## Phase D — Go-live criteria (Week 4)

All criteria from the audit report Part 5 Phase 7, plus:
```
□ All tests pass: pytest tests/ -v → 0 failures
□ ANN_FACTOR = 6048 used everywhere (not 252)
□ Volume gate is active and filtering trades (not always True)
□ TIB feature is non-zero for all symbols with tick data
□ Risk engine halts correctly on simulated daily DD breach
□ Visualizer shows all 7 panels with live data
□ Backtest IS/OOS Sharpe ratio < 2.0 (leakage removed)
□ Phase 6 validation passes 5/5 on OOS returns
□ 50+ paper trades on demo with positive net P&L
```

---

---

# PART 10 — WHAT TO BUILD IN WHAT ORDER

---

This is the precise implementation sequence. Each step produces something testable.

```
STEP 1 — new config.py
  Input: existing config.py
  Output: new config.py with all params from Part 7
  Test: python -c "import config; print(config.ANN_FACTOR)" → 6048
  Time: 30 minutes

STEP 2 — new model_spec.py
  Input: existing model_spec.py + config.py
  Output: model_spec.py with META_DROPOUT only (removed from config)
  Test: python -c "import model_spec; assert model_spec.META_INPUT_DIM == 17"
  Time: 15 minutes

STEP 3 — data_manager.py volume fix
  Input: existing data_pipeline/data_manager.py
  Output: adds mid-price and tib (tick imbalance) columns
  Test: re-download 1 symbol, check CSV has 'tib' column, no NaN
  Time: 1 hour

STEP 4 — unified features.py
  Input: v3 features.py + v4 feature_expansion.py
  Output: v5_engine/features.py returning 26D tensor, tib in dim 5
  Test: compute_unified_features(df).shape == (T, 26)
  Time: 2 hours

STEP 5 — preprocess.py with fit/transform split
  Input: existing preprocess.py
  Output: Preprocessor with fit(), transform(), no state reset
  Test: test_preprocess.py — causality test (OOS value must not change with more OOS data added)
  Time: 3 hours

STEP 6 — hmm.py causal forward pass
  Input: existing hmm.py
  Output: TVTPHMM with predict_proba_causal()
  Test: test that p_trend[t] does not change when bar t+1 is appended
  Time: 2 hours

STEP 7 — risk_engine/risk.py
  Input: scattered risk logic from execution.py, reports.py, live_bridge.py
  Output: RiskEngine class as specified in Part 5
  Test: test_risk.py — all 7 rules tested
  Time: 4 hours

STEP 8 — execution.py volume gate fix
  Input: existing execution.py
  Output: volume_pass = tib_z > config.VOLUME_THRESHOLD
  Test: test triple gate fails when tib_z is below threshold
  Time: 1 hour

STEP 9 — pipeline_live.py (unified inference)
  Input: v3 pipeline.py + v4 live_bridge.py
  Output: v5_engine/pipeline_live.py implementing full flow from Part 4.2
  Test: feed 200 bars of EURUSD, assert output is a dict with action key
  Time: 4 hours

STEP 10 — visualizer
  Input: API routes.py
  Output: visualizer/app.py + visualizer/dashboard.html
  Test: open http://localhost:8000/visualizer, all 7 panels render
  Time: 6 hours

STEP 11 — reports.py / pipeline_backtest.py stateless refactor
  Input: existing reports.py
  Output: run_tearsheet_dynamic() with no module-level globals
  Test: run on 2 symbols in same process, assert results differ
  Time: 3 hours

STEP 12 — Sharpe annualisation fix everywhere
  Input: all files with ann_factor = 252
  Output: all use config.ANN_FACTOR
  Test: re-run EURUSD backtest, Sharpe should be ~2.5× larger
  Time: 1 hour

STEP 13 — compile.py ONNX export
  Input: trained PyTorch weights, sklearn models
  Output: all 6 ONNX files in bin/
  Test: test_compiler.py passes
  Time: 1 hour (after training is complete)

STEP 14 — live_bridge_ctrader.py unified
  Input: v3 + v4 live bridges
  Output: single live_trading/live_bridge_ctrader.py using v5 pipeline
  Test: paper trade demo connection confirmed
  Time: 2 hours

TOTAL ESTIMATED TIME: 5-6 days focused work
```

---

**END OF V5 UNIFIED ENGINE DESIGN**

---

*This document supersedes the V3/V4 split architecture. The V5 engine is not a new engine — it is V3 and V4 merged, with all audit bugs fixed, compiled to ONNX where possible, with a dedicated risk module and a live visualizer.*

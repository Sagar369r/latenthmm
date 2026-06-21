# GROUND TRUTH SYSTEM AUDIT + V6 ARCHITECTURE PLAN
## Full Codebase Inspection · Bugs · Orphans · Architecture Redesign · Data Pipeline · Risk Engine · Prop Firm Simulator
**Snapshot:** CODEBASE_4.md — Repomix 2026-06-11T14:16:33.974Z  
**Files in repo:** 57 Python files across 9 directories  
**Status of V5:** Partially assembled, incomplete merge, multiple critical failures  

---

---

# SECTION 1 — FILE-BY-FILE ORPHAN AND DELETION AUDIT

---

## 1.1 Files to DELETE immediately — no references, duplicate, or superseded

```
DELETE  v3_engine/api/__init__.py          — empty file, no content
DELETE  v3_engine/api/routes.py            — EXACT DUPLICATE of api/routes.py (line-for-line identical minus import path)
DELETE  v3_engine/main.py                  — superseded by api/main.py
DELETE  v3_engine/conftest.py              — empty pytest config, no fixtures
DELETE  v3_engine/engine/__init__.py       — empty
DELETE  v3_engine/prop_firm_simulator.py   — hardcoded win_rate=0.6769, never called from any pipeline, no data hookup
DELETE  v4_engine/live_bridge_ctrader.py   — superseded by live_trading/live_bridge_ctrader.py (different version, causes confusion)
DELETE  v4_engine/live_bridge.py           — superseded by v5_engine/pipeline_live.py
DELETE  v4_engine/basket_backtester.py     — superseded by orchestrators/ensemble_pipeline.py
DELETE  v4_engine/compile_engine.py        — superseded by v5_engine/compile.py
DELETE  v4_engine/train_pipeline.py        — superseded by v5_engine/pipeline_train.py
DELETE  v4_engine/expert_layer.py          — superseded by v5_engine/experts.py
DELETE  v4_engine/feature_expansion.py     — superseded by v5_engine/features.py
DELETE  v4_engine/gmm_hmm.py               — superseded by v5_engine/router.py
DELETE  v4_engine/meta_data_prep.py        — superseded by v5_engine/meta_data_prep.py
DELETE  v4_engine/meta_learner.py          — superseded by v5_engine/meta.py
DELETE  v4_engine/triple_barrier.py        — superseded by v5_engine/triple_barrier.py
DELETE  v4_engine/vae_model.py             — superseded by v5_engine/vae.py
DELETE  v3_engine/engine/execution.py      — superseded by v5_engine/execution.py
DELETE  v3_engine/engine/features.py       — superseded by v5_engine/features.py
DELETE  v3_engine/engine/hmm.py            — superseded by v5_engine/hmm.py
DELETE  v3_engine/engine/kalman.py         — superseded by v5_engine/kalman.py
DELETE  v3_engine/engine/pipeline.py       — superseded by v5_engine/pipeline_backtest.py
DELETE  v3_engine/engine/preprocess.py     — superseded by v5_engine/preprocess.py
DELETE  v3_engine/engine/reports.py        — superseded by v5_engine/pipeline_backtest.py
DELETE  v3_engine/engine/surveillance.py   — superseded by v5_engine/surveillance.py
DELETE  v3_engine/engine/validation.py     — superseded by v5_engine/validation.py
```

**That is 27 files to delete. The entire v3_engine/ and v4_engine/ directories should be removed.**

## 1.2 Files to DELETE — orchestrators that import deleted modules

```
DELETE  orchestrators/master_pipeline.py   — imports forex_tearsheet (does not exist); no tests; superseded
DELETE  orchestrators/dynamic_tester.py    — imports from v3_engine.engine.reports directly; orphaned
```

## 1.3 Files to KEEP but fix

```
KEEP+FIX  api/main.py
KEEP+FIX  api/routes.py
KEEP+FIX  config.py
KEEP+FIX  model_spec.py
KEEP+FIX  data_pipeline/data_manager.py
KEEP+FIX  diagnostics/audit_real_data.py
KEEP+FIX  live_trading/get_token.py
KEEP+FIX  live_trading/live_bridge_ctrader.py
KEEP+FIX  orchestrators/ensemble_pipeline.py
KEEP+FIX  orchestrators/grid_search.py
KEEP+FIX  orchestrators/stage2_global_test.py
KEEP+FIX  risk_engine/__init__.py
KEEP+FIX  risk_engine/risk.py
KEEP+FIX  v3_engine/engine/tests/run_validation_suite.py  → MOVE to tests/
KEEP+FIX  v4_engine/tests/test_compiler.py               → MOVE to tests/
KEEP+FIX  v4_engine/tests/test_experts.py                → MOVE to tests/
KEEP+FIX  v4_engine/tests/test_meta.py                   → MOVE to tests/
KEEP+FIX  v4_engine/tests/test_router.py                 → MOVE to tests/
KEEP+FIX  v4_engine/tests/test_transformer.py            → MOVE to tests/
KEEP+FIX  v4_engine/tests/test_vae.py                    → MOVE to tests/
KEEP+FIX  v5_engine/ (all files)
KEEP+FIX  visualizer/app.py
```

## 1.4 Files to CREATE (currently missing or stub)

```
CREATE  visualizer/dashboard.html          — currently does not exist (app.py returns FileResponse to a missing file)
CREATE  tests/__init__.py
CREATE  tests/conftest.py
CREATE  tests/test_features.py
CREATE  tests/test_preprocess.py           — causality test (most important test missing)
CREATE  tests/test_risk.py
CREATE  tests/test_integration.py
CREATE  data_pipeline/volume_processor.py  — tick imbalance (TIB) extraction
```

---

---

# SECTION 2 — COMPLETE BUG LIST (NEW CODEBASE)

---

## BUG-N001 | CRITICAL | v5_engine/hmm.py uses backward smoother posteriors for prediction — data leakage

**File:** `v5_engine/hmm.py` — `TVTPHMM.predict_proba()`

```python
def predict_proba(self, X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    ...
    gamma, xi, ll, B, A_seq, alpha_norm = _forward_backward_jax(...)
    return np.array(gamma)   # ← SMOOTHER posteriors — uses FUTURE data
```

The `predict_proba` method returns `gamma`, which is the result of the full Baum-Welch forward-backward algorithm. At every bar t, `gamma[t]` incorporates backward messages from bars t+1, t+2, ..., T. For any walk-forward OOS evaluation, regime labels at OOS bar t are contaminated by future bars.

The code DOES have `_forward_only_jax` and `predict_proba_causal()` implemented. But:

```python
def predict_proba(self, ...):  # used by validation suite and pipeline
    ...
    return np.array(gamma)    # SMOOTHER — wrong for OOS

def predict_proba_causal(self, ...):  # exists but pipeline_backtest.py does NOT call it
    ...
    return np.array(alpha_norm)  # CAUSAL — correct
```

**In `v5_engine/pipeline_backtest.py`:**
```python
hmm_result = hmm.predict_proba(X_full, covariates)  # ← WRONG: uses smoother
```

The causal method exists but is never called by the backtest pipeline.

**Fix:** In `pipeline_backtest.py`, replace every `hmm.predict_proba(...)` with `hmm.predict_proba_causal(...)` for all OOS bars.

---

## BUG-N002 | CRITICAL | v5_engine/features.py — TIB column silently zeroed when not in dataframe

**File:** `v5_engine/features.py` — `compute_unified_features()`

```python
if "tib" in df_pl.columns:
    f_tib = df_pl["tib"]
else:
    f_tib = pl.Series(np.zeros(len(df_pl)))  # ← silently zero, no warning
```

The tick imbalance feature (TIB) — the only real volume signal in the system — silently falls back to all-zeros when the input dataframe does not have a `tib` column. Since `data_pipeline/data_manager.py` does NOT compute `tib` (it only has `volume`), every single backtest and live inference run uses TIB = 0.0 for all bars. The volume gate and all volume features are permanently broken without any indication.

**Fix:** Add a warning and raise if TIB is zero for more than 50% of bars:
```python
if "tib" in df_pl.columns:
    f_tib = df_pl["tib"]
    zero_frac = (f_tib.abs() < 1e-10).mean()
    if zero_frac > 0.5:
        import warnings
        warnings.warn(f"TIB column is zero for {zero_frac:.0%} of bars. Volume signal is inactive. Recompute with tick data.")
else:
    raise ValueError(
        "Input DataFrame missing 'tib' column (tick imbalance). "
        "Run data_pipeline/volume_processor.py to compute from tick data, "
        "or pass tib=np.zeros(len(df)) explicitly if tick data is unavailable."
    )
```

**Root cause:** `data_manager.py` resamples from tick data but discards the bid/ask split:
```python
# CURRENT (wrong):
df['volume'] = df['ask_volume'] + df['bid_volume']   # discards direction
ohlcv = df.resample(timeframe).agg({'bid': [...], 'volume': 'sum'})
```
Must be fixed as described in Section 4.

---

## BUG-N003 | CRITICAL | v5_engine/pipeline_backtest.py — run_tearsheet_dynamic() mutates global config

**File:** `v5_engine/pipeline_backtest.py` (inheriting the v3 bug)

```python
def run_tearsheet_dynamic(csv_path: str, params: dict) -> dict:
    ...
    config.ATR_SL_MULT = float(params["STOP_LOSS_ATR"])    # ← mutates global module
    config.ATR_TP_MULT = float(params["TAKE_PROFIT_ATR"])
    config.WF_OOS_BARS = int(params["HMM_WF_OOS_BARS"])
    exec_module.config.HMM_REGIME_CONF_THRESHOLD = float(params["VETO_THRESHOLD"])
    config.HMM_REGIME_CONF_THRESHOLD = float(params["VETO_THRESHOLD"])
    import os
    os.environ["TIME_EXIT_BARS"] = str(params["TIME_EXIT_BARS"])   # ← sets env var as side effect
```

When `orchestrators/ensemble_pipeline.py` runs multiple assets in a `ProcessPoolExecutor`, each worker process calls `run_tearsheet_dynamic`. If the orchestrator also runs single-threaded for any asset, it mutates the global config, affecting all subsequent calls in that process.

**Fix:** Make all parameters explicit function arguments, never mutate config:
```python
def run_tearsheet_dynamic(csv_path: str, params: dict) -> dict:
    local_cfg = SimpleNamespace(
        atr_sl_mult=float(params.get("STOP_LOSS_ATR", config.ATR_SL_MULT_TREND)),
        atr_tp_mult=float(params.get("TAKE_PROFIT_ATR", config.ATR_TP_MULT_TREND)),
        wf_oos_bars=int(params.get("HMM_WF_OOS_BARS", config.WF_OOS_BARS)),
        regime_conf_threshold=float(params.get("VETO_THRESHOLD", config.HMM_REGIME_CONF_THRESHOLD)),
        time_exit_bars=int(params.get("TIME_EXIT_BARS", config.TIME_EXIT_BARS)),
    )
    return main(ticker=csv_path, local_cfg=local_cfg, ...)  # pass as arg, not global
```

---

## BUG-N004 | CRITICAL | v5_engine/pipeline_train.py — VAE input_dim hardcoded to 26 but features.py produces variable count

**File:** `v5_engine/pipeline_train.py`

```python
model = VAE(input_dim=model_spec.VAE_INPUT_DIM, latent_dim=model_spec.VAE_LATENT_DIM)
```

`model_spec.VAE_INPUT_DIM = 26`. But `compute_unified_features()` returns a DataFrame with columns:
- 6 V3 features (vt, mvt, sigma_t, rho_t, ht, tib)
- 20 z-scored V4 features (z_trend_sma_10 ... z_volm_ofi_20)
- Plus the TIB derivatives: z_tib_raw, z_tib_ema, z_volm_surge_5_20, z_volm_ofi_10, z_volm_ofi_20

Actual column count from the function: 6 + 20 = 26. This works IF all V4 indicators produce exactly 20 columns. But the V4 column list is built dynamically:
```python
v4_df = pl.DataFrame(v4_raw)  # built by appending, count depends on loop iterations
```
If any `v4_raw.append()` is skipped (e.g. for a CSV without `high`/`low`), the count changes and VAE silently gets wrong-shaped input.

**Fix:** Add an explicit assertion in `compute_unified_features`:
```python
assert combined_pl.shape[1] == model_spec.VAE_INPUT_DIM + (1 if "date" in combined_pl.columns else 0), \
    f"Feature count mismatch: got {combined_pl.shape[1]-1}, expected {model_spec.VAE_INPUT_DIM}"
```

---

## BUG-N005 | CRITICAL | live_trading/live_bridge_ctrader.py imports from v3_engine (deleted path)

**File:** `live_trading/live_bridge_ctrader.py`

```python
from v3_engine.engine.pipeline import Pipeline, PipelineConfig
```

`v3_engine/engine/pipeline.py` is scheduled for deletion. The live bridge imports from the old deleted path. In production this throws `ModuleNotFoundError` immediately on startup.

**Fix:**
```python
from v5_engine.pipeline_live import LiveInferenceEngine
from v5_engine.pipeline_backtest import PipelineConfig
```

---

## BUG-N006 | HIGH | config.py — WF_TIMEFRAME is "1D" but TIMEFRAME is "1h" — contradiction

**File:** `config.py`

```python
TIMEFRAME    = "1h"        # hourly trading
WF_TIMEFRAME = "1D"        # walk-forward window described as daily
ANN_FACTOR   = int(252 * BARS_PER_DAY)  # = 6048 (hourly)
```

`WF_OOS_BARS = 126` with `WF_TIMEFRAME = "1D"` means 126 calendar days of OOS. But `ANN_FACTOR = 6048` means bars are hourly. If the WF loop counts `WF_OOS_BARS` as hourly bars, 126 bars = 5.25 days of data, which is far too little for a meaningful OOS period. If it counts them as daily bars, the Sharpe annualisation is wrong because bars are actually hourly.

**Fix:** Unify everything around hourly bars:
```python
TIMEFRAME       = "1h"
BARS_PER_DAY    = 24          # FX trades 24h; use 6.5 for equities
ANN_FACTOR      = int(252 * BARS_PER_DAY)   # = 6048
WF_TRAIN_BARS   = 504         # bars, not days (504 hourly = 21 days = 3 weeks of 24h FX)
WF_OOS_BARS     = 126         # bars (126 hourly = ~5 days)
# OR if you mean 126 DAYS of OOS:
WF_OOS_BARS     = 126 * 24    # = 3024 hourly bars = 126 days
```
Pick one and document it clearly. Right now the code is ambiguous and produces wrong OOS period lengths.

---

## BUG-N007 | HIGH | v5_engine/features.py — qt (Volume Delta Ratio) feature is missing, replaced by TIB but TIB is always zero

In the original V3, feature index 2 was `qt = Volume Delta Ratio`. In V5's `compute_unified_features`, the V3 block only has 5 named features (vt, mvt, sigma_t, rho_t, ht) plus tib. The `qt` feature was replaced by `tib` but:
1. `tib` is always zero (BUG-N002)
2. The original `qt` computation (which used GARCH variance) was never implemented in V5

This means the 6-dimensional regime geometry tensor fed to the HMM only has 5 real features + zeros for dimension 5 (tib). The HMM is being fit on a 6D tensor where one dimension is always 0, wasting a degree of freedom and reducing regime separation.

**Fix:** Either restore `qt` from V3 features.py until TIB is available, or explicitly document that the 6th dim is placeholder.

---

## BUG-N008 | HIGH | v5_engine/pipeline_backtest.py — Sharpe annualisation still uses 252 in some paths

**File:** `v5_engine/pipeline_backtest.py` (from the tearsheet inherited from v3)

```python
sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # line 5745
```

`config.ANN_FACTOR = 6048` is defined but this calculation still uses hardcoded 252. The basket backtester also has this:
```python
sharpe = ... * np.sqrt(252) if daily_returns.std() > 0 else 0.0
```

**Fix:** Replace all `np.sqrt(252)` with `np.sqrt(config.ANN_FACTOR)`. Search the whole codebase for `252` and replace.

---

## BUG-N009 | HIGH | orchestrators/stage2_global_test.py — leaderboard indentation bug still present

This was reported in the original audit and is STILL not fixed in CODEBASE_4.md.

```python
with concurrent.futures.ProcessPoolExecutor(...) as executor:
    for i, entry in enumerate(top_10, 1):
        ...
        futures = [executor.submit(...) for csv in csv_files]
        for future in concurrent.futures.as_completed(futures):
            ...
    if success_count > 0 and total_trades > 0:   # ← STILL outside the for loop
        ...
        final_leaderboard.append(...)
```

Only the last iteration's result is appended. The first 9 parameter sets are silently discarded from the leaderboard.

---

## BUG-N010 | HIGH | v5_engine/pipeline_backtest.py — Preprocessor fit/transform causality

From reading the code, `pipeline_backtest.py` calls:
```python
features = compute_unified_features(bars_df)
X = features[v3_cols].values
pp = Preprocessor()
X_white = pp.fit_transform(X)   # fits AND transforms in one call on FULL dataset
```

This fits the Preprocessor (Welford stats + PCA) on the entire IS+OOS window, then transforms all bars. The PCA eigenvectors are contaminated by OOS data. The `fit()` / `transform()` split exists in `v5_engine/preprocess.py` but is not used correctly in the pipeline.

**Verification required:** Check that pipeline_backtest.py calls `pp.fit(X[:train_end])` then `pp.transform(X)` separately per fold. If it calls `fit_transform(X)` on the full array, the leakage from the original audit report is still present.

---

## BUG-N011 | HIGH | visualizer/app.py — FileResponse returns path to non-existent dashboard.html

```python
@router.get("/visualizer")
async def serve_dashboard():
    return FileResponse(os.path.join(os.path.dirname(__file__), "dashboard.html"))
```

`visualizer/dashboard.html` does not exist in the repository. Visiting `/visualizer` will return HTTP 404 (FileNotFoundError). The visualizer is completely non-functional.

---

## BUG-N012 | HIGH | v5_engine/pipeline_train.py — train_vae uses hardcoded latent_dim=4 instead of model_spec

```python
vae = VAE(input_dim=model_spec.VAE_INPUT_DIM, latent_dim=4)   # hardcoded 4
router = GumbelSoftmaxRouter(latent_dim=4, n_regimes=3)        # hardcoded 4, 3
```

If `model_spec.VAE_LATENT_DIM` is changed, the train script builds the wrong architecture and saves weights that cannot be loaded.

**Fix:**
```python
vae = VAE(input_dim=model_spec.VAE_INPUT_DIM, latent_dim=model_spec.VAE_LATENT_DIM)
router = GumbelSoftmaxRouter(latent_dim=model_spec.ROUTER_LATENT_DIM, n_regimes=model_spec.ROUTER_N_REGIMES)
```

---

## BUG-N013 | MEDIUM | v5_engine/router.py — predict_proba() uses Baum-Welch smoother, not causal forward

**File:** `v5_engine/router.py` — `LatentRouter.predict_proba()`

```python
def predict_proba(self, latent_X: np.ndarray) -> tuple:
    probas = self.model.predict_proba(latent_X)   # ← hmmlearn smoother
```

`hmmlearn`'s `predict_proba()` uses the full forward-backward algorithm. For OOS evaluation this has the same leakage as BUG-N001. The causal version `predict_causal_proba()` exists and is correct. Everywhere `predict_proba()` is called on OOS data should use `predict_causal_proba()` instead.

---

## BUG-N014 | MEDIUM | risk_engine/risk.py — close_position() computes wrong PnL sign for shorts

```python
def close_position(self, symbol: str, exit_price: float, exit_time):
    pos = self.open_positions.pop(symbol, None)
    pnl = (exit_price - pos["entry_price"]) * pos["direction"] * pos["units"]
```

`pos["direction"]` is expected to be +1 (long) or -1 (short). But `evaluate()` returns a dict where `"direction"` is +1 or -1 as an integer. However, `register_position` stores whatever dict is passed in — and the live bridge passes `decision["direction"]` which is the string "BUY" or "SELL" from `process_live_tick()`. String "BUY" * float = TypeError.

**Fix:** Standardise direction to int at the risk engine boundary:
```python
def evaluate(self, ..., direction: int, ...):
    assert direction in (1, -1), f"direction must be 1 or -1, got {direction}"
```
And in live bridge, convert before calling risk engine:
```python
direction_int = 1 if decision["direction"] == "BUY" else -1
risk_result = risk_engine.evaluate(symbol, direction_int, ...)
```

---

## BUG-N015 | MEDIUM | v5_engine/hmm.py — LAMBDA_A, LAMBDA_MU, LAMBDA_BETA set at module level from config

```python
LAMBDA_A    = config.HMM_LAMBDA_A        # module-level
LAMBDA_MU   = config.HMM_LAMBDA_MU
LAMBDA_BETA = config.HMM_LAMBDA_BETA
RIDGE_SIGMA = config.HMM_RIDGE_SIGMA
```

These are module-level constants captured at import time. If `config.HMM_LAMBDA_A` is changed via environment variable (through `_apply_env_overrides()`), the change happens AFTER `hmm.py` is imported, so the JAX-compiled functions still use the old values. JAX `@jit` compiles with the captured value, not the current config value.

**Fix:** Pass regularisation parameters as function arguments to `_forward_backward_jax`:
```python
@jax.jit
def _update_gmm_emissions_jax(X, gamma, mu, sigma, pi_gmm, clean_mask, lambda_mu, ridge_sigma):
    ...
    mu_new = ... / (w_sum_safe + lambda_mu)   # use passed argument
```

---

## BUG-N016 | MEDIUM | data_pipeline/data_manager.py — resamples bid prices only for OHLCV (original bug unfixed)

```python
ohlcv = df.resample(timeframe).agg({
    'bid': ['first', 'max', 'min', 'last'],
    'volume': 'sum'
})
ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
```

Bid-price OHLCV without ask creates a systematic downward bias in:
- ATR (high and low are bid-side only, not true range)
- SL/TP placement (placed on mid-price but triggered on bid/ask)
- TIB computation (requires signed volume which is discarded)

The TIB column is never written to the CSV, which is why BUG-N002 exists.

---

## BUG-N017 | MEDIUM | v5_engine/validation.py — IS/OOS overfit ratio test uses synthetic hardcoded values

From the v3 validation suite test that was adopted:
```python
def test_is_oos_overfit_ratio_bounded(self):
    is_sharpe = 1.5       # hardcoded — not from actual pipeline output
    oos_sharpe = 1.0      # hardcoded
    overfit_ratio = is_sharpe / max(oos_sharpe, 0.01)
    assert overfit_ratio < 2.0, "Overfit ratio > 2.0"
```

This test always passes because 1.5/1.0 = 1.5 < 2.0 is a mathematical identity. It tests nothing about the actual pipeline. It gives false confidence that overfitting is under control.

**Fix:** Wire this test to a real pipeline run:
```python
def test_is_oos_overfit_ratio_bounded():
    # Run pipeline on synthetic AR(1) data
    # Compute real IS and OOS Sharpe from pipeline output
    # Assert ratio
    result = run_minimal_pipeline_on_synthetic_data(n_bars=1000)
    assert result.is_sharpe / max(result.oos_sharpe, 0.01) < 2.0
```

---

## BUG-N018 | MEDIUM | api/routes.py — healthz returns "v3.0" string in a v5 engine

```python
return {"status": "ok", "engine": "Latent Diffusion-HMM v3.0", ...}
```

The engine is now v5. Minor but creates confusion in production monitoring.

---

## BUG-N019 | LOW | v5_engine/features.py — imports both pandas and polars, converts between them multiple times

```python
df_pl = pl.from_pandas(df_copy.reset_index())   # pandas → polars
...
v4_df = pl.DataFrame(v4_raw)
...
combined_pl = pl.concat([v3_features, v4_z_df], how="horizontal")
df_combined = combined_pl.to_pandas()            # polars → pandas
```

There are 4 pandas↔polars conversions in one function call. Each conversion copies all data. For a 5000-bar dataset this is 4× unnecessary memory allocation. The function should be fully Polars-native end-to-end.

---

## BUG-N020 | LOW | orchestrators/ensemble_pipeline.py — Sharpe still uses 252 in fallback path

```python
sr = daily_mean / daily_vol * np.sqrt(config.ANN_FACTOR) if 'config' in sys.modules else daily_mean / daily_vol * np.sqrt(252)
```

The `if 'config' in sys.modules` condition is always True since config is imported at the top. The `np.sqrt(252)` fallback is dead code but its existence is confusing. Remove the conditional, use `config.ANN_FACTOR` directly.

---

## BUG-N021 | LOW | v5_engine/compile.py — exports VAE with hardcoded input_dim=20

```python
vae = VAE(input_dim=20, latent_dim=4)   # hardcoded, not from model_spec
```

`model_spec.VAE_INPUT_DIM = 26`. Compiling with 20 creates an ONNX model that does not match the trained weights and will fail at inference.

**Fix:**
```python
import model_spec
vae = VAE(input_dim=model_spec.VAE_INPUT_DIM, latent_dim=model_spec.VAE_LATENT_DIM)
```

---

## BUG-N022 | LOW | v4_engine/tests/test_transformer.py references eurgbp_daily.csv but data is hourly

```python
fx_path = "data/eurgbp_1h.csv"   # (already updated in this version)
```

This appears to be fixed in the new codebase. Verify the file is named consistently.

---

## BUG-N023 | LOW | v5_engine/triple_barrier.py imports config but uses tp_mult and sl_mult as function arguments, ignoring config values

The function signature:
```python
def apply_triple_barrier(df: pl.DataFrame, tp_mult=2.0, sl_mult=1.0, time_limit=24) -> pl.DataFrame:
```

Default values are hardcoded. In `pipeline_train.py` these are called with `config.BARRIER_TP_MULT` correctly:
```python
df = apply_triple_barrier(df, tp_mult=config.BARRIER_TP_MULT, sl_mult=config.BARRIER_SL_MULT, time_limit=config.BARRIER_TIME_LIMIT)
```

But the test in `test_experts.py` uses hardcoded values:
```python
out = apply_triple_barrier(df, tp_mult=8.0, sl_mult=4.0, time_limit=5)
```

The test uses different multipliers than config (config has BARRIER_TP_MULT=2.0, test uses 8.0). Tests should use config values or the test should explicitly note why it deviates.

---

---

# SECTION 3 — DATA PIPELINE: WHAT MUST BE REBUILT

---

## 3.1 The institutional-grade data requirement

The current system uses Yahoo Finance-style OHLCV resampled from Dukascopy tick data, using bid price only. This is not institutional grade. For prop firm challenges (FTMO, MyForexFunds, Funded Engineer, etc.) operating on FX pairs in the 1H timeframe, the minimum data standard is:

**Required data fields per completed bar:**
```
timestamp_open   — bar open time (milliseconds UTC)
timestamp_close  — bar close time
open_mid         — (bid_open + ask_open) / 2
high_mid         — max mid price during bar
low_mid          — min mid price during bar
close_mid        — (bid_close + ask_close) / 2
volume_total     — total tick count during bar
volume_buy       — ticks where ask was hit (buyer-initiated)
volume_sell      — ticks where bid was hit (seller-initiated)
tib              — (volume_buy - volume_sell) / volume_total  [tick imbalance]
spread_avg       — average bid-ask spread during bar (in pips)
spread_max       — maximum spread during bar
```

## 3.2 data_pipeline/data_manager.py — complete fix

The Dukascopy tick data format already has bid, ask, bid_volume, ask_volume per tick. The resampling must preserve all of this.

```python
# CORRECT resampling from Dukascopy ticks:

def resample_ticks_to_ohlcv(df_ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    df_ticks must have columns: timestamp, bid, ask, bid_volume, ask_volume
    All volume in thousands of units (Dukascopy format).
    """
    df = df_ticks.copy()
    df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['mid'] = (df['bid'] + df['ask']) / 2.0
    df['spread'] = df['ask'] - df['bid']
    df['vol_total'] = df['bid_volume'] + df['ask_volume']
    # Signed volume: positive = net buying, negative = net selling
    # Dukascopy convention: ask_volume = buyer-initiated (hit the ask)
    #                       bid_volume = seller-initiated (hit the bid)
    df['vol_signed'] = df['ask_volume'] - df['bid_volume']

    ohlcv = df.resample(timeframe).agg(
        open=('mid', 'first'),
        high=('mid', 'max'),
        low=('mid', 'min'),
        close=('mid', 'last'),
        volume=('vol_total', 'sum'),
        vol_signed=('vol_signed', 'sum'),
        spread_avg=('spread', 'mean'),
        spread_max=('spread', 'max'),
    ).dropna(subset=['open'])

    # Tick Imbalance Balance (TIB): [-1, +1]
    # +1 = all buyer-initiated, -1 = all seller-initiated
    ohlcv['tib'] = ohlcv['vol_signed'] / ohlcv['volume'].clip(lower=1e-10)
    ohlcv['tib'] = ohlcv['tib'].clip(-1.0, 1.0).fillna(0.0)

    # Convert spread from price units to pips
    # Will be adjusted per symbol in downstream processing
    ohlcv.index.name = 'date'
    return ohlcv[['open', 'high', 'low', 'close', 'volume', 'tib', 'spread_avg', 'spread_max']]
```

## 3.3 data_pipeline/volume_processor.py — new file (tick volume signals)

```python
# data_pipeline/volume_processor.py

import numpy as np
import pandas as pd

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes institutional-grade volume features from OHLCV+TIB data.
    
    Input: DataFrame with columns [open, high, low, close, volume, tib]
    Output: Input + additional volume feature columns
    
    Features added:
      tib_z         — Z-scored TIB over Z_SCORE_WINDOW bars
      tib_ema_z     — Z-scored EMA(20) of TIB
      vol_surge     — short_vol_ma / long_vol_ma (volume burst detector)
      vol_surge_z   — Z-scored vol_surge
      ofi_10        — Order Flow Imbalance EMA-10 (bar-direction × volume)
      ofi_20        — Order Flow Imbalance EMA-20
      ofi_10_z      — Z-scored OFI-10
      ofi_20_z      — Z-scored OFI-20
      cvd           — Cumulative Volume Delta (running sum of vol_signed)
      cvd_z         — Z-scored CVD rate of change
    """
    import config
    window = config.Z_SCORE_WINDOW

    result = df.copy()

    # TIB Z-score
    tib = df['tib'].values
    tib_ma = pd.Series(tib).rolling(window, min_periods=5).mean().values
    tib_std = pd.Series(tib).rolling(window, min_periods=5).std().clip(lower=1e-10).values
    result['tib_z'] = np.clip((tib - tib_ma) / tib_std, -10, 10)

    # TIB EMA Z-score
    tib_ema = pd.Series(tib).ewm(span=20, min_periods=1, adjust=False).mean().values
    tib_ema_ma = pd.Series(tib_ema).rolling(window, min_periods=5).mean().values
    tib_ema_std = pd.Series(tib_ema).rolling(window, min_periods=5).std().clip(lower=1e-10).values
    result['tib_ema_z'] = np.clip((tib_ema - tib_ema_ma) / tib_ema_std, -10, 10)

    # Volume surge (short/long ratio)
    vol = df['volume'].values + 1e-10
    vol_short_ma = pd.Series(vol).rolling(5, min_periods=1).mean().values
    vol_long_ma = pd.Series(vol).rolling(20, min_periods=1).mean().values.clip(min=1e-10)
    vol_surge = vol_short_ma / vol_long_ma
    vs_ma = pd.Series(vol_surge).rolling(window, min_periods=5).mean().values
    vs_std = pd.Series(vol_surge).rolling(window, min_periods=5).std().clip(lower=1e-10).values
    result['vol_surge_z'] = np.clip((vol_surge - vs_ma) / vs_std, -10, 10)

    # Bar-direction OFI (signed volume × bar direction)
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    bar_range = (high - low).clip(min=1e-10)
    direction = (close - open_) / bar_range
    ofi_raw = vol * direction

    for span in [10, 20]:
        ofi = pd.Series(ofi_raw).ewm(span=span, min_periods=1, adjust=False).mean().values
        vol_ema = pd.Series(vol).ewm(span=span, min_periods=1, adjust=False).mean().values.clip(min=1e-10)
        ofi_norm = ofi / vol_ema
        ofi_ma = pd.Series(ofi_norm).rolling(window, min_periods=5).mean().values
        ofi_std = pd.Series(ofi_norm).rolling(window, min_periods=5).std().clip(lower=1e-10).values
        result[f'ofi_{span}_z'] = np.clip((ofi_norm - ofi_ma) / ofi_std, -10, 10)

    # CVD rate of change Z-score
    vol_signed = df['tib'].values * df['volume'].values
    cvd = np.cumsum(vol_signed)
    cvd_roc = np.diff(cvd, prepend=cvd[0])
    cvd_ma = pd.Series(cvd_roc).rolling(window, min_periods=5).mean().values
    cvd_std = pd.Series(cvd_roc).rolling(window, min_periods=5).std().clip(lower=1e-10).values
    result['cvd_z'] = np.clip((cvd_roc - cvd_ma) / cvd_std, -10, 10)

    return result.fillna(0.0)
```

## 3.4 Why no yfinance — what to use instead

yfinance is inappropriate for this system because:
- Returns daily OHLCV with no bid/ask split → zero TIB, zero OFI
- Adjusted close prices distort fractional differentiation
- No tick volume data for FX pairs
- Volume data for FX is unreliable (aggregated from unknown sources)
- No spread data

**Correct data sources by instrument type:**

```
FX PAIRS (EURUSD, GBPUSD, etc.)
  Source: Dukascopy Historical Data Feed (free, tick-level)
  Library: tick-vault (pip install tick-vault) OR direct HTTP from dukascopy.com
  Format: bid, ask, bid_volume, ask_volume per tick, 1-minute minimum resolution
  Resampled to: 1h bars with TIB computed from tick imbalance

METALS (XAUUSD, XAGUSD)
  Source: Dukascopy (XAUUSD available)
  Fallback: Interactive Brokers historical data API (TWS API)

FUTURES (ES, NQ, CL, GC)
  Source: Interactive Brokers TWS API — historical bars with bid/ask volume
  Library: ib_insync (pip install ib_insync)
  Format: 1h bars, realtime subscription for live

OPTIONS
  Source: IBKR TWS API — implied vol, greeks, underlying price
  Library: ib_insync
  Note: Options require separate feature engineering (IV rank, skew, PCR)
  This is a separate feature set; do not mix with FX features

EQUITIES (SPY, QQQ, sector ETFs)
  Source: Polygon.io (free tier: daily; paid: minute + tick)
  Library: polygon-api-client (pip install polygon-api-client)
  Format: Trade-level: price, size, conditions; aggregated to bars

CRYPTO
  Source: Binance WebSocket API (tick-level order book)
  Library: python-binance or ccxt
  Format: Trade stream, aggregated to bars with signed volume
```

---

---

# SECTION 4 — V6 ARCHITECTURE: COMPLETE CLEAN DESIGN

---

## 4.1 Design principles

The V6 engine is NOT another incremental patch. It is the architecture that V5 was trying to be but failed to fully implement. The rules are:

```
RULE 1: ONE DIRECTORY — engine/
  All model code in engine/. No v3_, v4_, v5_ prefix directories.
  Old prefixed dirs are deleted.

RULE 2: ONE CONFIG — config.py
  No hardcoded numbers anywhere except in config.py and model_spec.py.
  Every module reads from config at runtime, not at import time.

RULE 3: CAUSAL ONLY — no smoother posteriors for OOS
  All inference uses forward-only algorithms.
  Training uses forward-backward (Baum-Welch) on IS data only.

RULE 4: ONNX FOR ALL INFERENCE
  Every model that runs more than once per hour is compiled to ONNX.
  Python-only paths (JAX HMM) run offline only.
  Live inference: all through ONNX runtime.

RULE 5: SEPARATE STRATEGIES
  MOMENTUM and MEAN_REV are separate strategy classes.
  They share the same feature pipeline and regime model.
  They have different execution parameters.
  They can run simultaneously on different symbols.

RULE 6: REAL VOLUME OR NONE
  If TIB data is not available, the volume gate passes (no false signal suppression).
  If TIB data IS available, it gates trades actively.
  Never silently zero out a feature.

RULE 7: PROP FIRM MODE
  Prop firm simulator is data-driven (uses actual backtest results).
  Targets: 3+ trades per week, max 5% daily DD, max 10% total DD.
  Challenge parameters configurable per firm (FTMO, MFF, TopstepFX).
```

## 4.2 Final directory structure (V6)

```
project_root/
│
├── config.py                  # ALL parameters — single source of truth
├── model_spec.py              # Architecture constants only
├── .env.example               # Template for environment variables
│
├── data_pipeline/
│   ├── data_manager.py        # Dukascopy tick download + mid-price OHLCV + TIB
│   └── volume_processor.py   # Volume features (TIB, OFI, CVD, vol surge)
│
├── engine/                    # THE ONLY ENGINE — no v3/v4/v5 prefixes
│   ├── __init__.py
│   ├── features.py            # Unified 26D feature tensor (V3+V4 merged, TIB active)
│   ├── preprocess.py          # fit() + transform() split, no global state
│   ├── kalman.py              # Kalman filter + CUSUM, O(T) incremental
│   ├── hmm.py                 # TVTP-HMM: train with forward-backward, infer causal-only
│   ├── vae.py                 # VAE encoder-decoder
│   ├── router.py              # Gumbel-Softmax router + GMM-HMM LatentRouter
│   ├── experts.py             # Expert layer (RF, XGBoost, IsolationForest)
│   ├── meta.py                # ResidualMetaLearner (correction network)
│   ├── execution.py           # TripleGate (regime + momentum + volume gates)
│   ├── surveillance.py        # Wasserstein W1 distribution monitor
│   ├── validation.py          # DSR, MC permutation, CPCV, WF-CV, tx-cost
│   └── triple_barrier.py      # Triple-barrier labelling for supervised learning
│
├── strategies/
│   ├── __init__.py
│   ├── base.py                # BaseStrategy abstract class
│   ├── momentum.py            # MOMENTUM strategy (TREND regime, long/short directional)
│   ├── mean_rev.py            # MEAN_REV strategy (MEAN_REV regime, fade extremes)
│   └── compression.py        # COMPRESSION strategy (breakout plays, pending orders)
│
├── risk_engine/
│   ├── __init__.py
│   └── risk.py                # RiskEngine: Kelly + ATR sizing + kill-switches (fixed)
│
├── prop_sim/
│   ├── __init__.py
│   └── simulator.py           # Prop firm challenge simulator (data-driven, not hardcoded)
│
├── pipeline/
│   ├── __init__.py
│   ├── train.py               # Offline training: VAE + experts + meta-learner
│   ├── backtest.py            # Walk-forward backtest (stateless, fold-correct IS/OOS)
│   ├── live.py                # Live inference (all ONNX, <5ms per bar)
│   └── compile.py             # ONNX export for all trainable models
│
├── bin/                       # Compiled ONNX models
│   ├── vae.onnx
│   ├── router.onnx
│   ├── meta_judge.onnx
│   ├── expert_TREND.onnx
│   ├── expert_MEAN_REV.onnx
│   └── expert_COMPRESSION.onnx
│
├── models/                    # Trained weights (PyTorch + sklearn)
│   ├── vae_weights.pth
│   ├── gumbel_router_weights.pth
│   ├── meta_judge.pth
│   ├── expert_TREND.pkl
│   ├── expert_MEAN_REV.pkl
│   └── expert_COMPRESSION.pkl
│
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI app (fixed CORS, version string)
│   └── routes.py              # All endpoints (fixed: loads bars, passes to pipeline)
│
├── visualizer/
│   ├── app.py                 # FastAPI router for visualizer endpoints
│   └── dashboard.html         # Single-page dashboard (Chart.js, no framework)
│
├── orchestrators/
│   ├── ensemble_pipeline.py   # Multi-asset backtest (fixed: column names, stateless)
│   ├── grid_search.py         # Hyperparameter sweep
│   └── stage2_global_test.py  # Cross-asset stress test (fixed: indentation bug)
│
├── live_trading/
│   ├── get_token.py           # OAuth (fixed: POST not GET)
│   └── live_bridge_ctrader.py # Unified bridge (imports from engine/, not v3_engine/)
│
├── diagnostics/
│   └── audit_pipeline.py      # Full pipeline audit on real data
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_features.py        # 26D tensor shape, NaN, stationarity
    ├── test_preprocess.py      # Causality test (KEY test — must exist)
    ├── test_kalman_hmm.py      # Kalman CUSUM + HMM Viterbi accuracy
    ├── test_execution.py       # Triple gate with volume gate active
    ├── test_surveillance.py    # Wasserstein circuit-breaker
    ├── test_validation.py      # Statistical validation (real PSR test, not hardcoded)
    ├── test_vae.py             # VAE reconstruction + KL divergence
    ├── test_router.py          # Causal router probabilities
    ├── test_experts.py         # Triple barrier + OOF
    ├── test_meta.py            # Meta-learner architecture
    ├── test_risk.py            # All 7 risk rules
    ├── test_compiler.py        # ONNX vs PyTorch (fixed: uses model_spec)
    └── test_integration.py     # Full pipeline smoke test on synthetic data
```

## 4.3 Prop firm simulator — data-driven

```python
# prop_sim/simulator.py
# Reads actual backtest returns, runs Monte Carlo for challenge pass probability

class PropFirmChallenge:
    """
    Configurable per firm. Key parameters:
      - profit_target_pct: e.g. 0.08 for 8%
      - max_daily_dd_pct: e.g. 0.05 for 5% (prop firm rule)
      - max_total_dd_pct: e.g. 0.10 for 10%
      - trading_days: e.g. 30 days minimum
      - min_trading_days: e.g. 10 days with at least one trade
      - position_size_limit: max lot size
    """
    FIRM_PROFILES = {
        "FTMO_100K": {
            "profit_target_pct": 0.10,
            "max_daily_dd_pct": 0.05,
            "max_total_dd_pct": 0.10,
            "trading_days": 30,
            "min_trading_days": 10,
        },
        "MFF_200K": {
            "profit_target_pct": 0.08,
            "max_daily_dd_pct": 0.05,
            "max_total_dd_pct": 0.10,
            "trading_days": 30,
            "min_trading_days": 5,
        },
        "TOPSTEP_FX_150K": {
            "profit_target_pct": 0.06,
            "max_daily_dd_pct": 0.04,
            "max_total_dd_pct": 0.06,
            "trading_days": None,  # no time limit
            "min_trading_days": None,
        },
    }

    def simulate(
        self,
        daily_returns: np.ndarray,  # actual backtest daily returns
        firm_profile: str = "FTMO_100K",
        n_simulations: int = 10_000,
        random_state: int = 42,
    ) -> dict:
        """
        Runs N Monte Carlo simulations by bootstrapping from actual daily returns.
        Each simulation represents one challenge attempt.
        """
        rng = np.random.default_rng(random_state)
        params = self.FIRM_PROFILES[firm_profile]
        passes = 0
        max_loss_reasons = []
        not_enough_trades_reasons = []

        for _ in range(n_simulations):
            # Bootstrap sample of daily returns (with replacement, maintaining
            # temporal autocorrelation by sampling blocks of 5 days)
            n_days = params["trading_days"] or 30
            indices = rng.integers(0, max(1, len(daily_returns) - 5), size=n_days)
            sim_returns = np.concatenate([daily_returns[i:i+5] for i in indices])[:n_days]

            equity = np.cumprod(1 + sim_returns)
            peak = np.maximum.accumulate(equity)
            total_dd = np.min((equity - peak) / peak)
            daily_dd = np.min(np.diff(equity, prepend=1.0) / np.maximum(equity[:-1], 1e-10))
            total_return = equity[-1] - 1.0
            trading_days_with_trades = np.sum(sim_returns != 0.0)

            # Check challenge conditions
            failed = False
            if total_dd < -params["max_total_dd_pct"]:
                max_loss_reasons.append("total_dd")
                failed = True
            if daily_dd < -params["max_daily_dd_pct"]:
                max_loss_reasons.append("daily_dd")
                failed = True
            if params["min_trading_days"] and trading_days_with_trades < params["min_trading_days"]:
                not_enough_trades_reasons.append("inactive")
                failed = True

            if not failed and total_return >= params["profit_target_pct"]:
                passes += 1

        return {
            "firm": firm_profile,
            "n_simulations": n_simulations,
            "pass_rate": passes / n_simulations,
            "expected_passes": passes,
            "top_failure_mode": max(set(max_loss_reasons), key=max_loss_reasons.count) if max_loss_reasons else "none",
        }
```

## 4.4 Strategies as separate classes

```python
# strategies/momentum.py
class MomentumStrategy(BaseStrategy):
    """
    Active in TREND regime. Goes long when p_trend > threshold AND
    TIB > VOLUME_THRESHOLD (buyer-initiated flow). Short when TIB < -VOLUME_THRESHOLD.
    Uses wide ATR-based stops (3 ATR SL, 8 ATR TP).
    Target: 3-5 trades per week on 1H chart.
    """
    REGIME = "TREND"
    ATR_SL_MULT = config.ATR_SL_MULT_TREND
    ATR_TP_MULT = config.ATR_TP_MULT_TREND

# strategies/mean_rev.py
class MeanRevStrategy(BaseStrategy):
    """
    Active in MEAN_REV regime. Fades momentum extremes (MVT > threshold = potential reversal).
    Uses tight stops (1.5 ATR SL) and moderate targets (3 ATR TP).
    Target: 2-3 trades per week.
    """
    REGIME = "MEAN_REV"
    ATR_SL_MULT = config.ATR_SL_MULT_MEAN_REV
    ATR_TP_MULT = config.ATR_TP_MULT_MEAN_REV

# strategies/compression.py
class CompressionStrategy(BaseStrategy):
    """
    Active in COMPRESSION regime. Detects coiling (low ATR, high Hurst compression).
    Places pending orders at breakout levels. Wider TP (5 ATR) to capture the full breakout.
    Target: 1-2 trades per week (lower frequency, higher R:R).
    """
    REGIME = "COMPRESSION"
    ATR_SL_MULT = config.ATR_SL_MULT_COMPRESSION
    ATR_TP_MULT = config.ATR_TP_MULT_COMPRESSION
```

---

---

# SECTION 5 — LIBRARIES AND DEPENDENCIES (COMPLETE CANONICAL LIST)

---

## 5.1 Core dependencies (pinned versions for reproducibility)

```
# Data
dukascopy-data==0.3.1         # or tick-vault — Dukascopy tick download
polars==0.20.31               # primary dataframe library (fast, no GIL)
pandas==2.2.2                 # secondary (for sklearn/hmmlearn interop)
numpy==1.26.4

# Statistical modelling
scipy==1.13.0
arch==6.4.0                   # GARCH modelling (sigma_t feature)
statsmodels==0.14.2           # ADF test, autocorrelation

# Machine learning
scikit-learn==1.5.0
xgboost==2.1.0
hmmlearn==0.3.2               # GMM-HMM LatentRouter

# Deep learning
torch==2.3.0
onnx==1.16.0
onnxruntime==1.18.0           # ONNX inference (all live inference goes here)
onnxmltools==1.12.0           # XGBoost → ONNX conversion
skl2onnx==1.17.0              # sklearn → ONNX conversion

# Bayesian / JAX
jax==0.4.28                   # TVTP-HMM training (JIT compiled)
jaxlib==0.4.28

# Optimal Transport (Wasserstein)
pot==0.9.3                    # Python Optimal Transport

# Compilation
numba==0.59.1                 # Kalman filter inner loops, triple barrier

# API
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1

# Live trading
ctrader-open-api==2.0.4       # cTrader connectivity
twisted==24.3.0               # async reactor for cTrader
python-dotenv==1.0.1

# Interactive Brokers (for futures/options)
ib_insync==0.9.86

# Visualizer
# No JS build step. Chart.js loaded from CDN in dashboard.html.

# Testing
pytest==8.2.2
pytest-asyncio==0.23.7

# Utilities
joblib==1.4.2
tqdm==4.66.4
```

## 5.2 What NOT to use

```
yfinance         — no tick volume, adjusted prices, no FX volume, unreliable
ta               — slow, pandas-based; implement indicators in polars directly
backtrader       — full framework with its own state machine; incompatible with V6 design
zipline          — outdated, no FX support, unmaintained
bt               — too simple, no regime-conditional logic
vectorbt         — good but adds unnecessary abstraction over our custom backtest
ccxt             — use only for crypto; not for FX or futures
ta-lib           — requires C compilation, hard to deploy; use polars rolling ops
```

---

---

# SECTION 6 — TEST AUDIT: WHAT IS CORRECT, WHAT MUST CHANGE

---

## 6.1 Tests that are CORRECT and should be kept

```
test_kalman_hmm.py — TestKalman.test_hmm_viterbi_accuracy_on_3state_synthetic
  → Correct: Hungarian assignment, 90% threshold on pure synthetic data

test_kalman_hmm.py — TestKalman.test_cusum_fires_on_large_shock
  → Correct: 10σ shock, 3-bar detection window

test_surveillance.py — TestSurveillance.test_wasserstein_halt_on_distribution_shift
  → Correct: checks both halt=True on crisis and halt=False on calm

test_validation.py — test_monte_carlo_shuffled_returns_psr
  → Correct: PSR < 0.95 on random returns, tightened threshold from 0.85

test_validation.py — test_cpcv_fold_sharpe_no_systematic_outlier
  → CORRECTED threshold: now max(2.0, 3.0 * std) (was max(5.0, 4.0*std))

test_preprocess.py — test_pca_whitening_causality
  → Correct: pp1.fit(X[:500]) then compare transform at bar 500
```

## 6.2 Tests that are WRONG and must be rewritten

```
test_is_oos_overfit_ratio_bounded (BUG-N017)
  WRONG: uses hardcoded is_sharpe=1.5, oos_sharpe=1.0 — always passes
  FIX: run actual pipeline on synthetic AR(1) data, compute real ratio

test_monte_carlo_shuffled_regime_probs_near_uniform
  WRONG: HMM is fitted on X then tested on X[permutation].
         The HMM emission model was fitted on the unpermuted data.
         Testing on permuted data with the same emission model is not
         a proper null test. The regime labels will still be somewhat
         meaningful because the emission parameters were fitted on the
         same distribution.
  FIX: Train HMM on purely random X (no autocorrelation).
       Assert PSR < 0.95 on the resulting signals.

test_onnx_compilation_accuracy (v4_engine test)
  WRONG: loads from hardcoded path, skips if weights missing
  FIX: train minimal VAE on synthetic data in the test, compile,
       then assert numerical equivalence. Should not require pre-trained weights.
```

## 6.3 Tests that are MISSING and must be created

```
test_preprocess.py::test_fit_transform_fold_independence
  Assert: Preprocessor fitted on fold A does not change when fold B data is appended.

test_features.py::test_tib_is_nonzero_on_real_data
  Assert: After loading Dukascopy tick data and running data_manager.py,
          TIB column is non-zero for at least 80% of bars.

test_execution.py::test_volume_gate_active_when_tib_below_threshold
  Assert: TripleGate.evaluate() returns all_pass=False when tib_z < VOLUME_THRESHOLD.

test_risk.py::test_daily_drawdown_halt
  Assert: After trading_halted is set, evaluate() always returns VETO.

test_risk.py::test_pnl_sign_correct_for_short
  Assert: close_position() with direction=-1 computes positive PnL when price falls.

test_integration.py::test_full_pipeline_smoke
  Assert: pipeline/backtest.py run on 2000-bar synthetic data produces:
          - at least 1 signal
          - Sharpe computation does not throw
          - IS/OOS split is respected (no NaN bleed)
          - All returned metrics are finite floats

test_prop_sim.py::test_ftmo_challenge_pass_rate_on_good_strategy
  Assert: A strategy with daily returns from N(0.001, 0.005) passes FTMO
          at rate > 60% over 10,000 simulations.
```

---

---

# SECTION 7 — NEXT PHASES IN ORDER

---

## Phase 1 — Delete and flatten (Day 1, 2 hours)

```
1. Delete all 27 orphaned files listed in Section 1.1
2. Delete v3_engine/ directory entirely
3. Delete v4_engine/ directory entirely (keep test files — move to tests/)
4. Rename v5_engine/ to engine/
5. Move all test files to tests/
6. Update all imports to reference engine/ not v3_engine.engine or v4_engine
7. Verify: python -c "from engine.features import compute_unified_features" passes
```

## Phase 2 — Data pipeline (Day 2-3, 4 hours)

```
1. Fix data_manager.py: mid-price OHLCV + TIB from tick imbalance
2. Create volume_processor.py
3. Run: python data_pipeline/data_manager.py --symbols EURUSD GBPUSD CHFJPY EURNZD AUDCAD XAUUSD --start 2018-01-01 --end 2024-12-31 --timeframe 1h
4. Verify output CSVs have 'tib' column, no NaN, spread_avg column
5. Run test_features.py::test_tib_is_nonzero_on_real_data → must pass
```

## Phase 3 — Fix critical bugs (Day 3-4, 6 hours)

```
Fix BUG-N001: pipeline/backtest.py calls hmm.predict_proba_causal() on OOS bars
Fix BUG-N002: features.py raises on missing TIB instead of silently zeroing
Fix BUG-N003: run_tearsheet_dynamic uses local_cfg not global mutation
Fix BUG-N005: live_bridge imports from engine/ not v3_engine/
Fix BUG-N006: unify WF_TRAIN_BARS and WF_OOS_BARS as hourly bar counts
Fix BUG-N008: replace all np.sqrt(252) with np.sqrt(config.ANN_FACTOR)
Fix BUG-N009: stage2 indentation bug — final_leaderboard.append inside loop
Fix BUG-N010: pipeline uses pp.fit() then pp.transform() separately per fold
Fix BUG-N012: pipeline_train uses model_spec dims not hardcoded 4
Fix BUG-N014: risk engine direction is int(+1/-1), not string
Fix BUG-N015: HMM regularisation passed as arguments, not module-level constants
Fix BUG-N016: data_manager uses mid-price OHLCV
Fix BUG-N021: compile.py uses model_spec dims
```

## Phase 4 — Strategies as separate classes (Day 4-5, 3 hours)

```
1. Create strategies/base.py with BaseStrategy abstract class
2. Create strategies/momentum.py, mean_rev.py, compression.py
3. Wire TripleGate to use strategy-specific ATR multipliers
4. Verify 3 strategies can run simultaneously on different symbols
```

## Phase 5 — Prop firm simulator (Day 5, 2 hours)

```
1. Create prop_sim/simulator.py as specified in Section 4.3
2. Wire to ensemble_pipeline.py output (reads daily returns from backtest)
3. Add FTMO_100K, MFF_200K, TOPSTEP_FX_150K profiles
4. Run on EURUSD backtest output and print pass rate
5. Target: > 60% pass rate on FTMO at 0% spread assumption
```

## Phase 6 — Tests (Day 6, 4 hours)

```
1. Rewrite test_is_oos_overfit_ratio_bounded to use real pipeline
2. Create test_preprocess.py::test_fit_transform_fold_independence
3. Create test_features.py::test_tib_is_nonzero
4. Create test_execution.py::test_volume_gate_active
5. Create test_risk.py (all 7 rules)
6. Create test_integration.py (full smoke test)
7. Create test_prop_sim.py
8. Run full test suite: pytest tests/ -v → must be 0 failures
```

## Phase 7 — ONNX compilation and live inference (Day 7-8)

```
1. Run pipeline/train.py on all 6 symbols (2018-2023 training data)
2. Run pipeline/compile.py → produces 6 ONNX files in bin/
3. Run test_compiler.py → ONNX vs PyTorch numerical equivalence
4. Test pipeline/live.py with 200 bars of EURUSD → assert action in (EXECUTE, VETO, HOLD)
5. Start live demo bridge, run for 48 hours, verify at least 1 paper trade per symbol
```

## Phase 8 — Visualizer (Day 8-9)

```
1. Create visualizer/dashboard.html with Chart.js panels
2. Wire visualizer/app.py to read from pipeline result cache
3. Test: open http://localhost:8000/visualizer → all 7 panels render with data
4. Test WebSocket feed: assert new bar data pushed within 2 seconds of bar close
```

## Phase 9 — Prop firm go-live (Day 10+)

```
1. Run full ensemble backtest on 6 symbols (2022-2024 OOS)
2. Run prop firm simulator on OOS results → print pass rates per firm
3. If FTMO pass rate < 50%: tune VOLUME_THRESHOLD, TIME_EXIT_BARS via grid_search
4. If FTMO pass rate > 60%: start 1 real challenge account (demo first)
5. Monitor for 2 weeks: ≥ 3 trades per week, no daily DD breach
```

---

---

# SECTION 8 — FINAL PROOF OF COMPLETION CHECKLIST

---

All of these must be simultaneously true. No partial credit.

```
ARCHITECTURE
 □ v3_engine/ directory does not exist
 □ v4_engine/ directory does not exist
 □ engine/ directory contains exactly the files listed in Section 4.2
 □ strategies/ directory contains 3 strategy classes
 □ No hardcoded numbers in any file outside config.py and model_spec.py
 □ grep -r "np.sqrt(252)" . → zero results
 □ grep -r "ann_factor = 252" . → zero results

DATA PIPELINE
 □ data_manager.py produces CSV with columns: open, high, low, close, volume, tib, spread_avg
 □ test_features.py::test_tib_is_nonzero passes on real Dukascopy data
 □ TIB is non-zero for > 80% of all bars across all 6 symbols

TESTS
 □ pytest tests/ -v → 0 failures, 0 errors (minimum 20 tests)
 □ test_preprocess.py::test_fit_transform_fold_independence PASSES
 □ test_risk.py — all 7 rules tested and passing
 □ test_is_oos_overfit_ratio_bounded uses real pipeline output (not hardcoded)
 □ test_integration.py::test_full_pipeline_smoke PASSES

MODEL AND INFERENCE
 □ All 6 ONNX files exist in bin/
 □ test_compiler.py: ONNX vs PyTorch difference < 1e-5
 □ pipeline/live.py processes 200 bars in < 100ms total

BACKTEST RESULTS (on 2022-2024 OOS)
 □ IS/OOS Sharpe ratio < 2.0 for all 6 symbols (leakage removed)
 □ Phase 6 validation: all 5 tests pass (WF-CV, MC, DSR, CPCV, tx-cost)
 □ Minimum 3 trades per week average across the OOS period
 □ Maximum daily drawdown never exceeded 5% in OOS

PROP FIRM SIMULATOR
 □ FTMO_100K pass rate > 50% on OOS daily returns
 □ Challenge simulator reads from actual backtest output (not hardcoded win rate)

LIVE TRADING
 □ live_bridge_ctrader.py starts without ModuleNotFoundError
 □ Demo account connects, symbol IDs resolve for all target symbols
 □ 50+ paper trades executed on demo with Sharpe > 0.5 annualised
 □ Daily DD kill-switch triggered and correctly halts trading in test scenario
 □ Visualizer shows regime probabilities, TIB, and signals for all symbols
```

---

**END OF GROUND TRUTH AUDIT AND V6 PLAN**

---

*Document covers CODEBASE_4.md (10,487 lines, 57 Python files). All line references are to that snapshot.*  
*V6 implementation requires ~10 focused working days. Do not start live trading before Phase 6 (IS/OOS overfit ratio verified < 2.0).*

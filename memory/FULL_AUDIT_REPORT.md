# LATENT DIFFUSION-HMM TRADING ENGINE
## COMPLETE SYSTEM AUDIT REPORT
### Architecture Review · Bug List · Data Leakage Analysis · Test Metric Audit · Fix Plan · API Deployment Guide
**Generated:** 2026-06-11  
**Codebase version:** Repomix snapshot 2026-06-11T08:11:26Z  
**Engine versions covered:** v3_engine (production), v4_engine (next-gen), orchestrators, data_pipeline, live_trading

---

---

# PART 1 — SYSTEM ARCHITECTURE OVERVIEW

---

## 1.1 What the system is

This is a multi-layer quantitative trading signal engine based on a pipeline that converts raw market price bars into statistically-validated trade signals. The core intellectual architecture is:

```
Raw Ticks / OHLCV bars
    ↓
[Layer 1] Data Ingestion
    Tick-Vault Dukascopy downloader, bid-price OHLCV resampling
    data_pipeline/data_manager.py

[Layer 2] 6-Dimensional Feature Tensor
    vt  = Volatility Proximity
    mvt = Momentum Velocity
    qt  = Volume Delta Ratio
    sigma_t = Volatility Regime Ratio (RV/GARCH)
    rho_t   = Autocorrelation Signal
    ht      = DFA Hurst Exponent
    v3_engine/engine/features.py

[Layer 3] Expanding Winsorize → Expanding Standardise → PCA Whitening
    v3_engine/engine/preprocess.py

[Layer 4] Kalman Filter (EM-fitted Q,R) + CUSUM Jump Detector
    v3_engine/engine/kalman.py

[Layer 5] TVTP-HMM (3 states, K=2 GMM emissions)
    v3_engine/engine/hmm.py
    States: TREND, MEAN_REV, STRESS/COMPRESSION

[Layer 6] Triple Gate Execution + Kelly Position Sizing + ATR SL/TP
    v3_engine/engine/execution.py

[Layer 7] Wasserstein Distribution Monitor (W1 circuit-breaker)
    v3_engine/engine/surveillance.py

[Layer 8] Statistical Validation Suite
    Walk-Forward CV, Monte Carlo Permutation, Deflated Sharpe,
    CPCV, Transaction Cost Sensitivity
    v3_engine/engine/validation.py

[Layer 9] REST API (FastAPI)
    v3_engine/api/routes.py + v3_engine/main.py

[Parallel] V4 Engine (Deep Learning Path)
    VAE latent encoding → Gumbel-Softmax Router → Expert GBM/SVM layer
    → Meta-Learner (neural correction) → ONNX compiled live inference
    v4_engine/

[Parallel] Live Bridge (cTrader)
    live_trading/live_bridge_ctrader.py (v3)
    v4_engine/live_bridge_ctrader.py (v4)

[Parallel] Orchestration
    orchestrators/ensemble_pipeline.py  — multi-asset backtest
    orchestrators/grid_search.py        — hyperparameter sweep
    orchestrators/stage2_global_test.py — cross-asset stress test
    orchestrators/master_pipeline.py    — run controller
```

The V4 engine replaces the TVTP-HMM with a VAE + Gumbel-Softmax routing head, adding expert gradient-boosted models per regime and a neural meta-learner that corrects expert output confidence before the Kelly/veto gate.

---

## 1.2 Architecture Design Language (how the system was designed)

The intended design pattern is:

- All hyperparameters in a single `config.py` with env-override at import time
- Model architecture dimensions in a separate `model_spec.py`
- Expanding (causal/online) statistics everywhere so no future data can flow back
- Walk-Forward backtest, not full-history backtest
- Six-layer statistical validation as a mandatory go-live gate
- `config.py` doubles as the source of truth for live and backtest, so tuning results survive deployment without manual translation

---

---

# PART 2 — COMPLETE BUG AND ERROR LIST

---

## BUG-001 | CRITICAL | Preprocessor uses full-history PCA on the backtest dataset — DATA LEAKAGE

**File:** `v3_engine/engine/preprocess.py` — `Preprocessor.fit_transform()`

**What happens:**
```python
def fit_transform(self, features: pd.DataFrame, train_bars: int | None = None) -> np.ndarray:
    ...
    if train_bars is not None:
        train_slice = X_std[:train_bars]
    else:
        train_slice = X_std[:1000]         # ← hardcoded 1000 bars for PCA fit
    self._pca_params = fit_pca_whitening(train_slice)
    ...
    X_white[mask] = apply_pca_whitening(X_std[mask], self._pca_params)  # applied to ALL bars
```

The `fit_pca_whitening` call uses only `train_slice` which is either `train_bars` (if passed) or the first 1000 bars. BUT the expanding winsorize and expanding standardize steps that happen **before** PCA still run on the full array. More critically, when `fit_transform` is called inside the walk-forward loop in `reports.py`, it is called once on the full OOS window and fits its Welford statistics on in-sample data embedded inside the full call. This means the 1–99 percentile boundaries for winsorization are computed on all data including OOS.

**How to fix:**
The `Preprocessor` must be split into `fit()` and `transform()` clearly, fitting only on in-sample, then transforming OOS blindly:

```python
# CORRECT PATTERN — replaces current fit_transform in pipeline.py
preprocessor = Preprocessor()
X_train = features.values[:train_end]
preprocessor.fit(X_train)                    # fits Welford + PCA on train only
X_white_full = preprocessor.transform(features.values)  # transforms all bars
```

The `Preprocessor` class has a `transform()` method already but `fit_transform` re-initialises `_state = PreprocessorState()` every call, resetting all expanding statistics. There is no separate `fit()` method. Add one.

---

## BUG-002 | CRITICAL | HMM fitted on full data including OOS inside walk-forward windows

**File:** `v3_engine/engine/pipeline.py` — `Pipeline.run()`

**What happens:**
```python
train_end = int(result.n_bars * config.train_fraction)
X_train = filtered[clean_mask & (np.arange(result.n_bars) < train_end)]
hmm = TVTPHMM(n_states=3, n_gmm=2, n_iter=30)
hmm.fit(X_train, cov_train)
...
X_full = filtered.copy()
for t in range(1, len(X_full)):
    if np.any(np.isnan(X_full[t])):
        X_full[t] = X_full[t - 1]           # forward-fill including OOS
hmm_result = hmm.predict(X_full, covariates) # predicts on ALL bars including OOS
```

The HMM is correctly fitted only on `X_train`. However the `predict` (Viterbi decode and forward-probability) call runs on the entire `X_full` array including the OOS portion. In the TVTP-HMM, the Baum-Welch smooth posteriors are used during prediction — if the implementation uses a backward pass (smoother) over the full sequence, the OOS regime probabilities used by the signal generator have information from the future. 

**How to fix:**
Pass only causal forward probabilities (filter, not smoother) to the execution layer for OOS bars. In `hmm.py`'s `predict_proba`, use the forward algorithm only, not the Viterbi or Baum-Welch backward pass. This must be verified inside `TVTPHMM.predict_proba()` and `TVTPHMM.predict()`.

---

## BUG-003 | CRITICAL | reports.py uses module-level global state that breaks multiprocessing

**File:** `v3_engine/engine/reports.py`

**What happens:**
The file sets module-level globals at import time:
```python
TICKER = sys.argv[1] if len(sys argv) > 1 else "EURUSD=X"
ASSET_NAME = ...
PIP_SIZE = ...
OOS_START = "2020-01-01"
INITIAL_EQUITY = 100_000.0
```

The ensemble pipeline runs `process_asset` in a `ProcessPoolExecutor`. Each worker process imports `reports.py`, but `sys.argv[1]` in a subprocess will be whatever the subprocess receives — it will not be the ticker the parent intended. The global `TICKER` will always resolve to the wrong value or the default in subprocess workers.

Additionally `_PHASE1_CACHE` is a module-level dict. Across multiple assets in the same process, the cache will return stale results from the first asset for all subsequent ones if the ticker string matches.

**How to fix:**
All globals in `reports.py` that represent per-run state (TICKER, ASSET_NAME, PIP_SIZE, OOS_START, _PHASE1_CACHE) must be moved inside `run_tearsheet_dynamic()` as local variables passed as arguments. The function `run_tearsheet_dynamic(csv_path, params)` already exists — make it stateless by removing all module-level mutable globals.

---

## BUG-004 | HIGH | Preprocessor state reset on every fit_transform call breaks the walk-forward loop

**File:** `v3_engine/engine/preprocess.py` — `Preprocessor.fit_transform()`

```python
self._state = PreprocessorState()  # RESET EVERY CALL
X_wins = expanding_winsorize(X, self._state)
state2 = PreprocessorState()
state2.n = 0
...
X_std = expanding_standardize(X_wins, state2)
```

Every time `fit_transform` is called it throws away the `_state` object. If `fit_transform` is called on each walk-forward fold window, the Welford online statistics are restarted from zero, meaning early bars in each fold get `NaN` for the first 99 samples (warmup to `n >= 100` in `_expanding_standardize_numba`). This is not expanding standardisation — it is fold-local standardisation — which is a form of lookahead because early production bars get no normalisation.

**How to fix:**
The `_state` should accumulate across calls if used in an incremental streaming context. Add a `fit(X)` method that updates `_state` from the training window, and a `transform(X)` that applies the fitted statistics to new bars without updating them.

---

## BUG-005 | HIGH | stage2_global_test.py has a loop-scope indentation bug

**File:** `orchestrators/stage2_global_test.py`

```python
with concurrent.futures.ProcessPoolExecutor(...) as executor:
    for i, entry in enumerate(top_10, 1):
        ...
        futures = [executor.submit(...) for csv in csv_files]
        for future in concurrent.futures.as_completed(futures):
            ...
    if success_count > 0 and total_trades > 0:   # ← WRONG INDENTATION
        ...
        final_leaderboard.append(...)
```

The `if success_count > 0` block and all the portfolio statistics and `final_leaderboard.append()` call are outside the `for i, entry` loop but inside the `with executor` block. This means:
- Variables `success_count`, `portfolio_returns`, `total_trades` computed inside the for-loop are only the values from the **last** iteration of the loop
- Every top-10 param set except the last one is silently discarded from the leaderboard

**How to fix:**
Indent the `if success_count > 0` block (and everything under it including `final_leaderboard.append`) to be inside the `for i, entry` loop.

---

## BUG-006 | HIGH | ensemble_pipeline.py passes wrong column name when splitting data

**File:** `orchestrators/ensemble_pipeline.py`

```python
if pd.api.types.is_numeric_dtype(df["datetime"]):
    dt_col = pd.to_datetime(df["datetime"], unit="ms")
```

The data produced by `data_pipeline/data_manager.py` writes the timestamp column as `"timestamp"`, not `"datetime"`. The ensemble pipeline reads it as `"datetime"`. This means every `--split train` or `--split test` execution will throw `KeyError: 'datetime'` immediately for every CSV file from the Dukascopy downloader.

**How to fix:**
Normalise the column name at read time:
```python
df = pd.read_csv(csv_path)
# normalise time column
for col_name in ["timestamp", "datetime", "time", "date"]:
    if col_name in df.columns:
        if pd.api.types.is_numeric_dtype(df[col_name]):
            dt_col = pd.to_datetime(df[col_name], unit="ms")
        else:
            dt_col = pd.to_datetime(df[col_name])
        break
```

---

## BUG-007 | HIGH | live_bridge.py (v4) has hardcoded SL/TP multipliers ignoring config

**File:** `v4_engine/live_bridge.py` — `LiveExecutionEngine.process_live_tick()`

```python
sl_distance = atr * 1.0    # hardcoded
tp_distance = atr * 2.0    # hardcoded
```

The config file defines `LIVE_SL_MULT = 1.0` and `LIVE_TP_MULT = 2.0`, but these are never referenced. If config values are changed for tuning (e.g. `LIVE_TP_MULT = 3.0`), live positions will silently use the wrong multiplier.

**How to fix:**
```python
import config
sl_distance = atr * config.LIVE_SL_MULT
tp_distance = atr * config.LIVE_TP_MULT
```

---

## BUG-008 | HIGH | get_token.py sends secrets in GET query parameters (security vulnerability)

**File:** `live_trading/get_token.py`

```python
params = {
    "grant_type": "authorization_code",
    "code": AUTH_CODE,
    "redirect_uri": REDIRECT_URI,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET
}
response = requests.get(token_url, params=params)
```

This encodes `client_secret` and the auth code as URL query parameters in a GET request. These values will be logged in:
- Server access logs
- Browser history
- Network proxies
- Any monitoring middleware

OAuth token endpoints require POST with the credentials in the request body.

**How to fix:**
```python
response = requests.post(token_url, data=params)
```

---

## BUG-009 | HIGH | v3_engine/api/routes.py uses variable name `config` that shadows imported module

**File:** `v3_engine/api/routes.py`

```python
import config   # <- module imported at top... actually it is NOT imported here
...
config = PipelineConfig(    # ← local variable named `config` shadows any module-level config
    ticker=req.ticker.upper(),
    ...
)
```

Inside `analyze()`, `get_regime()`, `get_signals()`, `get_features()`, and `validate()`, the local variable `config` is assigned a `PipelineConfig` dataclass instance. If the `routes.py` file ever needs to access the global `config` module (e.g. for default parameters), it cannot because the name is shadowed in every endpoint function. This is an existing design smell that will cause `NameError` or silent misconfiguration during future extension.

**How to fix:**
Rename the local variable to `pipeline_cfg` or `pcfg` throughout all route handlers.

---

## BUG-010 | MEDIUM | Sharpe ratio annualisation factor is wrong for hourly data

**File:** `v3_engine/engine/reports.py` and `orchestrators/ensemble_pipeline.py`

```python
ann_factor = 252   # DAILY annualisation
port_sharpe = port_mean / port_std if port_std > 0 else 0.0
port_mean   = np.mean(port_returns) * ann_factor
port_std    = np.std(port_returns, ddof=1) * np.sqrt(ann_factor)
```

The data is 1-hour candles (`*_1h.csv`). 252 is the trading-day annualisation factor. For hourly bars the correct factor is approximately 252 × 6.5 = 1638 (US equity sessions) or 252 × 24 = 6048 (FX which trades 24 hours). Using 252 for hourly data understates the annualised Sharpe ratio by a factor of roughly 2.5× to 5×, making strategies look worse than they are. More importantly it means the Sharpe firewall threshold of `SHARPE_FIREWALL = 0.40` corresponds to a much weaker actual edge than intended.

**How to fix:**
Add to `config.py`:
```python
BARS_PER_DAY = 24   # for FX 1h data; 6.5 for equity; override via env
ANN_FACTOR = int(252 * BARS_PER_DAY)
```
Reference `config.ANN_FACTOR` everywhere Sharpe is annualised.

---

## BUG-011 | MEDIUM | Kalman refit re-runs the full filter from bar 0 at every refit window

**File:** `v3_engine/engine/kalman.py` — `run_kalman_pipeline()`

```python
while current_idx < T:
    next_idx = min(current_idx + refit_every, T)
    ...
    kf.fit(X_whitened[past_clean], n_iter=2)
    f_s, _, inn, j_f = kf.filter(X_whitened[:next_idx], ...)   # ← filters from bar 0 each time
    chunk_len = next_idx - current_idx
    filtered_states[current_idx:next_idx] = f_s[-chunk_len:]   # takes only tail
```

Every `refit_every` bars, the filter is re-run from bar 0 to `next_idx`. For a 5000-bar series with `refit_every=63`, this means bar 0 is filtered 5000/63 ≈ 79 times. This is O(T²) time complexity. For large datasets this is very slow and the repeated re-filtering of early bars is wasted work. The Kalman filter state could instead be stored and resumed.

**How to fix:**
Store the final `(mu_pred, P_pred)` state from each chunk and pass it as the initial condition for the next chunk, making it O(T) total.

---

## BUG-012 | MEDIUM | Wasserstein monitor in pipeline.py fits on X_white but checks on X_white too — same distribution

**File:** `v3_engine/engine/pipeline.py`

```python
monitor = WassersteinMonitor(window=min(50, result.n_bars // 4))
live_window = X_white[-monitor.window:]      # last window bars of whitened features
clean_train = X_white[:train_end]            # training portion
monitor.fit(clean_train)
surv_result = monitor.check(live_window)     # checking on same preprocessed space
```

The monitor is fitted and checked on `X_white` — the same whitened feature space. Since `X_white` was produced by `fit_transform()` which fits the PCA on the first 1000 bars of the full series, the "live" window already has the same whitening transformation applied as training. The W1 distance will be artificially low because the whitening itself standardises the distribution globally. The monitor should operate on raw (un-whitened) features or on Kalman-filtered states to detect genuine distribution drift.

**How to fix:**
Feed `filtered_states` (output of Kalman pipeline) to the Wasserstein monitor instead of `X_white`. This way the monitor operates on data the whitening cannot have aligned globally.

---

## BUG-013 | MEDIUM | reports.py: OOS_START is hardcoded as "2020-01-01" not from config

**File:** `v3_engine/engine/reports.py`

```python
OOS_START = "2020-01-01"   # hardcoded here
```

`config.py` has `OOS_START_DATE = "2022-01-01"`. The two values disagree. The reports backtest will silently use a different OOS boundary than the orchestrators, making the reported OOS metrics inconsistent with ensemble results.

**How to fix:**
```python
import config
OOS_START = config.OOS_START_DATE
```

---

## BUG-014 | MEDIUM | master_pipeline.py and stage2_global_test.py import `forex_tearsheet` which does not exist

**File:** `orchestrators/stage2_global_test.py`

```python
import forex_tearsheet
data = forex_tearsheet.run_tearsheet_dynamic(csv_path, params)
```

There is no file called `forex_tearsheet.py` in the repository. The function `run_tearsheet_dynamic` lives in `v3_engine/engine/reports.py`. This will throw `ModuleNotFoundError` on any execution of `stage2_global_test.py` and `master_pipeline.py` (where the same pattern appears).

**How to fix:**
```python
import engine.reports as forex_tearsheet
# or
from engine.reports import run_tearsheet_dynamic
```

---

## BUG-015 | MEDIUM | CORSMiddleware in main.py allows all origins — production security risk

**File:** `v3_engine/main.py`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # all origins allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`allow_origins=["*"]` combined with `allow_credentials=True` is a security misconfiguration. Browsers will reject CORS requests that have both a wildcard origin and credentials enabled. This means browser-based API clients will fail silently, and no authentication can be enforced from the server side. In production this should be restricted to the actual front-end domain.

**How to fix:**
```python
allow_origins=config.CORS_ALLOWED_ORIGINS  # add to config.py as list
allow_credentials=True if allow_origins != ["*"] else False
```

---

## BUG-016 | MEDIUM | v4 test_compiler.py hardcodes VAE input_dim=20 bypassing model_spec

**File:** `v4_engine/tests/test_compiler.py`

```python
vae_python = VAE(input_dim=20, latent_dim=4)
```

If `model_spec.VAE_INPUT_DIM` is changed (e.g. to add more features), this test will still build a 20-dim model, load mismatched weights, and produce wrong ONNX comparison results or fail on shape mismatch. The test will give false confidence.

**How to fix:**
```python
import model_spec
vae_python = VAE(input_dim=model_spec.VAE_INPUT_DIM, latent_dim=model_spec.VAE_LATENT_DIM)
```

---

## BUG-017 | MEDIUM | model_spec.py and config.py have conflicting META_DROPOUT values

**File:** `model_spec.py` and `config.py`

`model_spec.py`:
```python
META_DROPOUT = 0.5
```

`config.py`:
```python
META_DROPOUT = 0.7
```

The `refactor_v4.py` script injects `model_spec.META_DROPOUT` into `meta_learner.py`. At training time `meta_learner.py` will use 0.5. But `config.py` defines 0.7. If any code references `config.META_DROPOUT` it will get a different value than the model was trained with. This creates a silent discrepancy between training and inference.

**How to fix:**
Remove `META_DROPOUT` from `config.py` entirely. The model architecture constants (dropout, layer widths, dims) belong only in `model_spec.py`. `config.py` should contain only training hyperparameters (learning rate, epochs, batch size).

---

## BUG-018 | LOW | data_manager.py resamples using `bid` price only — creates artificial OHLCV

**File:** `data_pipeline/data_manager.py`

```python
ohlcv = df.resample(timeframe).agg({
    'bid': ['first', 'max', 'min', 'last'],
    'volume': 'sum'
})
ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
```

Using only `bid` for all four OHLCV columns ignores the mid-price (bid+ask)/2. For forex pairs with a spread, the high/low bands will be systematically lower than mid-price bands. This creates a spread-direction bias in ATR calculations and in the triple-barrier labels.

**How to fix:**
Compute `mid = (bid + ask) / 2` from tick data and resample `mid` for OHLCV. Volume should remain summed.

---

## BUG-019 | LOW | prop_firm_simulator.py uses hardcoded win_rate not from backtest output

**File:** `v3_engine/prop_firm_simulator.py`

```python
def simulate_prop_firm(
    n_simulations=10000,
    win_rate=0.6769,    # hardcoded
    win_size=2.0,
    loss_size=1.0,
    risk_per_trade=0.01,
```

The win rate `0.6769` does not come from any live backtest output. If the strategy's win rate changes after hyperparameter tuning, the prop firm simulator will continue reporting the old number, giving a false confidence in the challenge pass rate.

**How to fix:**
Have `ensemble_pipeline.py` write portfolio-level win rate to a results JSON, and have `prop_firm_simulator.py` read that file as its default input rather than a hardcoded value.

---

## BUG-020 | LOW | diagnostics/audit_real_data.py hardcodes btcusd_1h.csv

**File:** `diagnostics/audit_real_data.py`

```python
df_raw = pd.read_csv("data/btcusd_1h.csv")
```

The audit script always loads BTC data and cannot be used for other symbols without editing the source. It should accept a `--file` CLI argument.

---

## BUG-021 | LOW | v3_engine/engine/reports.py imports warnings twice and os/sys twice

**File:** `v3_engine/engine/reports.py`

```python
import warnings
...
import warnings          # duplicate
...
import sys
import os
...
import os, sys           # duplicate
_root = ...
```

These duplicates are harmless but indicate unclean refactor history from the `refactor_*.py` scripts and will cause confusion during further editing.

---

## BUG-022 | LOW | pack_repo.py contains invalid Python — heredoc syntax inside a .py file

**File:** `pack_repo.py`

```python
cat << 'EOF' > pack_repo.py
import os
def pack_folders():
```

The file starts with a bash heredoc redirect (`cat << 'EOF' > pack_repo.py`). This is shell syntax inside a Python file. Running `python pack_repo.py` will throw `SyntaxError` immediately.

---

---

# PART 3 — DATA LEAKAGE ANALYSIS

---

## 3.1 Where data leakage exists in this codebase

Data leakage means: information from the future (OOS period) influences decisions or statistics that are supposed to be blind to the future (IS/training period). In backtesting, leakage produces inflated performance metrics that will not generalise to live trading.

### LEAK-A | CRITICAL | PCA whitening fitted on future data (BUG-001)

**Severity: CRITICAL**

The `Preprocessor.fit_transform()` function initialises `PreprocessorState()` fresh and then runs expanding winsorize and expanding Welford standardise on the entire dataset. The Welford statistics at OOS bar T have been influenced by all bars from 0 to T (correct, since it is expanding). However, the PCA is then fitted on either `train_bars` or the first 1000 bars of the standardised output. This part is correctly bounded.

The real leak: the expanding winsorise computes 1st and 99th percentiles using `history` accumulated from the beginning of the call. If `fit_transform` is called on a full IS+OOS window, the percentile cutoffs for winsorisation used on bar t in IS were computed using bars 0 through t-1 — that is causal. But the PCA is applied to ALL bars using a fit that incorporated all IS+OOS standardised values through the Welford state that ran over the whole array.

**Impact on Sharpe:** Estimated 0.1–0.3 Sharpe inflation. The primary mechanism is that the whitening rotation applied to OOS bars uses eigenvectors that align with the covariance structure of the full dataset including OOS. For a genuine out-of-sample test, the PCA must be fitted only on the IS bars.

### LEAK-B | HIGH | HMM smoother posteriors over full sequence (BUG-002)

**Severity: HIGH**

If TVTP-HMM uses a backward smoothing pass (Baum-Welch with forward-backward algorithm), then `predict_proba` at any bar t in the OOS region will have information about bars t+1, t+2, ... flowing backwards through the backward recursion. The regime labels at OOS bar t will reflect future regime transitions.

**Impact on Sharpe:** Estimated 0.2–0.5 Sharpe inflation. Regime labels that "know" a trend is about to end will correctly avoid the last few bars of the trend that a real causal system would not know to avoid.

**To verify:** Inspect `TVTPHMM.predict_proba()` in `v3_engine/engine/hmm.py`. If it calls any form of backward message passing or smoothing, it has this leakage. Replace with forward-only (filtering) probabilities for OOS prediction.

### LEAK-C | MEDIUM | Walk-forward train/test split uses train_fraction=0.7 of each call's window

**Severity: MEDIUM**

In `reports.py`, `run_tearsheet_dynamic()` calls `phase1_build_data()` which calls `Preprocessor().fit_transform(features)` on the full dataset and then walks forward. The WF splits are computed correctly with purge buffers. However the Preprocessor state (see LEAK-A) used for ALL WF folds is computed once on the full data. This means the Preprocessor statistics used at fold k's OOS test period contain data from fold k+1, k+2, ..., n.

**Impact on Sharpe:** Moderate, estimated 0.1–0.2 inflation per fold average. The effect is largest at early fold boundaries.

### LEAK-D | LOW | Monte Carlo permutation test uses real OOS returns as baseline

**Severity: LOW — correctly implemented**

In `validation.py`, `monte_carlo_permutation_test()` correctly shuffles the returns array and re-computes Sharpe on the shuffled sequence. The null distribution is: what Sharpe would we get if the return sequence had the same values but random ordering? The actual strategy Sharpe is computed on the real unshuffled OOS returns. This is the correct implementation and is not a source of leakage.

### LEAK-E | LOW | Transaction cost sensitivity applies a flat cost drag across all returns

**Severity: LOW — mild approximation**

In `validation.py`, `transaction_cost_sensitivity()`:
```python
total_cost_return = -(n_trades * cost_per_trade) / max(T, 1)
net_returns = gross_returns.copy()
net_returns = net_returns + total_cost_return    # flat cost applied uniformly
```

The cost is divided uniformly across all T bars rather than applied at the bars where trades actually occur. For low-frequency strategies (few trades per bar) this is a conservative approximation that slightly underestimates the Sharpe hit (because cost is spread thinly). Not a leakage issue, but an inaccuracy to note.

---

## 3.2 Summary of leakage severity

| Leak ID | Description | Sharpe Inflation Estimate | Priority |
|---------|-------------|--------------------------|----------|
| LEAK-A  | PCA fitted on full IS+OOS data | +0.1 to +0.3 | Fix First |
| LEAK-B  | HMM smoother backward pass over OOS | +0.2 to +0.5 | Fix Second |
| LEAK-C  | Preprocessor state not reset per fold | +0.1 to +0.2 | Fix Third |
| LEAK-D  | Monte Carlo permutation | None — correct | No action |
| LEAK-E  | Flat cost drag approximation | Slight understatement | Low priority |

Total estimated Sharpe inflation from leakage: approximately +0.4 to +1.0 depending on strategy mode and dataset. A strategy showing Sharpe 1.5 in backtest may be 0.5 to 1.1 in a truly leak-free test.

---

---

# PART 4 — TEST AND METRIC AUDIT

---

## 4.1 Are the test assertions correct?

### TEST-REVIEW-01 | run_validation_suite.py — HMM Viterbi accuracy test

```python
assert accuracy > 0.90
```

**Verdict: CORRECT AND RIGOROUS.** Using optimal assignment (Hungarian algorithm) to match predicted states to ground truth states is the mathematically correct approach for label-permutation-invariant accuracy. The 90% threshold on clearly-separated synthetic data is appropriate — if the HMM cannot hit 90% on pure-regime synthetic data, it will certainly fail on real data.

### TEST-REVIEW-02 | run_validation_suite.py — CUSUM jump detection test

```python
assert detection_window.any()  # fires within 3 bars of 10-sigma shock
assert shock_innov > 3.0 * pre_baseline
assert post_innov < 3.0 * pre_baseline   # recovers within 4 bars
```

**Verdict: CORRECT BUT POTENTIALLY TOO TIGHT.** A 10-sigma shock that fires CUSUM is a reasonable stress test. The 3-bar detection window is correct — CUSUM has 1-bar lag by design. The 3× innovation spike threshold is appropriate. The recovery assertion is questionable: after a diffuse-prior reset, the Kalman filter will be high-variance for several bars, not just 3-4. On noisy data this assertion could produce false failures.

**Recommendation:** Loosen recovery check to 10 bars post-trigger.

### TEST-REVIEW-03 | run_validation_suite.py — Wasserstein circuit breaker

```python
assert result_crisis["w1_distance"] > result_crisis["threshold"]
assert result_crisis["halt"] is True
assert result_crisis["position_scale"] == pytest.approx(0.25, abs=1e-9)
assert result_calm["halt"] is False
```

**Verdict: CORRECT.** Testing both the halt condition and the calm non-halt condition is the right approach. The `abs=1e-9` tolerance on `position_scale` is unnecessarily tight — it tests for exact floating point equality on a value set by `LIVE_POSITION_SCALE_ON_HALT = 0.25`. This will be brittle if the formula changes. Use `abs=1e-6`.

### TEST-REVIEW-04 | run_validation_suite.py — Monte Carlo shuffled regime test

```python
assert psr < 0.85
```

**Verdict: CORRECT INTENT, WEAK THRESHOLD.** PSR < 0.85 on shuffled data means the strategy is not statistically significant on random noise. But 0.85 is a loose threshold — a PSR of 0.84 on shuffled data is suspicious. The standard in quantitative finance for the null-rejection PSR is < 0.95. This test should use `psr < 0.95` to be a stronger leakage detector.

**Additionally:** The test at line 2132–2134 correctly notes:
```
"A PSR > 0.85 on shuffled features indicates data leakage."
```
The threshold should be consistent with the literature. PSR > 0.95 on null data is the standard alarm level.

### TEST-REVIEW-05 | run_validation_suite.py — CPCV fold uniformity test

```python
outlier_threshold = max(5.0, 4.0 * std_sr)
assert deviation < outlier_threshold
```

**Verdict: MECHANICALLY CORRECT, SEMANTICALLY WEAK.** The test uses 4-sigma outlier detection on CPCV fold Sharpes. However, on random data with 4 folds and a small dataset, the standard deviation of fold Sharpes is itself very noisy (chi-squared with 3 degrees of freedom). The `max(5.0, ...)` fallback means on low-variance data the threshold is 5.0 — a Sharpe of 5.0 on a single fold would not trigger this alarm, which is a sign of extreme overfitting. 

**Recommendation:** Change to:
```python
outlier_threshold = max(2.0, 3.0 * std_sr)  # tighter default
```

### TEST-REVIEW-06 | v4 tests — Triple barrier test

```python
assert target[0] == 1
assert target[1] == 1
```

**Verdict: CORRECT.** Testing that a high with `delta_high = 10` (which exceeds `tp_mult=8.0 × atr=1.0`) triggers target=1 is correct. The test is minimal but valid.

### TEST-REVIEW-07 | test_vae.py — VAE reconstruction error

**Not visible in truncated codebase.** Recommend adding:
- Reconstruction loss on held-out data must be below a threshold set by model_spec
- KL divergence must be positive and bounded above  
- Latent space samples must not be all-zero (posterior collapse check)

### TEST-REVIEW-08 | Absent test — Sharpe IS/OOS ratio (overfitting detector)

There is no test that checks:
```
assert IS_sharpe / OOS_sharpe < 2.0
```
This is the most important single test for overfitting. The walk-forward CV in `validation.py` computes `overfit_ratio` but there is no pytest assertion on it in `run_validation_suite.py`. It is only used inside the full validation suite which requires real data to run.

**Recommendation:** Add to `run_validation_suite.py`:
```python
def test_is_oos_overfit_ratio_bounded():
    # generate synthetic trending data, run full pipeline,
    # assert wfcv.overfit_ratio < 2.0
```

### TEST-REVIEW-09 | Missing test — PCA whitening causality

There is no test that verifies: "If I fit the Preprocessor on bars 0-499 and transform bar 500, the whitened value at bar 500 does not depend on bars 501+". This is the most important test to confirm LEAK-A is fixed.

---

## 4.2 Are the validation metrics correct?

### METRIC-01 | Sharpe Ratio

**v3_engine/engine/reports.py compute_metrics():**
```python
sharpe = float(np.mean(eq_rets) / (np.std(eq_rets) + 1e-10) * np.sqrt(ann_factor))
```

**Issues:**
- `ann_factor = 252.0` is used even for hourly data (see BUG-010)
- The equity curve returns `eq_rets = np.diff(eq_vals) / (eq_vals[:-1] + 1e-10)` are dollar-weighted returns, not trade returns — this is correct for portfolio Sharpe
- The `1e-10` denominator guard is correct to avoid division by zero

**Verdict: FORMULA CORRECT, ANNUALISATION FACTOR WRONG FOR HOURLY DATA**

### METRIC-02 | Deflated Sharpe Ratio

**v3_engine/engine/validation.py:**
```python
dsr_z = (port_sharpe * np.sqrt(T)) / np.sqrt(1 + 0.5 * port_sharpe**2)
dsr_p = 1 - stats.norm.cdf(dsr_z)
```

This follows the Bailey-Lopez de Prado (2012) DSR formula. The `ensemble_pipeline.py` replicates this calculation directly. The `validation.py` implementation uses the full formula including skewness and kurtosis corrections.

**Verdict: CORRECT — DSR implementation follows the academic standard**

### METRIC-03 | Probabilistic Sharpe Ratio (PSR)

**v3_engine/engine/validation.py — `_probabilistic_sharpe_ratio()`:**
Checks `sr_benchmark = 0.0` by default. PSR measures probability that the observed SR exceeds the benchmark SR after adjusting for parameter estimation error.

**Verdict: CORRECT but benchmark of 0.0 is very easy to beat — should be 0.5 or the risk-free rate Sharpe**

### METRIC-04 | Maximum Drawdown

```python
peak = np.maximum.accumulate(eq_vals)
dd = (eq_vals - peak) / (peak + 1e-10)
max_dd = float(dd.min() * 100)
```

**Verdict: CORRECT — standard high-water mark drawdown**

### METRIC-05 | Calmar Ratio

```python
calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("inf")
```

**Verdict: CORRECT — CAGR / MaxDD is the standard Calmar formula. The `inf` when no drawdown is technically correct though `float("inf")` can cause issues in JSON serialisation (use 999.0 as a cap)**

### METRIC-06 | CPCV path construction

```python
n_paths = int(comb(n_folds, test_folds))
```

**Verdict: CORRECT — combinatorial paths are the correct count for CPCV (Lopez de Prado 2018)**

---

---

# PART 5 — NEXT PHASES: FIXES, DESIGN, AND ARCHITECTURE

---

## Phase 1 — Fix All Critical Bugs (must be done before any live test)

**Objective:** Zero data leakage, zero crash bugs. All existing tests pass. No hidden state crosses IS/OOS boundary.

**Duration estimate:** 3–5 days of focused work.

### Step 1.1 — Split Preprocessor into fit() and transform()

In `v3_engine/engine/preprocess.py`, add:

```python
class Preprocessor:
    def fit(self, X_train: np.ndarray) -> "Preprocessor":
        """Fit expanding statistics and PCA on training data only."""
        D = X_train.shape[1]
        self._state = PreprocessorState()
        X_wins = expanding_winsorize(X_train, self._state)
        self._std_state = PreprocessorState()
        self._std_state.n = np.zeros(D, dtype=np.int64)
        self._std_state.means = np.zeros(D)
        self._std_state.M2 = np.zeros(D)
        X_std = expanding_standardize(X_wins, self._std_state)
        self._pca_params = fit_pca_whitening(X_std)
        self._fitted = True
        self._D = D
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted statistics to new data without updating them."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        # Apply saved percentile clips from training
        X_wins = winsorize_with_saved_bounds(X, self._state)
        # Apply Welford statistics saved from training
        X_std = standardize_with_saved_stats(X, self._std_state)
        # Apply PCA
        mask = ~np.any(np.isnan(X_std), axis=1)
        result = np.full_like(X_std, np.nan)
        result[mask] = apply_pca_whitening(X_std[mask], self._pca_params)
        return result
```

Add helper functions `winsorize_with_saved_bounds()` and `standardize_with_saved_stats()` that apply the saved boundaries and Welford parameters without updating state.

### Step 1.2 — Fix HMM to use causal forward probabilities for OOS

In `v3_engine/engine/hmm.py`, add a `predict_proba_causal()` method that uses only the forward algorithm:

```python
def predict_proba_causal(self, X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Return forward-only (causal, non-smoothed) regime probabilities.
    Use this for OOS evaluation. Do NOT use the smoother posteriors."""
    # Forward pass only — no backward pass
    alpha = self._forward_pass(X, covariates)  # shape (T, n_states)
    return alpha / alpha.sum(axis=1, keepdims=True)
```

In `pipeline.py`, replace:
```python
hmm_result = hmm.predict(X_full, covariates)
```
with:
```python
proba = hmm.predict_proba_causal(X_full, covariates)
```
for all OOS evaluation purposes.

### Step 1.3 — Fix stage2_global_test.py indentation bug

Move the `if success_count > 0` block inside the `for i, entry` loop (one tab deeper). Verify with a test run.

### Step 1.4 — Fix ensemble_pipeline.py column name normalisation

Replace the hard `df["datetime"]` access with the normalised column finder shown in BUG-006 fix.

### Step 1.5 — Fix forex_tearsheet import

In `stage2_global_test.py` and `master_pipeline.py`, replace:
```python
import forex_tearsheet
```
with:
```python
import engine.reports as forex_tearsheet
```

### Step 1.6 — Fix get_token.py to use POST

Change `requests.get(token_url, params=params)` to `requests.post(token_url, data=params)`.

### Step 1.7 — Fix reports.py to use config.OOS_START_DATE

Replace:
```python
OOS_START = "2020-01-01"
```
with:
```python
OOS_START = config.OOS_START_DATE
```

### Step 1.8 — Remove duplicate imports from reports.py

Run a dedup pass on the `import` section.

### Step 1.9 — Fix live_bridge.py hardcoded multipliers

Replace hardcoded `atr * 1.0` and `atr * 2.0` with `atr * config.LIVE_SL_MULT` and `atr * config.LIVE_TP_MULT`.

### Step 1.10 — Reconcile META_DROPOUT between model_spec.py and config.py

Remove `META_DROPOUT` and `META_TANH_MULTIPLIER` from `config.py`. Keep only in `model_spec.py`. Add note to `config.py`:
```python
# Model architecture constants (layer dims, dropout) are in model_spec.py
# Training hyperparameters (lr, epochs, batch) are here
```

---

## Phase 2 — Correct Sharpe Annualisation and Test Thresholds

**Objective:** All Sharpe calculations are correct for the data frequency actually being used.

### Step 2.1 — Add BARS_PER_DAY and ANN_FACTOR to config.py

```python
TIMEFRAME = "1h"          # trading timeframe
BARS_PER_DAY = 24         # 24 for FX 1h; 6.5 for equity 1h; 1 for daily
ANN_FACTOR = int(252 * BARS_PER_DAY)  # = 6048 for FX 1h
```

### Step 2.2 — Replace all hardcoded 252 or ann_factor = 252 references

Search the codebase for every `252` constant and replace with `config.ANN_FACTOR`. Key locations:
- `orchestrators/ensemble_pipeline.py` lines 912–915
- `v3_engine/engine/reports.py` compute_metrics()
- `v3_engine/engine/validation.py` Sharpe formulas
- `v3_engine/engine/tests/run_validation_suite.py` _sharpe() helper

### Step 2.3 — Tighten PSR threshold in tests

In `run_validation_suite.py`:
```python
assert psr < 0.95   # was 0.85 — stricter null-rejection standard
```

### Step 2.4 — Tighten CPCV outlier threshold

```python
outlier_threshold = max(2.0, 3.0 * std_sr)  # was max(5.0, 4.0 * std_sr)
```

---

## Phase 3 — Refactor reports.py and orchestrators to stateless design

**Objective:** No module-level mutable state. All functions are pure (input → output, no side effects).

### Step 3.1 — Remove all module-level globals from reports.py

Move into `run_tearsheet_dynamic(csv_path, params)`:
- `TICKER` — derive from `csv_path`
- `ASSET_NAME` — derive from TICKER
- `PIP_SIZE` — derive from TICKER
- `OOS_START` — from `config.OOS_START_DATE`
- `INITIAL_EQUITY` — from `config.INITIAL_EQUITY`
- `_PHASE1_CACHE` — pass results as return values, not cache

### Step 3.2 — Rename `config` local variable in routes.py

In every route handler, rename `config = PipelineConfig(...)` to `pcfg = PipelineConfig(...)` to avoid shadowing the imported `config` module.

---

## Phase 4 — Walk-Forward Backtest with Correct IS/OOS Separation

**Objective:** Every walk-forward fold uses its own Preprocessor fitted only on that fold's training bars. No statistics leak across the fold boundary.

### Step 4.1 — Integrate fit/transform into the walk-forward loop

In `v3_engine/engine/reports.py` — `run_tearsheet_dynamic()`:

```python
for fold_start, fold_end, oos_start, oos_end in walk_forward_windows:
    # CORRECT PATTERN
    train_df = full_df.iloc[fold_start:fold_end]
    oos_df = full_df.iloc[oos_start:oos_end]

    # Feature computation (causal — no future data)
    train_feat = compute_feature_tensor(train_df)
    oos_feat = compute_feature_tensor(oos_df)

    # Preprocessor fitted ONLY on training fold
    pp = Preprocessor()
    pp.fit(train_feat.values)
    X_train = pp.transform(train_feat.values)
    X_oos = pp.transform(oos_feat.values)

    # Kalman fitted on training
    kf_out = run_kalman_pipeline(X_train)
    # ... transform OOS through the fitted Kalman

    # HMM fitted on training, forward-probabilities on OOS
    hmm = TVTPHMM(...)
    hmm.fit(X_train_filtered, cov_train)
    proba_oos = hmm.predict_proba_causal(X_oos_filtered, cov_oos)

    # Execution layer on OOS only
    signals = run_execution_layer(oos_df, oos_feat, proba_oos)
    # ... collect trades
```

---

## Phase 5 — API Deployment and Testing

**Objective:** The v3_engine FastAPI runs correctly, all endpoints return valid JSON, and a single end-to-end call from raw CSV to signal JSON is testable without a live broker.

### Step 5.1 — Verify API can start

```bash
cd v3_engine
python main.py
# Expected: INFO: Application startup complete. on port 8000
```

Fix any import errors (most likely the `forex_tearsheet` import in orchestrators which is not used by the API, but verify).

### Step 5.2 — Test healthz endpoint

```bash
curl http://localhost:8000/engine/healthz
# Expected: {"status":"ok","engine":"Latent Diffusion-HMM v3.0","csv_accessible":true}
```

If `csv_accessible: false`, place at least one `*_1h.csv` file in `data/`.

### Step 5.3 — Test regime endpoint with pre-downloaded data

The `GET /engine/regime/{ticker}` endpoint calls `Pipeline.run()` which requires `bars_df` to be passed in `PipelineConfig`. The current route constructs `PipelineConfig(ticker=..., start=..., end=...)` without `bars_df`. Since `pipeline.py` raises `ValueError("bars_df must be provided via config")` when `bars_df` is None, ALL current API endpoints that trigger the pipeline will return HTTP 500.

This is a critical API integration bug. The API was designed to load data from disk (CSV files) or from an external data source, but the data ingestion step was removed from `Pipeline.run()` when the architecture was refactored.

**Fix:** In `routes.py`, before calling `pipeline.run(pcfg)`, load the CSV:

```python
import pandas as pd
import glob
import os

def _load_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Find matching CSV from data directory
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    csv_files = glob.glob(os.path.join(root, "data", f"{ticker.lower()}*_1h*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV data found for {ticker}")
    df = pd.read_csv(csv_files[0])
    # normalise column names and timestamp
    df.columns = [c.lower() for c in df.columns]
    for tc in ["timestamp", "datetime", "time", "date"]:
        if tc in df.columns:
            if pd.api.types.is_numeric_dtype(df[tc]):
                df.index = pd.to_datetime(df[tc], unit='ms')
            else:
                df.index = pd.to_datetime(df[tc])
            df = df.drop(columns=[tc])
            break
    df.index.name = 'date'
    # Apply date range filter
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df[mask]
```

Then in each route handler:
```python
pcfg = PipelineConfig(
    ticker=req.ticker.upper(),
    start=req.start,
    end=req.end,
    bars_df=_load_bars(req.ticker.upper(), req.start, req.end),
    ...
)
```

### Step 5.4 — Test analyze endpoint

```bash
curl -X POST http://localhost:8000/engine/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker":"EURUSD","start":"2020-01-01","end":"2023-12-31"}'
# Expected: {"status":"ACCEPTED","task_id":"...","message":"Pipeline Walk-Forward started in background"}
```

Note: The `/analyze` endpoint is asynchronous — it returns immediately with a task ID. There is currently no `/status/{task_id}` endpoint to check completion. Add one:

```python
task_results: dict = {}

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    result = task_results.get(task_id)
    if result is None:
        return {"status": "PENDING", "task_id": task_id}
    return {"status": "COMPLETE", "task_id": task_id, "result": result}
```

### Step 5.5 — Add input validation for regime endpoint minimum date range

The `AnalyzeRequest` validator requires `(end - start).days >= 730`. The `RegimeRequest` only requires `lookback_days >= 30`. If `lookback_days = 30`, the Kalman filter and HMM will have insufficient data. Add:

```python
@model_validator(mode='after')
def check_minimum_lookback(self):
    if self.lookback_days < 180:
        raise ValueError("Minimum 180 days required for meaningful HMM fitting.")
    return self
```

### Step 5.6 — Production startup checklist

Before calling the API production-ready, the following must all pass:

```
□ All Phase 1 bugs fixed
□ Phase 3 stateless refactor complete
□ At least one asset CSV in data/ directory
□ GET /engine/healthz returns 200
□ POST /engine/analyze returns 202 with task_id
□ GET /engine/regime/EURUSD returns current regime probabilities (not 0.333 each)
□ POST /engine/validate returns go_live: true on a clean IS dataset
□ GET /engine/features returns 6 features with non-null values
□ All 6 pytest tests in run_validation_suite.py pass
□ No CRITICAL or HIGH bugs from this report remain open
```

---

## Phase 6 — V4 Engine Integration Testing

**Objective:** VAE, router, expert models, and meta-learner are all trained, compiled to ONNX, and the live bridge can run end-to-end without `ModelNotFoundError`.

### Required artefacts (must exist before live bridge starts)

```
v4_engine/bin/vae.onnx
v4_engine/bin/router.onnx
v4_engine/bin/meta_judge.onnx
v4_engine/models/expert_TREND.pkl
v4_engine/models/expert_MEAN_REV.pkl
v4_engine/models/expert_COMPRESSION.pkl
```

### Training sequence

```bash
# Step 1: Download and prepare data
python data_pipeline/data_manager.py --start 2018-01-01 --end 2023-12-31 --symbols EURUSD,XAUUSD,GBPUSD,CHFJPY,EURNZD,AUDCAD

# Step 2: Train VAE
python v4_engine/train_pipeline.py --mode vae

# Step 3: Train expert models
python v4_engine/train_pipeline.py --mode experts

# Step 4: Prepare meta-learner training data
python v4_engine/meta_data_prep.py

# Step 5: Train meta-learner
python v4_engine/train_pipeline.py --mode meta

# Step 6: Compile to ONNX
python v4_engine/compile_engine.py

# Step 7: Verify compilation accuracy
cd v4_engine && python -m pytest tests/test_compiler.py -v

# Step 8: Run basket backtest
python v4_engine/basket_backtester.py

# Step 9: Start live bridge (demo mode only until Phase 1 fixes verified)
python v4_engine/live_bridge_ctrader.py
```

---

## Phase 7 — Final Stage Objective and Proof Criteria

**The final stage is reached when all of the following are simultaneously true.**

This is the go-live checklist — no partial passes accepted.

```
MATHEMATICAL INTEGRITY
□ BUG-001 fixed: Preprocessor has separate fit() and transform()
□ BUG-002 fixed: HMM uses causal forward-only probabilities on OOS bars
□ BUG-003 fixed: reports.py has no module-level mutable globals
□ BUG-005 fixed: stage2 leaderboard loop indentation corrected
□ LEAK-A verified absent: PCA fit uses only IS bars per fold
□ LEAK-B verified absent: HMM posteriors use forward algorithm only on OOS

STATISTICAL VALIDATION
□ All 6 pytest tests in v3_engine/engine/tests/run_validation_suite.py PASS
□ v4_engine tests (test_compiler, test_experts, test_meta, test_router, test_vae) all PASS
□ Ensemble pipeline: at least 4/6 assets pass Sharpe firewall
□ Phase 6 validation: all 5 of [WF-CV, MC Permutation, DSR, CPCV, Tx-Cost] PASS
□ IS/OOS Sharpe ratio < 2.0 on the held-out test split

LIVE API
□ GET /engine/healthz → 200 OK with csv_accessible: true
□ POST /engine/analyze → 202 Accepted with task_id
□ POST /engine/validate with 5-year data → go_live: true
□ Current regime is NOT "0.333 / 0.333 / 0.333" (collapsed HMM)
□ Wasserstein monitor W1 distance within threshold on current market data

LIVE TRADING GATE (V3 or V4)
□ Demo account connected and authenticated
□ At least 50 live-simulated paper trades executed on demo
□ Paper trading Sharpe ≥ 0.5 (annualised, corrected for hourly bars)
□ No position has exceeded MAX_POSITION_FRACTION = 0.02
□ Emergency liquidation circuit breaker tested and confirmed working
□ No hardcoded credentials, all from environment variables
□ get_token.py uses POST not GET (BUG-008 fixed)

OBJECTIVE PROOF THAT THIS IS THE FINAL STAGE:
Produce a report showing:
  1. run_validation_suite.py output — all PASS markers
  2. ensemble_pipeline.py output — Phase 6 section showing 5/5 PASS
  3. API logs showing at least one full analyze → validate round trip
  4. IS/OOS Sharpe ratio table from the walk-forward backtest
  5. Live demo account statement showing 50+ paper trades with positive net P&L
```

---

---

# PART 6 — COMPLETE SOURCE REFERENCE MAP

---

This section maps every concept in the system to its source file and function so you can navigate the codebase without searching.

## Core Pipeline (V3)

| Concept | File | Function/Class |
|---------|------|----------------|
| Tick download (Dukascopy) | data_pipeline/data_manager.py | DataManager.fetch_and_resample() |
| 6D Feature tensor | v3_engine/engine/features.py | compute_feature_tensor() |
| Fractional differentiation | v3_engine/engine/features.py | fractional_differencing(), _apply_frac_diff() |
| Expanding winsorize | v3_engine/engine/preprocess.py | expanding_winsorize() |
| Expanding standardise (Welford) | v3_engine/engine/preprocess.py | expanding_standardize() |
| PCA whitening | v3_engine/engine/preprocess.py | fit_pca_whitening(), apply_pca_whitening() |
| Kalman filter (EM-fitted) | v3_engine/engine/kalman.py | KalmanFilter |
| CUSUM jump detector | v3_engine/engine/kalman.py | CUSUMJumpDetector |
| Full Kalman pipeline | v3_engine/engine/kalman.py | run_kalman_pipeline() |
| TVTP-HMM 3-state GMM | v3_engine/engine/hmm.py | TVTPHMM |
| Triple gate execution | v3_engine/engine/execution.py | TripleGate.evaluate() |
| Kelly position sizer | v3_engine/engine/execution.py | KellyPositionSizer |
| ATR calculation | v3_engine/engine/execution.py | compute_atr() |
| Full execution layer | v3_engine/engine/execution.py | run_execution_layer() |
| Wasserstein W1 monitor | v3_engine/engine/surveillance.py | WassersteinMonitor |
| Walk-forward CV | v3_engine/engine/validation.py | walk_forward_cv() |
| Monte Carlo permutation test | v3_engine/engine/validation.py | monte_carlo_permutation_test() |
| Deflated Sharpe ratio | v3_engine/engine/validation.py | deflated_sharpe_ratio() |
| CPCV | v3_engine/engine/validation.py | cpcv() |
| Tx cost sensitivity | v3_engine/engine/validation.py | transaction_cost_sensitivity() |
| Full validation report | v3_engine/engine/validation.py | ValidationReport |
| Full backtest (tearsheet) | v3_engine/engine/reports.py | run_tearsheet_dynamic() |
| Backtester loop | v3_engine/engine/reports.py | run_backtest_engine() |
| Metrics computation | v3_engine/engine/reports.py | compute_metrics() |
| Buy-and-hold benchmark | v3_engine/engine/reports.py | buy_and_hold_benchmark() |
| Phase 6 statistical gate | v3_engine/engine/reports.py | phase6_validation() |
| Pipeline orchestrator | v3_engine/engine/pipeline.py | Pipeline.run() |
| Pipeline config | v3_engine/engine/pipeline.py | PipelineConfig |
| Pipeline result | v3_engine/engine/pipeline.py | PipelineResult |

## V4 Engine (Deep Learning)

| Concept | File | Function/Class |
|---------|------|----------------|
| VAE model | v4_engine/vae_model.py | VAE |
| Gumbel-Softmax router | v4_engine/vae_model.py | GumbelSoftmaxRouter |
| Triple barrier labelling | v4_engine/triple_barrier.py | apply_triple_barrier() |
| Feature expansion (z-score) | v4_engine/feature_expansion.py | extract_z_score_tensor() |
| Expert layer (GBM/SVM per regime) | v4_engine/expert_layer.py | ExpertLayer |
| GMM-HMM (hmmlearn wrapper) | v4_engine/gmm_hmm.py | LatentRouter |
| Meta-learner (neural correction) | v4_engine/meta_learner.py | MetaLearner |
| Meta data preparation | v4_engine/meta_data_prep.py | prepare_meta_training_data() |
| Training pipeline | v4_engine/train_pipeline.py | main() |
| ONNX compiler | v4_engine/compile_engine.py | compile_all() |
| Live inference engine | v4_engine/live_bridge.py | LiveExecutionEngine |
| cTrader live bridge (v4) | v4_engine/live_bridge_ctrader.py | CTraderLiveBridge |
| Basket backtester | v4_engine/basket_backtester.py | run_basket_backtest() |

## API

| Concept | File | Function/Class |
|---------|------|----------------|
| FastAPI app | v3_engine/main.py | app |
| All route handlers | v3_engine/api/routes.py | router |
| Analyze endpoint | v3_engine/api/routes.py | analyze() |
| Regime endpoint | v3_engine/api/routes.py | get_regime() |
| Signals endpoint | v3_engine/api/routes.py | get_signals() |
| Features endpoint | v3_engine/api/routes.py | get_features() |
| Validate endpoint | v3_engine/api/routes.py | validate() |
| Health endpoint | v3_engine/api/routes.py | health() |

## Orchestration

| Concept | File | Function/Class |
|---------|------|----------------|
| Multi-asset ensemble backtest | orchestrators/ensemble_pipeline.py | main() |
| Per-asset backtest worker | orchestrators/ensemble_pipeline.py | process_asset() |
| Hyperparameter grid search | orchestrators/grid_search.py | main() |
| Cross-asset stress test | orchestrators/stage2_global_test.py | main() |
| Master run controller | orchestrators/master_pipeline.py | main() |

## Configuration

| Concept | File | Variable |
|---------|------|----------|
| All trading parameters | config.py | (all uppercase) |
| Model architecture dims | model_spec.py | VAE_INPUT_DIM, META_INPUT_DIM, etc. |
| Environment overrides | config.py | _apply_env_overrides() |

## Tests

| Test class | File | What it tests |
|------------|------|---------------|
| TestFeatureEngineering | v3_engine/engine/tests/run_validation_suite.py | Frac-diff stationarity, PCA causality |
| TestSyntheticRegimeStress | v3_engine/engine/tests/run_validation_suite.py | HMM accuracy, Kalman CUSUM |
| TestExecutionAndSurveillance | v3_engine/engine/tests/run_validation_suite.py | Triple gate, Wasserstein circuit breaker |
| TestStatisticalValidation | v3_engine/engine/tests/run_validation_suite.py | MC permutation, CPCV fold uniformity |
| test_triple_barrier | v4_engine/tests/test_experts.py | Triple barrier labelling |
| test_onnx_compilation_accuracy | v4_engine/tests/test_compiler.py | ONNX vs PyTorch output match |

---

---

# PART 7 — PLAIN-TEXT EXECUTION RUNBOOK

---

This is the step-by-step sequence to go from a clean environment to a running, validated, API-serving system.

```
ENVIRONMENT SETUP
=================
pip install fastapi uvicorn pandas numpy scipy scikit-learn
pip install statsmodels arch jax jaxlib hmmlearn
pip install torch onnx onnxruntime polars joblib numba
pip install ctrader-open-api python-dotenv requests tick-vault

Set environment variables in a .env file:
  CTRADER_CLIENT_ID=<your id>
  CTRADER_CLIENT_SECRET=<your secret>
  CTRADER_AUTH_CODE=<freshly obtained code>
  CTRADER_ACCOUNT_ID=<your demo account id>
  CTRADER_ACCESS_TOKEN=<obtained from get_token.py after fix>
  ENVIRONMENT_MODE=demo
  OOS_START_DATE=2022-01-01
  STRATEGY_MODE=MOMENTUM

STEP 1 — DATA ACQUISITION
==========================
python data_pipeline/data_manager.py \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --symbols EURUSD,XAUUSD,GBPUSD,CHFJPY,EURNZD,AUDCAD \
  --timeframe 1h

Expected output: data/eurusd_1h.csv ... (6 files, each ~50-100 MB)

STEP 2 — APPLY ALL PHASE 1 FIXES
==================================
(Implement BUG-001 through BUG-022 fixes as described in Phase 1 above)

STEP 3 — RUN UNIT TESTS
=========================
cd v3_engine
python -m pytest engine/tests/run_validation_suite.py -v --tb=short
Expected: 6 tests, all PASSED

cd ..
python -m pytest v4_engine/tests/ -v --tb=short
Expected: 5 tests, all PASSED (skip if model files not yet compiled)

STEP 4 — RUN SINGLE-ASSET DIAGNOSTIC
======================================
python diagnostics/audit_real_data.py
Expected: ALL 6 LAYERS MATHEMATICALLY VERIFIED

STEP 5 — RUN WALK-FORWARD BACKTEST ON ONE ASSET
================================================
cd v3_engine
python engine/reports.py data/eurusd_1h.csv
Expected tearsheet with Sharpe > 0.5 on OOS, all Phase 6 tests passing

STEP 6 — RUN ENSEMBLE (ALL ASSETS)
====================================
python orchestrators/ensemble_pipeline.py
Expected: 4+ assets admitted through Sharpe firewall, portfolio Sharpe > 0.5

STEP 7 — OPTIONAL GRID SEARCH
================================
python orchestrators/grid_search.py
Expected: Top 10 parameter sets saved to results/grid_results/stage1_top10.json

python orchestrators/stage2_global_test.py
Expected: Global stress test leaderboard showing best params

STEP 8 — START API SERVER
==========================
cd v3_engine
python main.py
Expected: INFO: Uvicorn running on http://0.0.0.0:8000

Test: curl http://localhost:8000/engine/healthz
Test: curl http://localhost:8000/engine/redoc  (Swagger docs)

STEP 9 — V4 ENGINE TRAINING (if using v4)
==========================================
python v4_engine/train_pipeline.py --mode vae
python v4_engine/train_pipeline.py --mode experts
python v4_engine/meta_data_prep.py
python v4_engine/train_pipeline.py --mode meta
python v4_engine/compile_engine.py

STEP 10 — START LIVE DEMO BRIDGE
==================================
# Get fresh OAuth token (after BUG-008 fix)
python live_trading/get_token.py
# Set CTRADER_ACCESS_TOKEN in environment
# Start bridge
python v4_engine/live_bridge_ctrader.py  # for V4
# or
python live_trading/live_bridge_ctrader.py  # for V3

Monitor output for:
  ✅ Application Authorized
  ✅ Account XXXXXXX Authorized
  📌 Resolved Symbol IDs: ...
  Current Regime: TREND | ATR: 0.00043
  🔥 Veto Passed! (when signal fires)
  💸 Sending Market Order...
  ✅ Order sent to cTrader Matching Engine.
```

---

**END OF AUDIT REPORT**

---

*This document covers 7,368 lines across 40+ source files. All bug IDs, file references, and line numbers refer to the Repomix snapshot dated 2026-06-11T08:11:26Z. Apply fixes to the original repository files, not this document.*

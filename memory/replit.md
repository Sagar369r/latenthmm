# Latent Diffusion-HMM Trading Engine v3.0

A full implementation of the quantitative trading pipeline from the Latent Diffusion-HMM Architecture v3.0 spec. Implements all 6 layers: Dollar-Volume bar sampling → 6D feature tensor → Kalman denoising → TVTP-HMM regime classification → Triple Gate execution → Wasserstein distribution surveillance.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the Node.js API (port 8080)
- `cd artifacts/python-engine && python main.py` — run the Python engine (port 8000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- **Python Engine**: FastAPI + uvicorn, port 8000, served at `/engine`
- **Node.js API**: Express 5, port 8080, served at `/api`
- DB: PostgreSQL + Drizzle ORM
- Python scientific stack: numpy, scipy, pandas, statsmodels, scikit-learn, hmmlearn, arch, POT (Optimal Transport)
- Data: Yahoo Finance (yfinance)
- Build: esbuild (CJS bundle for Node.js)

## Where things live

- `artifacts/python-engine/` — full Python signal engine
  - `engine/data.py` — Layer 1: Dollar-Volume bars, MMMS fractional differentiation
  - `engine/features.py` — Layer 2: 6D feature tensor (vt, mvt, qt, σt, ρt, Ht)
  - `engine/preprocess.py` — Winsorise → standardise → Robust PCA whitening
  - `engine/kalman.py` — Layer 3: Kalman filter (EM-fitted) + CUSUM jump detector
  - `engine/hmm.py` — Layer 4: TVTP-HMM, 3 states, K=2 GMM, Baum-Welch + Viterbi
  - `engine/execution.py` — Layer 5: Triple Gate, Half-Kelly sizing, ATR SL/TP
  - `engine/surveillance.py` — Wasserstein W1 distribution monitor (Sinkhorn)
  - `engine/validation.py` — Layer 6: WF-CV, MC Permutation, DSR, CPCV, Tx Cost
  - `engine/pipeline.py` — Full pipeline orchestrator
  - `api/routes.py` — FastAPI route handlers
  - `main.py` — FastAPI app entry point
- `artifacts/api-server/` — Node.js Express API (health + extensible)
- `lib/api-spec/openapi.yaml` — API contract (OpenAPI)

## API Endpoints (Python Engine at `/engine`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/engine/healthz` | Health check |
| POST | `/engine/analyze` | Full pipeline run (all 6 layers) |
| GET | `/engine/regime/{ticker}` | Quick regime classification |
| POST | `/engine/signals` | Trade signals (Triple Gate output) |
| POST | `/engine/features` | 6D feature tensor for a ticker |
| POST | `/engine/validate` | Full statistical validation suite |
| GET | `/engine/docs-summary` | Architecture summary |
| GET | `/engine/redoc` | Interactive API docs (ReDoc) |

### Example: analyze SPY
```bash
curl -X POST /engine/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker":"SPY","start":"2019-01-01","end":"2024-01-01"}'
```

### Example: get current regime
```bash
curl /engine/regime/SPY?lookback_days=365
```

## Architecture

### Layer 1 — Data Ingestion
- **Dollar-Volume Bars**: triggers when Σ(P×V) ≥ D* (EMA daily DV / target_bars_per_day)
- **MMMS Fractional Differentiation**: d* = min{d: ADF p-value < 0.05}, expanding window, re-estimated every 252 bars

### Layer 2 — 6D Observation Tensor
- **vt** — Volatility Proximity = (P_t - Donchian_mid) / (2.1 × ATR₁₄), clipped [-1, +1]
- **mvt** — Momentum Velocity = log(P_t/P_{t-n}) / σ₅₀, n from ACF peak of |returns|
- **qt** — Volume Delta Ratio = V_t / EMA₂₀(V_t)
- **σt** — Volatility Regime Ratio = RV_t / GARCH(1,1)_t
- **ρt** — Autocorrelation Signal = Corr(r_t, r_{t-1}) on 30-bar rolling window
- **Ht** — DFA Hurst Exponent (replaces biased R/S estimator)
- Preprocessing: expanding winsorise (1/99 pct) → expanding z-score → Robust PCA whitening

### Layer 3 — Kalman Filter + CUSUM
- Linear Gaussian state-space model; Q, R estimated via EM on 504-bar window
- CUSUM jump detector: g_t = max(0, g_{t-1} + |νt|/σν - κ), κ=0.5, h=5.0
- Jump trigger resets Kalman to diffuse prior

### Layer 4 — TVTP-HMM
- 3 states: TREND (0), MEAN_REV (1), STRESS (2)
- K=2 GMM emission per state with Dirichlet prior
- TVTP conditioned on [σt, ρt] only: 18 β params (vs 594 in v2)
- Baum-Welch EM with L2 regularisation (λA=0.1, λμ=0.01, λβ=0.05)
- Viterbi decoding for max-likelihood state sequence

### Layer 5 — Execution + Surveillance
- **Triple Gate**: P(TREND) > 0.65 AND |mvt| > 1.0 AND qt > 1.3
- **Half-Kelly**: ft = 0.5 × f* × P(TREND), hard cap 2% equity
- **ATR SL/TP**: SL = Entry ∓ ATR₁₄ × 1.5, TP = Entry ± ATR₁₄ × 3.0
- **Wasserstein Monitor**: W1(live 50-bar, training) > 0.3σ → reduce positions to 25%

### Layer 6 — Statistical Validation
- WF-CV: expanding window, IS/OOS Sharpe ratio < 2.0
- Monte Carlo Permutation: actual Sharpe > 95th pct of 10k permuted series
- Deflated Sharpe Ratio (DSR): corrects for selection bias across M trials
- CPCV: MinSR > 0.3, PSR > 0.95
- Tx Cost Sensitivity: Sharpe > 0.6 at 2 bps per side

## Architecture Decisions

- **Python over TypeScript for the engine**: The pipeline requires GARCH, HMM, DFA, PCA, Sinkhorn — all have mature Python scientific libraries. Adding a Python FastAPI service alongside Node.js was the correct tradeoff.
- **TVTP-HMM implemented from scratch**: `hmmlearn` doesn't support time-varying transition probabilities. Built Baum-Welch + Viterbi manually with TVTP conditioning on [σt, ρt].
- **Kalman filter replaces neural DDPM**: Zero deep-learning dependency; Q and R estimated via EM. This is the spec's primary anti-overfit change from v2.
- **Expanding-window statistics throughout**: Winsorisation, standardisation, fractional diff d*, and GARCH all use expanding windows — never in-sample statistics.
- **Port 8000 for Python engine, 8080 for Node.js**: Both routed through the shared proxy via artifact.toml service blocks.

## Product

Users can submit any liquid ticker (equity, ETF, futures) and date range to receive:
- Current market regime (TREND / MEAN_REV / STRESS) with state probabilities
- The full 6D feature tensor with economic interpretation of each feature
- Trade signals satisfying all three entry gates, with position sizing and ATR-based stops
- Wasserstein drift monitoring status
- Full statistical validation results before live capital deployment

## User Preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- The GARCH(1,1) fitting requires at least 252 bars of history; sigma_t is NaN before that
- DFA Hurst requires at least 100 bars; ht is NaN in early bars
- Fractional differentiation requires 500 bars before d* estimation begins; earlier bars use d=0.40 default
- Regime probabilities will be roughly equal (~33%) until the HMM has enough data to separate states
- Signals require P(TREND) > 0.65 — on mixed/choppy markets this deliberately produces few or no signals
- Yahoo Finance rate limits: avoid batching too many tickers simultaneously
- The validation suite (POST /engine/validate) is slow due to Monte Carlo — use n_mc_permutations=200 for quick checks

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
- Python engine lives entirely in `artifacts/python-engine/`; it is standalone and not a pnpm package

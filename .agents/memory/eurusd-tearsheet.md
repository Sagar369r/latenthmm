---
name: EUR/USD forex tear sheet results
description: Walk-forward Make-or-Break verdict for EURUSD daily bars, key diagnostics, and next-iteration recommendations.
---

## Configuration
- Data: EURUSD=X daily bars 2016-2024 (2344 bars)
- WF windows: 504-bar train / 126-bar OOS (15 blocks)
- Stops: SL=3.0×ATR, TP=5.0×ATR, trail trigger=2.0×ATR
- Position sizing: ATR-risk-based — `units = (frac × equity) / (atr × ATR_SL_MULT)`
- qt proxy: range-ratio = (H-L)/EMA20(H-L) — forex has no volume

## Results
| Metric | Value | Threshold |
|---|---|---|
| OOS Sharpe | 0.26 | BANG>1.2, WASTE<0.3 |
| Profit Factor | 1.28 | BANG>1.5, WASTE<1.0 |
| Max Drawdown | 4.5% | BANG<15%, WASTE>25% |
| Win Rate | 47.5% | — |
| Total Return | +7.9% gross | — |
| Completed trades | 40 | — |

**VERDICT: WASTE** — Sharpe 0.26 < 0.3 waste threshold.

## Layer 6 validation
- 6.1 WF-CV: PASS ✓ (OOS SR=0.27, IS/OOS=1.40 — no overfitting)
- 6.2 MC Permutation: FAIL ✗ (actual SR=-0.19 in last 30%, p=0.75)
- 6.3 DSR: FAIL ✗ (z=-51, expected max SR for 15 trials >> observed 0.26)
- 6.4 CPCV: FAIL ✗ (MinSR=-0.60 driven by EUR Recovery 2023 losing folds)
- 6.5 Tx Cost: FAIL ✗ (net SR=0.21 < 0.6 realistic minimum)

## Key diagnostics

### What works
- WF-CV passes: IS/OOS=1.40 — the HMM is NOT overfitted. The regime signal is real.
- Short side wins: short win rate 56% vs long 41%. EUR/USD trends down more cleanly.
- COVID Crash and 2024 both profitable (PF=9.54 and 2.12 respectively).

### What fails
1. **Only 40 trades in 7 years** (5.7/year) — statistically insufficient for DSR/CPCV.
   Root cause: range-ratio gate passes only 16.6% of WF bars.
2. **30/40 exits via stop-loss** — TP at 5×ATR rarely reached on daily bars.
   EUR/USD daily ATR ~50 pips, TP = 250 pips; major trends need weeks to deliver.
3. **EUR Recovery 2023 losing period** (PF=0.57) — 8 trades, mostly stopped out.
   ECB rate peak / end of trend regime → HMM correctly identified stress but still
   fired signals on momentum reversals.
4. **DSR z=-51**: with 15 WF blocks, expected max Sharpe by chance >> 0.26.
   Need Sharpe >0.8 to survive DSR correction at 15 trials.

## Next iterations to investigate

### Path A — More signals, wider gate
- Lower range-ratio threshold from 1.3× to 1.1× (volume(range) gate: 17% → ~30%)
- Expected: 80-100 trades over 7 years, better statistical power
- Risk: more false signals in ranging markets

### Path B — Tighter SL, smaller TP ratio
- SL=2.0×ATR, TP=3.5×ATR (keeps 1.75× reward:risk but exits earlier)
- More TP hits, lower time-in-trade, reduces 2023 exposure
- Expected: PF improves but Sharpe may not change much

### Path C — Short-only EUR/USD
- Short win rate 56% vs long 41%. EUR has structural downside vs USD in rate-hike cycles.
- Run short-only backtest to see if the directional asymmetry is actionable.

### Path D — Different instrument
- The HMM regime detector itself passes WF-CV. Try: GBP/USD, USD/JPY, EUR/GBP.
- Or: Gold (XAU/USD) which has clearer trending regimes.

## Position sizing note
Correct formula for forex backtests:
```python
risk_amount = frac * current_equity      # $ at risk per trade
sl_distance = atr * ATR_SL_MULT          # price distance to SL
units = risk_amount / sl_distance        # EUR units
```
Using `units = notional / price` (dollar-notional) gives positions ~70× too small.

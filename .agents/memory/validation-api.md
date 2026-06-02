---
name: Validation module API
description: Correct function signatures for engine/validation.py — all five Layer 6 tests.
---

## The problem
engine/validation.py functions do NOT accept a raw returns array as the sole argument.
Passing `walk_forward_cv(returns_array)` silently fails and returns SR=0.00.

## Correct signatures

| Function | Signature | simulate_fn contract |
|---|---|---|
| `walk_forward_cv` | `(simulate_fn, T, train_bars, test_bars, min_bars)` | `(ts, te, ss, se) → (is_rets, oos_rets)` |
| `monte_carlo_permutation_test` | `(daily_returns, simulate_fn, n_permutations, train_split)` | `(shuffled_returns) → oos_sharpe` |
| `deflated_sharpe_ratio` | `(returns, n_trials, sr_benchmark)` | — direct array |
| `cpcv` | `(simulate_fn, T, n_folds, test_folds, embargo_bars)` | `(train_idx, test_idx) → (is_rets, oos_rets)` |
| `transaction_cost_sensitivity` | `(gross_returns, n_trades, T)` | — direct array |

## Correct simulate_fn wrappers for pre-computed equity returns

```python
# 6.1 WF-CV
def wf_fn(ts, te, ss, se):
    return eq_rets[ts:te], eq_rets[ss:se]

# 6.2 MC — shuffled returns, OOS = last 30%
def mc_fn(shuffled):
    oos_start = int(len(shuffled) * 0.7)
    return _sharpe_ratio(shuffled[oos_start:])

# 6.4 CPCV
def cpcv_fn(train_idx, test_idx):
    return eq_rets[train_idx], eq_rets[test_idx]
```

**Why:** engine/validation.py was designed for a full pipeline re-run per window,
not for slicing pre-computed returns. The tearsheet uses slicing as a valid
approximation since equity returns already capture realised P&L bar-by-bar.

**How to apply:** Any new tearsheet or backtest script that calls Layer 6 validation
must use these wrappers. Never pass a bare returns array to wf_cv, mc, or cpcv.

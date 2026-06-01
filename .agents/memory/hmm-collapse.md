---
name: TVTP-HMM mode collapse on long training windows
description: The TVTP-HMM collapses to a single absorbing state when trained on >~1000 bars of real SPY data. Walk-forward with 504-bar windows is the fix.
---

## The Rule
Never fit TVTPHMM on more than ~1000 bars of real market data in a single call. Use a rolling walk-forward scheme with ≤504 bars per training window.

**Why:** The Baum-Welch EM algorithm with K=2 GMM emissions and 6D whitened features collapses to a single-state solution (all probability mass to one state) when the training set is large. With 2512 bars (10 years), it reliably collapses to 100% MEAN_REV; with 499 bars (2 years), it collapses to 100% TREND. The direction of collapse depends on data, but in all cases Viterbi returns a single constant state. Root cause: GMM mode collapse — one Gaussian grows to cover all data, others shrink to near zero.

**How to apply:** In `oos_tearsheet.py`, Phase 2 uses a walk-forward loop:
- WF_TRAIN_BARS = 504 (2-year window)
- WF_OOS_BARS = 126 (6-month OOS block)
- Fit TVTPHMM(n_states=3, n_gmm=2, n_iter=30) on each training block
- Predict on the corresponding OOS block only
- Concatenate OOS predictions to get full OOS regime_proba

This produces meaningful regime variation: TREND/MEAN_REV/STRESS all observed across the 10-block OOS period (2020-2024).

**Do not fix by:** increasing n_iter, changing random_seed, or calling Pipeline.run() on the full date range — these all have the same collapse issue on long windows.

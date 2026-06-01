---
name: TVTP-HMM Custom Implementation
description: hmmlearn does not support time-varying transition probabilities; TVTP-HMM is fully implemented from scratch.
---

**Why custom**: `hmmlearn.GaussianHMM` uses a fixed transition matrix. The spec requires TVTP conditioned on [σt, ρt], which demands a custom Baum-Welch.

**Implementation**: `engine/hmm.py` — `TVTPHMM` class with:
- `_compute_transition_matrix(cov_t)` — softmax(β^T [σt, ρt]) per source state
- `_forward_backward(X, covariates)` — scaled alpha-beta with time-varying A_seq
- `_update_gmm_emissions(X, gamma)` — GMM M-step with Dirichlet prior
- `_update_tvtp_beta(covariates, gamma, xi)` — gradient ascent on TVTP log-likelihood
- `decode(X, covariates)` — log-space Viterbi

**How to apply**: When adding HMM features, always pass covariates=(σt, ρt) arrays of same length as X. NaN rows in X are handled by substituting uniform emission.

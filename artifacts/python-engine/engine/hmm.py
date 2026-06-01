"""
Layer 4: TVTP-HMM (Time-Varying Transition Probability HMM)

Architecture:
  - 3 hidden states: TREND (0), MEAN-REV (1), STRESS (2)
  - GMM emission model: K=2 components per state
  - TVTP conditioned on [σ_t, ρ_t] only (18 β params, not 594)
  - Baum-Welch EM with L2 regularisation
  - Viterbi decoding

Spec parameters (fixed, not optimised):
  K = 3 states, K_gmm = 2 GMM components
  λ_A = 0.1, λ_μ = 0.01, λ_β = 0.05
  TVTP: P(S_t=j | S_{t-1}=i, σ_t, ρ_t) = softmax(β_{ij}^T [σ_t, ρ_t])
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

STATE_NAMES = {0: "TREND", 1: "MEAN_REV", 2: "STRESS"}
STATE_LABELS = ["TREND", "MEAN_REV", "STRESS"]

# Regularisation constants (from spec §7.2)
LAMBDA_A = 0.1
LAMBDA_MU = 0.01
LAMBDA_BETA = 0.05
RIDGE_SIGMA = 0.01   # Σ_sk ← Σ_sk + λI


class TVTPHMM:
    """
    Time-Varying Transition Probability HMM with GMM emissions.

    Parameters
    ----------
    n_states : int     Number of hidden states (3)
    n_gmm    : int     GMM components per state (2)
    n_iter   : int     Baum-Welch iterations
    """

    def __init__(
        self,
        n_states: int = 3,
        n_gmm: int = 2,
        n_iter: int = 50,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_gmm = n_gmm
        self.n_iter = n_iter
        self.random_state = random_state
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def _init_params(self, X: np.ndarray) -> None:
        S, K, D = self.n_states, self.n_gmm, X.shape[1]
        np.random.seed(self.random_state)

        # k-means warm start for state assignment
        km = KMeans(n_clusters=S, n_init=10, random_state=self.random_state)
        clean = X[~np.any(np.isnan(X), axis=1)]
        labels = km.fit_predict(clean) if len(clean) >= S else np.zeros(len(clean), int)

        # Initial state distribution
        self.pi0_ = np.ones(S) / S

        # GMM mixing weights per state
        self.pi_gmm_ = np.ones((S, K)) / K           # (S, K)

        # GMM means: init around cluster centres
        self.mu_ = np.zeros((S, K, D))               # (S, K, D)
        self.sigma_ = np.array(
            [np.eye(D) * 0.5 for _ in range(S * K)]
        ).reshape(S, K, D, D)                        # (S, K, D, D)

        centers = km.cluster_centers_ if len(clean) >= S else np.zeros((S, D))
        for s in range(S):
            idx = np.where(labels == s)[0]
            if len(idx) >= K:
                half = len(idx) // K
                for k in range(K):
                    slice_ = clean[idx[k * half:(k + 1) * half]]
                    self.mu_[s, k] = slice_.mean(axis=0) if len(slice_) > 0 else centers[s]
            else:
                for k in range(K):
                    self.mu_[s, k] = centers[s] + np.random.randn(D) * 0.1

        # TVTP β parameters: (S, S, n_cov) — conditioned on [σ_t, ρ_t]
        self.n_cov = 2
        self.beta_ = np.zeros((S, S, self.n_cov))   # (S, S, 2)

    # ------------------------------------------------------------------ #
    # TVTP computation                                                     #
    # ------------------------------------------------------------------ #

    def _compute_transition_matrix(self, cov_t: np.ndarray) -> np.ndarray:
        """
        Compute S×S transition matrix from softmax of β^T [σ_t, ρ_t].

        cov_t shape: (n_cov,)
        Returns: (S, S) row-stochastic matrix
        """
        S = self.n_states
        A = np.zeros((S, S))
        for i in range(S):
            logits = self.beta_[i] @ cov_t  # (S,)
            logits -= logits.max()           # numerical stability
            exp_l = np.exp(logits)
            A[i] = exp_l / (exp_l.sum() + 1e-300)
        return A

    def _compute_all_transitions(self, covariates: np.ndarray) -> np.ndarray:
        """covariates: (T, n_cov). Returns (T, S, S)."""
        T = len(covariates)
        A = np.zeros((T, self.n_states, self.n_states))
        for t in range(T):
            A[t] = self._compute_transition_matrix(covariates[t])
        return A

    # ------------------------------------------------------------------ #
    # Emission probabilities                                               #
    # ------------------------------------------------------------------ #

    def _emission_prob(self, x: np.ndarray, s: int) -> float:
        """P(x | state s) = Σ_k π_{sk} N(x; μ_{sk}, Σ_{sk})"""
        prob = 0.0
        for k in range(self.n_gmm):
            try:
                prob += float(self.pi_gmm_[s, k]) * float(
                    multivariate_normal.pdf(
                        x, mean=self.mu_[s, k],
                        cov=self.sigma_[s, k] + np.eye(x.shape[0]) * RIDGE_SIGMA,
                    )
                )
            except Exception:
                pass
        return max(prob, 1e-300)

    def _compute_emission_matrix(self, X: np.ndarray) -> np.ndarray:
        """Returns (T, S) emission probability matrix."""
        T = X.shape[0]
        B = np.zeros((T, self.n_states))
        for t in range(T):
            if np.any(np.isnan(X[t])):
                B[t] = 1.0 / self.n_states   # uniform for missing
            else:
                for s in range(self.n_states):
                    B[t, s] = self._emission_prob(X[t], s)
        return B

    # ------------------------------------------------------------------ #
    # Forward–Backward (scaled)                                            #
    # ------------------------------------------------------------------ #

    def _forward_backward(
        self, X: np.ndarray, covariates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Scaled forward-backward algorithm with TVTP transitions.

        Returns
        -------
        gamma : (T, S) state marginals
        xi    : (T-1, S, S) joint state marginals
        log_likelihood : float
        """
        T, D = X.shape
        S = self.n_states
        B = self._compute_emission_matrix(X)
        A_seq = self._compute_all_transitions(covariates)

        # Forward (scaled)
        alpha = np.zeros((T, S))
        scale = np.zeros(T)

        alpha[0] = self.pi0_ * B[0]
        scale[0] = alpha[0].sum()
        if scale[0] < 1e-300:
            scale[0] = 1e-300
        alpha[0] /= scale[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A_seq[t - 1]) * B[t]
            scale[t] = alpha[t].sum()
            if scale[t] < 1e-300:
                scale[t] = 1e-300
            alpha[t] /= scale[t]

        log_likelihood = float(np.sum(np.log(scale + 1e-300)))

        # Backward (scaled)
        beta_bw = np.ones((T, S))
        for t in range(T - 2, -1, -1):
            beta_bw[t] = A_seq[t] @ (B[t + 1] * beta_bw[t + 1])
            s = beta_bw[t].sum()
            if s > 0:
                beta_bw[t] /= s

        # Gamma
        gamma = alpha * beta_bw
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-300, 1.0, row_sums)
        gamma /= row_sums

        # Xi
        xi = np.zeros((T - 1, S, S))
        for t in range(T - 1):
            num = (alpha[t:t+1].T * A_seq[t]) * (B[t + 1] * beta_bw[t + 1])
            denom = num.sum()
            xi[t] = num / max(denom, 1e-300)

        return gamma, xi, log_likelihood

    # ------------------------------------------------------------------ #
    # M-step helpers                                                       #
    # ------------------------------------------------------------------ #

    def _update_gmm_emissions(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """Update GMM (π, μ, Σ) for each state."""
        T, D = X.shape
        S, K = self.n_states, self.n_gmm

        clean_mask = ~np.any(np.isnan(X), axis=1)

        for s in range(S):
            # Component responsibilities r_{sk,t}
            r = np.zeros((T, K))
            for k in range(K):
                try:
                    r[:, k] = self.pi_gmm_[s, k] * multivariate_normal.pdf(
                        X, mean=self.mu_[s, k],
                        cov=self.sigma_[s, k] + np.eye(D) * RIDGE_SIGMA,
                    )
                except Exception:
                    r[:, k] = 1.0 / K
            r[~clean_mask] = 1.0 / K

            r_sum = r.sum(axis=1, keepdims=True)
            r_sum = np.where(r_sum < 1e-300, 1.0, r_sum)
            r /= r_sum

            for k in range(K):
                w = gamma[:, s] * r[:, k] * clean_mask.astype(float)
                w_sum = w.sum()
                if w_sum < 1e-10:
                    continue

                # L2-regularised mean update
                self.mu_[s, k] = (w @ X) / (w_sum + LAMBDA_MU)

                # Covariance with ridge
                diff = X - self.mu_[s, k]
                sigma_new = (w[:, None, None] * (
                    diff[:, :, None] * diff[:, None, :]
                )).sum(axis=0) / (w_sum + 1e-10)
                self.sigma_[s, k] = sigma_new + np.eye(D) * RIDGE_SIGMA

            # Renormalise mixing weights (with Dirichlet-like clipping)
            state_weight = gamma[:, s].sum()
            for k in range(K):
                w = gamma[:, s] * r[:, k] * clean_mask.astype(float)
                self.pi_gmm_[s, k] = max(w.sum() / max(state_weight, 1e-10), 1e-6)
            self.pi_gmm_[s] /= self.pi_gmm_[s].sum()

    def _update_tvtp_beta(
        self, covariates: np.ndarray, gamma: np.ndarray, xi: np.ndarray
    ) -> None:
        """
        Update TVTP β via gradient ascent (multinomial logit for each source state).
        L2-regularised with λ_β = 0.05.
        """
        T, S = gamma.shape
        lr = 0.05
        inner_steps = 15

        for _ in range(inner_steps):
            A_seq = self._compute_all_transitions(covariates)
            grad_beta = np.zeros_like(self.beta_)

            for t in range(T - 1):
                cov_t = covariates[t]
                g_s = gamma[t]
                for i in range(S):
                    if g_s[i] < 1e-10:
                        continue
                    for j in range(S):
                        # d log p / d β_{ij} contribution
                        delta_ij = xi[t, i, j] - g_s[i] * A_seq[t, i, j]
                        grad_beta[i, j] += delta_ij * cov_t

            # L2 regularisation
            self.beta_ += lr * (grad_beta - LAMBDA_BETA * self.beta_)

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, covariates: np.ndarray) -> "TVTPHMM":
        """
        Fit TVTP-HMM via Baum-Welch EM.

        Parameters
        ----------
        X          : (T, D) whitened, denoised observations
        covariates : (T, 2) array of [σ_t, ρ_t] for TVTP conditioning
        """
        self._init_params(X)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            # E-step
            gamma, xi, ll = self._forward_backward(X, covariates)

            # M-step
            self.pi0_ = gamma[0]
            self._update_gmm_emissions(X, gamma)
            self._update_tvtp_beta(covariates, gamma, xi)

            # Convergence check
            if abs(ll - prev_ll) < 1e-4 and iteration > 5:
                break
            prev_ll = ll

        self._fitted = True
        self._train_ll = prev_ll
        return self

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict_proba(
        self, X: np.ndarray, covariates: np.ndarray
    ) -> np.ndarray:
        """
        Return state probability sequence P_t = [P(S_t=0), P(S_t=1), P(S_t=2)]^T.

        Returns (T, 3) array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        gamma, _, _ = self._forward_backward(X, covariates)
        return gamma

    def decode(self, X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding — maximum-likelihood hidden state sequence.
        Returns (T,) integer array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        T, D = X.shape
        S = self.n_states
        B = self._compute_emission_matrix(X)
        A_seq = self._compute_all_transitions(covariates)

        # Log-space Viterbi
        log_delta = np.full((T, S), -np.inf)
        psi = np.zeros((T, S), dtype=int)

        log_delta[0] = np.log(self.pi0_ + 1e-300) + np.log(B[0] + 1e-300)

        for t in range(1, T):
            for j in range(S):
                log_trans = log_delta[t - 1] + np.log(A_seq[t - 1, :, j] + 1e-300)
                psi[t, j] = int(log_trans.argmax())
                log_delta[t, j] = log_trans.max() + np.log(B[t, j] + 1e-300)

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T - 1] = int(log_delta[T - 1].argmax())
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict(self, X: np.ndarray, covariates: np.ndarray) -> dict:
        """
        Run full HMM inference.

        Returns
        -------
        dict with:
            proba          : (T, 3) state probabilities
            viterbi_states : (T,) decoded state sequence
            state_labels   : (T,) list of label strings
            regime_summary : latest regime probabilities
        """
        proba = self.predict_proba(X, covariates)
        states = self.decode(X, covariates)
        labels = [STATE_LABELS[s] for s in states]

        latest = proba[-1]
        return {
            "proba": proba,
            "viterbi_states": states,
            "state_labels": labels,
            "regime_summary": {
                "TREND": float(latest[0]),
                "MEAN_REV": float(latest[1]),
                "STRESS": float(latest[2]),
                "dominant_regime": STATE_LABELS[int(latest.argmax())],
            },
        }

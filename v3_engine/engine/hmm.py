"""
Layer 4: TVTP-HMM (Time-Varying Transition Probability HMM)

Architecture:
  - 3 hidden states: TREND (1), MEAN-REV (0), STRESS (2)
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

from __future__ import annotations

import warnings
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

STATE_NAMES = {0: "MEAN_REV", 1: "TREND", 2: "STRESS"}
STATE_LABELS = ["MEAN_REV", "TREND", "STRESS"]

LAMBDA_A = 0.1
LAMBDA_MU = 0.01
LAMBDA_BETA = 0.05
RIDGE_SIGMA = 1e-5   

@jax.jit
def _compute_transition_matrix(beta: jnp.ndarray, covariates: jnp.ndarray) -> jnp.ndarray:
    logits = jnp.einsum('ijc,tc->tij', beta, covariates)
    logits = logits - jnp.max(logits, axis=2, keepdims=True)
    exp_l = jnp.exp(logits)
    return exp_l / (jnp.sum(exp_l, axis=2, keepdims=True) + 1e-300)

@jax.jit
def _compute_log_emission_matrix(X: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, pi_gmm: jnp.ndarray, clean_mask: jnp.ndarray) -> jnp.ndarray:
    T, D = X.shape
    S, K = pi_gmm.shape
    
    X_exp = X[:, None, None, :]
    mu_exp = mu[None, :, :, :]
    sigma_exp = sigma[None, :, :, :, :]
    
    logpdf_vals = multivariate_normal.logpdf(X_exp, mean=mu_exp, cov=sigma_exp)
    log_pi = jnp.log(jnp.maximum(pi_gmm, 1e-15))
    
    log_state_prob = jax.scipy.special.logsumexp(log_pi[None, :, :] + logpdf_vals, axis=2)
    
    log_state_prob = jnp.where(clean_mask[:, None], log_state_prob, -jnp.log(S))
    return log_state_prob

@jax.jit
def _forward_backward_jax(X: jnp.ndarray, covariates: jnp.ndarray, pi0: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, pi_gmm: jnp.ndarray, beta: jnp.ndarray, clean_mask: jnp.ndarray):
    T, D = X.shape
    S = pi0.shape[0]
    
    log_B = _compute_log_emission_matrix(X, mu, sigma, pi_gmm, clean_mask)
    A_seq = _compute_transition_matrix(beta, covariates)
    log_A_seq = jnp.log(jnp.maximum(A_seq, 1e-15))
    
    def forward_step(carry, elems):
        log_alpha_prev = carry
        log_A_t, log_B_t = elems
        log_alpha_t = log_B_t + jax.scipy.special.logsumexp(log_alpha_prev[:, None] + log_A_t, axis=0)
        return log_alpha_t, log_alpha_t
        
    log_alpha_0 = jnp.log(jnp.maximum(pi0, 1e-15)) + log_B[0]
    
    _, log_alpha_rest = jax.lax.scan(
        forward_step, 
        log_alpha_0, 
        (log_A_seq[0:-1], log_B[1:])
    )
    
    log_alpha = jnp.concatenate([log_alpha_0[None, :], log_alpha_rest], axis=0)
    log_likelihood = jax.scipy.special.logsumexp(log_alpha[-1])
    
    def backward_step(carry, elems):
        log_beta_next = carry
        log_A_t, log_B_next = elems
        log_beta_t = jax.scipy.special.logsumexp(log_A_t + log_B_next[None, :] + log_beta_next[None, :], axis=1)
        return log_beta_t, log_beta_t

    log_beta_T = jnp.zeros(S)
    
    _, log_beta_rest = jax.lax.scan(
        backward_step,
        log_beta_T,
        (log_A_seq[:-1], log_B[1:]),
        reverse=True
    )
    log_beta_bw = jnp.concatenate([log_beta_rest, log_beta_T[None, :]], axis=0)
    
    log_gamma = log_alpha + log_beta_bw
    gamma = jnp.exp(log_gamma - jax.scipy.special.logsumexp(log_gamma, axis=1, keepdims=True))
    gamma = jnp.clip(gamma, 1e-15, 1.0)
    gamma = gamma / jnp.sum(gamma, axis=1, keepdims=True)
    
    log_alpha_t = log_alpha[:-1, :, None]
    log_A_t = log_A_seq[:-1]
    log_beta_next = log_beta_bw[1:, None, :]
    log_B_next = log_B[1:, None, :]
    
    log_xi = log_alpha_t + log_A_t + log_B_next + log_beta_next
    xi = jnp.exp(log_xi - jax.scipy.special.logsumexp(log_xi, axis=(1, 2), keepdims=True))
    
    B = jnp.exp(log_B)
    
    alpha_norm = jnp.exp(log_alpha - jax.scipy.special.logsumexp(log_alpha, axis=1, keepdims=True))
    alpha_norm = jnp.clip(alpha_norm, 1e-15, 1.0)
    alpha_norm = alpha_norm / jnp.sum(alpha_norm, axis=1, keepdims=True)
    
    return gamma, xi, log_likelihood, B, A_seq, alpha_norm

@jax.jit
def _update_gmm_emissions_jax(X: jnp.ndarray, gamma: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, pi_gmm: jnp.ndarray, clean_mask: jnp.ndarray):
    T, D = X.shape
    S, K = pi_gmm.shape
    
    X_exp = X[:, None, None, :]
    mu_exp = mu[None, :, :, :]
    sigma_exp = sigma[None, :, :, :, :]
    
    logpdf_vals = multivariate_normal.logpdf(X_exp, mean=mu_exp, cov=sigma_exp)
    log_r = jnp.log(jnp.maximum(pi_gmm[None, :, :], 1e-15)) + logpdf_vals
    
    log_r = jnp.where(clean_mask[:, None, None], log_r, -jnp.log(K))
    log_r_norm = log_r - jax.scipy.special.logsumexp(log_r, axis=2, keepdims=True)
    r = jnp.exp(log_r_norm)
    
    w = gamma[:, :, None] * r * clean_mask[:, None, None]
    w_sum = jnp.sum(w, axis=0) 
    
    w_sum_safe = jnp.maximum(w_sum, 1e-10)
    
    mu_new = jnp.einsum('tsk,td->skd', w, X) / (w_sum_safe[:, :, None] + LAMBDA_MU)
    mu_new = jnp.where((w_sum < 1e-10)[:, :, None], mu, mu_new)
    
    diff = X[:, None, None, :] - mu_new[None, :, :, :]
    cov_new = jnp.einsum('tsk,tskd,tske->skde', w, diff, diff) / w_sum_safe[:, :, None, None]
    
    EPSILON = 1e-4
    sigma_new = cov_new + jnp.eye(D)[None, None, :, :] * EPSILON
    sigma_new = jnp.where((w_sum < 1e-10)[:, :, None, None], sigma, sigma_new)
    
    state_weight = jnp.maximum(jnp.sum(gamma, axis=0), 1e-10)
    pi_gmm_new = jnp.maximum(w_sum / state_weight[:, None], 1e-6)
    pi_gmm_new = pi_gmm_new / jnp.sum(pi_gmm_new, axis=1, keepdims=True)
    
    return mu_new, sigma_new, pi_gmm_new

@jax.jit
def _update_tvtp_beta_jax(beta: jnp.ndarray, covariates: jnp.ndarray, gamma: jnp.ndarray, xi: jnp.ndarray):
    lr = 0.05
    def body_fun(i, b):
        A_seq = _compute_transition_matrix(b, covariates)
        delta = xi - gamma[:-1, :, None] * A_seq[:-1, :, :]
        grad_b = jnp.einsum('tij,tc->ijc', delta, covariates[:-1])
        return b + lr * (grad_b - LAMBDA_BETA * b)
    
    return jax.lax.fori_loop(0, 15, body_fun, beta)


class TVTPHMM:
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

    def _init_params(self, X: np.ndarray) -> None:
        S, K, D = self.n_states, self.n_gmm, X.shape[1]
        np.random.seed(self.random_state)

        km = KMeans(n_clusters=S, n_init=10, random_state=self.random_state)
        clean = X[~np.any(np.isnan(X), axis=1)]
        labels = km.fit_predict(clean) if len(clean) >= S else np.zeros(len(clean), int)

        pi0_ = np.ones(S) / S
        pi_gmm_ = np.ones((S, K)) / K           
        mu_ = np.zeros((S, K, D))               
        sigma_ = np.array([np.eye(D) * 0.5 for _ in range(S * K)]).reshape(S, K, D, D)                        

        centers = km.cluster_centers_ if len(clean) >= S else np.zeros((S, D))
        for s in range(S):
            idx = np.where(labels == s)[0]
            if len(idx) >= K:
                half = len(idx) // K
                for k in range(K):
                    slice_ = clean[idx[k * half:(k + 1) * half]]
                    mu_[s, k] = slice_.mean(axis=0) if len(slice_) > 0 else centers[s]
            else:
                for k in range(K):
                    mu_[s, k] = centers[s] + np.random.randn(D) * 0.1

        self.n_cov = 2
        beta_ = np.zeros((S, S, self.n_cov))   
        
        self.pi0_ = jnp.array(pi0_)
        self.pi_gmm_ = jnp.array(pi_gmm_)
        self.mu_ = jnp.array(mu_)
        self.sigma_ = jnp.array(sigma_)
        self.beta_ = jnp.array(beta_)

    def fit(self, X: np.ndarray, covariates: np.ndarray) -> "TVTPHMM":
        self._init_params(X)
        
        clean_mask = ~np.any(np.isnan(X), axis=1)
        X_safe = np.where(np.isnan(X), 0.0, X)
        
        X_jnp = jnp.array(X_safe)
        cov_jnp = jnp.array(covariates)
        mask_jnp = jnp.array(clean_mask)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            gamma, xi, ll, B, A_seq, _ = _forward_backward_jax(
                X_jnp, cov_jnp, self.pi0_, self.mu_, self.sigma_, self.pi_gmm_, self.beta_, mask_jnp
            )
            self.pi0_ = gamma[0]
            self.mu_, self.sigma_, self.pi_gmm_ = _update_gmm_emissions_jax(
                X_jnp, gamma, self.mu_, self.sigma_, self.pi_gmm_, mask_jnp
            )
            self.beta_ = _update_tvtp_beta_jax(self.beta_, cov_jnp, gamma, xi)

            if abs(float(ll) - prev_ll) < 1e-4 and iteration > 5:
                break
            prev_ll = float(ll)

        vol_means = np.array(self.mu_[:, 0, 3])
        proper_order = np.argsort(vol_means)

        self.pi0_ = self.pi0_[proper_order]
        self.pi_gmm_ = self.pi_gmm_[proper_order]
        self.mu_ = self.mu_[proper_order]
        self.sigma_ = self.sigma_[proper_order]
        self.beta_ = self.beta_[proper_order][:, proper_order]

        self._fitted = True
        self._train_ll = prev_ll
        return self

    def predict_proba(self, X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        clean_mask = ~np.any(np.isnan(X), axis=1)
        X_safe = np.where(np.isnan(X), 0.0, X)
        
        _, _, _, _, _, alpha_norm = _forward_backward_jax(
            jnp.array(X_safe), jnp.array(covariates), self.pi0_, self.mu_, self.sigma_, self.pi_gmm_, self.beta_, jnp.array(clean_mask)
        )
        return np.array(alpha_norm)

    def decode(self, X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        
        clean_mask = ~np.any(np.isnan(X), axis=1)
        X_safe = np.where(np.isnan(X), 0.0, X)
        
        log_B = _compute_log_emission_matrix(jnp.array(X_safe), self.mu_, self.sigma_, self.pi_gmm_, jnp.array(clean_mask))
        A_seq = _compute_transition_matrix(self.beta_, jnp.array(covariates))
        
        log_B = np.array(log_B)
        A_seq = np.array(A_seq)
        pi0 = np.array(self.pi0_)
        
        T, D = X.shape
        S = self.n_states

        log_delta = np.full((T, S), -np.inf)
        psi = np.zeros((T, S), dtype=int)

        log_delta[0] = np.log(pi0 + 1e-300) + log_B[0]

        for t in range(1, T):
            for j in range(S):
                log_trans = log_delta[t - 1] + np.log(A_seq[t - 1, :, j] + 1e-300)
                psi[t, j] = int(log_trans.argmax())
                log_delta[t, j] = log_trans.max() + log_B[t, j]

        states = np.zeros(T, dtype=int)
        states[T - 1] = int(log_delta[T - 1].argmax())
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict(self, X: np.ndarray, covariates: np.ndarray) -> dict:
        proba = self.predict_proba(X, covariates)
        states = self.decode(X, covariates)
        labels = [STATE_LABELS[s] for s in states]

        latest = proba[-1]
        return {
            "proba": proba,
            "viterbi_states": states,
            "state_labels": labels,
            "regime_summary": {
                "MEAN_REV": float(latest[0]),  
                "TREND": float(latest[1]),     
                "STRESS": float(latest[2]),    
                "dominant_regime": STATE_LABELS[int(latest.argmax())],
            },
        }

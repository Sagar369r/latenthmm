from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

@jax.jit
def _em_step_jax(X: jnp.ndarray, A: jnp.ndarray, H: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray, clean_mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    T, D = X.shape
    d = A.shape[0]
    
    eye_d = jnp.eye(d)
    eye_D = jnp.eye(D)
    
    def forward_step(carry, elems):
        mu_pred, P_pred = carry
        x, is_clean = elems
        
        S = H @ P_pred @ H.T + R
        S_inv = jnp.linalg.pinv(S + eye_D * 1e-6)
        K = P_pred @ H.T @ S_inv
        
        innov_clean = x - H @ mu_pred
        innov = jnp.where(is_clean, innov_clean, jnp.zeros(D))
        
        mu_update = jnp.where(is_clean, mu_pred + K @ innov, mu_pred)
        P_update = jnp.where(is_clean, (eye_d - K @ H) @ P_pred, P_pred)
        
        mu_pred_next = A @ mu_update
        P_pred_next = A @ P_update @ A.T + Q
        
        return (mu_pred_next, P_pred_next), (mu_update, P_update, mu_pred_next, P_pred_next, K)
    
    mu_pred_0 = jnp.zeros(d)
    P_pred_0 = eye_d * 10.0
    
    _, (mu, P, mu_pred_nexts, P_pred_nexts, K_gains) = jax.lax.scan(
        forward_step,
        (mu_pred_0, P_pred_0),
        (X, clean_mask)
    )
    
    mu_pred = jnp.concatenate([mu_pred_0[None, :], mu_pred_nexts[:-1]], axis=0)
    P_pred = jnp.concatenate([P_pred_0[None, :, :], P_pred_nexts[:-1]], axis=0)
    
    def backward_step(carry, elems):
        mu_s_next, P_s_next = carry
        mu_t, P_t, mu_pred_next, P_pred_next = elems
        
        P_pred_inv = jnp.linalg.pinv(P_pred_next + eye_d * 1e-6)
        G_t = P_t @ A.T @ P_pred_inv
        
        mu_s_t = mu_t + G_t @ (mu_s_next - mu_pred_next)
        P_s_t = P_t + G_t @ (P_s_next - P_pred_next) @ G_t.T
        
        return (mu_s_t, P_s_t), (mu_s_t, P_s_t, G_t)
    
    _, (mu_s_rev, P_s_rev, G_rev) = jax.lax.scan(
        backward_step,
        (mu[-1], P[-1]),
        (mu[:-1], P[:-1], mu_pred[1:], P_pred[1:]),
        reverse=True
    )
    
    mu_s = jnp.concatenate([mu_s_rev, mu[-1][None, :]], axis=0)
    P_s = jnp.concatenate([P_s_rev, P[-1][None, :, :]], axis=0)
    G = G_rev
    
    Pst = jnp.einsum('tij,tkj->tik', P_s[1:], G) + jnp.einsum('ti,tj->tij', mu_s[1:], mu_s[:-1])
    diff = mu_s[1:] - jnp.einsum('ij,tj->ti', A, mu_s[:-1])
    
    Q_terms = (
        P_s[1:] 
        - jnp.einsum('tij,kj->tik', Pst, A) 
        - jnp.einsum('ij,tkj->tik', A, Pst) 
        + jnp.einsum('ij,tjk,lk->til', A, P_s[:-1], A)
        + jnp.einsum('ti,tj->tij', diff, diff)
    )
    Q_new = jnp.sum(Q_terms, axis=0) / (T - 1)
    Q_new = (Q_new + Q_new.T) / 2 + eye_d * 1e-6
    
    diff_obs = X - jnp.einsum('ij,tj->ti', H, mu_s)
    R_terms = jnp.einsum('ti,tj->tij', diff_obs, diff_obs) + jnp.einsum('ij,tjk,lk->til', H, P_s, H)
    
    R_terms_clean = jnp.where(clean_mask[:, None, None], R_terms, 0.0)
    clean_count = jnp.maximum(jnp.sum(clean_mask), 1.0)
    R_new = jnp.sum(R_terms_clean, axis=0) / clean_count
    R_new = (R_new + R_new.T) / 2 + eye_D * 1e-6
    
    return Q_new, R_new


@jax.jit
def _filter_jax(X: jnp.ndarray, A: jnp.ndarray, H: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray, clean_mask: jnp.ndarray, use_cusum: bool, kappa: float, threshold: float, warmup: int):
    T, D = X.shape
    d = A.shape[0]
    
    eye_d = jnp.eye(d)
    eye_D = jnp.eye(D)
    
    def step_fn(carry, elems):
        mu_pred, P_pred, cusum_g, cusum_sum, cusum_sum2, t = carry
        x, is_clean = elems
        
        S = H @ P_pred @ H.T + R
        S_inv = jnp.linalg.pinv(S + eye_D * 1e-6)
        K = P_pred @ H.T @ S_inv
        
        innov_clean = x - H @ mu_pred
        innov = jnp.where(is_clean, innov_clean, jnp.zeros(D))
        
        mu_update = jnp.where(is_clean, mu_pred + K @ innov, mu_pred)
        P_update = jnp.where(is_clean, (eye_d - K @ H) @ P_pred, P_pred)
        
        abs_innov = jnp.mean(jnp.abs(innov))
        
        # update cusum stats
        c_sum_next = cusum_sum + abs_innov
        c_sum2_next = cusum_sum2 + abs_innov ** 2
        n = t + 1.0
        mu_nu = c_sum_next / n
        var_nu = jnp.maximum(c_sum2_next / n - mu_nu ** 2, 1e-8)
        sigma_nu = jnp.sqrt(var_nu)
        
        normed = abs_innov / jnp.maximum(sigma_nu, 1e-8)
        
        new_g_candidate = jnp.maximum(0.0, cusum_g + normed - kappa)
        is_jump = (new_g_candidate > threshold) & (t >= warmup) & use_cusum
        
        new_g = jnp.where(is_jump, 0.0, new_g_candidate)
        new_g = jnp.where((t >= warmup) & is_clean, new_g, cusum_g)
        
        # predict next
        mu_pred_next = A @ mu_update
        P_pred_next = A @ P_update @ A.T + Q
        
        # reset P if jump
        P_pred_next = jnp.where(is_jump, eye_d * 10.0, P_pred_next)
        
        # If not clean, we don't update cusum stats properly for that step
        c_sum_next = jnp.where(is_clean, c_sum_next, cusum_sum)
        c_sum2_next = jnp.where(is_clean, c_sum2_next, cusum_sum2)
        
        carry_next = (mu_pred_next, P_pred_next, new_g, c_sum_next, c_sum2_next, t + 1.0)
        return carry_next, (mu_update, P_update, innov, is_jump, new_g)

    mu_pred_0 = jnp.zeros(d)
    P_pred_0 = eye_d * 10.0
    
    _, (mu, P, innovations, jump_flags, g_seq) = jax.lax.scan(
        step_fn,
        (mu_pred_0, P_pred_0, 0.0, 0.0, 0.0, 0.0),
        (X, clean_mask)
    )
    
    return mu, P, innovations, jump_flags, g_seq


class KalmanFilter:
    def __init__(self, dim_obs: int = 6, dim_state: int | None = None) -> None:
        self.dim_obs = dim_obs
        self.dim_state = dim_state or dim_obs

        d = self.dim_state
        self.A = np.eye(d)
        self.H = np.eye(d, dim_obs)
        self.Q = np.eye(d) * 0.1
        self.R = np.eye(dim_obs) * 1.0
        self._fitted = False

    def _em_step(self, X: np.ndarray) -> None:
        clean_mask = ~np.any(np.isnan(X), axis=1)
        X_safe = np.where(np.isnan(X), 0.0, X)
        
        Q_new, R_new = _em_step_jax(
            jnp.array(X_safe), jnp.array(self.A), jnp.array(self.H), jnp.array(self.Q), jnp.array(self.R), jnp.array(clean_mask)
        )
        self.Q = np.array(Q_new)
        self.R = np.array(R_new)

    def fit(self, X: np.ndarray, n_iter: int = 10) -> "KalmanFilter":
        for _ in range(n_iter):
            self._em_step(X)
        self._fitted = True
        return self

    def filter(self, X: np.ndarray, cusum_detector: "CUSUMJumpDetector" | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        use_cusum = cusum_detector is not None
        kappa = cusum_detector.kappa if use_cusum else 0.5
        threshold = cusum_detector.threshold if use_cusum else 5.0
        warmup = cusum_detector.warmup if use_cusum else 30
        
        clean_mask = ~np.any(np.isnan(X), axis=1)
        X_safe = np.where(np.isnan(X), 0.0, X)
        
        mu, P, innovations, jump_flags, g_seq = _filter_jax(
            jnp.array(X_safe), jnp.array(self.A), jnp.array(self.H), jnp.array(self.Q), jnp.array(self.R), jnp.array(clean_mask),
            use_cusum, kappa, threshold, warmup
        )
        
        if use_cusum:
            cusum_detector.g_history = np.array(g_seq)
            
        return np.array(mu), np.array(P), np.array(innovations), np.array(jump_flags)


class CUSUMJumpDetector:
    def __init__(self, kappa: float = 0.5, threshold: float = 5.0, warmup: int = 30) -> None:
        self.kappa = kappa
        self.threshold = threshold
        self.warmup = warmup
        self.g_history = None


def run_kalman_pipeline(
    X_whitened: np.ndarray,
    fit_window: int = 504,
    refit_every: int = 63,
) -> dict:
    T, D = X_whitened.shape
    kf = KalmanFilter(dim_obs=D, dim_state=D)
    cusum = CUSUMJumpDetector()

    clean_idx = np.where(~np.any(np.isnan(X_whitened), axis=1))[0]
    if len(clean_idx) >= fit_window:
        kf.fit(X_whitened[clean_idx[:fit_window]], n_iter=5)
    elif len(clean_idx) >= 50:
        kf.fit(X_whitened[clean_idx], n_iter=5)

    filtered_states, _, innovations, jump_flags = kf.filter(X_whitened, cusum_detector=cusum)
    
    cusum_g = cusum.g_history if cusum.g_history is not None else np.zeros(T)

    return {
        "filtered_states": filtered_states,
        "innovations": innovations,
        "cusum_g": cusum_g,
        "jump_flags": jump_flags,
        "kalman_filter": kf,
    }

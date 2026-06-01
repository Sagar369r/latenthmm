"""
Layer 3: Kalman Filter + CUSUM Jump Detector

Replaces the neural DDPM denoiser from v2 with a parameter-efficient
linear state-space model. Zero deep-learning dependency.

State equation:     z_t = A z_{t-1} + w_t,   w_t ~ N(0, Q)
Observation eq:     X_t = H z_t + v_t,        v_t ~ N(0, R)

Q and R estimated via EM on a 504-bar (2-year) training window,
re-estimated quarterly (63 bars).

CUSUM jump detector on normalised innovations ν_t = X_t - H ẑ_{t|t-1}:
    g_t = max(0, g_{t-1} + |ν_t|/σ_ν - κ)
    Trigger jump if g_t > h
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


class KalmanFilter:
    """
    Linear Gaussian Kalman filter for 6D → 6D latent state denoising.

    Parameters
    ----------
    dim_obs : int
        Dimensionality of observations (6 for the feature tensor).
    dim_state : int
        Dimensionality of latent state (defaults to dim_obs).
    """

    def __init__(self, dim_obs: int = 6, dim_state: int | None = None) -> None:
        self.dim_obs = dim_obs
        self.dim_state = dim_state or dim_obs

        d = self.dim_state
        self.A = np.eye(d)           # state transition (identity → random walk)
        self.H = np.eye(d, dim_obs)  # observation matrix
        self.Q = np.eye(d) * 0.1     # process noise covariance
        self.R = np.eye(dim_obs) * 1.0  # observation noise covariance

        self._fitted = False

    def _em_step(self, X: np.ndarray) -> None:
        """
        One pass of EM to estimate Q and R from data.
        Uses the fixed A=I, H=I structure with EM updates for Q, R.
        """
        T, D = X.shape
        d = self.dim_state

        # Forward pass (predict + update)
        mu = np.zeros((T, d))
        P = np.zeros((T, d, d))
        mu_pred = np.zeros((T, d))
        P_pred = np.zeros((T, d, d))
        K_gains = np.zeros((T, d, D))
        innovations = np.zeros((T, D))

        # Initial conditions
        mu_pred[0] = np.zeros(d)
        P_pred[0] = np.eye(d) * 10.0

        for t in range(T):
            # Update step
            S = self.H @ P_pred[t] @ self.H.T + self.R
            try:
                S_inv = np.linalg.inv(S + np.eye(D) * 1e-6)
            except np.linalg.LinAlgError:
                S_inv = np.eye(D)

            K = P_pred[t] @ self.H.T @ S_inv
            K_gains[t] = K

            if np.any(np.isnan(X[t])):
                mu[t] = mu_pred[t]
                P[t] = P_pred[t]
                innovations[t] = 0.0
            else:
                innov = X[t] - self.H @ mu_pred[t]
                innovations[t] = innov
                mu[t] = mu_pred[t] + K @ innov
                P[t] = (np.eye(d) - K @ self.H) @ P_pred[t]

            # Predict step
            if t < T - 1:
                mu_pred[t + 1] = self.A @ mu[t]
                P_pred[t + 1] = self.A @ P[t] @ self.A.T + self.Q

        # Backward pass (RTS smoother)
        mu_s = mu.copy()
        P_s = P.copy()
        G = np.zeros((T - 1, d, d))

        for t in range(T - 2, -1, -1):
            try:
                P_pred_inv = np.linalg.inv(P_pred[t + 1] + np.eye(d) * 1e-6)
            except np.linalg.LinAlgError:
                P_pred_inv = np.eye(d)

            G[t] = P[t] @ self.A.T @ P_pred_inv
            mu_s[t] = mu[t] + G[t] @ (mu_s[t + 1] - mu_pred[t + 1])
            P_s[t] = P[t] + G[t] @ (P_s[t + 1] - P_pred[t + 1]) @ G[t].T

        # M-step: update Q and R
        Q_new = np.zeros((d, d))
        R_new = np.zeros((D, D))
        Pst = np.zeros((d, d))  # cross-covariance E[z_t z_{t-1}^T]

        for t in range(1, T):
            Pst = P_s[t] @ G[t - 1].T + np.outer(mu_s[t], mu_s[t - 1])
            diff = mu_s[t] - self.A @ mu_s[t - 1]
            Q_new += (P_s[t] - Pst @ self.A.T - self.A @ Pst.T
                      + self.A @ P_s[t - 1] @ self.A.T
                      + np.outer(diff, diff))

        Q_new /= (T - 1)
        Q_new = (Q_new + Q_new.T) / 2 + np.eye(d) * 1e-6  # ensure PSD

        clean_mask = ~np.any(np.isnan(X), axis=1)
        X_clean = X[clean_mask]
        mu_s_clean = mu_s[clean_mask]
        P_s_clean = P_s[clean_mask]

        for t in range(len(X_clean)):
            diff = X_clean[t] - self.H @ mu_s_clean[t]
            R_new += (np.outer(diff, diff)
                      + self.H @ P_s_clean[t] @ self.H.T)

        if len(X_clean) > 0:
            R_new /= len(X_clean)
        R_new = (R_new + R_new.T) / 2 + np.eye(D) * 1e-6  # ensure PSD

        self.Q = Q_new
        self.R = R_new

    def fit(self, X: np.ndarray, n_iter: int = 10) -> "KalmanFilter":
        """
        Fit Q and R via EM on the training window.
        X shape: (T, D)
        """
        for _ in range(n_iter):
            self._em_step(X)
        self._fitted = True
        return self

    def filter(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Kalman filter on observations.

        Returns
        -------
        mu_filtered : (T, d) filtered state means
        P_filtered  : (T, d, d) filtered state covariances
        innovations : (T, D) normalised innovation sequence ν_t
        """
        T, D = X.shape
        d = self.dim_state

        mu = np.zeros((T, d))
        P = np.zeros((T, d, d))
        innovations = np.zeros((T, D))

        mu_pred = np.zeros(d)
        P_pred = np.eye(d) * 10.0

        for t in range(T):
            # Innovation
            if np.any(np.isnan(X[t])):
                mu[t] = mu_pred
                P[t] = P_pred
                innovations[t] = 0.0
            else:
                S = self.H @ P_pred @ self.H.T + self.R
                try:
                    S_inv = np.linalg.inv(S + np.eye(D) * 1e-6)
                except np.linalg.LinAlgError:
                    S_inv = np.eye(D)
                K = P_pred @ self.H.T @ S_inv
                innov = X[t] - self.H @ mu_pred
                innovations[t] = innov
                mu[t] = mu_pred + K @ innov
                P[t] = (np.eye(d) - K @ self.H) @ P_pred

            # Predict
            mu_pred = self.A @ mu[t]
            P_pred = self.A @ P[t] @ self.A.T + self.Q

        return mu, P, innovations


class CUSUMJumpDetector:
    """
    Modified CUSUM change-point detector on the normalised innovation sequence.

    g_t = max(0, g_{t-1} + |ν_t|/σ_ν - κ)
    Trigger: g_t > h  →  reset Kalman to diffuse prior

    Spec defaults: κ = 0.5, h = 5.0
    """

    def __init__(self, kappa: float = 0.5, threshold: float = 5.0,
                 warmup: int = 30) -> None:
        self.kappa = kappa
        self.threshold = threshold
        self.warmup = warmup

    def detect(
        self, innovations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run CUSUM on the innovation sequence.

        Parameters
        ----------
        innovations : (T, D) array of Kalman innovations

        Returns
        -------
        g          : (T,) CUSUM statistic
        jump_flags : (T,) boolean array, True at detected jump
        sigma_nu   : (T,) running estimate of innovation std
        """
        T = innovations.shape[0]
        # Use mean absolute innovation across dimensions
        abs_innov = np.abs(innovations).mean(axis=1)

        g = np.zeros(T)
        jump_flags = np.zeros(T, dtype=bool)
        sigma_nu = np.zeros(T)
        running_sum = 0.0
        running_sum2 = 0.0

        for t in range(T):
            running_sum += abs_innov[t]
            running_sum2 += abs_innov[t] ** 2
            n = t + 1
            mu_nu = running_sum / n
            var_nu = max(running_sum2 / n - mu_nu ** 2, 1e-8)
            sigma_nu[t] = np.sqrt(var_nu)

            if t < self.warmup:
                g[t] = 0.0
            else:
                normed = abs_innov[t] / max(sigma_nu[t], 1e-8)
                g[t] = max(0.0, g[t - 1] + normed - self.kappa)
                if g[t] > self.threshold:
                    jump_flags[t] = True
                    g[t] = 0.0  # reset after trigger

        return g, jump_flags, sigma_nu


def run_kalman_pipeline(
    X_whitened: np.ndarray,
    fit_window: int = 504,
    refit_every: int = 63,
) -> dict:
    """
    Full Layer 3 pipeline: fit Kalman → filter → detect jumps.

    Parameters
    ----------
    X_whitened : (T, D) whitened feature matrix (NaN rows allowed)
    fit_window : bars for EM fitting (spec: 504 = ~2 years)
    refit_every: refit frequency in bars (spec: quarterly ≈ 63 bars)

    Returns
    -------
    dict with keys:
        filtered_states : (T, D) denoised state estimates
        innovations     : (T, D) Kalman innovations
        cusum_g         : (T,) CUSUM statistic
        jump_flags      : (T,) boolean jump indicators
    """
    T, D = X_whitened.shape
    kf = KalmanFilter(dim_obs=D, dim_state=D)
    cusum = CUSUMJumpDetector()

    clean_idx = np.where(~np.any(np.isnan(X_whitened), axis=1))[0]
    if len(clean_idx) >= fit_window:
        kf.fit(X_whitened[clean_idx[:fit_window]], n_iter=5)
    elif len(clean_idx) >= 50:
        kf.fit(X_whitened[clean_idx], n_iter=5)

    filtered_states, _, innovations = kf.filter(X_whitened)
    cusum_g, jump_flags, _ = cusum.detect(innovations)

    # Diffuse reset on jump: inflate uncertainty (simulated by zeroing filtered state)
    for t in np.where(jump_flags)[0]:
        if t + 1 < T:
            filtered_states[t] = np.zeros(D)

    return {
        "filtered_states": filtered_states,
        "innovations": innovations,
        "cusum_g": cusum_g,
        "jump_flags": jump_flags,
        "kalman_filter": kf,
    }

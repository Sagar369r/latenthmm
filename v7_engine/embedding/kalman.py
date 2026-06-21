"""
1-D Kalman Filter — constant-velocity model for price smoothing.
State: [position, velocity]. Used for regime detection in the embedding layer.
"""
from __future__ import annotations
import numpy as np


class KalmanFilter1D:
    """
    2-state linear Kalman filter: x = [price, velocity].
    Measurement: price only.  dt comes from real tick timestamps.
    """

    __slots__ = ("x", "P", "q_c", "R", "H", "is_initialized")

    def __init__(
        self,
        process_noise:     float = 1e-5,
        measurement_noise: float = 1e-4,
    ):
        self.x  = np.zeros(2,      dtype=np.float64)
        self.P  = np.eye(2,        dtype=np.float64)
        self.q_c = float(process_noise)
        self.R  = np.array([[measurement_noise]], dtype=np.float64)
        self.H  = np.array([[1.0, 0.0]],         dtype=np.float64)
        self.is_initialized = False

    def update(self, measurement: float, dt: float) -> tuple[float, float]:
        """
        Predict + correct one step.

        Parameters
        ----------
        measurement : observed price
        dt          : seconds since last tick

        Returns
        -------
        (filtered_price, filtered_velocity)
        """
        if not self.is_initialized:
            self.x[0] = measurement
            self.x[1] = 0.0
            self.is_initialized = True
            return float(self.x[0]), float(self.x[1])

        # Unroll 2x2 matrix operations into pure Python floats to avoid numpy overhead
        dt = max(dt, 1e-6)
        x0, x1 = self.x[0], self.x[1]
        P00, P01 = self.P[0, 0], self.P[0, 1]
        P10, P11 = self.P[1, 0], self.P[1, 1]
        # Process Noise Covariance (White Noise Acceleration Model)
        # Q = G * q_c * dt * G.T where G = [dt^2/2, dt]
        # Q00 = q_c * dt^5 / 4 (We'll use standard dt^4 / 4)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        Q00 = self.q_c * dt4 / 4.0
        Q01 = self.q_c * dt3 / 2.0
        Q10 = Q01
        Q11 = self.q_c * dt2
        
        R00 = self.R[0, 0]

        # Predict (x = F @ x)
        new_x0 = x0 + dt * x1
        new_x1 = x1

        # Predict P (P = F @ P @ F.T + Q)
        P00_pred = P00 + dt * (P10 + P01) + dt2 * P11 + Q00
        P01_pred = P01 + dt * P11 + Q01
        P10_pred = P10 + dt * P11 + Q10
        P11_pred = P11 + Q11

        # Update (y = z - H @ x)
        y = measurement - new_x0
        
        # S = H @ P_pred @ H.T + R
        S = P00_pred + R00
        
        # K = P_pred @ H.T / S
        K0 = P00_pred / S
        K1 = P10_pred / S

        # x = x_pred + K * y
        self.x[0] = new_x0 + K0 * y
        self.x[1] = new_x1 + K1 * y

        # Joseph Form Update: P = (I - KH)P(I - KH)^T + KRK^T
        # This mathematically guarantees P remains positive-definite and perfectly symmetric.
        # I - KH = [[1 - K0, 0], [-K1, 1]]
        M00 = 1.0 - K0
        M01 = 0.0
        M10 = -K1
        M11 = 1.0

        # M @ P
        MP00 = M00 * P00_pred + M01 * P10_pred
        MP01 = M00 * P01_pred + M01 * P11_pred
        MP10 = M10 * P00_pred + M11 * P10_pred
        MP11 = M10 * P01_pred + M11 * P11_pred

        # (M @ P) @ M.T
        MPM00 = MP00 * M00 + MP01 * M01
        MPM01 = MP00 * M10 + MP01 * M11
        MPM10 = MP10 * M00 + MP11 * M01
        MPM11 = MP10 * M10 + MP11 * M11

        # K @ R @ K.T
        KRK00 = K0 * R00 * K0
        KRK01 = K0 * R00 * K1
        KRK10 = K1 * R00 * K0
        KRK11 = K1 * R00 * K1

        # Final P
        self.P[0, 0] = MPM00 + KRK00
        self.P[0, 1] = MPM01 + KRK01
        self.P[1, 0] = MPM10 + KRK10
        self.P[1, 1] = MPM11 + KRK11

        return float(self.x[0]), float(self.x[1])

    def to_dict(self) -> dict:
        return {
            "x": self.x.tolist(),
            "P": self.P.tolist(),
            "q_c": self.q_c,
            "R": self.R.tolist(),
            "H": self.H.tolist(),
            "is_initialized": self.is_initialized
        }
        
    def from_dict(self, state: dict) -> None:
        self.x = np.array(state["x"], dtype=np.float64)
        self.P = np.array(state["P"], dtype=np.float64)
        self.q_c = float(state["q_c"])
        self.R = np.array(state["R"], dtype=np.float64)
        self.H = np.array(state["H"], dtype=np.float64)
        self.is_initialized = bool(state["is_initialized"])

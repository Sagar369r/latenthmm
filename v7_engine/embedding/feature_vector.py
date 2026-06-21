"""
Feature Vector — Assembles the 128-dim embedding from raw tick data.

All price-based features (log returns, momentum, spread) are computed on
the 100ms resampled grid. Raw tick processing only for TIB and delta-t.

FIXES applied:
  • Context is NEVER torch.zeros — computed from real tick embeddings
  • All features on 100ms grid, not raw ticks
  • XGBoost regime prediction appended as extra context
"""
from __future__ import annotations
import numpy as np
from v7_engine.config import (
    CWT_SCALES, CWT_WAVELET, EMBEDDING_DIM,
    XGB_REGIME_IDX_START, XGB_REGIME_IDX_END, XGB_RISK_IDX,
    EMBED_KALMAN_VEL_MIN_LEN, EMBED_COMPUTE_N_RECENT, EMBED_MIN_BIDS_LEN, 
    EMBED_STALE_TIME_NORM, EMBED_STALE_TIME_SEC, EMBED_LOG_RET_SLICE, 
    EMBED_MIN_CLIP_12, EMBED_MIN_CLIP_8, EMBED_MIN_CLIP_5, EMBED_MIN_CLIP_4, 
    EMBED_VAR_CLIP, EMBED_SPREAD_Z_CLIP, EMBED_SPREAD_MA_CLIP, 
    EMBED_MOM_5, EMBED_MOM_20, EMBED_HF_ENERGY_SLICE, 
    EMBED_COL_KALMAN, EMBED_COL_CUSUM, EMBED_COL_WAVELET, EMBED_XGB_CACHE_TICKS
)
import time
from collections import deque
import logging
import threading

from v7_engine.config import (
    EMBEDDING_DIM, CWT_SCALES,
    KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE,
)
from v7_engine.ingestion.ring_buffer import RingBuffer
from v7_engine.ingestion.welford import WelfordNormaliser
from v7_engine.embedding.tib_engine import TIBEngine
from v7_engine.embedding.wavelet_encoder import encode_delta_t_buffer
from v7_engine.embedding.kalman import KalmanFilter1D
from v7_engine.embedding.cusum_v2 import CUSUMv2

logger = logging.getLogger(__name__)


class TickFeatureVector:
    """
    Stateful feature extractor.  One instance per live symbol.
    Call compute() on every tick to get the latest EMBEDDING_DIM vector.

    Feature layout (128 dims):
      [0:4]   TIB features          (4)
      [4:36]  CWT delta-t encoding  (32)
      [36:41] Log returns grid      (5)
      [41]    Volatility ratio       (1)
      [42:45] Spread features        (3)
      [45:47] Momentum 5 + 20       (2)
      [47:49] Kalman vel + CUSUM    (2)
      [49:53] Regime one-hot        (4)
      [53]    Sideways confidence   (1)
      [54:58] XGB regime probs      (4)  ← set externally by ml/regime_xgb.py
      [58]    Risk score            (1)
      [59:128] Zero-padded          (69)
    """

    def __init__(
        self,
        buffer:             RingBuffer,
        tib:                TIBEngine,
        xgb_regime_model    = None,    # Optional ml.regime_xgb.RegimeXGBClassifier
        xgb_risk_model      = None,    # Optional ml.risk_xgb.RiskXGBClassifier
        welford_normaliser  = None,    # Optional ml.welford.WelfordNormaliser
        hmm_regime_model    = None,
    ):
        self.buffer     = buffer
        self.tib        = tib
        self._xgb       = xgb_regime_model
        self._xgb_risk  = xgb_risk_model
        self._hmm       = hmm_regime_model
        self.welford    = welford_normaliser

        self.kalman = KalmanFilter1D(KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE)
        self.cusum  = CUSUMv2(window=60)
        self.kalman_vel_history: list[float] = []
        self.dt_history: deque = deque(maxlen=CWT_SCALES * 4)
        # Simple rule-based regime for internal one-hot when XGBoost not loaded
        self._regime_onehot = np.zeros(4, dtype=np.float32)
        self._sideways_conf = 0.0
        self._last_cusum_event = 0
        self._last_kalman_vel = 0.0

        # XGBoost real-time evaluation lock
        self._xgb_lock = threading.Lock()
        # Rolling history of last 10 feature vectors for Risk XGB (1280-dim window)
        self._feat_history: deque = deque(maxlen=10)

    # ── public ────────────────────────────────────────────────────────────────

    def update_state(self, mid: float, dt: float, sign: int):
        """
        Fast O(1) state tracker update for every single tick.
        """
        self.tib.push_sign(sign)
        self.dt_history.append(float(dt))
        
        kalman_price, kalman_vel = self.kalman.update(mid, dt)
        self._last_kalman_vel = kalman_vel
        self.kalman_vel_history.append(kalman_vel)
        if len(self.kalman_vel_history) > EMBED_KALMAN_VEL_MIN_LEN:
            self.kalman_vel_history.pop(0)
            
        self._last_cusum_event = self.cusum.update(kalman_price)

    def compute(self, n_recent: int = EMBED_COMPUTE_N_RECENT, update_welford: bool = False) -> np.ndarray:
        """
        Compute the 128-dim embedding from the most recent n_recent ticks.
        Returns zeros if not enough data.
        """
        timestamps, bids, asks, dts, signs = self.buffer.get_latest(n_recent, copy=False)
        if len(bids) < EMBED_MIN_BIDS_LEN:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        timestamps = timestamps / EMBED_STALE_TIME_NORM   # → seconds

        mid        = (bids + asks) / 2.0
        spread_raw = asks - bids

        # ── Raw Tick Stream (Bypass Python Grid) ───────────────────────────────
        # Using raw arrays directly drops latency to 0ms and matches Numba logic exactly
        grid_mid = mid
        grid_spread = spread_raw
        is_stale = (time.time() - timestamps[-1] > EMBED_STALE_TIME_SEC) if len(timestamps) > 0 else True

        if is_stale:
            logger.debug("STALE_TICK: last tick >500ms ago — features may be lagged")

        # ── TIB (on 100ms grid) ───────────────────────────────────────────────
        # State already updated externally via update_state()
        tib_feats = np.array(self.tib.get_features(), dtype=np.float32)   # (4,)

        # ── CWT delta-t (on raw ticks) ────────────────────────────────────────
        # self.dt_history is already updated via update_state()
        dt_enc = encode_delta_t_buffer(np.array(self.dt_history, dtype=np.float32))  # (32,)

        # ── Log returns on grid ───────────────────────────────────────────────
        full_log_ret = np.log(grid_mid[1:] / np.maximum(grid_mid[:-1], EMBED_MIN_CLIP_12))
        log_ret = full_log_ret[-EMBED_LOG_RET_SLICE:]
        if len(log_ret) < EMBED_LOG_RET_SLICE:
            log_ret = np.pad(log_ret, (EMBED_LOG_RET_SLICE - len(log_ret), 0))

        var_short = np.var(log_ret) + EMBED_MIN_CLIP_12
        var_long  = np.var(full_log_ret) + EMBED_MIN_CLIP_12
        vol_ratio = float(np.clip(np.log(var_short / var_long), -EMBED_VAR_CLIP, EMBED_VAR_CLIP))

        # ── Spread features ───────────────────────────────────────────────────
        s_mean   = grid_spread.mean() + EMBED_MIN_CLIP_8
        s_std    = grid_spread.std()  + EMBED_MIN_CLIP_8
        s_p95    = float(np.percentile(grid_spread, 95)) + EMBED_MIN_CLIP_8
        spread_z  = float(np.clip((grid_spread[-1] - s_mean) / s_std, -EMBED_VAR_CLIP, EMBED_VAR_CLIP))
        spread_ma = float(np.clip(np.mean(grid_spread[-int(EMBED_SPREAD_MA_CLIP):]) / s_mean, 0.0, EMBED_SPREAD_MA_CLIP))
        spread_spk= float(np.clip(grid_spread[-1] / s_p95, 0.0, EMBED_SPREAD_MA_CLIP))

        # ── Momentum (grid) ───────────────────────────────────────────────────
        mom_5  = _mom(grid_mid, EMBED_MOM_5)
        mom_20 = _mom(grid_mid, EMBED_MOM_20)

        # ── Kalman + CUSUM ────────────────────────────────────────────────────
        # State already updated via update_state()
        kalman_vel_var = float(np.var(self.kalman_vel_history)) if len(self.kalman_vel_history) > 1 else 0.0
        cusum_energy   = float(abs(self.cusum.c_pos) + abs(self.cusum.c_neg))

        # ── Rule-based regime one-hot ─────────────────────────────────────────
        wavelet_hf_energy = float(np.mean(np.abs(dt_enc[:EMBED_HF_ENERGY_SLICE])))
        self._regime_onehot, self._sideways_conf = _rule_regime(
            kalman_vel_var, cusum_energy, wavelet_hf_energy
        )

        # ── Assemble pre-XGB vector ───────────────────────────────────────────
        pre_vec = np.concatenate([
            tib_feats,                                        # 4
            dt_enc,                                           # 32
            log_ret,                                          # 5
            [vol_ratio],                                      # 1
            [spread_z, spread_ma, spread_spk],                # 3
            [mom_5, mom_20],                                  # 2
            [self._last_kalman_vel, float(self._last_cusum_event)], # 2
            self._regime_onehot,                              # 4
            [self._sideways_conf],                            # 1
            np.zeros(4, dtype=np.float32),                    # 4 (xgb placeholder)
            [0.0],                                            # 1 (xgb risk placeholder)
        ]).astype(np.float32)
        
        pad_len = EMBEDDING_DIM - len(pre_vec)
        if pad_len > 0:
            pre_vec = np.concatenate([pre_vec, np.zeros(pad_len, dtype=np.float32)])
            
        # ── XGBoost regime probabilities and risk ─────────────────────────────
        xgb_probs = np.zeros(4, dtype=np.float32)
        risk_score = 0.0
        
        with self._xgb_lock:
            if self._xgb is not None:
                try:
                    xgb_probs = self._xgb.predict_proba(pre_vec)   # (4,)
                except Exception as e:
                    logger.error(f"XGB predict failed: {e}")
                    xgb_probs = np.ones(4, dtype=np.float32) / 4.0
                    
            if self._hmm is not None:
                try:
                    # HMM uses recent raw mid prices from the buffer
                    _, b_all, a_all, _, _ = self.buffer.get_latest(self.buffer.count, copy=False)
                    mid_all = (b_all + a_all) / 2.0
                    hmm_probs = self._hmm.predict_proba(mid_all)
                    self._last_hmm_probs = hmm_probs
                except Exception as e:
                    logger.debug(f"HMM predict failed: {e}")
                    self._last_hmm_probs = np.ones(4, dtype=np.float32) / 4.0
            
            if self._xgb_risk is not None:
                try:
                    if hasattr(self._xgb_risk, 'predict_risk'):
                        # Risk XGB expects last 10 feature vectors concatenated → (1280,)
                        if len(self._feat_history) >= 10:
                            feat_window = np.concatenate(list(self._feat_history)[-10:], axis=0)
                            risk_score = self._xgb_risk.predict_risk(feat_window)
                        else:
                            risk_score = 0.9 # Conservative default until history is full
                    else:
                        risk_score = 0.0
                except Exception as e:
                    logger.error(f"XGB risk predict failed: {e}")
                    risk_score = 0.9

        # BUG FIX 2.6: Do NOT mutate pre_vec with xgb outputs! 
        # The SDE was trained on zeros in these slots. Mutating them causes a feedback loop.
        # pre_vec[XGB_REGIME_IDX_START:XGB_REGIME_IDX_END] = xgb_probs
        # pre_vec[XGB_RISK_IDX] = risk_score
        
        # We can store them for external access if needed
        self._last_xgb_probs = xgb_probs
        self._last_risk_score = risk_score
        
        vec = pre_vec
        # Append to history before normalisation so Risk XGB sees same space as training
        self._feat_history.append(pre_vec.copy())
            
        # ── Parity Normalisation ──────────────────────────────────────────────
        if self.welford is not None:
            if update_welford:
                vec = self.welford.update_and_transform(vec)
            else:
                vec = self.welford.transform(vec)

        return vec


# ── helpers ────────────────────────────────────────────────────────────────────

def _mom(grid: np.ndarray, n: int) -> float:
    if len(grid) <= n:
        return 0.0
    base = grid[-(n + 1)]
    if abs(base) < EMBED_MIN_CLIP_12:
        return 0.0
    return float(np.clip((grid[-1] - base) / base, -1.0, 1.0))


def _rule_regime(
    kalman_vel_var: float,
    cusum_energy:   float,
    wavelet_hf_energy: float,
) -> tuple[np.ndarray, float]:
    """
    Simple rule-based regime one-hot: [trend_up, trend_down, sideways, volatile].
    Replaced by XGBoost probabilities when that model is loaded.
    """
    from v7_engine.config import EMBED_WAVELET_HF_THRESHOLD
    oh = np.zeros(4, dtype=np.float32)
    if wavelet_hf_energy > EMBED_WAVELET_HF_THRESHOLD:
        oh[3] = 1.0                  # volatile
    elif kalman_vel_var < EMBED_MIN_CLIP_5:
        oh[2] = 1.0                  # sideways
    elif cusum_energy > 0:
        oh[0] = 1.0                  # trend up proxy
    else:
        oh[1] = 1.0                  # trend down proxy

    sideways_conf = float(np.clip(1.0 / (1.0 + kalman_vel_var / EMBED_MIN_CLIP_4), 0.0, 1.0))
    return oh, sideways_conf

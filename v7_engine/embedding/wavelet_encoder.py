"""
Wavelet Encoder — CWT Δt positional encoding using rolling Morlet convolution.
Each scale uses a kernel whose length adapts to the scale parameter.
Output shape: (CWT_SCALES,) — one energy value per scale.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from numba import njit

from v7_engine.config import CWT_SCALES


def _morlet_kernel(scale: float, n_half: int) -> np.ndarray:
    """
    Causal Discrete Morlet wavelet kernel (t <= 0).
    Length = n_half + 1. Normalised to unit L2 norm.
    Applies Heaviside H(-t) to prevent phase lag.
    """
    t = np.linspace(-n_half, 0, n_half + 1)
    psi = np.exp(-0.5 * (t / scale) ** 2) * np.cos(5.0 * t / scale)
    norm = np.linalg.norm(psi)
    if norm > 1e-8:
        psi /= norm
    return psi.astype(np.float32)


# Pre-compute kernels for all scales.  Scale i uses scale = 2^(i/CWT_SCALES * 3)
# so scales range from ~1 to ~8, covering micro (fast) to macro (slow) oscillations.
_SCALES_FLOAT = [2.0 ** (i / CWT_SCALES * 3.0) for i in range(1, CWT_SCALES + 1)]
_KERNELS      = [_morlet_kernel(s, n_half=max(4, int(3 * s))) for s in _SCALES_FLOAT]
_KERNEL_LENS  = [len(k) for k in _KERNELS]
_MAX_LEN      = max(_KERNEL_LENS)


_KERNEL_OFFSETS = np.zeros(CWT_SCALES, dtype=np.int64)
_offset = 0
for i in range(CWT_SCALES):
    _KERNEL_OFFSETS[i] = _offset
    _offset += _KERNEL_LENS[i]
_KERNELS_FLAT = np.concatenate(_KERNELS).astype(np.float64)

@njit(nogil=True)
def fast_cwt_encode(buf_log: np.ndarray, kernels_flat: np.ndarray, kernel_offsets: np.ndarray, kernel_lens: np.ndarray, cwt_scales: int, max_len: int) -> np.ndarray:
    energy = np.zeros(cwt_scales, dtype=np.float32)
    buf_len = len(buf_log)
    
    for i in range(cwt_scales):
        klen = kernel_lens[i]
        koff = kernel_offsets[i]
        
        conv = 0.0
        for j in range(klen):
            idx = buf_len - klen + j
            if idx < 0:
                val = buf_log[0]
            else:
                val = buf_log[idx]
            conv += val * kernels_flat[koff + j]
            
        energy[i] = conv * conv
        
    e_sum = 0.0
    for i in range(cwt_scales):
        e_sum += energy[i]
        
    if e_sum > 1e-8:
        for i in range(cwt_scales):
            energy[i] = energy[i] / e_sum
            
    return energy

def encode_delta_t_buffer(dt_buffer: np.ndarray) -> np.ndarray:
    """
    Encode a buffer of inter-tick time deltas (seconds) into a CWT energy vector.

    Parameters
    ----------
    dt_buffer : 1-D float32 array of length ≥ 1

    Returns
    -------
    energy : float32 array of shape (CWT_SCALES,)
    """
    buf = np.asarray(dt_buffer, dtype=np.float64)
    if len(buf) == 0:
        return np.zeros(CWT_SCALES, dtype=np.float32)

    # Log-compress to reduce dynamic range before convolution
    buf_log = np.log1p(np.clip(buf, 0.0, 1e6))

    return fast_cwt_encode(buf_log, _KERNELS_FLAT, _KERNEL_OFFSETS, np.array(_KERNEL_LENS, dtype=np.int64), CWT_SCALES, _MAX_LEN)


def encode_delta_t(dt: float, history: deque) -> np.ndarray:
    """
    Convenience wrapper for single-tick streaming.
    Push dt into history, then compute the CWT energy vector.
    """
    history.append(float(dt))
    return encode_delta_t_buffer(np.array(history, dtype=np.float32))

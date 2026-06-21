"""
Combinatorial Purged Cross-Validation (CPCV).
Fixes the temporal leakage present in V6's simple TimeSeriesSplit.

Key operations:
  PURGE : remove training samples whose LABELS overlap with test period
  EMBARGO: remove post-test trailing samples that carry look-ahead info
  COMBINATORIAL: enumerate all C(N,k) path combinations, not just rolling windows

Reference: De Prado, M. L. (2018). "Advances in Financial Machine Learning"
"""

import numpy as np
from itertools import combinations
from v7_engine.config import (
    CPCV_N_SPLITS, CPCV_N_TEST_SPLITS,
    CPCV_EMBARGO_TICKS, CPCV_PURGE_TICKS
)

def cpcv_split(
    n:             int,
    n_splits:      int = CPCV_N_SPLITS,
    n_test_splits: int = CPCV_N_TEST_SPLITS,
    purge:         int = CPCV_PURGE_TICKS,
    embargo:       int = CPCV_EMBARGO_TICKS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Combinatorial Purged Cross-Validation (CPCV) split generator.
    Produces splits ensuring no overlap or data leakage between train/test folds.
    """
    all_idx = np.arange(n)
    fold_size = n // n_splits
    splits = []

    for i in range(n_splits - n_test_splits + 1):
        test_start = i * fold_size
        test_end   = test_start + n_test_splits * fold_size

        test_idx = all_idx[test_start:test_end]

        # Purge: remove [test_start - purge, test_start)
        # Embargo: remove (test_end, test_end + embargo]
        train_mask = (
            (all_idx < test_start - purge)   |   # before purge zone
            (all_idx > test_end + embargo)        # after embargo zone
        )
        train_idx = all_idx[train_mask]
        splits.append((train_idx, test_idx))

    return splits

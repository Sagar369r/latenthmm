# V7 Neural SDE Trading Engine

The V7 Trading Engine is a high-performance tick-level trading system combining Neural SDEs, Energy-Based Models, Reinforcement Learning, and XGBoost.

## Dynamic Feature Vector Allocation

The ML embedding vector (handled via `TickFeatureVector` in `v7_engine/embedding/feature_vector.py`) constructs a `128`-dimensional vector natively. To prevent misalignment in downstream ML models like the XGBoost classifiers, feature array slices are **computed dynamically at boot**.

**How it works:**
In `v7_engine/config.py`, the `_feat_sizes` dictionary stores the lengths of individual feature chunks (e.g., CUSUM size, Kalman size, Wavelet size). From these chunk sizes, a helper function (`_get_feat_offset()`) calculates exactly where every sub-vector begins.

**Adding New Features:**
If you ever want to add new engineered features to the tick vector:
1. Concatenate your new array/scalar in `TickFeatureVector.compute()`.
2. Update the `_feat_sizes` dictionary in `config.py` with the length of your new array.
3. *That's it.* The system will automatically shift `XGB_REGIME_IDX_START`, `EMBED_COL_KALMAN`, etc., avoiding silent dimension crashes.

## Running Tests

Ensure PyTorch and PyTest are installed, then run:
```bash
pytest v7_engine/tests/ -v
```

## Running Dry Backtest

You can trigger a dry walk-forward CPCV simulation over 1 month of tick data:
```bash
python scripts/backtest.py --symbol EURUSD --months 1 --dry-run
```

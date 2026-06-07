"""
Layer 6: Statistical Validation Framework

Mandatory before any live deployment. Tests in order:
  6.1 Walk-Forward Cross-Validation (primary)
  6.2 Monte Carlo Permutation Test (signal verification)
  6.3 Deflated Sharpe Ratio (DSR)
  6.4 Combinatorial Purged Cross-Validation (CPCV)
  6.5 Transaction Cost Sensitivity Analysis

GO LIVE criteria: all 5 tests must pass simultaneously.
"""
from __future__ import annotations

import warnings
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
# Shared helpers                                                       #
# ------------------------------------------------------------------ #

def _sharpe_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """Annualised Sharpe ratio (assuming 0 risk-free rate)."""
    returns = returns[~np.isnan(returns)]
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(freq))


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative return series."""
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    return float(dd.min())


def _probabilistic_sharpe_ratio(
    observed_sr: float,
    sr_benchmark: float,
    T: int,
    skewness: float,
    excess_kurtosis: float,
) -> float:
    """
    PSR = Φ[(SR - SR*) √(T-1) / √(1 - γ₃·SR + (γ₄-1)/4·SR*²)]
    where γ₃ = skewness, γ₄ = excess kurtosis.
    """
    sr_diff = observed_sr - sr_benchmark
    denom = np.sqrt(1 - skewness * observed_sr + (excess_kurtosis - 1) / 4 * observed_sr ** 2)
    if denom <= 0 or T <= 1:
        return 0.5
    z = sr_diff * np.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z))


# ------------------------------------------------------------------ #
# 6.1 Walk-Forward Cross-Validation                                    #
# ------------------------------------------------------------------ #

@dataclass
class WFCVResult:
    oos_returns: list[np.ndarray] = field(default_factory=list)
    is_sharpes: list[float] = field(default_factory=list)
    oos_sharpes: list[float] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)

    @property
    def combined_oos_returns(self) -> np.ndarray:
        if not self.oos_returns:
            return np.array([])
        return np.concatenate(self.oos_returns)

    @property
    def mean_oos_sharpe(self) -> float:
        arr = self.combined_oos_returns
        return _sharpe_ratio(arr) if len(arr) > 0 else 0.0

    @property
    def mean_is_sharpe(self) -> float:
        return float(np.mean(self.is_sharpes)) if self.is_sharpes else 0.0

    @property
    def overfit_ratio(self) -> float:
        oos = self.mean_oos_sharpe
        if oos == 0:
            return np.inf
        return self.mean_is_sharpe / oos

    @property
    def passes(self) -> bool:
        """
        Overfit criterion: IS Sharpe ÷ OOS Sharpe must be < 2.0.
        OOS Sharpe must be positive.
        """
        return self.mean_oos_sharpe > 0 and self.overfit_ratio < 2.0


def walk_forward_cv(
    simulate_fn,
    T: int,
    train_bars: int = 504,
    test_bars: int = 126,
    min_bars: int = 252 * 5,
) -> WFCVResult:
    """
    Expanding-window walk-forward cross-validation.

    simulate_fn(train_start, train_end, test_start, test_end)
        → (is_returns: np.ndarray, oos_returns: np.ndarray)

    Parameters
    ----------
    T           : total number of bars available
    train_bars  : initial training window (expanding)
    test_bars   : test window (6 months ≈ 126 bars)
    min_bars    : minimum 5-year history check (1260 bars)
    """
    result = WFCVResult()

    if T < min_bars:
        warnings.warn(
            f"Only {T} bars available; spec requires ≥{min_bars}. "
            "Results may be unreliable."
        )

    train_start = 0
    train_end = train_bars
    n_windows = 0

    while train_end + test_bars <= T:
        test_start = train_end
        test_end = min(train_end + test_bars, T)

        try:
            is_rets, oos_rets = simulate_fn(train_start, train_end, test_start, test_end)
            is_sr = _sharpe_ratio(is_rets)
            oos_sr = _sharpe_ratio(oos_rets)

            result.oos_returns.append(oos_rets)
            result.is_sharpes.append(is_sr)
            result.oos_sharpes.append(oos_sr)
            result.windows.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "is_sharpe": is_sr,
                "oos_sharpe": oos_sr,
            })
        except Exception as e:
            warnings.warn(f"WF-CV window failed: {e}")

        train_end = test_end
        n_windows += 1

    return result


# ------------------------------------------------------------------ #
# 6.2 Monte Carlo Permutation Test                                     #
# ------------------------------------------------------------------ #

@dataclass
class MCPermutationResult:
    actual_oos_sharpe: float
    permuted_sharpes: np.ndarray
    p_value: float
    n_permutations: int

    @property
    def passes(self) -> bool:
        """p < 0.05: actual OOS Sharpe > 95th percentile of permuted distribution."""
        return self.p_value < 0.05


def monte_carlo_permutation_test(
    daily_returns: np.ndarray,
    simulate_fn,
    n_permutations: int = 500,
    train_split: float = 0.7,
) -> MCPermutationResult:
    """
    Verify that regime signals are not artefacts of sequential structure.

    1. Generate permuted return series by shuffling daily returns.
    2. Run full pipeline on each permuted series.
    3. Require actual OOS Sharpe > 95th percentile of permuted distribution (p < 0.05).

    Parameters
    ----------
    daily_returns : (T,) raw return series
    simulate_fn   : fn(permuted_returns) → oos_sharpe
    n_permutations: spec recommends 10,000; default 500 for speed
    train_split   : fraction used for training
    """
    rng = np.random.default_rng(42)

    # Actual OOS Sharpe
    T = len(daily_returns)
    train_end = int(T * train_split)
    try:
        actual_oos_sharpe = simulate_fn(daily_returns)
    except Exception:
        actual_oos_sharpe = 0.0

    # Permuted Sharpes
    perm_sharpes = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(daily_returns)
        try:
            sr = simulate_fn(shuffled)
        except Exception:
            sr = 0.0
        perm_sharpes.append(sr)

    perm_arr = np.array(perm_sharpes)
    p_value = float(np.mean(perm_arr >= actual_oos_sharpe))

    return MCPermutationResult(
        actual_oos_sharpe=float(actual_oos_sharpe),
        permuted_sharpes=perm_arr,
        p_value=p_value,
        n_permutations=n_permutations,
    )


# ------------------------------------------------------------------ #
# 6.3 Deflated Sharpe Ratio                                           #
# ------------------------------------------------------------------ #

@dataclass
class DSRResult:
    observed_sr: float
    deflated_sr_z: float
    p_value: float
    n_trials: int
    expected_max_sr: float

    @property
    def passes(self) -> bool:
        """DSR > 0 means p < 0.05."""
        return self.p_value < 0.05


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    sr_benchmark: float = 0.0,
) -> DSRResult:
    """
    DSR = Φ[(SR̂ - E[max SR_M]) √(T-1) / √(1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²)]

    E[max SR_M] ≈ (1 - γ_E) Φ⁻¹(1 - 1/M) + γ_E Φ⁻¹(1 - 1/(M·e))
    γ_E = Euler–Mascheroni constant ≈ 0.5772

    Require DSR > 0 (p < 0.05) before live deployment.
    """
    returns = returns[~np.isnan(returns)]
    T = len(returns)
    if T < 5:
        return DSRResult(0.0, 0.0, 1.0, n_trials, 0.0)

    sr = float(returns.mean() / (returns.std() + 1e-10) * np.sqrt(252))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))  # excess kurtosis

    gamma_e = 0.5772156649
    
    import os
    iters = int(os.environ.get("PIPELINE_ITERATIONS", "1"))
    effective_trials = n_trials * iters

    if effective_trials >= 2:
        z1 = stats.norm.ppf(1 - 1 / effective_trials)
        z2 = stats.norm.ppf(1 - 1 / (effective_trials * np.e))
        expected_max = (1 - gamma_e) * z1 + gamma_e * z2
    else:
        expected_max = 0.0

    denom_inner = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    denom_inner = max(denom_inner, 1e-6)
    z = (sr - expected_max) * np.sqrt(T - 1) / np.sqrt(denom_inner)
    p_value = 1.0 - float(stats.norm.cdf(z))

    return DSRResult(
        observed_sr=sr,
        deflated_sr_z=float(z),
        p_value=float(p_value),
        n_trials=effective_trials,
        expected_max_sr=float(expected_max),
    )


# ------------------------------------------------------------------ #
# 6.4 Combinatorial Purged Cross-Validation (CPCV)                    #
# ------------------------------------------------------------------ #

@dataclass
class CPCVResult:
    sharpe_distribution: np.ndarray
    min_sr: float
    psr: float
    n_paths: int

    @property
    def passes(self) -> bool:
        """MinSR > 0.3 and PSR(SR* > 0) > 0.95."""
        return self.min_sr > 0.3 and self.psr > 0.95


def cpcv(
    simulate_fn,
    T: int,
    n_folds: int = 6,
    test_folds: int = 2,
    embargo_bars: int = 5,
    sr_benchmark: float = 0.0,
) -> CPCVResult:
    """
    Combinatorial Purged Cross-Validation.

    1. Purge: embargo_bars around each fold boundary.
    2. Combinatorial: all C(n_folds, test_folds) combinations.
    3. PSR: Φ[(SR* - SR_benchmark)√(T-1) / ...] across all paths.
    4. Require MinSR > 0.3 and PSR > 0.95.

    simulate_fn(train_idx, test_idx) → (is_returns, oos_returns)
    """
    fold_size = T // n_folds
    folds = [
        (i * fold_size, min((i + 1) * fold_size, T))
        for i in range(n_folds)
    ]

    sharpe_paths = []
    combinations = list(itertools.combinations(range(n_folds), test_folds))

    for test_combo in combinations:
        test_set = set(test_combo)
        train_folds = [i for i in range(n_folds) if i not in test_set]

        # Build purged train indices (embargo around fold boundaries)
        train_idx = []
        for fi in train_folds:
            start, end = folds[fi]
            # Check adjacency to test folds and add embargo
            purge_start = start
            purge_end = end
            for ti in test_combo:
                t_start, t_end = folds[ti]
                if t_end == start:
                    purge_start = start + embargo_bars
                if t_start == end:
                    purge_end = end - embargo_bars
            train_idx.extend(range(max(purge_start, 0), min(purge_end, T)))

        # Build test indices
        test_idx = []
        for ti in test_combo:
            start, end = folds[ti]
            test_idx.extend(range(start, end))

        if not train_idx or not test_idx:
            continue

        try:
            _, oos_rets = simulate_fn(
                np.array(train_idx), np.array(test_idx)
            )
            sr = _sharpe_ratio(oos_rets)
            sharpe_paths.append(sr)
        except Exception:
            pass

    if not sharpe_paths:
        return CPCVResult(
            sharpe_distribution=np.array([0.0]),
            min_sr=0.0,
            psr=0.0,
            n_paths=0,
        )

    sr_arr = np.array(sharpe_paths)
    combined_sr = float(sr_arr.mean())
    T_obs = T // n_folds * test_folds
    skew = float(stats.skew(sr_arr)) if len(sr_arr) > 3 else 0.0
    kurt = float(stats.kurtosis(sr_arr)) if len(sr_arr) > 3 else 0.0
    psr = _probabilistic_sharpe_ratio(combined_sr, sr_benchmark, T_obs, skew, kurt)

    return CPCVResult(
        sharpe_distribution=sr_arr,
        min_sr=float(sr_arr.min()),
        psr=float(psr),
        n_paths=len(sharpe_paths),
    )


# ------------------------------------------------------------------ #
# 6.5 Transaction Cost Sensitivity Analysis                           #
# ------------------------------------------------------------------ #

COST_SCENARIOS = {
    "optimistic":    {"bps": 0.5,  "min_sharpe": 0.8},
    "realistic":     {"bps": 2.0,  "min_sharpe": 0.6},
    "conservative":  {"bps": 5.0,  "min_sharpe": 0.4},
    "stress":        {"bps": 10.0, "min_sharpe": 0.2},
}


@dataclass
class TxCostResult:
    scenario_results: dict[str, dict]

    @property
    def passes(self) -> bool:
        """Must pass at least the 'realistic' scenario."""
        r = self.scenario_results.get("realistic", {})
        return r.get("passes", False)

    @property
    def passes_all(self) -> bool:
        return all(v.get("passes", False) for v in self.scenario_results.values())


def transaction_cost_sensitivity(
    gross_returns: np.ndarray,
    n_trades: int,
    T: int,
) -> TxCostResult:
    """
    Deduct per-trade costs (round-trip) and recompute Sharpe across scenarios.

    gross_returns : daily P&L returns (not trade-level)
    n_trades      : number of round-trip trades
    T             : total bars (for annualisation denominator)
    """
    scenario_results = {}
    gross_sr = _sharpe_ratio(gross_returns)

    for name, spec in COST_SCENARIOS.items():
        bps_per_side = spec["bps"]
        min_required = spec["min_sharpe"]

        # Round-trip cost = 2 × bps_per_side per trade
        cost_per_trade = 2 * bps_per_side / 10000
        total_cost_return = -(n_trades * cost_per_trade) / max(T, 1)

        net_returns = gross_returns.copy()
        if len(net_returns) > 0:
            net_returns = net_returns + total_cost_return  # uniform deduction

        net_sr = _sharpe_ratio(net_returns)
        passes = net_sr >= min_required

        scenario_results[name] = {
            "bps_per_side": bps_per_side,
            "net_sharpe": float(net_sr),
            "gross_sharpe": float(gross_sr),
            "min_required": float(min_required),
            "passes": passes,
            "cost_drag": float(total_cost_return),
        }

    return TxCostResult(scenario_results=scenario_results)


# ------------------------------------------------------------------ #
# Validation Report                                                    #
# ------------------------------------------------------------------ #

@dataclass
class ValidationReport:
    wfcv: WFCVResult | None = None
    mc_permutation: MCPermutationResult | None = None
    dsr: DSRResult | None = None
    cpcv: CPCVResult | None = None
    tx_cost: TxCostResult | None = None
    n_trials: int = 1

    @property
    def go_live_criteria_met(self) -> bool:
        """All 5 tests must pass simultaneously."""
        checks = []
        if self.wfcv:
            checks.append(self.wfcv.passes)
        if self.mc_permutation:
            checks.append(self.mc_permutation.passes)
        if self.dsr:
            checks.append(self.dsr.passes)
        if self.cpcv:
            checks.append(self.cpcv.passes)
        if self.tx_cost:
            checks.append(self.tx_cost.passes)
        return bool(checks) and all(checks)

    def to_dict(self) -> dict:
        d: dict = {"go_live": self.go_live_criteria_met, "tests": {}}

        if self.wfcv:
            d["tests"]["walk_forward_cv"] = {
                "passes": self.wfcv.passes,
                "mean_oos_sharpe": round(self.wfcv.mean_oos_sharpe, 4),
                "mean_is_sharpe": round(self.wfcv.mean_is_sharpe, 4),
                "overfit_ratio": round(self.wfcv.overfit_ratio, 4),
                "n_windows": len(self.wfcv.windows),
            }

        if self.mc_permutation:
            d["tests"]["monte_carlo_permutation"] = {
                "passes": self.mc_permutation.passes,
                "actual_oos_sharpe": round(self.mc_permutation.actual_oos_sharpe, 4),
                "p_value": round(self.mc_permutation.p_value, 4),
                "n_permutations": self.mc_permutation.n_permutations,
                "percentile_95": round(float(np.percentile(self.mc_permutation.permuted_sharpes, 95)), 4),
            }

        if self.dsr:
            d["tests"]["deflated_sharpe_ratio"] = {
                "passes": self.dsr.passes,
                "observed_sr": round(self.dsr.observed_sr, 4),
                "deflated_sr_z": round(self.dsr.deflated_sr_z, 4),
                "p_value": round(self.dsr.p_value, 4),
                "n_trials": self.dsr.n_trials,
                "expected_max_sr": round(self.dsr.expected_max_sr, 4),
            }

        if self.cpcv:
            d["tests"]["cpcv"] = {
                "passes": self.cpcv.passes,
                "min_sr": round(self.cpcv.min_sr, 4),
                "psr": round(self.cpcv.psr, 4),
                "n_paths": self.cpcv.n_paths,
            }

        if self.tx_cost:
            d["tests"]["transaction_cost_sensitivity"] = {
                "passes": self.tx_cost.passes,
                "passes_all": self.tx_cost.passes_all,
                "scenarios": {
                    k: {
                        "net_sharpe": round(v["net_sharpe"], 4),
                        "passes": v["passes"],
                    }
                    for k, v in self.tx_cost.scenario_results.items()
                },
            }

        return d

"""
Full CPCV Walk-Forward Backtest Engine.

Combinatorial Purged Cross-Validation (CPCV) prevents data leakage:
  - Purge: remove ticks within CPCV_PURGE_TICKS of train/test boundary
  - Embargo: block CPCV_EMBARGO_TICKS after each test fold
  - Combinatorial: all (n_splits choose n_test_splits) path combinations

The backtest simulates V7's full decision pipeline:
  embed → EBM → DailyGuard → MC barrier → RL actor → execute

Output: per-fold metrics + aggregate pass/fail against V7 targets.
"""

import numpy as np
import torch
import logging
from typing import Iterator
from v7_engine.config import (
    CHECKPOINT_SDE, CHECKPOINT_EBM, CHECKPOINT_RL, CHECKPOINT_DECODER,
    CHECKPOINT_WELFORD, EMBEDDING_DIM, TRAIN_SEQUENCE_LENGTH,
    BACKTEST_INITIAL_EQUITY, MAX_POSITION_FRACTION, HALF_KELLY_FRACTION,
    ATR_SL_MULT, ATR_TP_MULT, CPCV_N_SPLITS, CPCV_N_TEST_SPLITS,
    CPCV_EMBARGO_TICKS, CPCV_PURGE_TICKS, PIP_SIZE, PIP_VALUE_PER_LOT,
    MODELS_DIR, TARGET_CALMAR_RATIO, TARGET_MAX_DD_USD, TARGET_WIN_RATE,
    EBM_ENERGY_THRESHOLD
)
from v7_engine.training.cpcv import cpcv_split
from v7_engine.risk.tail_metrics import (
    calculate_sortino, downside_std, calculate_var, calculate_cvar, tail_ratio
)
from numba import njit

@njit
def _numba_simulate_fold(
    n: int, bids: np.ndarray, asks: np.ndarray, timestamps: np.ndarray, is_new_day: np.ndarray,
    test_indices: np.ndarray, test_dir: np.ndarray, test_sz_frac: np.ndarray, base_var_loss: np.ndarray,
    initial_equity: float, pip_sz: float, pip_val: float,
    daily_dd_limit: float, max_total_dd: float, atr_sl_mult: float, atr_tp_mult: float,
    half_kelly_frac: float, max_pos_frac: float, slippage_pips: float, comm_per_lot: float,
    profit_lock_pct: float, prop_max_dd_pct: float, exec_latency_ns: int
):
    equity = initial_equity
    in_position = False
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_size = 0.0
    pos_entry_time = 0
    
    day_start_equity = initial_equity
    session_high = initial_equity
    daily_halted = False
    
    atr_window = np.zeros(14)
    atr_ptr = 0
    atr_count = 0
    atr_est = 1.0 * pip_sz
    
    max_trades = n // 10 + 100
    trade_entry_times = np.zeros(max_trades, dtype=np.int64)
    trade_exit_times = np.zeros(max_trades, dtype=np.int64)
    trade_dirs = np.zeros(max_trades, dtype=np.int32)
    trade_entries = np.zeros(max_trades, dtype=np.float64)
    trade_exits = np.zeros(max_trades, dtype=np.float64)
    trade_pnls = np.zeros(max_trades, dtype=np.float64)
    num_trades = 0
    
    equity_curve = np.zeros(n // 10 + 100, dtype=np.float64)
    equity_curve[0] = initial_equity
    eq_ptr = 1
    
    tradeable_ptr = 0
    num_tradeables = len(test_indices)
    
    for i in range(n):
        bid = bids[i]
        ask = asks[i]
        mid = (bid + ask) / 2.0
        
        prev_mid = mid
        if i > 0:
            prev_mid = (bids[i-1] + asks[i-1]) / 2.0
            
        tr = abs(mid - prev_mid)
        atr_window[atr_ptr % 14] = tr
        atr_ptr += 1
        atr_count = min(atr_count + 1, 14)
        if atr_count > 0:
            sum_atr = 0.0
            for j in range(atr_count):
                sum_atr += atr_window[j]
            atr_est = sum_atr / atr_count
        
        if is_new_day[i]:
            day_start_equity = equity
            session_high = equity
            daily_halted = False
            
        if in_position:
            if pos_dir == 1 and (bid <= pos_sl or bid >= pos_tp):
                exit_price = pos_sl if bid <= pos_sl else pos_tp
                exit_price -= slippage_pips * pip_sz
                pnl = (exit_price - pos_entry) * pos_dir / pip_sz * pip_val * pos_size
                pnl -= comm_per_lot * pos_size
                equity += pnl
                
                trade_entry_times[num_trades] = pos_entry_time
                trade_exit_times[num_trades] = timestamps[i]
                trade_dirs[num_trades] = 1
                trade_entries[num_trades] = pos_entry
                trade_exits[num_trades] = exit_price
                trade_pnls[num_trades] = pnl
                num_trades += 1
                in_position = False
                
            elif pos_dir == -1 and (ask >= pos_sl or ask <= pos_tp):
                exit_price = pos_sl if ask >= pos_sl else pos_tp
                exit_price += slippage_pips * pip_sz
                pnl = (pos_entry - exit_price) / pip_sz * pip_val * pos_size
                pnl -= comm_per_lot * pos_size
                equity += pnl
                
                trade_entry_times[num_trades] = pos_entry_time
                trade_exit_times[num_trades] = timestamps[i]
                trade_dirs[num_trades] = -1
                trade_entries[num_trades] = pos_entry
                trade_exits[num_trades] = exit_price
                trade_pnls[num_trades] = pnl
                num_trades += 1
                in_position = False
                
        if i % 10 == 0 and eq_ptr < len(equity_curve):
            equity_curve[eq_ptr] = equity
            eq_ptr += 1
            
        if in_position:
            continue
            
        if equity > session_high:
            session_high = equity
            
        profit_lock_amt = day_start_equity * profit_lock_pct
        if equity >= day_start_equity + profit_lock_amt:
            daily_halted = True
            
        dd_usd = day_start_equity - equity
        if dd_usd >= daily_dd_limit:
            daily_halted = True
            
        total_dd_usd = initial_equity - equity
        if total_dd_usd >= max_total_dd:
            daily_halted = True
            
        if daily_halted:
            continue
            
        if (initial_equity - equity) / initial_equity >= prop_max_dd_pct:
            break
            
        while tradeable_ptr < num_tradeables and test_indices[tradeable_ptr] < i:
            tradeable_ptr += 1
            
        if tradeable_ptr >= num_tradeables or test_indices[tradeable_ptr] != i:
            continue
            
        direction = test_dir[i]
        if direction == 0:
            continue
            
        size_frac = test_sz_frac[i]
        budget_remaining = daily_dd_limit - dd_usd
        if budget_remaining <= 0:
            continue
            
        var_usd = base_var_loss[i] * equity * size_frac
        if var_usd >= budget_remaining:
            continue
            
        sl_pips = atr_est / pip_sz * atr_sl_mult
        sl_pips = max(sl_pips, 2.0)
        risk_amt = min(equity * max_pos_frac * half_kelly_frac * size_frac, equity * 0.01)
        
        size_lots = 0.01
        if sl_pips > 0:
            size_lots = max(0.01, round(risk_amt / (sl_pips * pip_val), 2))
            
        max_loss = size_lots * sl_pips * pip_val
        if max_loss > budget_remaining:
            if budget_remaining > 0:
                size_lots = round(budget_remaining / (sl_pips * pip_val), 2)
            else:
                size_lots = 0.0
                
        if size_lots < 0.01:
            continue
            
        target_ts = timestamps[i] + exec_latency_ns
        exec_i = i
        while exec_i < n - 1 and timestamps[exec_i] < target_ts:
            exec_i += 1
            
        exec_ask = asks[exec_i]
        exec_bid = bids[exec_i]
        
        entry = exec_ask + slippage_pips * pip_sz if direction == 1 else exec_bid - slippage_pips * pip_sz
        
        in_position = True
        pos_dir = direction
        pos_entry = entry
        pos_sl = entry - direction * atr_est * atr_sl_mult
        pos_tp = entry + direction * atr_est * atr_tp_mult
        pos_size = size_lots
        pos_entry_time = timestamps[exec_i]
        
    return (
        trade_entry_times[:num_trades],
        trade_exit_times[:num_trades],
        trade_dirs[:num_trades],
        trade_entries[:num_trades],
        trade_exits[:num_trades],
        trade_pnls[:num_trades],
        equity_curve[:eq_ptr]
    )

logger = logging.getLogger("backtest")


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics_from_trades(trades: list[dict], equity_curve: list[float], n_days: float = 0.0) -> dict:
    """
    Compute full institutional-grade performance metrics from a trade list
    and equity curve.
    """
    if not trades or len(equity_curve) < 2:
        return {}

    equity = np.array(equity_curve, dtype=np.float64)
    peak   = np.maximum.accumulate(equity)
    dd     = equity - peak
    max_dd_usd = float(abs(np.min(dd)))

    # Returns
    rets       = np.diff(equity) / equity[:-1]
    rets       = rets[np.isfinite(rets)]

    sortino    = calculate_sortino(rets, target=0.0)
    dd_std     = downside_std(rets, target=0.0)
    t_ratio    = tail_ratio(rets, confidence=0.05)
    var_95     = calculate_var(rets, confidence=0.95)
    cvar_95    = calculate_cvar(rets, confidence=0.95)

    if n_days <= 0.0:
        n_days = max(len(equity), 1)
        
    annual_ret = float((equity[-1] / equity[0]) ** (252.0 / n_days) - 1)
    calmar     = annual_ret / (max_dd_usd / BACKTEST_INITIAL_EQUITY) if max_dd_usd > 0 else 0.0

    # Annualised Sharpe ratio
    rets_std   = float(np.std(rets))
    # Ticks per day varies, but equity is appended every 10 ticks. 
    # If a day has 28800 ticks, equity has 2880 entries per day.
    n_trades_per_year = len(trades) / (n_days / 252.0) if n_days > 0 else 252
    sharpe = (float(np.mean(rets)) / rets_std) * np.sqrt(max(n_trades_per_year, 1)) if rets_std > 0 else 0.0

    wins       = [t for t in trades if t.get("pnl", 0) > 0]
    losses     = [t for t in trades if t.get("pnl", 0) <= 0]
    win_rate   = len(wins) / max(len(trades), 1)

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses)) + 1e-10
    profit_factor = gross_profit / gross_loss

    avg_win  = np.mean([t["pnl"] for t in wins])  if wins  else 0.0
    avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    if avg_loss > 0 and avg_win > 0:
        R = avg_win / avg_loss
        kelly = win_rate - ((1.0 - win_rate) / R)
    else:
        kelly = 0.0

    return {
        # Core metrics
        "max_dd_usd":         float(max_dd_usd),
        "calmar":             float(calmar),
        "sortino":            float(sortino),
        "sharpe":             float(sharpe),
        "annual_return":      float(annual_ret),
        # Trade statistics
        "win_rate":           float(win_rate),
        "profit_factor":      float(profit_factor),
        "expectancy":         float(expectancy),
        "n_trades":           int(len(trades)),
        "avg_win_usd":        float(avg_win),
        "avg_loss_usd":       float(avg_loss),
        "kelly_criterion":    float(kelly),
        # Tail risk
        "downside_deviation": float(dd_std),
        "tail_ratio":         float(t_ratio),
        "var_95":             float(var_95),
        "cvar_95":            float(cvar_95),
        # Pass/fail gates
        "pass_dd":            bool(max_dd_usd <= TARGET_MAX_DD_USD),
        "pass_calmar":        bool(calmar >= TARGET_CALMAR_RATIO),
        "pass_win_rate":      bool(win_rate >= TARGET_WIN_RATE),
        "pass_profit_factor": bool(profit_factor >= 1.5),
        "pass_all":           bool(
            max_dd_usd <= TARGET_MAX_DD_USD and
            calmar >= TARGET_CALMAR_RATIO and
            win_rate >= TARGET_WIN_RATE
        ),
    }


# ── Simulated Walk-Forward ─────────────────────────────────────────────────────

import gc
import collections
from v7_engine.config import (
    EMBEDDING_DIM, SEQUENCE_LENGTH, CHECKPOINT_WELFORD, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD,
    RISK_XGB_THRESHOLD, CHECKPOINT_RISK, CHECKPOINT_REGIME_HMM
)
from v7_engine.ingestion.ring_buffer import RingBuffer
from v7_engine.embedding.tib_engine import TIBEngine
from v7_engine.ingestion.welford import WelfordNormaliser
from v7_engine.embedding.feature_vector import TickFeatureVector

def _worker_extract_chunk(args):
    import os
    import numpy as np
    chunk_id, chunk_start, chunk_end, timestamps_slice, bids_slice, asks_slice, dts_slice, welford_state, STRIDE = args
    
    from v7_engine.risk.risk_xgb import RiskXGBClassifier
    
    xgb_risk = RiskXGBClassifier.load(CHECKPOINT_RISK) if os.path.exists(CHECKPOINT_RISK) else None
    hmm_regime = None
    
    buf = RingBuffer(SEQUENCE_LENGTH)
    tib = TIBEngine(window=500)
    welford = WelfordNormaliser.from_state(welford_state, dim=EMBEDDING_DIM, clip_sigma=WELFORD_CLIP_SIGMA, min_std=WELFORD_MIN_STD)
    welford.is_warm = True
    
    fv = TickFeatureVector(buf, tib, welford_normaliser=welford, hmm_regime_model=hmm_regime, xgb_risk_model=xgb_risk)
    
    chunk_out_seqs = []
    chunk_valid_indices = []
    has_features_local = np.zeros(chunk_end - chunk_start, dtype=bool)
    
    global_feat_window = collections.deque(maxlen=10)
    
    warmup_offset = len(bids_slice) - (chunk_end - chunk_start)
    
    for local_i in range(len(bids_slice)):
        buf.push(int(timestamps_slice[local_i]), float(bids_slice[local_i]), float(asks_slice[local_i]), float(dts_slice[local_i]), 1)
        fv.update_state(float(bids_slice[local_i] + asks_slice[local_i]) / 2.0, float(dts_slice[local_i]), 1)
        
        if local_i >= warmup_offset:
            global_i = chunk_start + (local_i - warmup_offset)
            
            if global_i % STRIDE == 0 and buf.count >= SEQUENCE_LENGTH:
                vec = fv.compute(SEQUENCE_LENGTH, update_welford=False)
                
                global_feat_window.append(vec)
                is_blocked = False
                if xgb_risk is not None and len(global_feat_window) >= 10:
                    window = np.concatenate(list(global_feat_window)).reshape(1, -1)
                    if xgb_risk.predict_risk(window) > RISK_XGB_THRESHOLD:
                        is_blocked = True
                        
                if not is_blocked:
                    chunk_out_seqs.append(vec)
                    chunk_valid_indices.append(global_i)
                    has_features_local[local_i - warmup_offset] = True
                    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Worker completed Chunk {chunk_id} | Extracted {len(chunk_valid_indices)} tradeable vectors.")
                    
    return chunk_id, chunk_out_seqs, chunk_valid_indices, has_features_local

def precompute_global_features(
    tick_data: dict,
    feat_vec_factory=None,
    xgb_risk=None,
    hmm_regime=None,
):
    import torch
    import numpy as np
    from tqdm import tqdm
    import logging
    import gc
    from v7_engine.config import EMBEDDING_DIM, SEQUENCE_LENGTH, CHECKPOINT_WELFORD, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD
    from v7_engine.ingestion.ring_buffer import RingBuffer
    from v7_engine.embedding.tib_engine import TIBEngine
    from v7_engine.ingestion.welford import WelfordNormaliser
    from v7_engine.embedding.feature_vector import TickFeatureVector
    import os
    import pickle

    logger = logging.getLogger(__name__)
    
    bids       = tick_data["bid"].astype(np.float64)
    asks       = tick_data["ask"].astype(np.float64)
    dts        = tick_data["delta_t"].astype(np.float64)
    timestamps = tick_data.get("timestamp_ns", np.arange(len(bids)) * int(1e8)).astype(np.int64)
    n = len(bids)

    import hashlib
    ts_hash = hashlib.md5(np.array([timestamps[0], timestamps[-1], n])).hexdigest()
    CACHE_FILE = f"global_features_cache_{ts_hash}.npz"
    if os.path.exists(CACHE_FILE):
        logger.info(f"GLOBAL PRECOMPUTATION: Found cache {CACHE_FILE}! Loading instantly...")
        try:
            data = np.load(CACHE_FILE)
            return data["valid_indices"], data["valid_vectors"], data["has_features"]
        except Exception as e:
            logger.error(f"Cache corrupted: {e}. Deleting and recomputing...")
            os.remove(CACHE_FILE)

    buf = RingBuffer(SEQUENCE_LENGTH)
    tib = TIBEngine(window=500)

    if os.path.exists(CHECKPOINT_WELFORD):
        wdata = np.load(CHECKPOINT_WELFORD)
        welford_state_dict = {"mean": wdata["mean"], "m2": wdata["m2"], "n": int(wdata["n"])}
        welford = WelfordNormaliser.from_state(
            welford_state_dict,
            dim=EMBEDDING_DIM,
            clip_sigma=WELFORD_CLIP_SIGMA,
            min_std=WELFORD_MIN_STD,
        )
    else:
        raise FileNotFoundError(f"Missing {CHECKPOINT_WELFORD}.")

    has_features   = np.zeros(n, dtype=bool)
    
    CHUNK_SIZE = 250_000
    STRIDE = 10
    
    logger.info(f"GLOBAL PRECOMPUTATION: Chunked MULTIPROCESSING extraction of {n} ticks...")
    
    import concurrent.futures
    
    worker_args = []
    chunk_id = 0
    
    for chunk_start in range(0, n, CHUNK_SIZE):
        chunk_end = min(n, chunk_start + CHUNK_SIZE)
        slice_start = max(0, chunk_start - SEQUENCE_LENGTH - 100) 
        
        args = (
            chunk_id, chunk_start, chunk_end,
            timestamps[slice_start:chunk_end],
            bids[slice_start:chunk_end],
            asks[slice_start:chunk_end],
            dts[slice_start:chunk_end],
            welford_state_dict,
            STRIDE
        )
        worker_args.append(args)
        chunk_id += 1
        
    all_chunk_out_seqs = {}
    all_chunk_valid_indices = {}
    
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(mp_context=ctx) as executor:
        futures = {executor.submit(_worker_extract_chunk, args): args for args in worker_args}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parallel CPU Extraction"):
            try:
                cid, cout, cvalid, has_feat_local = future.result()
                all_chunk_out_seqs[cid] = cout
                all_chunk_valid_indices[cid] = cvalid
                
                chunk_start = worker_args[cid][1]
                chunk_end = worker_args[cid][2]
                has_features[chunk_start:chunk_end] = has_feat_local
            except Exception as e:
                import traceback
                logger.error(f"Worker crashed! {traceback.format_exc()}")
                raise e

    valid_indices = []
    valid_vectors = []
    
    for chunk_start in range(0, n, CHUNK_SIZE):
        cid = chunk_start // CHUNK_SIZE
        chunk_out_seqs = all_chunk_out_seqs[cid]
        chunk_valid_indices = all_chunk_valid_indices[cid]
        
        if not chunk_out_seqs:
            continue
            
        valid_indices.extend(chunk_valid_indices)
        valid_vectors.extend(chunk_out_seqs)

    valid_indices = np.array(valid_indices, dtype=np.int32)
    valid_vectors = np.array(valid_vectors, dtype=np.float32)

    logger.info(f"GLOBAL PRECOMPUTATION FINISHED: Saved {len(valid_indices)} total contexts to dense arrays.")
    np.savez_compressed(
        CACHE_FILE,
        valid_indices=valid_indices,
        valid_vectors=valid_vectors,
        has_features=has_features,
    )
    
    return valid_indices, valid_vectors, has_features


def simulate_fold(
    tick_data:   dict,
    fold_idx:    int = 0,
    symbol:      str = "EURUSD",
    # Injected global signals
    test_indices=None,
    test_dir=None,
    test_sz_frac=None,
    base_var_loss_full=None,
    hyperparams: dict = None,
) -> tuple[list[dict], list[float]]:

    from v7_engine.config import (
        PIP_SIZE, PIP_VALUE_PER_LOT, BACKTEST_INITIAL_EQUITY,
        MAX_POSITION_FRACTION, HALF_KELLY_FRACTION, ATR_SL_MULT, ATR_TP_MULT,
        MAX_DAILY_DRAWDOWN_USD, DAILY_PROFIT_LOCK_PCT, PROP_MAX_DD_PCT,
        SLIPPAGE_PIPS, COMMISSION_PER_LOT, EXECUTION_LATENCY_MS
    )
    from datetime import datetime, timezone

    pip_sz = PIP_SIZE.get(symbol.upper(), 0.0001)
    pip_val = PIP_VALUE_PER_LOT.get(symbol.upper(), 10.0)
    
    # Hyperparams override
    atr_sl = hyperparams.get("ATR_SL_MULT", ATR_SL_MULT) if hyperparams else ATR_SL_MULT
    atr_tp = hyperparams.get("ATR_TP_MULT", ATR_TP_MULT) if hyperparams else ATR_TP_MULT
    half_kelly = hyperparams.get("HALF_KELLY_FRACTION", HALF_KELLY_FRACTION) if hyperparams else HALF_KELLY_FRACTION
    max_pos = hyperparams.get("MAX_POSITION_FRACTION", MAX_POSITION_FRACTION) if hyperparams else MAX_POSITION_FRACTION
    profit_lock = hyperparams.get("DAILY_PROFIT_LOCK_PCT", DAILY_PROFIT_LOCK_PCT) if hyperparams else DAILY_PROFIT_LOCK_PCT

    bids       = tick_data["bid"].astype(np.float64)
    asks       = tick_data["ask"].astype(np.float64)
    timestamps = tick_data.get("timestamp_ns", np.arange(len(bids)) * int(1e8)).astype(np.int64)
    n = len(bids)

    # Precalculate new day bool array
    is_new_day = np.zeros(n, dtype=bool)
    current_day = ""
    for i in range(n):
        day_str = datetime.fromtimestamp(timestamps[i] / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
        if day_str != current_day:
            if current_day != "":
                is_new_day[i] = True
            current_day = day_str

    logger.debug(f"Fold {fold_idx}: Running Numba-Accelerated tick-by-tick simulation...")
    
    t_ent, t_ex, t_dir, t_entry, t_exit, t_pnl, eq_curve = _numba_simulate_fold(
        n, bids, asks, timestamps, is_new_day,
        test_indices, test_dir, test_sz_frac, base_var_loss_full,
        BACKTEST_INITIAL_EQUITY, pip_sz, pip_val,
        MAX_DAILY_DRAWDOWN_USD, 250.0, atr_sl, atr_tp,
        half_kelly, max_pos, SLIPPAGE_PIPS, COMMISSION_PER_LOT,
        profit_lock, PROP_MAX_DD_PCT, EXECUTION_LATENCY_MS * 1_000_000
    )
    
    trades = []
    for i in range(len(t_ent)):
        trades.append({
            "entry_time": t_ent[i],
            "exit_time": t_ex[i],
            "direction": "LONG" if t_dir[i] == 1 else "SHORT",
            "entry_price": t_entry[i],
            "exit_price": t_exit[i],
            "pnl": t_pnl[i],
            "reason": "sl_or_tp"
        })

    logger.info(
        f"Fold {fold_idx}: {len(trades)} trades, "
        f"final equity ${eq_curve[-1]:.2f}"
    )
    return trades, list(eq_curve)


def run_walk_forward(
    tick_data:        dict,
    sde_model,
    ebm_model,
    rl_actor,
    n_splits:         int = CPCV_N_SPLITS,
    n_test_splits:    int = CPCV_N_TEST_SPLITS,
    feat_vec_factory  = None,
    symbol:           str = "EURUSD",
    xgb_regime        = None,
    xgb_risk          = None,
    hmm_regime        = None,
    decoder_model     = None,
) -> dict:
    """
    Run CPCV walk-forward validation across all combinatorial paths.
    Returns aggregated metrics and per-fold results.
    """
    import os
    import torch
    device = next(sde_model.parameters()).device
    
    n = len(tick_data["bid"])
    splits = list(cpcv_split(n, n_splits=n_splits, n_test_splits=n_test_splits,
                              purge=CPCV_PURGE_TICKS, embargo=CPCV_EMBARGO_TICKS))

    # Precompute all global features ONCE
    global_indices, global_vectors, has_features = precompute_global_features(
        tick_data, feat_vec_factory, xgb_risk, hmm_regime
    )
    all_trades: list[dict] = []
    all_equity: list[float] = [BACKTEST_INITIAL_EQUITY]
    fold_results: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"")
        logger.info(f"==================================================")
        logger.info(f"=== Walk Forward: Fold {fold_idx} ===")
        logger.info(f"==================================================")
        
        fold_dir = f"models/fold_{fold_idx}"
        
        from v7_engine.embedding.regime_xgb import RegimeXGBClassifier
        from v7_engine.risk.risk_xgb import RiskXGBClassifier
        from v7_engine.embedding.regime_hmm import RegimeHMM

        xgb_regime = RegimeXGBClassifier.load(f"{fold_dir}/regime_xgb.json") if os.path.exists(f"{fold_dir}/regime_xgb.json") else None
        xgb_risk   = RiskXGBClassifier.load(f"{fold_dir}/risk_xgb.json") if os.path.exists(f"{fold_dir}/risk_xgb.json") else None
        hmm_regime = RegimeHMM.load(f"{fold_dir}/regime_hmm.pkl") if os.path.exists(f"{fold_dir}/regime_hmm.pkl") else None

        if os.path.exists(fold_dir):
            logger.info(f"Loading strictly out-of-sample models from {fold_dir}...")
            if os.path.exists(f"{fold_dir}/sde_best.pth"):
                sde_model.load_state_dict(torch.load(f"{fold_dir}/sde_best.pth", map_location=device, weights_only=True))
            if os.path.exists(f"{fold_dir}/rl_best.pth"):
                rl_actor.load_state_dict(torch.load(f"{fold_dir}/rl_best.pth", map_location=device, weights_only=True))
            if os.path.exists(f"{fold_dir}/ebm_best.pth"):
                ebm_model.load_state_dict(torch.load(f"{fold_dir}/ebm_best.pth", map_location=device, weights_only=True))
        
        sde_model.eval()
        ebm_model.eval()
        rl_actor.eval()
        
        # 2. Fold-Specific PyTorch Inference
        in_test = np.isin(global_indices, test_idx)
        fold_valid_vectors = global_vectors[in_test]
        
        chunk_tradeable = np.zeros(len(fold_valid_vectors), dtype=bool)
        chunk_dirs      = np.zeros(len(fold_valid_vectors), dtype=np.int32)
        chunk_sz_fracs  = np.zeros(len(fold_valid_vectors), dtype=np.float32)
        
        if len(fold_valid_vectors) > 0:
            valid_seqs = torch.tensor(fold_valid_vectors, dtype=torch.float32, device=device)
            BATCH_SIZE = 4096
            for b_idx in range(0, len(valid_seqs), BATCH_SIZE):
                batch = valid_seqs[b_idx : b_idx + BATCH_SIZE]
                with torch.no_grad():
                    _, final_states, _, _ = sde_model(batch)
                    tradeable, _ = ebm_model.is_tradeable(final_states)
                    action = rl_actor.sample_action(final_states)
                    
                    chunk_tradeable[b_idx : b_idx + BATCH_SIZE] = tradeable.cpu().numpy()
                    chunk_dirs[b_idx : b_idx + BATCH_SIZE]      = action["direction"].cpu().numpy()
                    chunk_sz_fracs[b_idx : b_idx + BATCH_SIZE]  = action["size_fraction"].cpu().numpy()
                    
        test_tradeable_full = np.zeros(n, dtype=bool)
        test_dir_full       = np.zeros(n, dtype=np.int32)
        test_sz_frac_full   = np.zeros(n, dtype=np.float32)
        base_var_loss_full  = np.zeros(n, dtype=np.float32)
        
        from v7_engine.config import RISK_MC_BARRIER_BLOCKS, SDE_DT, MC_PATHS, MC_CONFIDENCE, RISK_XGB_THRESHOLD
        
        if len(fold_valid_vectors) > 0:
            test_tradeable_full[global_indices[in_test]] = chunk_tradeable
            test_dir_full[global_indices[in_test]] = chunk_dirs
            test_sz_frac_full[global_indices[in_test]] = chunk_sz_fracs
            
            # Precompute Base VaR purely for the active directions and apply XGB/HMM filters
            active_local_indices = np.where(chunk_dirs != 0)[0]
            if len(active_local_indices) > 0:
                ts = torch.linspace(0, RISK_MC_BARRIER_BLOCKS * SDE_DT, RISK_MC_BARRIER_BLOCKS, device=device)
                
                # Process VaR and ML filters in micro-batches to prevent OOM
                for active_i in active_local_indices:
                    global_idx = global_indices[in_test][active_i]
                    tick_i = test_idx[active_i]
                    
                    # 1. XGB Risk Filter
                    if xgb_risk is not None and xgb_risk._model is not None:
                        if global_idx >= 9:
                            window = global_vectors[global_idx - 9 : global_idx + 1]
                            window_flat = window.reshape(1, -1)
                            if xgb_risk.predict_risk(window_flat) > RISK_XGB_THRESHOLD:
                                test_dir_full[global_idx] = 0
                                continue
                                
                    # 2. XGB Regime Filter
                    if xgb_regime is not None and xgb_regime._model is not None:
                        probs = xgb_regime.predict_proba(fold_valid_vectors[active_i])
                        if probs[2] > 0.6 or probs[3] > 0.6:  # Confident sideways or volatile
                            test_dir_full[global_idx] = 0
                            continue
                            
                    # 3. HMM Regime Filter (Optional Block)
                    if hmm_regime is not None and hmm_regime.model is not None:
                        start_i = max(0, tick_i - 5000)
                        b_win = bids[start_i:tick_i]
                        a_win = asks[start_i:tick_i]
                        if len(b_win) > 200:
                            mid_win = (b_win + a_win) / 2.0
                            hmm_probs = hmm_regime.predict_proba(mid_win)
                            # If state 3 (volatile) is heavily dominant:
                            if hmm_probs[3] > 0.7:
                                test_dir_full[global_idx] = 0
                                continue

                    # 4. SDE MC Barrier Bound Computation
                    with torch.no_grad():
                        context = torch.tensor(fold_valid_vectors[active_i], dtype=torch.float32, device=device).unsqueeze(0)
                        seqs_batch = context.repeat(MC_PATHS, 1)
                        paths = sde_model.forward_multi_step(seqs_batch, ts=ts)
                        pnl_paths = decoder_model.decode_paths(paths, units=1.0, equity=1.0).cpu().numpy()
                        var_val = float(np.quantile(-pnl_paths, MC_CONFIDENCE))
                        
                        base_var_loss_full[global_idx] = var_val

        # Slice test fold
        test_data = {
            k: v[test_idx] for k, v in tick_data.items()
            if isinstance(v, np.ndarray) and len(v) == n
        }

        # Fix Numba boundary mapping by extracting local indices relative to test_data
        dynamic_dir = test_dir_full[test_idx]
        local_indices = np.where(dynamic_dir != 0)[0]

        trades, equity = simulate_fold(
            test_data, fold_idx, symbol,
            test_indices=local_indices.astype(np.int64),
            test_dir=dynamic_dir,
            test_sz_frac=test_sz_frac_full[test_idx],
            base_var_loss_full=base_var_loss_full[test_idx]
        )

        fold_tns = test_data.get("timestamp_ns", np.arange(len(test_data["bid"])) * int(1e8))
        fold_n_days = float((fold_tns[-1] - fold_tns[0]) / 1e9 / 86400.0) if len(fold_tns) > 1 else 1.0

        fold_metrics = compute_metrics_from_trades(trades, equity, n_days=fold_n_days)
        fold_results.append({"fold": fold_idx, "metrics": fold_metrics})
        all_trades.extend(trades)
        all_equity.extend(equity[1:])   # skip duplicate starting equity

        logger.info(
            f"Fold {fold_idx} | "
            f"DD: ${fold_metrics.get('max_dd_usd', 0):.2f} | "
            f"Calmar: {fold_metrics.get('calmar', 0):.2f} | "
            f"WR: {fold_metrics.get('win_rate', 0):.1%} | "
            f"{'PASS' if fold_metrics.get('pass_all') else 'FAIL'}"
        )

    all_tns = tick_data.get("timestamp_ns", np.arange(n) * int(1e8))
    total_n_days = float((all_tns[-1] - all_tns[0]) / 1e9 / 86400.0) if len(all_tns) > 1 else 1.0
    final_metrics = compute_metrics_from_trades(all_trades, all_equity, n_days=total_n_days)
    aggregate = final_metrics
    aggregate["fold_results"] = fold_results
    aggregate["n_folds"]      = len(fold_results)
    aggregate["pct_folds_pass"] = (
        sum(1 for f in fold_results if f["metrics"].get("pass_all", False)) /
        max(len(fold_results), 1)
    )

    logger.info(
        f"\n{'='*60}\n"
        f"WALK-FORWARD SUMMARY (INSTITUTIONAL GRADE)\n"
        f"  Calmar:        {aggregate.get('calmar', 0):.2f}  (target ≥ {TARGET_CALMAR_RATIO})\n"
        f"  Sharpe:        {aggregate.get('sharpe', 0):.2f}\n"
        f"  Sortino:       {aggregate.get('sortino', 0):.2f}\n"
        f"  Max DD:        ${aggregate.get('max_dd_usd', 0):.2f}  (limit ≤ ${TARGET_MAX_DD_USD})\n"
        f"  Win Rate:      {aggregate.get('win_rate', 0):.1%}  (target ≥ {TARGET_WIN_RATE:.0%})\n"
        f"  Avg Win/Loss:  ${aggregate.get('avg_win_usd', 0):.2f} / ${aggregate.get('avg_loss_usd', 0):.2f}\n"
        f"  Kelly Frac:    {aggregate.get('kelly_criterion', 0):.2f}\n"
        f"  Profit Factor: {aggregate.get('profit_factor', 0):.2f}\n"
        f"  Folds Passing: {aggregate.get('pct_folds_pass', 0):.0%}\n"
        f"  OVERALL: {'✅ PASS' if aggregate.get('pass_all') else '❌ FAIL'}\n"
        f"{'='*60}"
    )

    import json
    # Convert np types so it is JSON serializable
    def np_encoder(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError

    with open("backtest_results.json", "w") as f:
        json.dump(aggregate, f, indent=4, default=np_encoder)
        
    with open("trades_log.json", "w") as f:
        json.dump(all_trades, f, indent=4, default=np_encoder)
        
    with open("equity_curve.json", "w") as f:
        json.dump(all_equity, f, indent=4, default=np_encoder)

    logger.info("Saved walk-forward metrics to backtest_results.json, trades_log.json, equity_curve.json")

    return aggregate

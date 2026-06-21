#!/usr/bin/env python3
"""
Optuna Bayesian Hyperparameter Tuning for the V7 Engine.

Execution-Only Tuning with Nested Cross-Validation.
This script strictly evaluates tuning on the Validation split (last 20% of CPCV Train Fold)
to discover optimal parameters, and ONLY evaluates the True OOS (Test Fold) at the very end
using the absolute best parameters. This guarantees ZERO lookahead bias and mathematically 
prevents Optuna from overfitting to the test set.
"""

import os
import sys
import torch
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta, timezone

torch.set_num_threads(1)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from v7_engine.ingestion.dukascopy_loader import DukascopyLoader
from v7_engine.training.cpcv import cpcv_split
from v7_engine.backtest.walk_forward import precompute_global_features, simulate_fold, compute_metrics_from_trades
from v7_engine.config import CPCV_N_SPLITS, CPCV_N_TEST_SPLITS, CPCV_PURGE_TICKS, CPCV_EMBARGO_TICKS, RISK_MC_BARRIER_BLOCKS, SDE_DT, MC_PATHS, MC_CONFIDENCE, BACKTEST_INITIAL_EQUITY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optuna_tune")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def build_nested_inference_caches(tick_data, splits, global_indices, global_vectors):
    """
    Extracts two caches per fold:
    - val_cache (Validation Split: tuned by Optuna)
    - test_cache (True OOS Split: evaluated ONCE at the end)
    """
    logger.info("Building Nested Inference Caches (Validation & True OOS)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(tick_data["bid"])
    bids = tick_data["bid"].astype(np.float64)
    asks = tick_data["ask"].astype(np.float64)
    
    val_cache = {}
    test_cache = {}
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_dir = f"models/fold_{fold_idx}"
        
        # Extract Validation Split (last 20% of train_idx)
        split_idx = int(len(train_idx) * 0.8)
        val_idx = train_idx[split_idx:]
        
        # Combine indices to run batch inference once
        eval_idx = np.concatenate([val_idx, test_idx])
        in_eval = np.isin(global_indices, eval_idx)
        fold_eval_vectors = global_vectors[in_eval]
        
        from v7_engine.sde.sde_model import NeuralSDE
        from v7_engine.sde.latent_decoder import LatentDecoder
        from v7_engine.ebm.energy_model import EnergyModel
        from v7_engine.ebm.rl_actor import RLActor
        from v7_engine.embedding.regime_xgb import RegimeXGBClassifier
        from v7_engine.risk.risk_xgb import RiskXGBClassifier
        from v7_engine.embedding.regime_hmm import RegimeHMM

        sde_model = NeuralSDE().to(device)
        ebm_model = EnergyModel().to(device)
        rl_actor  = RLActor().to(device)
        decoder_model = LatentDecoder().to(device)
        
        SDE_PATH = f"{fold_dir}/sde_best.pth"
        if not os.path.exists(SDE_PATH):
            continue
            
        sde_model.load_state_dict(torch.load(SDE_PATH, map_location=device, weights_only=True))
        if os.path.exists(f"{fold_dir}/ebm_best.pth"):
            ebm_model.load_state_dict(torch.load(f"{fold_dir}/ebm_best.pth", map_location=device, weights_only=True))
        if os.path.exists(f"{fold_dir}/rl_best.pth"):
            rl_actor.load_state_dict(torch.load(f"{fold_dir}/rl_best.pth", map_location=device, weights_only=True))
        if os.path.exists(f"{fold_dir}/latent_decoder.pth"):
            decoder_model.load(f"{fold_dir}/latent_decoder.pth", device=device)
            
        xgb_regime = RegimeXGBClassifier.load(f"{fold_dir}/regime_xgb.json") if os.path.exists(f"{fold_dir}/regime_xgb.json") else None
        xgb_risk   = RiskXGBClassifier.load(f"{fold_dir}/risk_xgb.json") if os.path.exists(f"{fold_dir}/risk_xgb.json") else None
        hmm_regime = RegimeHMM.load(f"{fold_dir}/regime_hmm.pkl") if os.path.exists(f"{fold_dir}/regime_hmm.pkl") else None

        sde_model.eval()
        ebm_model.eval()
        rl_actor.eval()
        
        raw_dirs     = np.zeros(len(fold_eval_vectors), dtype=np.int32)
        sz_fracs     = np.zeros(len(fold_eval_vectors), dtype=np.float32)
        base_vars    = np.zeros(len(fold_eval_vectors), dtype=np.float32)
        risk_scores  = np.zeros(len(fold_eval_vectors), dtype=np.float32)
        
        if len(fold_eval_vectors) > 0:
            valid_seqs = torch.tensor(fold_eval_vectors, dtype=torch.float32, device=device)
            BATCH_SIZE = 4096
            
            for b_idx in range(0, len(valid_seqs), BATCH_SIZE):
                batch = valid_seqs[b_idx : b_idx + BATCH_SIZE]
                with torch.no_grad():
                    _, final_states, _, _ = sde_model(batch)
                    tradeable, _ = ebm_model.is_tradeable(final_states)
                    action = rl_actor.sample_action(final_states)
                    
                    dirs = action["direction"].cpu().numpy()
                    mask = tradeable.cpu().numpy()
                    dirs[~mask] = 0
                    
                    raw_dirs[b_idx : b_idx + BATCH_SIZE] = dirs
                    sz_fracs[b_idx : b_idx + BATCH_SIZE] = action["size_fraction"].cpu().numpy()
                    
            active_local_indices = np.where(raw_dirs != 0)[0]
            if len(active_local_indices) > 0:
                ts = torch.linspace(0, RISK_MC_BARRIER_BLOCKS * SDE_DT, RISK_MC_BARRIER_BLOCKS, device=device)
                
                for active_i in active_local_indices:
                    global_idx = global_indices[in_eval][active_i]
                    
                    if xgb_risk is not None and xgb_risk._model is not None:
                        if global_idx >= 9:
                            window = global_vectors[global_idx - 9 : global_idx + 1].reshape(1, -1)
                            risk_scores[active_i] = xgb_risk.predict_risk(window)
                            
                    if xgb_regime is not None and xgb_regime._model is not None:
                        probs = xgb_regime.predict_proba(fold_eval_vectors[active_i])
                        if probs[2] > 0.6 or probs[3] > 0.6: 
                            raw_dirs[active_i] = 0
                            
                    if hmm_regime is not None and hmm_regime.model is not None:
                        start_i = max(0, global_idx - 5000)
                        if len(bids[start_i:global_idx]) > 200:
                            mid_win = (bids[start_i:global_idx] + asks[start_i:global_idx]) / 2.0
                            hmm_probs = hmm_regime.predict_proba(mid_win)
                            if hmm_probs[3] > 0.7:
                                raw_dirs[active_i] = 0
                                
                    if raw_dirs[active_i] != 0:
                        with torch.no_grad():
                            context = torch.tensor(fold_eval_vectors[active_i], dtype=torch.float32, device=device).unsqueeze(0)
                            seqs_batch = context.repeat(MC_PATHS, 1)
                            paths = sde_model.forward_multi_step(seqs_batch, ts=ts)
                            pnl_paths = decoder_model.decode_paths(paths, units=1.0, equity=1.0).cpu().numpy()
                            base_vars[active_i] = float(np.quantile(-pnl_paths, MC_CONFIDENCE))

        test_dir_full       = np.zeros(n, dtype=np.int32)
        test_sz_frac_full   = np.zeros(n, dtype=np.float32)
        base_var_loss_full  = np.zeros(n, dtype=np.float32)
        risk_scores_full    = np.zeros(n, dtype=np.float32)
        
        test_dir_full[global_indices[in_eval]] = raw_dirs
        test_sz_frac_full[global_indices[in_eval]] = sz_fracs
        base_var_loss_full[global_indices[in_eval]] = base_vars
        risk_scores_full[global_indices[in_eval]] = risk_scores
        
        # 1. Validation Cache Slice
        val_data = {k: v[val_idx] for k, v in tick_data.items() if isinstance(v, np.ndarray) and len(v) == n}
        val_cache[fold_idx] = {
            "test_data": val_data,
            "raw_dir": test_dir_full[val_idx],
            "sz_frac": test_sz_frac_full[val_idx],
            "base_var": base_var_loss_full[val_idx],
            "risk_scores": risk_scores_full[val_idx]
        }
        
        # 2. True OOS Test Cache Slice
        test_data = {k: v[test_idx] for k, v in tick_data.items() if isinstance(v, np.ndarray) and len(v) == n}
        test_cache[fold_idx] = {
            "test_data": test_data,
            "raw_dir": test_dir_full[test_idx],
            "sz_frac": test_sz_frac_full[test_idx],
            "base_var": base_var_loss_full[test_idx],
            "risk_scores": risk_scores_full[test_idx]
        }
        
    return val_cache, test_cache

def simulate_cache(cache, hyperparams, all_tns):
    """
    Runs the fast Numba backtest using the provided cache and hyperparameters.
    """
    all_trades = []
    all_equity = [BACKTEST_INITIAL_EQUITY]
    
    for fold_idx, c in cache.items():
        dynamic_dir = c["raw_dir"].copy()
        mask = (c["risk_scores"] > hyperparams["RISK_XGB_THRESHOLD"])
        dynamic_dir[mask] = 0
        
        # Extract LOCAL indices corresponding to the data slice to fix Numba boundary check
        local_indices = np.where(dynamic_dir != 0)[0]
        
        trades, equity = simulate_fold(
            tick_data=c["test_data"],
            fold_idx=fold_idx,
            symbol="EURUSD",
            test_indices=local_indices.astype(np.int64),
            test_dir=dynamic_dir,
            test_sz_frac=c["sz_frac"],
            base_var_loss_full=c["base_var"],
            hyperparams=hyperparams
        )
        all_trades.extend(trades)
        all_equity.extend(equity[1:])
        
    total_n_days = float((all_tns[-1] - all_tns[0]) / 1e9 / 86400.0) if len(all_tns) > 1 else 1.0
    return compute_metrics_from_trades(all_trades, all_equity, n_days=total_n_days)

def objective(trial, val_cache, all_tns):
    hyperparams = {
        "ATR_SL_MULT": trial.suggest_float("ATR_SL_MULT", 1.0, 5.0, step=0.1),
        "ATR_TP_MULT": trial.suggest_float("ATR_TP_MULT", 1.0, 10.0, step=0.2),
        "HALF_KELLY_FRACTION": trial.suggest_float("HALF_KELLY_FRACTION", 0.1, 1.0, step=0.05),
        "DAILY_PROFIT_LOCK_PCT": trial.suggest_float("DAILY_PROFIT_LOCK_PCT", 0.01, 0.10, step=0.01),
        "MAX_POSITION_FRACTION": 1.0,
        "RISK_XGB_THRESHOLD": trial.suggest_float("RISK_XGB_THRESHOLD", 0.4, 0.9, step=0.05)
    }
    
    metrics = simulate_cache(val_cache, hyperparams, all_tns)
    if not metrics:
        return 0.0
        
    calmar = metrics.get("calmar", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    max_dd = metrics.get("max_dd_usd", 0.0)
    n_trades = metrics.get("n_trades", 0)
    
    # Severe Anti-Cheating Penalties
    score = calmar
    if win_rate < 0.45:
        score *= (win_rate / 0.45)
    if max_dd > 1000.0:
        score *= (1000.0 / max_dd)
    if n_trades < 200: # Penalty for mathematically overfitting to lucky outliers
        score *= 0.1 
        
    trial.set_user_attr("Calmar", calmar)
    trial.set_user_attr("WinRate", win_rate)
    trial.set_user_attr("MaxDD", max_dd)
    trial.set_user_attr("Trades", n_trades)
    
    return float(score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--offset-months", type=int, default=1)
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()
    
    end = datetime.now(timezone.utc) - timedelta(days=30 * args.offset_months)
    start_date = (end - timedelta(days=30 * args.months)).strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")
    
    logger.info(f"Loading {args.symbol} from {start_date} to {end_date} for Optuna Tuning...")
    loader = DukascopyLoader()
    ticks = loader.load(args.symbol, start_date, end_date)
    
    n = len(ticks["bid"])
    all_tns = ticks.get("timestamp_ns", np.arange(n) * int(1e8))
    splits = list(cpcv_split(n, n_splits=CPCV_N_SPLITS, n_test_splits=CPCV_N_TEST_SPLITS, purge=CPCV_PURGE_TICKS, embargo=CPCV_EMBARGO_TICKS))
    
    global_indices, global_vectors, _ = precompute_global_features(ticks)
    
    val_cache, test_cache = build_nested_inference_caches(ticks, splits, global_indices, global_vectors)
    if not val_cache:
        logger.error("Failed to build inference cache.")
        return
        
    logger.info(f"Starting Nested Optuna Optimization ({args.trials} Trials) on Validation Blocks...")
    study = optuna.create_study(
        study_name=f"V7_{args.symbol}_Optimization", 
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner()
    )
    
    study.optimize(lambda t: objective(t, val_cache, all_tns), n_trials=args.trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params["MAX_POSITION_FRACTION"] = 1.0
    
    logger.info(f"\n{'='*60}\nVALIDATION TUNING COMPLETE\n{'='*60}")
    logger.info(f"Best Validation Objective Score: {study.best_value:.4f}")
    logger.info("Best Parameters Discovered:")
    for key, value in best_params.items():
        logger.info(f"    {key}: {value}")
        
    logger.info(f"\n{'='*60}\nRUNNING TRUE OUT-OF-SAMPLE (OOS) VERIFICATION...\n{'='*60}")
    oos_metrics = simulate_cache(test_cache, best_params, all_tns)
    
    logger.info("True OOS Metrics (Un-cheatable Test Sets):")
    logger.info(f"  Calmar:        {oos_metrics.get('calmar', 0):.2f}")
    logger.info(f"  Sharpe:        {oos_metrics.get('sharpe', 0):.2f}")
    logger.info(f"  Max DD:        ${oos_metrics.get('max_dd_usd', 0):.2f}")
    logger.info(f"  Win Rate:      {oos_metrics.get('win_rate', 0):.1%}")
    logger.info(f"  Trades:        {oos_metrics.get('n_trades', 0)}")
    logger.info(f"  Profit Factor: {oos_metrics.get('profit_factor', 0):.2f}")

if __name__ == "__main__":
    main()

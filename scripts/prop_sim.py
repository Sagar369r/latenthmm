#!/usr/bin/env python3
"""
Prop Firm Challenge Simulator
Evaluates a trained V7 model against strict Prop Firm rules (e.g., The 5%ers).
"""
import os
import sys
import torch
import argparse
import logging
import numpy as np
from datetime import datetime, timedelta, timezone

torch.set_num_threads(1)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v7_engine.ingestion.dukascopy_loader import DukascopyLoader
from v7_engine.sde.sde_model import NeuralSDE
from v7_engine.ebm.energy_model import EnergyModel
from v7_engine.ebm.rl_actor import RLActor
from v7_engine.sde.latent_decoder import LatentDecoder
from v7_engine.embedding.regime_xgb import RegimeXGBClassifier
from v7_engine.risk.risk_xgb import RiskXGBClassifier
from v7_engine.embedding.regime_hmm import RegimeHMM
from v7_engine.backtest.walk_forward import precompute_global_features, _numba_simulate_fold
from v7_engine.config import (
    PROP_FIRM_PROFILES, PIP_SIZE, PIP_VALUE_PER_LOT, SLIPPAGE_PIPS, 
    COMMISSION_PER_LOT, EXECUTION_LATENCY_MS, MAX_POSITION_FRACTION, 
    HALF_KELLY_FRACTION, ATR_SL_MULT, ATR_TP_MULT, RISK_XGB_THRESHOLD,
    MC_PATHS, MC_CONFIDENCE, SDE_DT, RISK_MC_BARRIER_BLOCKS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prop_sim")

def run_challenge(tick_data, global_indices, global_vectors, test_dir_full, test_sz_frac_full, base_var_loss_full, profile, equity_size, symbol):
    # This runs the actual simulation using the exact parameters of the prop firm
    bids = tick_data["bid"].astype(np.float64)
    asks = tick_data["ask"].astype(np.float64)
    timestamps = tick_data.get("timestamp_ns", np.arange(len(bids)) * int(1e8)).astype(np.int64)
    n = len(bids)

    is_new_day = np.zeros(n, dtype=bool)
    current_day = ""
    for i in range(n):
        day_str = datetime.fromtimestamp(timestamps[i] / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
        if day_str != current_day:
            if current_day != "":
                is_new_day[i] = True
            current_day = day_str

    pip_sz = PIP_SIZE.get(symbol.upper(), 0.0001)
    pip_val = PIP_VALUE_PER_LOT.get(symbol.upper(), 10.0)

    daily_dd_limit = equity_size * profile.get("max_daily_dd_pct", 0.05)
    max_total_dd_usd = equity_size * profile.get("max_total_dd_pct", 0.10)
    profit_target_usd = equity_size * profile.get("profit_target_pct", 0.10)

    dynamic_dir = test_dir_full
    local_indices = np.where(dynamic_dir != 0)[0]

    # Use the fast Numba simulator
    t_ent, t_ex, t_dir, t_entry, t_exit, t_pnl, eq_curve = _numba_simulate_fold(
        n, bids, asks, timestamps, is_new_day,
        local_indices.astype(np.int64), dynamic_dir, test_sz_frac_full, base_var_loss_full,
        equity_size, pip_sz, pip_val,
        daily_dd_limit, max_total_dd_usd, ATR_SL_MULT, ATR_TP_MULT,
        HALF_KELLY_FRACTION, MAX_POSITION_FRACTION, SLIPPAGE_PIPS, COMMISSION_PER_LOT,
        0.02, profile.get("max_total_dd_pct", 0.10), EXECUTION_LATENCY_MS * 1_000_000
    )

    passed = False
    failed = False
    fail_reason = ""
    final_eq = equity_size
    
    if len(eq_curve) > 0:
        max_eq = np.max(eq_curve)
        min_eq = np.min(eq_curve)
        
        if max_eq >= equity_size + profit_target_usd:
            passed = True
        elif min_eq <= equity_size - max_total_dd_usd:
            failed = True
            fail_reason = "Max DD Hit"
        elif len(t_ent) > 0:
            # Check if simulation halted due to daily DD (eq_curve stopped early but neither profit nor max DD hit)
            # This is an approximation since numba halts instantly
            last_tick_eq = eq_curve[-1]
            if last_tick_eq < equity_size and not passed:
                failed = True
                fail_reason = "Time/Daily Limit"
            
        final_eq = eq_curve[-1]

    return {
        "passed": passed,
        "failed": failed,
        "fail_reason": fail_reason,
        "final_equity": float(final_eq),
        "return_pct": float((final_eq - equity_size) / equity_size),
        "trades_count": len(t_ent),
        "max_dd_pct": float((equity_size - np.min(eq_curve)) / equity_size) if len(eq_curve) > 0 else 0.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--fold", type=int, default=0, help="Which trained fold to use")
    parser.add_argument("--profile", type=str, default="THE_5ERS_100K", help="Prop firm profile from config")
    parser.add_argument("--equity", type=float, default=100000.0, help="Starting equity")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--challenge-days", type=int, default=30, help="Days per challenge chunk")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.profile not in PROP_FIRM_PROFILES:
        logger.error(f"Profile {args.profile} not found in config/trading.yaml")
        return
        
    profile = PROP_FIRM_PROFILES[args.profile]
    logger.info(f"=== Prop Firm Simulator ===")
    logger.info(f"Profile: {args.profile}")
    logger.info(f"Equity: ${args.equity:,.2f}")
    logger.info(f"Target: {profile.get('profit_target_pct')*100}% | Max DD: {profile.get('max_total_dd_pct')*100}% | Daily DD: {profile.get('max_daily_dd_pct')*100}%")

    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30 * args.months)
        start_date, end_date = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        
    logger.info(f"Simulation Range: {start_date} to {end_date}")
    
    loader = DukascopyLoader()
    tick_data = loader.load(args.symbol, start_date, end_date)
    
    fold_dir = f"models/fold_{args.fold}"
    if not os.path.exists(fold_dir):
        logger.error(f"Model directory {fold_dir} not found. Please train first.")
        return
        
    sde_model = NeuralSDE().to(device)
    ebm_model = EnergyModel().to(device)
    rl_actor = RLActor().to(device)
    decoder_model = LatentDecoder().to(device)
    
    sde_model.load_state_dict(torch.load(f"{fold_dir}/sde_best.pth", map_location=device, weights_only=True))
    ebm_model.load_state_dict(torch.load(f"{fold_dir}/ebm_best.pth", map_location=device, weights_only=True))
    rl_actor.load_state_dict(torch.load(f"{fold_dir}/rl_best.pth", map_location=device, weights_only=True))
    decoder_model.load_state_dict(torch.load(f"{fold_dir}/latent_decoder.pth", map_location=device, weights_only=True))
    
    xgb_regime = RegimeXGBClassifier.load(f"{fold_dir}/regime_xgb.json") if os.path.exists(f"{fold_dir}/regime_xgb.json") else None
    xgb_risk = RiskXGBClassifier.load(f"{fold_dir}/risk_xgb.json") if os.path.exists(f"{fold_dir}/risk_xgb.json") else None
    hmm_regime = RegimeHMM.load(f"{fold_dir}/regime_hmm.pkl") if os.path.exists(f"{fold_dir}/regime_hmm.pkl") else None
    
    sde_model.eval()
    ebm_model.eval()
    rl_actor.eval()
    
    logger.info("Extracting global features for simulation period...")
    global_indices, global_vectors, _ = precompute_global_features(tick_data, None, xgb_risk, hmm_regime)
    
    n = len(tick_data["bid"])
    test_dir_full = np.zeros(n, dtype=np.int32)
    test_sz_frac_full = np.zeros(n, dtype=np.float32)
    base_var_loss_full = np.zeros(n, dtype=np.float32)
    
    if len(global_vectors) > 0:
        logger.info("Running RL Inference...")
        valid_seqs = torch.tensor(global_vectors, dtype=torch.float32, device=device)
        BATCH_SIZE = 4096
        
        chunk_dirs = np.zeros(len(valid_seqs), dtype=np.int32)
        chunk_sz_fracs = np.zeros(len(valid_seqs), dtype=np.float32)
        
        for b_idx in range(0, len(valid_seqs), BATCH_SIZE):
            batch = valid_seqs[b_idx : b_idx + BATCH_SIZE]
            with torch.no_grad():
                _, final_states, _, _ = sde_model(batch)
                action = rl_actor.sample_action(final_states)
                chunk_dirs[b_idx : b_idx + BATCH_SIZE] = action["direction"].cpu().numpy()
                chunk_sz_fracs[b_idx : b_idx + BATCH_SIZE] = action["size_fraction"].cpu().numpy()
                
        logger.info("Calculating VaR Risk Bounds...")
        active_local_indices = np.where(chunk_dirs != 0)[0]
        if len(active_local_indices) > 0:
            ts = torch.linspace(0, RISK_MC_BARRIER_BLOCKS * SDE_DT, RISK_MC_BARRIER_BLOCKS, device=device)
            from tqdm import tqdm
            BATCH_SIZE_MC = 512
            
            for i in tqdm(range(0, len(active_local_indices), BATCH_SIZE_MC), desc="MC VaR Risk Bounds"):
                batch_indices = active_local_indices[i : i + BATCH_SIZE_MC]
                global_idxs = global_indices[batch_indices]
                
                test_dir_full[global_idxs] = chunk_dirs[batch_indices]
                test_sz_frac_full[global_idxs] = chunk_sz_fracs[batch_indices]
                
                with torch.no_grad():
                    context = valid_seqs[batch_indices] # (B, 128)
                    B = len(context)
                    
                    # Interleave to create (B * MC_PATHS, 128)
                    context_repeated = context.repeat_interleave(MC_PATHS, dim=0)
                    
                    paths = sde_model.forward_multi_step(context_repeated, ts=ts)
                    pnl_paths = decoder_model.decode_paths(paths, units=1.0, equity=1.0).cpu().numpy()
                    
                    # Reshape to (B, MC_PATHS) and calculate quantile along MC_PATHS axis
                    pnl_paths = pnl_paths.reshape(B, MC_PATHS)
                    var_vals = np.quantile(-pnl_paths, MC_CONFIDENCE, axis=1)
                    
                    base_var_loss_full[global_idxs] = var_vals

    tns = tick_data.get("timestamp_ns", np.arange(n) * int(1e8))
    start_ns = tns[0]
    end_ns = tns[-1]
    challenge_ns = args.challenge_days * 24 * 60 * 60 * 1e9
    
    challenges = []
    current_start = start_ns
    
    logger.info("Running Challenge Simulations...")
    while current_start + challenge_ns <= end_ns:
        current_end = current_start + challenge_ns
        mask = (tns >= current_start) & (tns < current_end)
        
        chunk_data = {k: v[mask] for k, v in tick_data.items() if isinstance(v, np.ndarray) and len(v) == n}
        
        if len(chunk_data["bid"]) > 0:
            res = run_challenge(
                chunk_data, global_indices, global_vectors, 
                test_dir_full[mask], test_sz_frac_full[mask], base_var_loss_full[mask], 
                profile, args.equity, args.symbol
            )
            challenges.append(res)
            
        current_start += challenge_ns

    print("\n" + "="*50)
    print("🎯 PROP FIRM CHALLENGE RESULTS")
    print("="*50)
    
    passed = sum(1 for c in challenges if c["passed"])
    failed = sum(1 for c in challenges if not c["passed"])
    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"Total Challenges Simulated: {total}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    print("\nDetailed Breakdown:")
    for i, c in enumerate(challenges):
        status = "✅ PASS" if c["passed"] else "❌ FAIL"
        reason = f" ({c['fail_reason']})" if c['fail_reason'] else ""
        print(f"Challenge {i+1:02d}: {status} | Return: {c['return_pct']*100:6.2f}% | Max DD: {c['max_dd_pct']*100:5.2f}% | Trades: {c['trades_count']}{reason}")
        
    print("="*50)

if __name__ == "__main__":
    main()

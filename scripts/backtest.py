#!/usr/bin/env python3
"""
Backtest Execution Script

Runs the combinatorial purged cross-validation (CPCV) walk-forward 
backtest across your trained models to generate Sharpe, Calmar, and Max Drawdown metrics.
"""

import os
import sys
import torch

# Prevent PyTorch from spawning massive numbers of threads on the CPU
torch.set_num_threads(1)

# Automatically add the project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from datetime import datetime, timedelta

from v7_engine.ingestion.dukascopy_loader import DukascopyLoader
from v7_engine.backtest.walk_forward import run_walk_forward
from v7_engine.sde.sde_model import NeuralSDE
from v7_engine.sde.latent_decoder import LatentDecoder
from v7_engine.embedding.regime_xgb import RegimeXGBClassifier
from v7_engine.risk.risk_xgb import RiskXGBClassifier
from v7_engine.embedding.regime_hmm import RegimeHMM
from v7_engine.config import (
    CHECKPOINT_SDE, CHECKPOINT_EBM, CHECKPOINT_RL, 
    CHECKPOINT_REGIME, CHECKPOINT_REGIME_HMM, CHECKPOINT_RISK, CHECKPOINT_DECODER,
    SDE_LATENT_DIM, SDE_DRIFT_HEADS, SDE_DRIFT_LAYERS, SDE_DRIFT_DIM_FF,
    SDE_DIFFUSION_HIDDEN, EMBEDDING_DIM
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")


def get_date_range(months: int):
    end = datetime.utcnow()
    start = end - timedelta(days=30 * months)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Run a short slice")
    args = parser.parse_args()

    start_date, end_date = get_date_range(1 if args.dry_run else args.months)
    
    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date

    logger.info(f"=== Starting Walk-Forward Backtest ===")
    
    # 1. Load Data
    logger.info("Loading Dukascopy history...")
    loader = DukascopyLoader()
    try:
        tick_data = loader.load(args.symbol, start_date, end_date)
    except Exception as e:
        logger.error("No tick data loaded!")
        return

    # 2. Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading NeuralSDE...")
    sde_model = NeuralSDE().to(device)
    
    from v7_engine.ebm.energy_model import EnergyModel
    from v7_engine.ebm.rl_actor import RLActor
    from v7_engine.config import CHECKPOINT_EBM, CHECKPOINT_RL
    
    if not all(os.path.exists(p) for p in [CHECKPOINT_SDE, CHECKPOINT_EBM, CHECKPOINT_RL]):
        raise FileNotFoundError("Real models not found. Train first.")
    
    sde_model.load_state_dict(torch.load(CHECKPOINT_SDE, map_location=device, weights_only=True))
    
    ebm_model = EnergyModel().to(device)
    ebm_model.load_state_dict(torch.load(CHECKPOINT_EBM, map_location=device, weights_only=True))
    
    rl_actor = RLActor().to(device)
    rl_actor.load_state_dict(torch.load(CHECKPOINT_RL, map_location=device, weights_only=True))    
    xgb_regime = None
    if os.path.exists(CHECKPOINT_REGIME):
        xgb_regime = RegimeXGBClassifier.load(CHECKPOINT_REGIME)
        
    xgb_risk = None
    if os.path.exists(CHECKPOINT_RISK):
        xgb_risk = RiskXGBClassifier.load(CHECKPOINT_RISK)
        
    hmm_regime = None
    if os.path.exists(CHECKPOINT_REGIME_HMM):
        hmm_regime = RegimeHMM.load(CHECKPOINT_REGIME_HMM)
        
    decoder_model = None
    if os.path.exists(CHECKPOINT_DECODER):
        decoder_model = LatentDecoder().to(device)
        decoder_model.load_state_dict(torch.load(CHECKPOINT_DECODER, map_location=device, weights_only=True))

    # 3. Execute
    logger.info("Running CPCV Walk-Forward Validation...")
    results = run_walk_forward(
        tick_data=tick_data,
        sde_model=sde_model,
        ebm_model=ebm_model,
        rl_actor=rl_actor,
        symbol=args.symbol,
        xgb_regime=xgb_regime,
        xgb_risk=xgb_risk,
        hmm_regime=hmm_regime,
        decoder_model=decoder_model
    )
    
if __name__ == "__main__":
    main()

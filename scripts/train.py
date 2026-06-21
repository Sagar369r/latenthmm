#!/usr/bin/env python3
"""
Master Training Script for V7 Engine.

Runs the full end-to-end training pipeline on real Dukascopy CFD tick data:
1. Download historical ticks
2. Train SDE
3. Train Latent Decoder
4. Build EBM dataset and train EBM
5. Train RL Actor

Usage:
    python scripts/train.py --symbol EURUSD --months 12
"""

import os
import sys
import torch
import numpy as np

# Prevent PyTorch from spawning massive numbers of threads on the CPU
torch.set_num_threads(1)

# Automatically add the project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hashlib
import json
from v7_engine.config import (
    CHECKPOINT_SDE, CHECKPOINT_DECODER, CHECKPOINT_EBM, 
    CHECKPOINT_RL, CHECKPOINT_WELFORD, CHECKPOINT_REGIME, CHECKPOINT_RISK
)

import argparse
import logging
from datetime import datetime, timedelta

from v7_engine.ingestion.dukascopy_loader import DukascopyLoader
from v7_engine.training.train_sde import train_sde, prepare_sequences_from_ticks
from v7_engine.embedding.feature_vector import TickFeatureVector
from v7_engine.sde.latent_decoder import LatentDecoder
from v7_engine.training.build_ebm_dataset import build_dataset
from v7_engine.sde.sde_model import NeuralSDE

from v7_engine.embedding.regime_xgb import RegimeXGBClassifier, label_regimes_from_features
from v7_engine.risk.risk_xgb import RiskXGBClassifier, label_risk_windows

from v7_engine.config import (
    TRAIN_DEFAULT_MONTHS, TRAIN_VAL_SPLIT, TRAIN_STRIDE, TRAIN_STRIDE_DRY,
    TRAIN_SEQUENCE_LENGTH, TRAIN_SDE_EPOCHS_DRY, SDE_EPOCHS,
    TRAIN_DECODER_BATCH_SIZE, TRAIN_DECODER_EPOCHS_DRY, TRAIN_DECODER_EPOCHS,
    CHECKPOINT_DECODER, CHECKPOINT_SDE, TRAIN_RL_EPISODES, TRAIN_RL_EPISODES_DRY,
    CHECKPOINT_REGIME, CHECKPOINT_RISK, MODELS_DIR
)

# EBM training module might need to be imported or executed.
# Since train_ebm.py is in training/, we will import it if available, or just call it as a script.
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_master")

def get_date_range(months: int):
    end = datetime.utcnow()
    start = end - timedelta(days=30 * months)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--months", type=int, default=TRAIN_DEFAULT_MONTHS)
    parser.add_argument("--offset-months", type=int, default=1, help="Months to shift back (leaves recent data for backtest)")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick syntax check pipeline")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    
    end = datetime.utcnow() - timedelta(days=30 * args.offset_months)
    start_date = (end - timedelta(days=30 * args.months)).strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")
    
    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
        
    if args.dry_run:
        # Just 1 day for dry run
        end = datetime.utcnow() - timedelta(days=30 * args.offset_months)
        start_date = (end - timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")

    logger.info(f"=== Starting V7 Training Pipeline ===")
    logger.info(f"Symbol: {args.symbol} | Range: {start_date} to {end_date}")

    # 1. Download & Prepare Data
    logger.info("=== Phase 1: Data Ingestion & SDE Training ===")
    loader = DukascopyLoader()
    try:
        ticks = loader.load(args.symbol, start_date, end_date)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    n_ticks = len(ticks["bid"])
    split_idx = int(n_ticks * TRAIN_VAL_SPLIT)
    
    train_data = {k: v[:split_idx] for k, v in ticks.items()}
    val_data = {k: v[split_idx:] for k, v in ticks.items()}

    logger.info("Extracting feature sequences for SDE...")
    stride = TRAIN_STRIDE_DRY if args.dry_run else TRAIN_STRIDE
    
    logger.info(f"Extracting sliding windows (seq_len={TRAIN_SEQUENCE_LENGTH}, stride={stride})...")
    from v7_engine.ingestion.welford import WelfordNormaliser
    from v7_engine.config import WELFORD_CLIP_SIGMA, WELFORD_MIN_STD, WELFORD_WARMUP_TICKS, EMBEDDING_DIM
    train_normaliser = WelfordNormaliser(EMBEDDING_DIM, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD, warmup=WELFORD_WARMUP_TICKS)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    FEATURES_DIR = "data/features"
    os.makedirs(FEATURES_DIR, exist_ok=True)
    features_cache_path = os.path.join(FEATURES_DIR, f"extracted_features_{args.symbol}_{start_date}_{end_date}.npz")
    
    if os.path.exists(features_cache_path):
        logger.info(f"Loading extracted features from {features_cache_path}...")
        npz = np.load(features_cache_path)
        train_seqs = npz['train_seqs']
        train_vars = npz['train_vars']
        train_rets = npz['train_rets']
        train_barrier_labels = npz['train_barrier_labels']
        train_tns  = npz['train_tns']
        val_seqs   = npz['val_seqs']
        val_vars   = npz['val_vars']
        val_rets   = npz['val_rets']
        val_barrier_labels = npz['val_barrier_labels']
        val_tns    = npz['val_tns']
        
        # Restore welford normaliser state
        train_normaliser._mean = npz['welford_mean']
        train_normaliser._M2   = npz['welford_m2']
        train_normaliser._n    = int(npz['welford_count'])
        train_normaliser.is_warm = bool(npz['welford_warm'])
    else:
        logger.info("Running deep feature extraction (this may take hours on large datasets)...")
        train_seqs, train_vars, train_rets, train_barrier_labels, train_normaliser, train_tns = prepare_sequences_from_ticks(
            train_data, TickFeatureVector, seq_len=TRAIN_SEQUENCE_LENGTH, stride=stride, normaliser=train_normaliser, update_welford=True
        )
        val_seqs, val_vars, val_rets, val_barrier_labels, _, val_tns = prepare_sequences_from_ticks(
            val_data, TickFeatureVector, seq_len=TRAIN_SEQUENCE_LENGTH, stride=stride, normaliser=train_normaliser, update_welford=False
        )
        logger.info(f"Saving extracted features to {features_cache_path}...")
        np.savez_compressed(
            features_cache_path,
            train_seqs=train_seqs, train_vars=train_vars, train_rets=train_rets, 
            train_barrier_labels=train_barrier_labels, train_tns=train_tns,
            val_seqs=val_seqs, val_vars=val_vars, val_rets=val_rets, 
            val_barrier_labels=val_barrier_labels, val_tns=val_tns,
            welford_mean=train_normaliser._mean, welford_m2=train_normaliser._M2,
            welford_count=train_normaliser._n, welford_warm=train_normaliser.is_warm
        )

    if len(train_seqs) == 0:
        logger.error("Not enough data to train SDE.")
        return

    # Train SDE
    sde_epochs = TRAIN_SDE_EPOCHS_DRY if args.dry_run else SDE_EPOCHS
    
    cache_train = os.path.join(FEATURES_DIR, f"train_features_{args.symbol}_{start_date}_{end_date}.parquet")
    cache_val = os.path.join(FEATURES_DIR, f"val_features_{args.symbol}_{start_date}_{end_date}.parquet")
    
    import polars as pl
    def to_polars(seqs, vars_, labels, ts_array):
        data = {f"f_{i}": seqs[:, i] for i in range(seqs.shape[1])}
        data["var"] = vars_
        data["barrier_label"] = labels
        data["ts"] = ts_array
        return pl.DataFrame(data)
        
    df_train = to_polars(train_seqs, train_vars, train_barrier_labels, train_tns)
    df_train.write_parquet(cache_train, compression="zstd")
    
    df_val = to_polars(val_seqs, val_vars, val_barrier_labels, val_tns)
    df_val.write_parquet(cache_val, compression="zstd")
    
    import v7_engine.training.train_sde
    from v7_engine.sde.sde_model import NeuralSDE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(CHECKPOINT_SDE):
        logger.info(f"SDE checkpoint found at {CHECKPOINT_SDE}. Skipping training.")
        sde_model = NeuralSDE().to(device)
        sde_model.load_state_dict(torch.load(CHECKPOINT_SDE, map_location=device))
        sde_model.eval()
    else:
        logger.info("Training SDE...")
        sde_model = v7_engine.training.train_sde.train_sde(
            cache_train, cache_val, epochs=sde_epochs
        )

    # 2. Train Latent Decoder
    logger.info("=== Phase 2: Latent Decoder Training ===")
    decoder = LatentDecoder()
    
    sde_model.eval()
    
    latent_list = []
    with torch.no_grad():
        for i in range(0, len(train_seqs), TRAIN_DECODER_BATCH_SIZE):
            batch = train_seqs[i:i+TRAIN_DECODER_BATCH_SIZE]
            ctx = torch.tensor(batch, dtype=torch.float32, device=device)
            _, final_state, _, _ = sde_model(ctx)
            latent_list.append(final_state.cpu().numpy())
            
    latent_states = np.concatenate(latent_list, axis=0)
    
    if os.path.exists(CHECKPOINT_DECODER):
        logger.info(f"Latent Decoder checkpoint found at {CHECKPOINT_DECODER}. Skipping training.")
        decoder.load(CHECKPOINT_DECODER)
    else:
        decoder_epochs = TRAIN_DECODER_EPOCHS_DRY if args.dry_run else TRAIN_DECODER_EPOCHS
        decoder.train_on_pairs(latent_states, train_rets, epochs=decoder_epochs)
        decoder.save(CHECKPOINT_DECODER)

    # Save Welford normalizer state so backtest can load it
    welford_state = train_normaliser.save_state()
    np.savez(CHECKPOINT_WELFORD, **welford_state)
    logger.info(f"Welford normalizer saved → {CHECKPOINT_WELFORD}")

    # 3. Build EBM Dataset and Train EBM
    logger.info("=== Phase 3: EBM Dataset & Training ===")
    from v7_engine.training.build_ebm_dataset import build_dataset_from_memory
    build_dataset_from_memory(
        symbol=args.symbol,
        train_data=train_data,
        train_seqs=train_seqs,
        train_tns=train_tns,
        latent_states=latent_states,
        sde_checkpoint=CHECKPOINT_SDE,
    )
    
    logger.info("=== Phase 3: EBM Training ===")
    try:
        from v7_engine.training.train_ebm import main as train_ebm_main
        # We temporarily manipulate sys.argv to emulate the CLI for train_ebm if needed
        # Or better yet, train_ebm_main() doesn't need args if it uses config
        train_ebm_main()
    except Exception as e:
        logger.error(f"EBM training failed: {e}")
        return

    # 4. Train RL
    logger.info("=== Phase 4: RL Actor Training ===")
    from v7_engine.training.train_rl import train_rl
    rl_episodes = TRAIN_RL_EPISODES_DRY if args.dry_run else TRAIN_RL_EPISODES
    train_rl(
        contexts=train_seqs, 
        actual_returns=train_rets, 
        valid_tns=train_tns,
        sde=sde_model, 
        episodes=rl_episodes, 
        device_str=device.type
    )

    # 5. Train XGBoost Models
    logger.info("=== Phase 5: XGBoost Model Training ===")
    
    # train_seqs shape is (N, EMBEDDING_DIM) — already the last feature vector per window
    feat_matrix = train_seqs
    
    logger.info("Training RegimeXGBClassifier...")
    try:
        feat_matrix_np = feat_matrix.cpu().numpy() if hasattr(feat_matrix, "cpu") else feat_matrix
        regime_labels = label_regimes_from_features(
            feature_matrix=feat_matrix_np
        )
        regime_model = RegimeXGBClassifier()
        regime_model.fit(feat_matrix_np, regime_labels)
        regime_model.save(CHECKPOINT_REGIME)
    except Exception as e:
        logger.error(f"Failed to train RegimeXGBClassifier: {e}")

    logger.info("Training RiskXGBClassifier...")
    try:
        # Pass raw log returns to label_risk_windows
        train_rets_np = train_rets.cpu().numpy() if hasattr(train_rets, "cpu") else train_rets
        X_risk, y_risk = label_risk_windows(feat_matrix_np, train_rets_np)
        if len(X_risk) > 0:
            risk_model = RiskXGBClassifier()
            risk_model.fit(X_risk, y_risk)
            risk_model.save(CHECKPOINT_RISK)
        else:
            logger.warning("Not enough data to train RiskXGBClassifier.")
    except Exception as e:
        logger.error(f"Failed to train RiskXGBClassifier: {e}")

    logger.info("=== All Training Phases Complete ===")
    
    manifest = {}
    for path in [CHECKPOINT_SDE, CHECKPOINT_DECODER, CHECKPOINT_EBM, 
                 CHECKPOINT_RL, CHECKPOINT_WELFORD, CHECKPOINT_REGIME, CHECKPOINT_RISK]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                manifest[os.path.basename(path)] = hashlib.sha256(f.read()).hexdigest()
                
    manifest_path = os.path.join(os.path.dirname(CHECKPOINT_SDE), "model_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    logger.info(f"Model manifest generated: {manifest_path}")
    logger.info("Models saved in models/ directory.")

if __name__ == "__main__":
    main()

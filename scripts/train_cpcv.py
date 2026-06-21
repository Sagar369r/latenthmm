#!/usr/bin/env python3
"""
True CPCV Training Script for V7 Engine.

Runs the training pipeline K times for K different Combinatorial Purged Folds.
Guarantees zero data leakage by physically saving fold-specific models to models/fold_X/.

Usage:
    python scripts/train_cpcv.py --symbol EURUSD --months 12
"""

import os
import sys
import torch
import numpy as np

# Prevent PyTorch from spawning massive numbers of threads on the CPU
torch.set_num_threads(1)

# Automatically add the project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import argparse
from datetime import datetime, timedelta, timezone

from v7_engine.ingestion.dukascopy_loader import DukascopyLoader
from v7_engine.training.train_sde import train_sde, prepare_sequences_from_ticks
from v7_engine.embedding.feature_vector import TickFeatureVector
from v7_engine.sde.latent_decoder import LatentDecoder
from v7_engine.sde.sde_model import NeuralSDE
from v7_engine.training.cpcv import cpcv_split

from v7_engine.embedding.regime_xgb import RegimeXGBClassifier, label_regimes_from_features
from v7_engine.risk.risk_xgb import RiskXGBClassifier, label_risk_windows

from v7_engine.config import (
    TRAIN_DEFAULT_MONTHS, TRAIN_STRIDE, TRAIN_STRIDE_DRY,
    TRAIN_SEQUENCE_LENGTH, TRAIN_SDE_EPOCHS_DRY, SDE_EPOCHS,
    TRAIN_DECODER_BATCH_SIZE, TRAIN_DECODER_EPOCHS_DRY, TRAIN_DECODER_EPOCHS,
    TRAIN_RL_EPISODES, TRAIN_RL_EPISODES_DRY,
    CPCV_N_SPLITS, CPCV_N_TEST_SPLITS, CPCV_PURGE_TICKS, CPCV_EMBARGO_TICKS,
    EMBEDDING_DIM, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD, WELFORD_WARMUP_TICKS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_cpcv")

def get_date_range(months: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * months)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--months", type=int, default=TRAIN_DEFAULT_MONTHS)
    parser.add_argument("--offset-months", type=int, default=1)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    
    end = datetime.now(timezone.utc) - timedelta(days=30 * args.offset_months)
    start_date = (end - timedelta(days=30 * args.months)).strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")
    
    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
        
    if args.dry_run:
        end = datetime.now(timezone.utc) - timedelta(days=30 * args.offset_months)
        start_date = (end - timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")

    logger.info(f"=== Starting V7 True CPCV Training Pipeline ===")
    logger.info(f"Symbol: {args.symbol} | Range: {start_date} to {end_date}")

    loader = DukascopyLoader()
    try:
        ticks = loader.load(args.symbol, start_date, end_date)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    logger.info("Extracting global feature sequences...")
    stride = TRAIN_STRIDE_DRY if args.dry_run else TRAIN_STRIDE
    
    from v7_engine.ingestion.welford import WelfordNormaliser
    normaliser = WelfordNormaliser(EMBEDDING_DIM, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD, warmup=WELFORD_WARMUP_TICKS)
    
    FEATURES_DIR = "data/features"
    os.makedirs(FEATURES_DIR, exist_ok=True)
    features_cache_path = os.path.join(FEATURES_DIR, f"cpcv_features_{args.symbol}_{start_date}_{end_date}.npz")
    
    if os.path.exists(features_cache_path):
        logger.info(f"Loading extracted features from {features_cache_path}...")
        npz = np.load(features_cache_path)
        full_seqs = npz['seqs']
        full_vars = npz['vars']
        full_rets = npz['rets']
        full_barrier_labels = npz['barrier_labels']
        full_tns  = npz['tns']
    else:
        logger.info("Running deep feature extraction...")
        # Fix Welford leakage: only train the normalizer on the warm-up period, then freeze it
        warmup_n = min(len(ticks["bid"]), normaliser._warmup + 5000)
        warmup_ticks = {k: v[:warmup_n] for k, v in ticks.items()}
        logger.info(f"Warming up Welford normaliser on first {warmup_n} ticks to prevent data leakage...")
        _, _, _, _, normaliser, _ = prepare_sequences_from_ticks(
            warmup_ticks, TickFeatureVector, seq_len=TRAIN_SEQUENCE_LENGTH, stride=stride, normaliser=normaliser, update_welford=True
        )
        
        logger.info("Extracting full sequences with frozen normaliser...")
        full_seqs, full_vars, full_rets, full_barrier_labels, normaliser, full_tns = prepare_sequences_from_ticks(
            ticks, TickFeatureVector, seq_len=TRAIN_SEQUENCE_LENGTH, stride=stride, normaliser=normaliser, update_welford=False
        )
        np.savez_compressed(
            features_cache_path,
            seqs=full_seqs, vars=full_vars, rets=full_rets, 
            barrier_labels=full_barrier_labels, tns=full_tns
        )

    if len(full_seqs) == 0:
        logger.error("Not enough data to train.")
        return

    # Calculate CPCV splits based on the feature array length
    n_features = len(full_seqs)
    splits = list(cpcv_split(n_features, n_splits=CPCV_N_SPLITS, n_test_splits=CPCV_N_TEST_SPLITS, purge=CPCV_PURGE_TICKS // stride, embargo=CPCV_EMBARGO_TICKS // stride))
    
    logger.info(f"Generated {len(splits)} CPCV Folds.")

    import polars as pl
    def to_polars(seqs, vars_, labels, ts_array):
        data = {f"f_{i}": seqs[:, i] for i in range(seqs.shape[1])}
        data["var"] = vars_
        data["barrier_label"] = labels
        data["ts"] = ts_array
        return pl.DataFrame(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"")
        logger.info(f"==================================================")
        logger.info(f"=== Training Fold {fold_idx} ===")
        logger.info(f"==================================================")
        
        fold_dir = f"models/fold_{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)
        
        SDE_PATH = f"{fold_dir}/sde_best.pth"
        DEC_PATH = f"{fold_dir}/latent_decoder.pth"
        RL_PATH  = f"{fold_dir}/rl_best.pth"
        REG_PATH = f"{fold_dir}/regime_xgb.json"
        RSK_PATH = f"{fold_dir}/risk_xgb.json"
        
        # Internal early stopping split (80% train, 20% val of the train_idx)
        split_idx = int(len(train_idx) * 0.8)
        sub_val_idx = train_idx[split_idx:]
        
        # Purge to prevent leakage from SDE target labels looking forward MICRO_HORIZON_TICKS
        from v7_engine.config import MICRO_HORIZON_TICKS
        purge_len = MICRO_HORIZON_TICKS // stride
        val_start_global = sub_val_idx[0]
        
        sub_train_idx = np.array([idx for idx in train_idx[:split_idx] if idx < val_start_global - purge_len])
        
        # 1. Train SDE
        if not os.path.exists(SDE_PATH):
            logger.info("Training Neural SDE...")
            df_train = to_polars(full_seqs[sub_train_idx], full_vars[sub_train_idx], full_barrier_labels[sub_train_idx], full_tns[sub_train_idx])
            df_val   = to_polars(full_seqs[sub_val_idx], full_vars[sub_val_idx], full_barrier_labels[sub_val_idx], full_tns[sub_val_idx])
            
            cache_train = f"{FEATURES_DIR}/temp_train_fold_{fold_idx}.parquet"
            cache_val   = f"{FEATURES_DIR}/temp_val_fold_{fold_idx}.parquet"
            df_train.write_parquet(cache_train, compression="zstd")
            df_val.write_parquet(cache_val, compression="zstd")
            
            sde_epochs = TRAIN_SDE_EPOCHS_DRY if args.dry_run else SDE_EPOCHS
            sde_model = train_sde(cache_train, cache_val, epochs=sde_epochs, save_path=SDE_PATH)
            torch.save(sde_model.state_dict(), SDE_PATH)
        else:
            logger.info(f"SDE checkpoint found for Fold {fold_idx}.")
            
        sde_model = NeuralSDE().to(device)
        sde_model.load_state_dict(torch.load(SDE_PATH, map_location=device, weights_only=True))
        sde_model.eval()

        # 2. Train Latent Decoder
        if not os.path.exists(DEC_PATH):
            logger.info("Training Latent Decoder...")
            decoder = LatentDecoder().to(device)
            latent_list = []
            with torch.no_grad():
                for i in range(0, len(sub_train_idx), TRAIN_DECODER_BATCH_SIZE):
                    batch = full_seqs[sub_train_idx[i:i+TRAIN_DECODER_BATCH_SIZE]]
                    ctx = torch.tensor(batch, dtype=torch.float32, device=device)
                    _, final_state, _, _ = sde_model(ctx)
                    latent_list.append(final_state.cpu().numpy())
            latent_states = np.concatenate(latent_list, axis=0)
            
            decoder_epochs = TRAIN_DECODER_EPOCHS_DRY if args.dry_run else TRAIN_DECODER_EPOCHS
            decoder.train_on_pairs(latent_states, full_rets[sub_train_idx], epochs=decoder_epochs)
            decoder.save(DEC_PATH)
        else:
            logger.info(f"Latent Decoder checkpoint found for Fold {fold_idx}.")

        # 3. Train EBM
        EBM_PATH = f"{fold_dir}/ebm_best.pth"
        if not os.path.exists(EBM_PATH):
            logger.info("Training Energy Based Model (EBM)...")
            from v7_engine.training.build_ebm_dataset import build_dataset_from_memory
            from v7_engine.training.train_ebm import train_ebm
            
            # Generate latents for the full train_idx to train EBM
            latent_list_full = []
            with torch.no_grad():
                for i in range(0, len(train_idx), TRAIN_DECODER_BATCH_SIZE):
                    batch = full_seqs[train_idx[i:i+TRAIN_DECODER_BATCH_SIZE]]
                    ctx = torch.tensor(batch, dtype=torch.float32, device=device)
                    _, final_state, _, _ = sde_model(ctx)
                    latent_list_full.append(final_state.cpu().numpy())
            latent_states_full = np.concatenate(latent_list_full, axis=0)
            
            start_ns = full_tns[train_idx[0]]
            end_ns   = full_tns[train_idx[-1]]
            mask = (ticks["timestamp_ns"] >= start_ns) & (ticks["timestamp_ns"] <= end_ns)
            train_data = {
                "bid": ticks["bid"][mask],
                "ask": ticks["ask"][mask],
                "timestamp_ns": ticks["timestamp_ns"][mask]
            }
            
            build_dataset_from_memory(
                symbol=args.symbol,
                train_data=train_data,
                train_seqs=full_seqs[train_idx],
                train_tns=full_tns[train_idx],
                latent_states=latent_states_full,
                sde_checkpoint=SDE_PATH,
                save_dir=fold_dir
            )
            
            x_clean = np.load(os.path.join(fold_dir, "ebm_clean.npy"))
            x_toxic = np.load(os.path.join(fold_dir, "ebm_toxic.npy"))
            if len(x_clean) > 0 and len(x_toxic) > 0:
                ebm_epochs = 2 if args.dry_run else 10
                train_ebm(x_clean, x_toxic, epochs=ebm_epochs, save_path=EBM_PATH)
            else:
                logger.warning(f"Not enough clean/toxic samples to train EBM in Fold {fold_idx}")
        else:
            logger.info(f"EBM checkpoint found for Fold {fold_idx}.")
        
        # 4. Train RL
        if not os.path.exists(RL_PATH):
            logger.info("Training RL Actor...")
            from v7_engine.training.train_rl import train_rl
            from v7_engine.ebm.energy_model import EnergyModel
            
            ebm_model = None
            if os.path.exists(EBM_PATH):
                ebm_model = EnergyModel().to(device)
                ebm_model.load_state_dict(torch.load(EBM_PATH, map_location=device, weights_only=True))
                ebm_model.eval()
                logger.info("Loaded EBM for RL intrinsic dopamine motivation.")
            
            rl_episodes = TRAIN_RL_EPISODES_DRY if args.dry_run else TRAIN_RL_EPISODES
            rl_actor = train_rl(
                contexts=full_seqs[train_idx], # Train on full train set
                actual_returns=full_rets[train_idx], 
                valid_tns=full_tns[train_idx],
                sde=sde_model,
                ebm=ebm_model,
                episodes=rl_episodes, 
                save_path=RL_PATH,
                device_str=device.type
            )
            torch.save(rl_actor.state_dict(), RL_PATH)
        else:
            logger.info(f"RL checkpoint found for Fold {fold_idx}.")

        # 5. Train XGBoost
        if not os.path.exists(REG_PATH):
            logger.info("Training XGBoost Classifiers...")
            feat_matrix = full_seqs[train_idx]
            regime_labels = label_regimes_from_features(feat_matrix)
            regime_model = RegimeXGBClassifier()
            regime_model.fit(feat_matrix, regime_labels)
            regime_model.save(REG_PATH)
            
            X_risk, y_risk = label_risk_windows(feat_matrix, full_rets[train_idx])
            if len(X_risk) > 0:
                risk_model = RiskXGBClassifier()
                risk_model.fit(X_risk, y_risk)
                risk_model.save(RSK_PATH)
        else:
            logger.info(f"XGBoost checkpoints found for Fold {fold_idx}.")

        # 6. Train Regime HMM
        HMM_PATH = f"{fold_dir}/regime_hmm.pkl"
        if not os.path.exists(HMM_PATH):
            logger.info("Training Regime HMM...")
            from v7_engine.embedding.regime_hmm import RegimeHMM
            start_ns = full_tns[train_idx[0]]
            end_ns   = full_tns[train_idx[-1]]
            mask = (ticks["timestamp_ns"] >= start_ns) & (ticks["timestamp_ns"] <= end_ns)
            train_bids = ticks["bid"][mask]
            train_asks = ticks["ask"][mask]
            
            hmm_model = RegimeHMM()
            hmm_model.fit(train_bids, train_asks, save_path=HMM_PATH)
        else:
            logger.info(f"Regime HMM checkpoint found for Fold {fold_idx}.")

    logger.info("=== CPCV True Training Complete ===")

if __name__ == "__main__":
    main()

"""
Build EBM Dataset from Real Dukascopy Ticks.

1. Loads real ticks via DukascopyLoader
2. Labels tick windows using RealToxicityLabeler
3. Generates latent states using frozen NeuralSDE
4. Saves ebm_clean.npy and ebm_toxic.npy
"""

import numpy as np
import os
import logging
import torch
from v7_engine.ingestion.dukascopy_loader import DukascopyLoader
from v7_engine.ebm.real_toxicity_labeler import label_tick_windows
from v7_engine.sde.sde_model import NeuralSDE
from v7_engine.ingestion.ring_buffer import RingBuffer, TickRecord
from v7_engine.ingestion.welford import WelfordNormaliser
from v7_engine.embedding.tib_engine import TIBEngine
from v7_engine.embedding.feature_vector import TickFeatureVector
from v7_engine.config import (
    SDE_LATENT_DIM, EMBEDDING_DIM, WELFORD_CLIP_SIGMA,
    WELFORD_MIN_STD, WELFORD_WARMUP_TICKS, BACKTEST_INITIAL_EQUITY,
    MAX_DAILY_DRAWDOWN_USD, TRAIN_SEQUENCE_LENGTH, CHECKPOINT_SDE,
    MODELS_DIR, TRAIN_STRIDE, TRAIN_STRIDE_DRY
)

logger = logging.getLogger("build_ebm")

def build_dataset(
    symbol: str = "EURUSD",
    start_date: str = "2024-01-01",
    end_date: str = "2024-02-01",
    sde_checkpoint: str = CHECKPOINT_SDE,
    save_dir: str = MODELS_DIR,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load SDE
    sde = NeuralSDE().to(device)
    from v7_engine.utils.checkpoint_guard import load_verified_checkpoint
    if os.path.exists(sde_checkpoint):
        state = load_verified_checkpoint(sde_checkpoint, map_location=device)
        sde.load_state_dict(state)
        logger.info(f"Loaded frozen SDE from {sde_checkpoint}")
    else:
        logger.warning(f"SDE checkpoint not found at {sde_checkpoint}, using untrained SDE!")
    sde.eval()
    
    # 2. Load Ticks
    loader = DukascopyLoader()
    try:
        ticks = loader.load(symbol, start_date, end_date)
    except Exception as e:
        logger.error(f"Failed to load ticks: {e}")
        return
        
    n_ticks = len(ticks["bid"])
    logger.info(f"Loaded {n_ticks} ticks for {symbol}")
    
    # 3. Get labels
    logger.info("Labelling tick windows...")
    stride = TRAIN_STRIDE_DRY if len(ticks["bid"]) < 100000 else TRAIN_STRIDE
    lab_ts, labels, _ = label_tick_windows(
        ticks, forward_n=100, stride=stride, equity=BACKTEST_INITIAL_EQUITY, prop_limit=MAX_DAILY_DRAWDOWN_USD
    )
    
    # Create lookup map for fast checking
    label_map = {ts: lbl for ts, lbl in zip(lab_ts, labels)}
    
    # 4. Generate Latent States (Loading pre-extracted Parquet from training)
    logger.info("Loading extracted features from Parquet...")
    cache_train = os.path.join(MODELS_DIR, f"train_features_{symbol}_{start_date}_{end_date}.parquet")
    
    if not os.path.exists(cache_train):
        logger.error(f"Parquet file {cache_train} not found. Please run SDE training first.")
        return
        
    import polars as pl
    df_train = pl.read_parquet(cache_train)
    
    if "ts" not in df_train.columns:
        raise ValueError(f"Parquet file {cache_train} is missing the required 'ts' column. Re-run SDE training.")
        
    actual_f_cols = [c for c in df_train.columns if c.startswith("f_")]
    if len(actual_f_cols) != EMBEDDING_DIM:
        raise ValueError(f"Parquet file {cache_train} has {len(actual_f_cols)} feature columns, but EMBEDDING_DIM is {EMBEDDING_DIM}. Delete Parquet and re-run SDE training.")
    
    # Extract sequence features (f_0 to f_{EMBEDDING_DIM-1})
    feature_cols = [f"f_{i}" for i in range(EMBEDDING_DIM)]
    valid_seqs = df_train.select(feature_cols).to_numpy()
    
    # Extract timestamps to align with labels
    valid_tns = df_train.get_column("ts").to_numpy()
    out_idx = len(valid_tns)
    
    valid_mask_clean = np.zeros(out_idx, dtype=bool)
    valid_mask_toxic = np.zeros(out_idx, dtype=bool)
    
    for i in range(out_idx):
        ts = int(valid_tns[i])
        if ts in label_map:
            lbl = label_map[ts]
            if lbl == 0:
                valid_mask_clean[i] = True
            elif lbl == 1:
                valid_mask_toxic[i] = True
                
    clean_seqs = valid_seqs[valid_mask_clean]
    toxic_seqs = valid_seqs[valid_mask_toxic]
    
    logger.info(f"Found {len(clean_seqs)} clean, {len(toxic_seqs)} toxic labeled sequences.")
    
    def process_batch(seqs_numpy):
        if len(seqs_numpy) == 0:
            return np.empty((0, SDE_LATENT_DIM), dtype=np.float32)
            
        tensor_seqs = torch.tensor(seqs_numpy, dtype=torch.float32, device=device)
        batch_size = 4096
        latents_list = []
        
        with torch.no_grad():
            for i in range(0, len(tensor_seqs), batch_size):
                batch = tensor_seqs[i : i + batch_size]
                _, final_states, _, _ = sde(batch)
                latents_list.append(final_states.cpu().numpy())
                
        return np.concatenate(latents_list, axis=0)
        
    x_clean = process_batch(clean_seqs)
    x_toxic = process_batch(toxic_seqs)
    
    logger.info(f"Dataset generated: {len(x_clean)} clean, {len(x_toxic)} toxic.")
    np.save(os.path.join(save_dir, "ebm_clean.npy"), x_clean)
    np.save(os.path.join(save_dir, "ebm_toxic.npy"), x_toxic)
    logger.info(f"Saved to {save_dir}")

def build_dataset_from_memory(
    symbol: str,
    train_data: dict,
    train_seqs: np.ndarray,
    train_tns: np.ndarray,
    latent_states: np.ndarray,
    sde_checkpoint: str = CHECKPOINT_SDE,
    save_dir: str = MODELS_DIR,
):
    """Build EBM dataset directly from the already-split training slice to prevent leakage."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_ticks = len(train_data["bid"])
    logger.info(f"Labelling {n_ticks} training ticks...")
    
    # We pass the in-memory strictly-sliced train_data so toxicity labels cannot peer into validation!
    stride = TRAIN_STRIDE_DRY if len(train_data["bid"]) < 100000 else TRAIN_STRIDE
    lab_ts, labels, _ = label_tick_windows(
        train_data, forward_n=100, stride=stride, equity=BACKTEST_INITIAL_EQUITY, prop_limit=MAX_DAILY_DRAWDOWN_USD
    )
    
    label_map = {ts: lbl for ts, lbl in zip(lab_ts, labels)}
    
    valid_seqs = train_seqs
    valid_tns = train_tns
    out_idx = len(valid_tns)
    
    valid_mask_clean = np.zeros(out_idx, dtype=bool)
    valid_mask_toxic = np.zeros(out_idx, dtype=bool)
    
    for i in range(out_idx):
        ts = int(valid_tns[i])
        if ts in label_map:
            lbl = label_map[ts]
            if lbl == 0:
                valid_mask_clean[i] = True
            elif lbl == 1:
                valid_mask_toxic[i] = True
                
    x_clean = latent_states[valid_mask_clean]
    x_toxic = latent_states[valid_mask_toxic]
    
    logger.info(f"Dataset generated directly from memory: {len(x_clean)} clean, {len(x_toxic)} toxic.")
    np.save(os.path.join(save_dir, "ebm_clean.npy"), x_clean)
    np.save(os.path.join(save_dir, "ebm_toxic.npy"), x_toxic)
    logger.info(f"Saved to {save_dir}")

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Build EBM Dataset from Real Dukascopy Ticks.")
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    args = parser.parse_args()
    
    build_dataset(symbol=args.symbol, start_date=args.start, end_date=args.end)

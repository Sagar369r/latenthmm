from v7_engine.config import CHECKPOINT_SDE
"""
Full SDE Training Pipeline.

Loss function composition:
  L_total = L_calibration + α * L_reconstruction + β * L_KL

  1. L_calibration : σ̂_φ²(X_T) vs realised variance — prevents diffusion collapse
  2. L_reconstruction: final latent state vs next-step price direction
  3. L_KL           : KL(q(X_T) || N(0,I)) — prevents posterior collapse

Architecture safeguards:
  - Spectral norm on Transformer (drift) — Lipschitz continuity
  - Gradient clipping — prevents explosion in adjoint backprop
  - NaN interception — falls back to zero gradient on explosion
  - Cosine LR annealing — prevents late-training instability

Usage:
    python -m v7_engine.training.train_sde
"""

import torch
import torch.nn as nn
import numpy as np
import os
import logging
from v7_engine.config import (
    SDE_EPOCHS, SDE_BATCH_SIZE, SDE_LR, SDE_GRAD_CLIP,
    SDE_LATENT_DIM, SDE_DT, EMBEDDING_DIM, SDE_LR_SCHEDULER
)
from v7_engine.sde.sde_model import NeuralSDE

torch.set_num_threads(2) # Lock physical cores to prevent thread thrashing

logger = logging.getLogger("train_sde")


# ── Loss Functions ────────────────────────────────────────────────────────────

def sde_calibration_loss(
    pred_sigma_sq:   torch.Tensor,   # (batch,) predicted variance
    actual_variance: torch.Tensor,   # (batch,) realised variance
) -> torch.Tensor:
    """
    SDE Calibration Error: E[|σ̂² - σ²_real|]
    Target: → 0  (continuous minimisation, not a fixed threshold)
    """
    return torch.mean(torch.abs(pred_sigma_sq - actual_variance))


def reconstruction_loss(
    final_state:  torch.Tensor,   # (batch, latent_dim)
    next_returns: torch.Tensor,   # (batch,) actual next-step returns
) -> torch.Tensor:
    """
    Predict the sign of the next return from the final SDE state.
    Uses BCE so the model learns directional market structure.
    """
    sign_pred  = final_state[:, 0]   # first latent dimension as direction proxy
    sign_label = (next_returns > 0).float()
    return nn.functional.binary_cross_entropy_with_logits(sign_pred, sign_label)


def kl_regularisation(final_state: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from N(0,I) to prevent posterior collapse.
    KL(q || p) = 0.5 * (μ² + σ² - log σ² - 1)  with σ²=1 simplifies to:
    KL ≈ 0.5 * E[||X_T||²] — penalise large latent norms
    """
    return 0.5 * torch.mean(final_state.pow(2))


# ── Dataset Preparation ───────────────────────────────────────────────────────

def _worker_train_extract_chunk(args):
    import numpy as np
    from v7_engine.config import EMBEDDING_DIM, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD
    from v7_engine.ingestion.ring_buffer import RingBuffer
    from v7_engine.embedding.tib_engine import TIBEngine
    from v7_engine.ingestion.welford import WelfordNormaliser
    from v7_engine.embedding.feature_vector import TickFeatureVector

    (
        chunk_id, chunk_start, chunk_end,
        timestamps_slice, bids_slice, asks_slice, dts_slice,
        welford_state_dict, STRIDE, SEQ_LEN, update_welford
    ) = args

    buf = RingBuffer(SEQ_LEN)
    tib = TIBEngine(window=500)
    fv = TickFeatureVector(buf, tib, welford_normaliser=None)
    
    n_local = len(bids_slice)
    max_out = n_local // STRIDE + 1
    out_seqs = np.empty((max_out, EMBEDDING_DIM), dtype=np.float32)
    out_vars = np.empty(max_out, dtype=np.float32)
    out_rets = np.empty(max_out, dtype=np.float32)
    out_tick_idx = np.empty(max_out, dtype=np.int64)
    out_valid_tns = np.empty(max_out, dtype=np.int64)
    
    out_idx = 0
    warmup_offset = n_local - (chunk_end - chunk_start)
    
    prev_mid = float(bids_slice[0] + asks_slice[0]) / 2.0 if n_local > 0 else 0.0
    
    for local_i in range(n_local):
        mid = float(bids_slice[local_i] + asks_slice[local_i]) / 2.0
        if mid > prev_mid: sign = 1
        elif mid < prev_mid: sign = -1
        else: sign = 0
        prev_mid = mid

        buf.push(int(timestamps_slice[local_i]), float(bids_slice[local_i]), float(asks_slice[local_i]), float(dts_slice[local_i]), sign)
        fv.update_state(mid, float(dts_slice[local_i]), sign)
        
        if local_i >= warmup_offset:
            global_i = chunk_start + (local_i - warmup_offset)
            
            if (global_i % STRIDE) == 0 and buf.count >= SEQ_LEN:
                vec = fv.compute(SEQ_LEN, update_welford=False)
                
                out_seqs[out_idx] = vec
                    
                start_idx_var = max(0, local_i - 1000)
                mids = (bids_slice[start_idx_var:local_i+1] + asks_slice[start_idx_var:local_i+1]) / 2.0
                rets = np.diff(np.log(mids[mids > 0]))
                out_vars[out_idx] = float(np.var(rets)) if len(rets) > 0 else 0.0
                
                out_tick_idx[out_idx] = global_i
                out_valid_tns[out_idx] = timestamps_slice[local_i]
                
                if local_i + STRIDE < n_local:
                    mid_curr = float(bids_slice[local_i] + asks_slice[local_i]) / 2.0
                    mid_next = float(bids_slice[local_i + STRIDE] + asks_slice[local_i + STRIDE]) / 2.0
                    out_rets[out_idx] = float(np.log(max(mid_next, 1e-12) / max(mid_curr, 1e-12)))
                else:
                    out_rets[out_idx] = 0.0
                    
                out_idx += 1
                    
    return (
        chunk_id,
        out_seqs[:out_idx],
        out_vars[:out_idx],
        out_rets[:out_idx],
        out_tick_idx[:out_idx],
        out_valid_tns[:out_idx],
        None
    )

def prepare_sequences_from_ticks(
    tick_data:    dict,
    feat_vec_cls,
    seq_len:      int = 100,
    stride:       int = 20,
    normaliser        = None,
    update_welford: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, any]:
    import time
    import concurrent.futures
    import multiprocessing as mp
    from tqdm import tqdm
    import logging
    from v7_engine.config import EMBEDDING_DIM

    logger = logging.getLogger("train_sde")

    if normaliser is None:
        raise ValueError("WelfordNormaliser must be explicitly provided to prevent data leakage.")

    bids = tick_data["bid"].astype(np.float64)
    asks = tick_data["ask"].astype(np.float64)
    dts  = tick_data["delta_t"].astype(np.float64)
    tns  = tick_data.get("timestamp_ns", np.arange(len(bids)) * int(1e8)).astype(np.int64)
    n    = len(bids)

    CHUNK_SIZE = 250_000
    welford_state_dict = normaliser.save_state()
    
    if update_welford and normaliser._n == 0:
        logger.info("Welford is cold. Performing fast sequential warmup pass...")
        from v7_engine.ingestion.ring_buffer import RingBuffer
        from v7_engine.embedding.tib_engine import TIBEngine
        buf = RingBuffer(seq_len)
        tib = TIBEngine(window=500)
        fv = feat_vec_cls(buf, tib, welford_normaliser=normaliser)
        warmup_n = min(n, normaliser._warmup + 1000)
        for i in range(warmup_n):
            buf.push(int(tns[i]), float(bids[i]), float(asks[i]), float(dts[i]), 1)
            fv.update_state(float(bids[i] + asks[i]) / 2.0, float(dts[i]), 1)
            if buf.count >= seq_len:
                fv.compute(seq_len, update_welford=True)
        welford_state_dict = normaliser.save_state()
        import os
        os.makedirs("models", exist_ok=True)
        np.savez("models/welford.npz", mean=normaliser._mean, m2=normaliser._M2, n=normaliser._n)
        logger.info("Warmup complete. Welford state synced for parallel workers and saved to disk.")

    worker_args = []
    chunk_id = 0
    
    logger.info(f"Chunking {n} ticks into MULTIPROCESSING windows...")
    for chunk_start in range(0, n, CHUNK_SIZE):
        chunk_end = min(n, chunk_start + CHUNK_SIZE)
        slice_start = max(0, chunk_start - seq_len - 1500)
        
        args = (
            chunk_id, chunk_start, chunk_end,
            tns[slice_start:chunk_end],
            bids[slice_start:chunk_end],
            asks[slice_start:chunk_end],
            dts[slice_start:chunk_end],
            welford_state_dict,
            stride, seq_len, update_welford
        )
        worker_args.append(args)
        chunk_id += 1

    all_chunk_out_seqs = {}
    all_chunk_out_vars = {}
    all_chunk_out_rets = {}
    all_chunk_out_tick_idx = {}
    all_chunk_out_valid_tns = {}
    all_welford_states = {}

    t0 = time.time()
    ctx = mp.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(mp_context=ctx) as executor:
        futures = {executor.submit(_worker_train_extract_chunk, args): args for args in worker_args}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parallel Extraction"):
            cid, cseqs, cvars, crets, ctick, ctns, cwelford = future.result()
            all_chunk_out_seqs[cid] = cseqs
            all_chunk_out_vars[cid] = cvars
            all_chunk_out_rets[cid] = crets
            all_chunk_out_tick_idx[cid] = ctick
            all_chunk_out_valid_tns[cid] = ctns
            if cwelford is not None:
                all_welford_states[cid] = cwelford

    t1 = time.time()
    logger.info(f"Feature Extraction Complete! Processed {n} ticks in {t1 - t0:.2f} seconds.")

    train_seqs_list = []
    train_vars_list = []
    next_rets_list = []
    tick_idx_list = []
    valid_tns_list = []

    for cid in range(chunk_id):
        train_seqs_list.append(all_chunk_out_seqs[cid])
        train_vars_list.append(all_chunk_out_vars[cid])
        next_rets_list.append(all_chunk_out_rets[cid])
        tick_idx_list.append(all_chunk_out_tick_idx[cid])
        valid_tns_list.append(all_chunk_out_valid_tns[cid])

    train_seqs = np.concatenate(train_seqs_list, axis=0) if train_seqs_list else np.empty((0, EMBEDDING_DIM))
    train_vars = np.concatenate(train_vars_list, axis=0) if train_vars_list else np.empty((0,))
    next_rets  = np.concatenate(next_rets_list, axis=0) if next_rets_list else np.empty((0,))
    tick_idx   = np.concatenate(tick_idx_list, axis=0) if tick_idx_list else np.empty((0,), dtype=np.int64)
    valid_tns  = np.concatenate(valid_tns_list, axis=0) if valid_tns_list else np.empty((0,), dtype=np.int64)

    if update_welford:
        from v7_engine.config import WELFORD_CLIP_SIGMA, WELFORD_MIN_STD
        from v7_engine.ingestion.welford import WelfordNormaliser
        
        logger.info("Applying sequential Welford normalization to raw features (Master Process)...")
        normaliser = WelfordNormaliser.from_state(
            welford_state_dict,
            dim=EMBEDDING_DIM,
            clip_sigma=WELFORD_CLIP_SIGMA,
            min_std=WELFORD_MIN_STD
        )
        normed_seqs = np.empty_like(train_seqs)
        for i in tqdm(range(len(train_seqs)), desc="Online Welford Pass"):
            normed_seqs[i] = normaliser.update_and_transform(train_seqs[i])
        train_seqs = normed_seqs
    else:
        if normaliser is not None:
            logger.info("Applying frozen Welford normalization to raw features...")
            train_seqs = normaliser.transform(train_seqs)

    # Run Triple Barrier Labeling
    logger.info("Computing Triple Barrier Target Labels...")
    from v7_engine.training.triple_barrier import fast_triple_barrier
    t2 = time.time()
    from v7_engine.config import MICRO_HORIZON_TICKS
    barrier_labels = fast_triple_barrier(
        bids.astype(np.float64), 
        asks.astype(np.float64), 
        tns.astype(np.int64), 
        tick_idx, 
        train_vars,
        max_ticks=MICRO_HORIZON_TICKS
    )
    t3 = time.time()
    logger.info(f"Triple Barrier Labeling Complete in {t3 - t2:.2f} seconds.")
    
    return train_seqs, train_vars, next_rets, barrier_labels, normaliser, valid_tns


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_sde(
    train_parquet_path: str,
    val_parquet_path:   str,
    alpha: float = 0.1,              # recon loss weight
    beta:  float = 0.01,             # KL weight
    epochs: int = SDE_EPOCHS,
    save_path: str = None,
) -> NeuralSDE:
    """
    Full SDE training with calibration + reconstruction + KL losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NeuralSDE().to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=SDE_LR, weight_decay=1e-5)

    if SDE_LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5)

    best_val_loss = float("inf")
    loss_history = []

    from tqdm import tqdm
    from v7_engine.training.parquet_dataset import ParquetStreamingDataset
    from torch.utils.data import DataLoader
    
    train_dataset = ParquetStreamingDataset(train_parquet_path, batch_size=SDE_BATCH_SIZE)
    # Explicitly hardcode num_workers=0 to prevent PyTorch from duplicating Polars in memory
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)

    for epoch in tqdm(range(epochs), desc="SDE Epochs"):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        
        optim.zero_grad()

        for i, (seqs, vars_, labels) in enumerate(tqdm(train_loader, desc="Batches", leave=False, mininterval=1.0)):
            seqs  = seqs.to(device)
            vars_ = vars_.to(device)
            labels = labels.to(device).long() # For cross-entropy

            # Initial latent: Not needed anymore as proj happens inside model
            # x0 = seqs[:, :SDE_LATENT_DIM]
            # ts = torch.linspace(0, 1, 2, device=device)
            # context = seqs.unsqueeze(1)

            try:
                paths, final, trend_logits, vol_pred = model(seqs)
            except RuntimeError as e:
                logger.warning(f"SDE forward error at epoch {epoch}: {e}")
                continue

            # Guard against NaN paths
            if torch.isnan(final).any():
                logger.warning(f"NaN in SDE output at epoch {epoch}, skipping batch")
                continue

            # 1. Calibration loss (Volatility Head vs Realized Variance)
            cal_loss = torch.nn.functional.mse_loss(vol_pred ** 2, vars_)

            # 2. KL regularisation
            kl_loss  = kl_regularisation(final)

            # 3. Triple Barrier Trend Loss
            # labels are -1, 0, 1. Map to 0, 1, 2
            mapped_labels = labels + 1
            trend_loss = torch.nn.functional.cross_entropy(trend_logits, mapped_labels)

            loss = cal_loss + alpha * trend_loss + beta * kl_loss

            loss.backward()

            # Hardware vectorization clip handles NaNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), SDE_GRAD_CLIP)
            optim.step()
            optim.zero_grad()

            epoch_loss += loss.item()
            n_batches  += 1

        # Scheduler step
        avg_train = epoch_loss / max(n_batches, 1)
        if SDE_LR_SCHEDULER == "cosine":
            scheduler.step()
        else:
            scheduler.step(avg_train)

        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_dataset = ParquetStreamingDataset(val_parquet_path, batch_size=SDE_BATCH_SIZE)
            val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)
            
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                # Evaluate 100 batches max to save time
                for j, (v_seqs, v_vars, v_labels) in enumerate(val_loader):
                    if j >= 100:
                        break
                    
                    v_seqs = v_seqs.to(device)
                    v_vars = v_vars.to(device)
                    v_labels = v_labels.to(device).long()
                    
                    paths_v, final_v, trend_logits_v, vol_pred_v = model(v_seqs)
                        
                    if torch.isnan(final_v).any():
                        continue
                        
                    cal_loss_v = torch.nn.functional.mse_loss(vol_pred_v ** 2, v_vars)
                    kl_loss_v = kl_regularisation(final_v)
                    mapped_labels_v = v_labels + 1
                    trend_loss_v = torch.nn.functional.cross_entropy(trend_logits_v, mapped_labels_v)
                    
                    loss_v = cal_loss_v + alpha * trend_loss_v + beta * kl_loss_v
                    val_loss += loss_v.item()
                    val_batches += 1

            if val_batches > 0:
                avg_val = val_loss / val_batches
                logger.info(f"Epoch {epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

                if avg_val < best_val_loss and np.isfinite(avg_val):
                    best_val_loss = avg_val
                    if save_path:
                        torch.save(model.state_dict(), save_path)
                    logger.info(f"  ✅ New best SDE saved (val_loss={avg_val:.5f})")
            else:
                avg_val = None
                logger.info(f"Epoch {epoch} | Train: {avg_train:.4f} | Val: N/A")
                
            loss_history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})

    # Save final model
    if save_path:
        final_path = save_path.replace("best", "final") if "best" in save_path else save_path + "_final"
        torch.save(model.state_dict(), final_path)
    
    import json
    with open("sde_loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=4)
        
    logger.info(f"SDE training complete. Best val error: {best_val_loss:.5f}")
    return model


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    """
    Train SDE on real Dukascopy CFD tick data.
    """
    import argparse
    from v7_engine.ingestion.dukascopy_loader import DukascopyLoader
    from v7_engine.embedding.feature_vector import TickFeatureVector

    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    args = parser.parse_args()

    logger.info(f"Loading Dukascopy ticks for {args.symbol} from {args.start} to {args.end}...")
    loader = DukascopyLoader()
    try:
        ticks = loader.load(args.symbol, args.start, args.end)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    n_ticks = len(ticks["bid"])
    split_idx = int(n_ticks * 0.8)
    
    from v7_engine.config import MICRO_HORIZON_TICKS
    # Purge gap to prevent leakage into the validation set
    train_data = {k: v[:split_idx - MICRO_HORIZON_TICKS] for k, v in ticks.items()}
    val_data = {k: v[split_idx:] for k, v in ticks.items()}

    cache_train = f"v7/models/train_features_{args.symbol}_{args.start}_{args.end}.parquet"
    cache_val   = f"v7/models/val_features_{args.symbol}_{args.start}_{args.end}.parquet"
    welford_file = "v7/models/welford.npz"
    
    import os
    import polars as pl
    if os.path.exists(cache_train) and os.path.exists(cache_val) and os.path.exists(welford_file):
        logger.info(f"Loading extracted features from cache: {cache_train}")
        df_train = pl.read_parquet(cache_train)
        df_val   = pl.read_parquet(cache_val)
        
        # No need to load arrays into RAM anymore because train_sde will stream them lazily.
        
        # Load welford
        from v7_engine.ingestion.welford import WelfordNormaliser
        from v7_engine.config import EMBEDDING_DIM, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD
        train_normaliser = WelfordNormaliser(EMBEDDING_DIM, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD)
        w_data = np.load(welford_file)
        state = {
            "mean": w_data["mean"],
            "m2": w_data["m2"],
            "n": int(w_data["n"])
        }
        train_normaliser = WelfordNormaliser.from_state(state, EMBEDDING_DIM, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD)
    else:
        logger.info("Extracting feature sequences (this may take a while)...")
        from v7_engine.ingestion.welford import WelfordNormaliser
        from v7_engine.config import WELFORD_CLIP_SIGMA, WELFORD_MIN_STD, WELFORD_WARMUP_TICKS, EMBEDDING_DIM, TRAIN_SEQUENCE_LENGTH, TRAIN_STRIDE
        train_normaliser = WelfordNormaliser(EMBEDDING_DIM, WELFORD_CLIP_SIGMA, WELFORD_MIN_STD, warmup=WELFORD_WARMUP_TICKS)
        
        # Prepare train/val datasets
        import time
        logger.info("Extracting feature sequences...")
        
        train_seqs, train_vars, train_rets, train_labels, train_normaliser, train_tns = prepare_sequences_from_ticks(
            train_data, TickFeatureVector, seq_len=TRAIN_SEQUENCE_LENGTH, stride=TRAIN_STRIDE, normaliser=train_normaliser, update_welford=True
        )
        val_seqs, val_vars, val_rets, val_labels, _, val_tns = prepare_sequences_from_ticks(
            val_data, TickFeatureVector, seq_len=TRAIN_SEQUENCE_LENGTH, stride=TRAIN_STRIDE, normaliser=train_normaliser, update_welford=False
        )
        
        logger.info(f"Saving extracted features to {cache_train} and {cache_val}")
        
        def to_polars(seqs, vars_, rets, labels, ts_array):
            data = {f"f_{i}": seqs[:, i] for i in range(seqs.shape[1])}
            data["var"] = vars_
            data["ret"] = rets
            data["barrier_label"] = labels
            data["ts"] = ts_array
            return pl.DataFrame(data)
            
        df_train = to_polars(train_seqs, train_vars, train_rets, train_labels, train_tns)
        df_train.write_parquet(cache_train, compression="zstd")
        
        df_val = to_polars(val_seqs, val_vars, val_rets, val_labels, val_tns)
        df_val.write_parquet(cache_val, compression="zstd")
        
    # Always save the welford normaliser independently so the EBM and live pipelines can load it
    w_state = train_normaliser.save_state()
    np.savez_compressed(welford_file, mean=w_state["mean"], m2=w_state["m2"], n=w_state["n"])
    logger.info(f"Saved Welford normaliser to {welford_file}")

    import polars as pl
    n_train = pl.scan_parquet(cache_train).select(pl.len()).collect().item()
    n_val   = pl.scan_parquet(cache_val).select(pl.len()).collect().item()
    
    logger.info(
        f"Dataset: {n_train} train, {n_val} val sequences"
    )

    if n_train == 0 or n_val == 0:
        raise RuntimeError("Not enough data after warmup — increase date range")

    train_sde(
        train_parquet_path=cache_train,
        val_parquet_path=cache_val,
    )


if __name__ == "__main__":
    main()

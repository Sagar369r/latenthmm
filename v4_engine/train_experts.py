import os
import sys
import polars as pl
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v4_engine.feature_expansion import extract_z_score_tensor
from v4_engine.data_transformer import transform_to_dimensionless
from v4_engine.vae_model import VAE
from v4_engine.gmm_hmm import LatentRouter
from v4_engine.triple_barrier import apply_triple_barrier
from v4_engine.expert_layer import ExpertLayer

def generate_expert_oof(csv_paths: list[str], output_path: str):
    print(f"=== Phase 4: Generating Expert OOF for Multi-Asset Basket ===")
    
    vae = VAE(input_dim=20, latent_dim=4)
    vae_path = "v4_engine/models/vae_weights.pth"
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, weights_only=True))
    else:
        print("WARNING: Using untrained VAE weights!")
    vae.eval()
    
    router_path = "v4_engine/models/gmm_router.pkl"
    if not os.path.exists(router_path):
        raise FileNotFoundError("Router model not found. Run train_router.py first.")
    router = LatentRouter.load(router_path)
    
    all_dfs = []
    
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Skipping {csv_path} - not found.")
            continue
            
        print(f"Processing {csv_path}...")
        df = pl.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        
        # We need the 'time' column to sort chronologically across assets
        if "time" not in df.columns and "timestamp" in df.columns:
            df = df.rename({"timestamp": "time"})
            
        # 0. Phase 1 Data Transformer (Injects base ATR and base Z-Scores)
        df = transform_to_dimensionless(df, window=90)
        
        # 1. Triple Barrier Labeling
        df = apply_triple_barrier(df, tp_mult=2.0, sl_mult=1.0, time_limit=24)
        
        # 2. Phase 1 Data Extractor
        df, z_cols = extract_z_score_tensor(df, window=90)
        
        # We must skip the 90-period warmup
        df = df.slice(90)
        
        features_np = df.select(z_cols).to_numpy()
        features_tensor = torch.tensor(features_np, dtype=torch.float32)
        
        with torch.no_grad():
            mu, _ = vae.encode(features_tensor)
            latent_np = mu.numpy()
            
        # Append Latent features to the dataframe
        latent_cols = [f"latent_{i}" for i in range(4)]
        df = df.with_columns([pl.Series(name, latent_np[:, i]) for i, name in enumerate(latent_cols)])
        
        # 4. Phase 3 HMM Router
        # Use causal forward-filtering probabilities to prevent lookahead bias
        probas = router.predict_causal_proba(latent_np)
        
        # Argmax to find the dominant regime string
        regime_labels = [max(p.items(), key=lambda x: x[1])[0] for p in probas]
        df = df.with_columns(pl.Series("regime", regime_labels))
        
        pandas_df = df.to_pandas()
        pandas_df["asset"] = os.path.basename(csv_path).split('_')[0].split('-')[0]
        
        if "time" in pandas_df.columns:
            pandas_df["time"] = pd.to_datetime(pandas_df["time"])
            
        all_dfs.append(pandas_df)
        
    if not all_dfs:
        print("No valid data processed.")
        return
        
    print("\nMerging and sorting Multi-Asset DataFrame by Time...")
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    if "time" in master_df.columns:
        master_df = master_df.sort_values("time").reset_index(drop=True)
        
    # 5. Phase 4 Expert Layer Routing & OOF
    # We will pass BOTH Latent features and Z-score indicators as inputs
    expert_features = latent_cols + z_cols
    
    expert_layer = ExpertLayer(n_splits=5, embargo_bars=5)
    
    # We will store the final predictions back into the main DataFrame
    master_df["expert_pred"] = np.nan
    
    for regime in ["TREND", "MEAN_REV", "COMPRESSION"]:
        mask = master_df["regime"] == regime
        if mask.sum() < 50:
            print(f"Skipping {regime} - Not enough samples ({mask.sum()})")
            continue
            
        sub_df = master_df[mask].copy()
        print(f"Routing {mask.sum()} samples to {regime} Expert...")
        
        oof_preds = expert_layer.generate_oof_predictions(sub_df, expert_features, regime)
        
        # Place the OOF predictions back into the master dataset
        master_df.loc[mask, "expert_pred"] = oof_preds
        
        # Train on the FULL dataset for live execution and save the model
        full_X = sub_df[expert_features].to_numpy()
        full_y = sub_df["target"].to_numpy()
        global_model = expert_layer.models[regime]
        
        if regime == "COMPRESSION":
            global_model.fit(full_X)
        else:
            global_model.fit(full_X, full_y)
            
        import joblib
        os.makedirs("v4_engine/models", exist_ok=True)
        joblib.dump(global_model, f"v4_engine/models/expert_{regime}.pkl")
        
    # 6. Save the final Unified Output
    # Drop rows where OOF is NaN (the very first fold of TimeSeriesSplit doesn't get validated)
    final_df = master_df.dropna(subset=["expert_pred"]).copy()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✓ Purged OOF Loop complete. Unified output saved to {output_path}")
    print(f"Final OOF Dataset Size: {len(final_df)} bars.")

if __name__ == "__main__":
    target_pairs = [
        "data/audcad_1h.csv",
        "data/chfjpy_1h.csv",
        "data/eurnzd_1h.csv"
    ]
    generate_expert_oof(target_pairs, "v4_engine/data/master_oof_dataset.csv")

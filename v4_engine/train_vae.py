import os
import sys
import torch
import torch.optim as optim
import polars as pl
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v4_engine.feature_expansion import extract_z_score_tensor
from v4_engine.data_transformer import transform_to_dimensionless
from v4_engine.vae_model import VAE, vae_loss_function

def train_vae(csv_paths: list[str], output_model_path: str):
    print(f"=== Phase 2: Training PyTorch VAE on Multi-Asset Basket ===")
    
    # 1. Data Prep
    all_features = []
    
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Skipping {csv_path} - not found.")
            continue
            
        print(f"Processing {csv_path}...")
        df = pl.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        
        # ── Chronological Firewall (2022-12-31) ──
        UNSUPERVISED_TRAIN_END = 1672444800000
        if "timestamp" in df.columns:
            df = df.filter(pl.col("timestamp") <= UNSUPERVISED_TRAIN_END)
            
        df = transform_to_dimensionless(df, window=90)
        df_feat, z_cols = extract_z_score_tensor(df, window=90)
        
        features_np = df_feat.select(z_cols).to_numpy()[90:]
        all_features.append(features_np)
        
    if not all_features:
        print("No valid CSVs provided.")
        return
        
    combined_features = np.vstack(all_features)
    print(f"Combined Tensor Shape: {combined_features.shape}")
    
    X_t = torch.tensor(combined_features, dtype=torch.float32)
    
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 2. Initialize VAE
    model = VAE(input_dim=20, latent_dim=4)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # 3. Train
    epochs = 30
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in loader:
            batch_x = batch[0]
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(batch_x)
            loss, recon, kld = vae_loss_function(recon_x, batch_x, mu, logvar, beta=0.1) # Beta=0.1 to focus on reconstruction initially
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(loader.dataset)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {train_loss:.4f}")
            
    # 4. Save
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"✓ True VAE trained and saved to {output_model_path}")

if __name__ == "__main__":
    target_pairs = [
        "data/audcad_1h.csv",
        "data/chfjpy_1h.csv",
        "data/eurnzd_1h.csv"
    ]
    train_vae(target_pairs, "v4_engine/models/vae_weights.pth")

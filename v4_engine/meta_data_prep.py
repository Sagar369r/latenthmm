import pandas as pd
import numpy as np

def prepare_meta_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Ingests the Phase 4 Master OOF dataset and transforms it into the strict
    17-Dimensional tensor required for the Meta-Learner FFNN.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Calculate Target (The Delta)
    # Delta = Actual Outcome - Predicted Confidence
    df["delta"] = df["target"] - df["expert_pred"]
    
    # 2. Static Quantization of Expert Confidence (Causal Firewall)
    # We use pd.cut with fixed bins (0.0 to 1.0) to ensure a confidence score of 0.85 
    # in 2020 maps to the EXACT same bin as a 0.85 in 2024.
    bins = np.linspace(0, 1, 11) # 10 equal-width bins from 0.0 to 1.0
    df["expert_bin"] = pd.cut(df["expert_pred"], bins=bins, labels=False, include_lowest=True)
    
    # 3. One-Hot Encode the 10 Confidence Deciles
    # We force exactly 10 columns (0 to 9) to ensure tensor shape stability
    expert_ohe = pd.get_dummies(df["expert_bin"], prefix="conf_bin")
    # Ensure all 10 bins exist even if some are missing in small datasets
    for i in range(10):
        col_name = f"conf_bin_{i}"
        if col_name not in expert_ohe.columns:
            expert_ohe[col_name] = False
    # Sort columns to guarantee exact tensor order
    expert_ohe = expert_ohe[[f"conf_bin_{i}" for i in range(10)]].astype(float)
    
    # 4. One-Hot Encode the 3 HMM States
    regime_ohe = pd.get_dummies(df["regime"], prefix="regime")
    for r in ["regime_TREND", "regime_MEAN_REV", "regime_COMPRESSION"]:
        if r not in regime_ohe.columns:
            regime_ohe[r] = False
    regime_ohe = regime_ohe[["regime_TREND", "regime_MEAN_REV", "regime_COMPRESSION"]].astype(float)
    
    # 5. Extract the 4 Latent Features
    latent_cols = [col for col in df.columns if col.startswith("latent_")]
    latent_df = df[latent_cols].astype(float)
    
    # 6. Concatenate into the final 17-Dimensional Input Tensor
    # Order: [4 Latent] + [3 HMM] + [10 Quantized Expert]
    X_df = pd.concat([latent_df, regime_ohe, expert_ohe], axis=1)
    
    X = X_df.to_numpy()
    y = df["delta"].to_numpy().reshape(-1, 1)
    
    return X, y

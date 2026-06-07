import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from v4_engine.meta_data_prep import prepare_meta_dataset
from v4_engine.meta_learner import ResidualMetaLearner, meta_loss_function

def test_meta_data_prep():
    # 1. Create a dummy master_oof_dataset.csv
    data = {
        "target": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        "expert_pred": [0.95, 0.85, 0.15, 0.05, 0.50, 0.60, 0.40, 0.30, 0.75, 0.25],
        "regime": ["TREND", "MEAN_REV", "COMPRESSION", "TREND", "MEAN_REV", 
                   "COMPRESSION", "TREND", "MEAN_REV", "COMPRESSION", "TREND"],
        "latent_0": np.random.randn(10),
        "latent_1": np.random.randn(10),
        "latent_2": np.random.randn(10),
        "latent_3": np.random.randn(10)
    }
    df = pd.DataFrame(data)
    os.makedirs("v4_engine/tests/dummy_data", exist_ok=True)
    dummy_csv = "v4_engine/tests/dummy_data/test_oof.csv"
    df.to_csv(dummy_csv, index=False)
    
    # 2. Run the prep function
    X, y = prepare_meta_dataset(dummy_csv)
    
    # Assert exact tensor shape
    assert X.shape[1] == 17, f"Expected 17 features, got {X.shape[1]}"
    assert y.shape[1] == 1, "Target shape mismatch"
    
    # Assert Delta calculation
    # index 1: target=0, pred=0.85 -> delta = -0.85
    assert np.isclose(y[1][0].item(), -0.85)
    
    print("✓ Meta-Learner Data Prep: Delta calculation and 17-Dimensional Soft Quantization successful.")

def test_meta_architecture():
    # Generate fake [Batch, 17] tensor
    batch_size = 64
    x = torch.randn(batch_size, 17)
    y_true = torch.randn(batch_size, 1)
    
    model = ResidualMetaLearner(input_dim=17)
    model.train()
    
    preds = model(x)
    assert preds.shape == (batch_size, 1), "Output shape mismatch"
    
    loss = meta_loss_function(preds, y_true)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        
    print("✓ Meta-Learner Architecture: [Batch, 17] -> [Batch, 1] compression successful. Gradients intact.")

if __name__ == "__main__":
    test_meta_data_prep()
    test_meta_architecture()

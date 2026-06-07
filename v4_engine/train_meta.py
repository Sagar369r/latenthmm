import os
import sys
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v4_engine.meta_data_prep import prepare_meta_dataset
from v4_engine.meta_learner import ResidualMetaLearner, meta_loss_function

def train_meta_judge(csv_path: str, output_model_path: str):
    print(f"=== Phase 5: Training Meta-Learner on {csv_path} ===")
    
    # 1. Prepare Data
    X_np, y_np = prepare_meta_dataset(csv_path)
    print(f"Input Tensor Shape: {X_np.shape}")
    print(f"Target Tensor Shape: {y_np.shape}")
    
    # Simple Temporal Train/Test Split (80/20)
    split_idx = int(len(X_np) * 0.8)
    X_train, y_train = X_np[:split_idx], y_np[:split_idx]
    X_test, y_test = X_np[split_idx:], y_np[split_idx:]
    
    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 2. Initialize Model
    model = ResidualMetaLearner(input_dim=17)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 3. Training Loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = meta_loss_function(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
            test_loss = meta_loss_function(test_preds, y_test_t).item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
            
    # 4. Save the Frozen Meta-Judge
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"✓ Meta-Learner training complete. Frozen weights saved to {output_model_path}")

if __name__ == "__main__":
    train_meta_judge("v4_engine/data/master_oof_dataset.csv", "v4_engine/models/meta_judge.pth")

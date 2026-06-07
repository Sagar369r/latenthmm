import os
import sys
import torch
import joblib
import treelite
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v4_engine.vae_model import VAE
from v4_engine.meta_learner import ResidualMetaLearner

def export_pytorch_to_onnx():
    print("=== Exporting PyTorch Models to ONNX ===")
    os.makedirs("v4_engine/bin", exist_ok=True)
    
    # 1. Export VAE
    vae = VAE(input_dim=20, latent_dim=4)
    vae.load_state_dict(torch.load("v4_engine/models/vae_weights.pth", weights_only=True))
    vae.eval()
    
    dummy_input_vae = torch.randn(1, 20)
    torch.onnx.export(
        vae, dummy_input_vae, "v4_engine/bin/vae.onnx", 
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input'], output_names=['recon', 'mu', 'logvar'],
        dynamic_axes={'input': {0: 'batch_size'}, 'mu': {0: 'batch_size'}}
    )
    print("✓ VAE compiled to v4_engine/bin/vae.onnx")
    
    # 2. Export Meta-Learner
    meta = ResidualMetaLearner(input_dim=17)
    meta.load_state_dict(torch.load("v4_engine/models/meta_judge.pth", weights_only=True))
    meta.eval()
    
    dummy_input_meta = torch.randn(1, 17)
    torch.onnx.export(
        meta, dummy_input_meta, "v4_engine/bin/meta_judge.onnx",
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("✓ Meta-Learner compiled to v4_engine/bin/meta_judge.onnx")

def export_trees_to_treelite():
    print("=== Exporting Expert Trees to Treelite C++ ===")
    
    # Treelite requires gcc or clang to compile the .so
    # We will assume 'gcc' is available on the linux system.
    
    # 1. MEAN_REV (XGBoost) -> Converted to ONNX for stability
    xgb_path = "v4_engine/models/expert_MEAN_REV.pkl"
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        initial_type = [('float_input', FloatTensorType([None, 24]))]
        onx = convert_xgboost(xgb_model, initial_types=initial_type)
        with open("v4_engine/bin/expert_MEAN_REV.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        print("✓ XGBoost compiled to v4_engine/bin/expert_MEAN_REV.onnx")
    
    # 2. TREND (Random Forest)
    rf_path = "v4_engine/models/expert_TREND.pkl"
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        try:
            treelite_model = treelite.sklearn.import_model(rf_model)
            treelite_model.export_lib(toolchain='gcc', libpath='v4_engine/bin/expert_TREND.so', verbose=False)
            print("✓ Random Forest compiled to v4_engine/bin/expert_TREND.so")
        except Exception as e:
            print(f"Treelite sklearn error: {e}")
            print("Falling back to ONNX for Random Forest...")
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            initial_type = [('float_input', FloatTensorType([None, 24]))]
            onx = convert_sklearn(rf_model, initial_types=initial_type)
            with open("v4_engine/bin/expert_TREND.onnx", "wb") as f:
                f.write(onx.SerializeToString())
            print("✓ Random Forest compiled to v4_engine/bin/expert_TREND.onnx (ONNX Fallback)")
            
    # 3. COMPRESSION (Isolation Forest)
    # Treelite doesn't natively support Isolation Forest well. We will use ONNX natively.
    iso_path = "v4_engine/models/expert_COMPRESSION.pkl"
    if os.path.exists(iso_path):
        iso_model = joblib.load(iso_path)
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        try:
            initial_type = [('float_input', FloatTensorType([None, 24]))]
            onx = convert_sklearn(iso_model, initial_types=initial_type, target_opset={'': 14, 'ai.onnx.ml': 3})
            with open("v4_engine/bin/expert_COMPRESSION.onnx", "wb") as f:
                f.write(onx.SerializeToString())
            print("✓ Isolation Forest compiled to v4_engine/bin/expert_COMPRESSION.onnx")
        except Exception as e:
            print(f"Skipping Isolation Forest ONNX conversion due to compat error: {e}")

if __name__ == "__main__":
    export_pytorch_to_onnx()
    export_trees_to_treelite()

import os
import sys
import numpy as np
import polars as pl
import joblib

# ONNX Runtime for Deep Learning
import onnxruntime as ort

# Try Treelite for tree execution, fallback to Python if compiled binary missing
try:
    import treelite_runtime
    TREELITE_AVAILABLE = True
except ImportError:
    TREELITE_AVAILABLE = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v4_engine.feature_expansion import extract_z_score_tensor
from v4_engine.data_transformer import transform_to_dimensionless
from v4_engine.gmm_hmm import LatentRouter

class LiveExecutionEngine:
    def __init__(self):
        print("Initializing Universal Live Engine...")
        
        # 1. Load ONNX VAE
        self.vae_session = ort.InferenceSession("v4_engine/bin/vae.onnx")
        
        # 2. Load HMM Router (Python is fast enough for 4D vector routing)
        self.router = LatentRouter.load("v4_engine/models/gmm_router.pkl")
        
        # 3. Load ONNX Meta-Learner
        self.meta_session = ort.InferenceSession("v4_engine/bin/meta_judge.onnx")
        
        # 4. Load Expert Models (Try Treelite/ONNX, fallback to PKL for simplicity in this bridge)
        # In a strict production setup, we would strictly load .so files using treelite_runtime.Predictor
        self.experts = {}
        for regime in ["TREND", "MEAN_REV", "COMPRESSION"]:
            model_path = f"v4_engine/models/expert_{regime}.pkl"
            if os.path.exists(model_path):
                self.experts[regime] = joblib.load(model_path)
                
        print("✓ Live Engine Ready.")

    def process_live_tick(self, recent_data: pl.DataFrame):
        """
        recent_data: A DataFrame containing at least 90 bars of historical OHLCV data.
        """
        # 0. Calculate ATR natively for Dynamic Stops
        df_atr = recent_data.with_columns([
            (pl.col("high") - pl.col("low")).alias("tr1"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr2"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr3")
        ])
        df_atr = df_atr.with_columns(pl.max_horizontal(["tr1", "tr2", "tr3"]).alias("tr"))
        atr = float(df_atr["tr"].tail(14).mean())

        # 1. Dimensionless Transform (Lazy Polars)
        df = transform_to_dimensionless(recent_data, window=90)
        df_feat, z_cols = extract_z_score_tensor(df, window=90)
        
        # We only care about the final row (the live tick)
        live_row = df_feat.tail(1)
        z_tensor = live_row.select(z_cols).to_numpy().astype(np.float32)
        
        # 2. ONNX VAE Inference -> 4 Latent Features
        onnx_inputs = {self.vae_session.get_inputs()[0].name: z_tensor}
        _, mu, _ = self.vae_session.run(None, onnx_inputs)
        latent_features = mu.astype(np.float32)
        
        # 3. HMM Regime Routing
        probas = self.router.predict_causal_proba(latent_features)[0]
        active_regime = max(probas.items(), key=lambda x: x[1])[0]
        print(f"Current Regime: {active_regime} | ATR: {atr:.5f}")
        
        # 4. Expert Model Inference
        expert_input = np.concatenate([latent_features, z_tensor], axis=1)
        expert_model = self.experts.get(active_regime)
        if expert_model is None:
            return {"action": "HOLD", "reason": "No expert for regime"}
            
        if active_regime == "COMPRESSION":
            raw_pred = -expert_model.decision_function(expert_input)[0]
        else:
            raw_pred = expert_model.predict_proba(expert_input)[0, 1]
            
        print(f"Expert Raw Confidence: {raw_pred:.4f}")
        
        # Determine Direction and Dynamic Stops
        direction = "BUY" if raw_pred >= 0.5 else "SELL"
        sl_distance = atr * 1.0
        tp_distance = atr * 2.0
        
        # 5. Meta-Learner Data Prep
        bin_idx = int(raw_pred * 10)
        if bin_idx == 10: bin_idx = 9
        
        conf_ohe = np.zeros(10, dtype=np.float32)
        conf_ohe[bin_idx] = 1.0
        
        regime_ohe = np.zeros(3, dtype=np.float32)
        if active_regime == "TREND": regime_ohe[0] = 1.0
        elif active_regime == "MEAN_REV": regime_ohe[1] = 1.0
        elif active_regime == "COMPRESSION": regime_ohe[2] = 1.0
        
        meta_input = np.concatenate([latent_features[0], regime_ohe, conf_ohe]).astype(np.float32)
        meta_input = meta_input.reshape(1, 17)
        
        # 6. ONNX Meta-Learner Inference -> Delta
        meta_onnx_inputs = {self.meta_session.get_inputs()[0].name: meta_input}
        predicted_delta = self.meta_session.run(None, meta_onnx_inputs)[0][0, 0]
        
        # 7. The Veto Firewall
        p_final = raw_pred + predicted_delta
        print(f"Meta-Learner Predicted Delta: {predicted_delta:.4f}")
        print(f"P_final: {p_final:.4f}")
        
        if p_final >= 0.65:
            kelly_f = p_final - ((1.0 - p_final) / 2.0)
            print(f"🔥 Veto Passed! Kelly Allocation: {kelly_f * 100:.2f}% | Action: {direction}")
            return {
                "action": "EXECUTE", 
                "direction": direction,
                "p_final": p_final, 
                "kelly": kelly_f,
                "sl_distance": sl_distance,
                "tp_distance": tp_distance
            }
        else:
            print("🛡️ VETO. Trade Killed.")
            return {"action": "VETO", "p_final": p_final}

if __name__ == "__main__":
    engine = LiveExecutionEngine()
    
    # Simulate a live tick array (100 rows to satisfy 90-period warmup)
    dummy_data = pl.DataFrame({
        "close": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 101,
        "low": np.random.randn(100).cumsum() + 99,
        "open": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(100, 1000, 100)
    })
    
    engine.process_live_tick(dummy_data)

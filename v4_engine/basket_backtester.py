import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import onnxruntime as ort

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v4_engine.feature_expansion import extract_z_score_tensor
from v4_engine.data_transformer import transform_to_dimensionless
from v4_engine.gmm_hmm import LatentRouter
from v4_engine.triple_barrier import apply_triple_barrier

def calculate_exact_returns(df: pd.DataFrame, executed_indices: list, tp_mult=2.0, sl_mult=1.0, time_limit=24, slippage=0.0005) -> list:
    returns = []
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    atr_arr = df["atr"].values

    for idx in executed_indices:
        entry_price = close_arr[idx]
        atr = atr_arr[idx]
        
        upper_barrier = entry_price + (tp_mult * atr)
        lower_barrier = entry_price - (sl_mult * atr)
        
        trade_return = 0.0
        exited = False
        
        for i in range(1, time_limit + 1):
            if idx + i >= len(close_arr):
                break
                
            if high_arr[idx + i] >= upper_barrier:
                trade_return = ((upper_barrier - entry_price) / entry_price) - slippage
                exited = True
                break
            elif low_arr[idx + i] <= lower_barrier:
                trade_return = ((lower_barrier - entry_price) / entry_price) - slippage
                exited = True
                break
                
        if not exited:
            exit_price = close_arr[min(idx + time_limit, len(close_arr) - 1)]
            trade_return = ((exit_price - entry_price) / entry_price) - slippage
            
        returns.append(trade_return)
    return returns

def run_basket():
    print("=== Phase 8: Universal Basket Backtest ===")
    
    csv_files = [
        "data/btcusd_1h.csv",
        "data/eurgbp-h1-bid-2020-01-01-2024-12-31.csv",
        "data/xauusd-h1-bid-2020-01-01-2024-12-31.csv",
        "data/audcad_daily.csv",
        "data/chfjpy_daily.csv",
        "data/eurnzd_daily.csv"
    ]
    
    print("Loading compiled ONNX models...")
    vae_sess = ort.InferenceSession("v4_engine/bin/vae.onnx")
    meta_sess = ort.InferenceSession("v4_engine/bin/meta_judge.onnx")
    mean_rev_sess = ort.InferenceSession("v4_engine/bin/expert_MEAN_REV.onnx")
    trend_sess = ort.InferenceSession("v4_engine/bin/expert_TREND.onnx")
    router = LatentRouter.load("v4_engine/models/gmm_router.pkl")
    
    all_executed_trades = []
    
    threshold = 0.55
    win_reward = 0.02
    loss_risk = -0.01
    slippage = 0.0005
    
    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            print(f"Skipping {csv_path} - Not found")
            continue
            
        print(f"\nProcessing {csv_path}...")
        df = pl.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        
        # We need the 'time' column to sort chronologically across assets
        if "time" not in df.columns and "timestamp" in df.columns:
            df = df.rename({"timestamp": "time"})
            
        # Transform & Label
        df = transform_to_dimensionless(df, window=90)
        df = apply_triple_barrier(df, tp_mult=2.0, sl_mult=1.0, time_limit=24)
        df, z_cols = extract_z_score_tensor(df, window=90)
        
        df = df.slice(90) # Skip warmup
        
        # 1. VAE Inference
        features_np = df.select(z_cols).to_numpy().astype(np.float32)
        vae_inputs = {vae_sess.get_inputs()[0].name: features_np}
        recon_x, latent_np, logvar = vae_sess.run(None, vae_inputs)
        
        latent_cols = [f"latent_{i}" for i in range(4)]
        df = df.with_columns([pl.Series(name, latent_np[:, i]) for i, name in enumerate(latent_cols)])
        
        # 2. Router Inference
        probas = router.predict_causal_proba(latent_np)
        regime_labels = [max(p.items(), key=lambda x: x[1])[0] for p in probas]
        df = df.with_columns(pl.Series("regime", regime_labels))
        
        expert_features = latent_cols + z_cols
        pandas_df = df.to_pandas()
        pandas_df["expert_pred"] = 0.0
        
        # 3. Expert Inference
        for regime in ["TREND", "MEAN_REV"]:
            mask = pandas_df["regime"] == regime
            if mask.sum() == 0: continue
            
            sub_X = pandas_df.loc[mask, expert_features].to_numpy().astype(np.float32)
            
            if regime == "MEAN_REV":
                sess = mean_rev_sess
                # XGBoost ONNX outputs list of dicts for probas
                preds = sess.run(None, {sess.get_inputs()[0].name: sub_X})[1]
                pandas_df.loc[mask, "expert_pred"] = [p[1] for p in preds]
            else:
                sess = trend_sess
                # Random Forest ONNX also outputs list of dicts
                preds = sess.run(None, {sess.get_inputs()[0].name: sub_X})[1]
                pandas_df.loc[mask, "expert_pred"] = [p[1] for p in preds]
                
        # 4. Meta-Learner Inference
        meta_features = [
            "expert_pred",
            "volm_roc_5", "volm_surge_5_20", "volm_surge_10_50", "volm_ofi_10", "volm_ofi_20",
            "trend_sma_10", "trend_sma_20", "trend_sma_50", "trend_sma_100", "trend_sma_200",
            "mom_roc_5", "mom_roc_10", "mom_roc_20", "mom_roc_50", "mom_macd",
            "vol_atr_ratio"
        ]
        X_meta = pandas_df[meta_features].to_numpy().astype(np.float32)
        meta_inputs = {meta_sess.get_inputs()[0].name: X_meta}
        pred_delta = meta_sess.run(None, meta_inputs)[0].flatten()
        
        pandas_df["pred_delta"] = pred_delta
        pandas_df["p_final"] = pandas_df["expert_pred"] + pandas_df["pred_delta"]
        pandas_df["asset"] = os.path.basename(csv_path).split('_')[0].split('-')[0]
        
        # Apply Firewall
        executed = pandas_df[pandas_df["p_final"] >= threshold].copy()
        
        executed["trade_return"] = calculate_exact_returns(
            pandas_df, 
            executed.index.tolist(), 
            tp_mult=2.0, 
            sl_mult=1.0, 
            time_limit=24, 
            slippage=slippage
        )
        
        all_executed_trades.append(executed)
        print(f"   => {len(executed)} trades executed on {pandas_df['asset'].iloc[0]}.")
        
    print("\nAggregating Global Basket...")
    combined_df = pd.concat(all_executed_trades).copy()
    
    if "time" in combined_df.columns:
        combined_df["time"] = pd.to_datetime(combined_df["time"])
        combined_df = combined_df.sort_values("time")
        
    combined_df["equity"] = (1 + combined_df["trade_return"]).cumprod() * 100000.0
    
    final_equity = combined_df["equity"].iloc[-1]
    total_return_pct = ((final_equity / 100000.0) - 1.0) * 100.0
    
    wins = len(combined_df[combined_df["target"] == 1])
    losses = len(combined_df) - wins
    win_rate = (wins / len(combined_df)) * 100.0
    
    combined_df["peak_equity"] = combined_df["equity"].cummax()
    combined_df["drawdown"] = (combined_df["equity"] - combined_df["peak_equity"]) / combined_df["peak_equity"]
    max_drawdown = combined_df["drawdown"].min() * 100.0
    
    gross_profit = combined_df[combined_df["trade_return"] > 0]["trade_return"].sum()
    gross_loss = abs(combined_df[combined_df["trade_return"] < 0]["trade_return"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print("\n=============================================")
    print("  GLOBAL BASKET V4 ENGINE: TEAR SHEET  ")
    print("=============================================")
    print(f"Total Trades:        {len(combined_df)}")
    print(f"Starting Capital:    $100,000.00")
    print(f"Final Capital:       ${final_equity:,.2f}")
    print(f"Total Return:        {total_return_pct:+.2f}%")
    print(f"Max Drawdown:        {max_drawdown:+.2f}%")
    print(f"Win Rate:            {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Profit Factor:       {profit_factor:.2f}x")
    print("=============================================\n")

if __name__ == "__main__":
    run_basket()

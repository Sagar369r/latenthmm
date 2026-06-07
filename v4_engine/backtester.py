import os
import sys
import numpy as np
import pandas as pd
import onnxruntime as ort

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v4_engine.meta_data_prep import prepare_meta_dataset

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

def run_vectorized_backtest(csv_path: str, threshold=0.65, win_reward=0.02, loss_risk=-0.01, slippage=0.0005):
    print(f"=== Phase 7: Vectorized Backtester ===")
    print(f"Loading {csv_path}...")
    
    # 1. Load Data and Extract Meta-Learner 17-D Tensor
    df = pd.pd.read_csv(csv_path) if hasattr(pd, "pd") else pd.read_csv(csv_path) # Safe fallback
    X_meta, _ = prepare_meta_dataset(csv_path)
    X_meta = X_meta.astype(np.float32)
    
    # 2. Run ONNX Meta-Learner Inference
    print("Running ONNX Meta-Learner to generate Delta predictions for 34,000+ bars...")
    session = ort.InferenceSession("v4_engine/bin/meta_judge.onnx")
    onnx_inputs = {session.get_inputs()[0].name: X_meta}
    pred_delta = session.run(None, onnx_inputs)[0].flatten()
    
    # 3. Calculate Final Probability and Apply Veto Firewall
    df["pred_delta"] = pred_delta
    df["p_final"] = df["expert_pred"] + df["pred_delta"]
    
    print(f"DEBUG: Max Expert Pred: {df['expert_pred'].max():.4f}, Mean: {df['expert_pred'].mean():.4f}")
    print(f"DEBUG: Max Delta Pred: {df['pred_delta'].max():.4f}, Mean: {df['pred_delta'].mean():.4f}")
    print(f"DEBUG: Max P_Final: {df['p_final'].max():.4f}, Mean: {df['p_final'].mean():.4f}")
    
    # The Veto Logic
    executed_trades = df[df["p_final"] >= threshold].copy()
    vetoed_count = len(df) - len(executed_trades)
    
    print(f"Total Rows: {len(df)}")
    print(f"Trades Vetoed by Firewall: {vetoed_count}")
    print(f"Trades Executed: {len(executed_trades)}")
    
    if len(executed_trades) == 0:
        print("No trades executed. The Firewall blocked everything.")
        return
        
    # 4. Simulate Equity Curve
    # target == 1 means we hit TP (+8.0 ATR)
    # target == 0 means we hit SL (-4.0 ATR) or Time Exit (usually small loss)
    # We use exact intrabar PnL from raw price arrays instead of hardcoded illusions
    executed_trades["trade_return"] = calculate_exact_returns(
        df, 
        executed_trades.index.tolist(), 
        tp_mult=2.0, 
        sl_mult=1.0, 
        time_limit=24, 
        slippage=slippage
    )
    
    if "time" in executed_trades.columns:
        executed_trades["time"] = pd.to_datetime(executed_trades["time"])
        executed_trades = executed_trades.sort_values("time")
    
    # Cumulative log returns for equity compounding
    executed_trades["equity"] = (1 + executed_trades["trade_return"]).cumprod() * 100000.0 # $100k starting capital
    
    # 5. Calculate Tear Sheet Metrics
    final_equity = executed_trades["equity"].iloc[-1]
    total_return_pct = ((final_equity / 100000.0) - 1.0) * 100.0
    
    wins = len(executed_trades[executed_trades["target"] == 1])
    losses = len(executed_trades) - wins
    win_rate = (wins / len(executed_trades)) * 100.0
    
    # Max Drawdown
    executed_trades["peak_equity"] = executed_trades["equity"].cummax()
    executed_trades["drawdown"] = (executed_trades["equity"] - executed_trades["peak_equity"]) / executed_trades["peak_equity"]
    max_drawdown = executed_trades["drawdown"].min() * 100.0
    
    # Sharpe Ratio (Assuming roughly 252 trading days per year, wait, trades are irregular so we use per-trade Sharpe)
    # Annualized Sharpe roughly = (Mean Return / Std Dev) * sqrt(Trades per year)
    trades_per_year = len(executed_trades) / 5.0 # We have ~5 years of data
    mean_ret = executed_trades["trade_return"].mean()
    std_ret = executed_trades["trade_return"].std()
    
    sharpe_ratio = 0.0
    if std_ret > 0:
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(trades_per_year)
        
    # Profit Factor
    gross_profit = executed_trades[executed_trades["trade_return"] > 0]["trade_return"].sum()
    gross_loss = abs(executed_trades[executed_trades["trade_return"] < 0]["trade_return"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print("\n=============================================")
    print("      V4 ENGINE: QUANTITATIVE TEAR SHEET     ")
    print("=============================================")
    print(f"Starting Capital:    $100,000.00")
    print(f"Final Capital:       ${final_equity:,.2f}")
    print(f"Total Return:        {total_return_pct:+.2f}%")
    print(f"Max Drawdown:        {max_drawdown:+.2f}%")
    print(f"Win Rate:            {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Profit Factor:       {profit_factor:.2f}x")
    print(f"Annualized Sharpe:   {sharpe_ratio:.2f}")
    print("=============================================\n")

if __name__ == "__main__":
    # Threshold set to 0.55. (Breakeven for 1:2 R:R is 33.3%). This allows top-tier trades.
    run_vectorized_backtest(
        "v4_engine/data/master_oof_dataset.csv", 
        threshold=0.55, 
        win_reward=0.02,   # +2.0%
        loss_risk=-0.01,   # -1.0%
        slippage=0.0005    # -0.05% slippage cost
    )

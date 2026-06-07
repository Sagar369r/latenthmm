import numpy as np

def simulate_prop_firm(
    n_simulations=10000,
    win_rate=0.47,
    win_size=2.0,   # Multiplier (+2.0 ATR)
    loss_size=1.0,  # Multiplier (-1.0 ATR)
    risk_per_trade=0.01,
    max_trades_per_phase=500
):
    """
    Simulates the Prop Firm 3-Stage Challenge with Path Dependency.
    """
    start_capital = 2500.0
    
    # Prop Firm Rules
    p1_target = start_capital * 1.10
    p1_max_loss = start_capital * 0.90
    
    p2_target = start_capital * 1.05
    p2_max_loss = start_capital * 0.90
    
    funded_target = start_capital + 150.0
    funded_max_loss = start_capital * 0.90
    
    results = {
        "passed_p1": 0,
        "passed_p2": 0,
        "secured_payout": 0
    }
    
    for _ in range(n_simulations):
        # ==========================================
        # PHASE 1
        # ==========================================
        capital = start_capital
        passed_1 = False
        for _ in range(max_trades_per_phase):
            if np.random.rand() < win_rate:
                capital += (capital * risk_per_trade) * win_size
            else:
                capital -= (capital * risk_per_trade) * loss_size
                
            if capital >= p1_target:
                passed_1 = True
                break
            if capital <= p1_max_loss:
                break
                
        if not passed_1:
            continue
            
        results["passed_p1"] += 1
        
        # ==========================================
        # PHASE 2 (Account Resets!)
        # ==========================================
        capital = start_capital
        passed_2 = False
        for _ in range(max_trades_per_phase):
            if np.random.rand() < win_rate:
                capital += (capital * risk_per_trade) * win_size
            else:
                capital -= (capital * risk_per_trade) * loss_size
                
            if capital >= p2_target:
                passed_2 = True
                break
            if capital <= p2_max_loss:
                break
                
        if not passed_2:
            continue
            
        results["passed_p2"] += 1
        
        # ==========================================
        # FUNDED STAGE (Account Resets!)
        # ==========================================
        capital = start_capital
        secured_payout = False
        for _ in range(max_trades_per_phase):
            if np.random.rand() < win_rate:
                capital += (capital * risk_per_trade) * win_size
            else:
                capital -= (capital * risk_per_trade) * loss_size
                
            if capital >= funded_target:
                secured_payout = True
                break
            if capital <= funded_max_loss:
                break
                
        if secured_payout:
            results["secured_payout"] += 1
            
    print(f"--- Prop Firm Monte Carlo ({n_simulations} iterations) ---")
    print(f"Risk Per Trade:       {risk_per_trade*100:.1f}%")
    print(f"Phase 1 Pass Rate:    {results['passed_p1'] / n_simulations * 100:.1f}%")
    print(f"Phase 2 Pass Rate:    {results['passed_p2'] / n_simulations * 100:.1f}%")
    print(f"Funded Payout Rate:   {results['secured_payout'] / n_simulations * 100:.1f}%")
    print("")

if __name__ == "__main__":
    print("Simulating Gauntlet with 1.0% Risk...")
    simulate_prop_firm(risk_per_trade=0.01)
    
    print("Simulating Gauntlet with 2.0% Risk...")
    simulate_prop_firm(risk_per_trade=0.02)
    
    print("Simulating Gauntlet with 3.0% Risk (High Risk)...")
    simulate_prop_firm(risk_per_trade=0.03)

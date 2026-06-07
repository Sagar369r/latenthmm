#!/usr/bin/env python3
"""
Ensemble Portfolio Pipeline — Latent Diffusion-HMM v3.0

Runs the Daily Mean-Reversion Tearsheet across all *_daily.csv files in data/,
then aggregates the individual equity curves into a Portfolio-Level Sharpe.

Usage:
    uv run python ensemble_pipeline.py
"""

import os, sys, json, glob, time
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# 🎛️ ENSEMBLE CONTROL DIALS
# ============================================================
DATA_DIR         = "data"
STRATEGY_MODE    = "MEAN_REVERSION_EXHAUSTION"
TIMEFRAME        = "1H"
STOP_LOSS_ATR    = 4.0
TAKE_PROFIT_ATR  = 8.0
TIME_EXIT_BARS   = 12
INITIAL_EQUITY   = 100_000
RESULTS_DIR      = "results/ensemble_results"

# 🔥 SHARPE FIREWALL — Only deploy capital into pairs with proven alpha
SHARPE_FIREWALL  = 0.40   # Minimum individual Sharpe to enter portfolio
HALF_KELLY_RISK  = 0.02   # 2% account risk per trade (Half-Kelly)
USE_CACHED       = False  # Recompute every time and ask to save

# ============================================================
# Helper: pretty separator
# ============================================================
def _sep(title: str = "", width: int = 74) -> None:
    if title:
        pad = max(0, width - len(title) - 2)
        left = pad // 2
        right = pad - left
        print(f"{'─'*left} {title} {'─'*right}")
    else:
        print("─" * width)


def process_asset(csv_path, tmp_dir):
    import sys, os, time, numpy as np
    engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v3_engine"))
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import forex_tearsheet

    ticker = os.path.basename(csv_path).replace("_daily.csv", "").replace("_1h.csv", "").upper()
    result_path = os.path.join(tmp_dir, f"{ticker.lower()}_results.json")
    
    t0 = time.time()
    
    params = {
        "STOP_LOSS_ATR": STOP_LOSS_ATR,
        "TAKE_PROFIT_ATR": TAKE_PROFIT_ATR,
        "TIME_EXIT_BARS": TIME_EXIT_BARS,
        "HMM_WF_OOS_BARS": "240", # 2 weeks for 1H
        "VETO_THRESHOLD": "0.70"
    }
    
    try:
        os.environ["FOREX_RESULTS_PATH"] = os.path.abspath(result_path)
        os.environ["TIMEFRAME"] = TIMEFRAME
        os.environ["HMM_WF_TRAIN_BARS"] = "1512"
        
        import io
        from contextlib import redirect_stdout
        with io.StringIO() as buf, redirect_stdout(buf):
            data = forex_tearsheet.run_tearsheet_dynamic(csv_path, params)
            
        elapsed = time.time() - t0
        
        scenarios = data.get("pip_scenarios", {})
        gross_key = next((k for k in scenarios.keys() if "gross" in k), None)
        gross = scenarios.get(gross_key, {}) if gross_key else {}
        n_trades   = gross.get("n_trades", 0)
        win_rate   = gross.get("win_rate_pct", 0)
        sharpe     = gross.get("sharpe_ratio", 0)
        pf         = gross.get("profit_factor", 0)
        total_ret  = gross.get("total_ret_pct", 0)
        returns    = np.array(gross.get("returns_array", []))
        
        result_dict = {
            "returns": returns, "n_trades": n_trades,
            "win_rate": win_rate, "sharpe": sharpe,
            "pf": pf, "total_ret": total_ret,
            "trades": data.get("trade_log", []),
        }
        summary_dict = {
            "ticker": ticker, "trades": n_trades,
            "win_rate": win_rate, "sharpe": sharpe,
            "pf": pf, "return": total_ret,
        }
        return ticker, True, elapsed, (result_dict, summary_dict), None
    except Exception as e:
        return ticker, False, elapsed, None, str(e)

def main():
    import glob, time, os
    from datetime import datetime
    
    wall_t0 = time.time()
    print("=" * 74)
    print("  LATENT DIFFUSION-HMM v3.0 — ENSEMBLE PORTFOLIO ENGINE")
    print(f"  Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = os.path.join(RESULTS_DIR, f".tmp_{run_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"  Strategy: {STRATEGY_MODE}  |  TF: {TIMEFRAME}")
    print(f"  Risk: SL={STOP_LOSS_ATR}×ATR  TP={TAKE_PROFIT_ATR}×ATR  TimeExit={TIME_EXIT_BARS}b")
    print("=" * 74)
    print()

    # ── Phase 1: Discover assets ─────────────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_1h.csv")))
    if not csv_files:
        print("✗ No *_1h.csv files found in data/. Run download_basket.sh first.")
        return

    _sep(f"PHASE 1 — DISCOVERED {len(csv_files)} ASSETS")
    for f in csv_files:
        ticker = os.path.basename(f).replace("_daily.csv", "").upper()
        size_kb = os.path.getsize(f) / 1024
        print(f"  {ticker:>10}  →  {f}  ({size_kb:.0f} KB)")
    print()

    # ── Phase 2: Run individual tearsheets ───────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    asset_results = {}
    asset_summaries = []

    _sep(f"PHASE 2 — RUNNING {len(csv_files)} WALK-FORWARD BACKTESTS")
    print()
    import concurrent.futures

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
        for csv_path in csv_files:
            futures.append(executor.submit(process_asset, csv_path, tmp_dir))
            
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            ticker, success, elapsed, payload, err = future.result()
            
            if success:
                res_d, sum_d = payload
                asset_results[ticker] = res_d
                asset_summaries.append(sum_d)
                status = "✓" if sum_d['sharpe'] > 0 else "─"
                print(f"  [{completed:>2}/{len(csv_files)}] {ticker:<8} {status}  trades={sum_d['trades']:>3}  WR={sum_d['win_rate']:>5.1f}%  PF={sum_d['pf']:>5.2f}  SR={sum_d['sharpe']:>+6.2f}  Ret={sum_d['return']:>+6.1f}%  ({elapsed:.1f}s)")
            else:
                print(f"  [{completed:>2}/{len(csv_files)}] {ticker:<8} ✗ FAILED ({elapsed:.1f}s) {err}")

    print()

    if not asset_results:
        print("✗ No assets produced valid results. Cannot build portfolio.")
        return

    # ── Phase 2.5: Sharpe Firewall ───────────────────────────────────────
    _sep(f"PHASE 2.5 — SHARPE FIREWALL (min SR > {SHARPE_FIREWALL})")
    print()
    filtered_results = {}
    rejected = []
    for ticker, data in asset_results.items():
        if data["sharpe"] >= SHARPE_FIREWALL:
            filtered_results[ticker] = data
            print(f"  ✓ {ticker:<10} SR={data['sharpe']:>+.2f}  PF={data['pf']:>.2f}  → ADMITTED")
        else:
            rejected.append(ticker)
            print(f"  ✗ {ticker:<10} SR={data['sharpe']:>+.2f}  PF={data['pf']:>.2f}  → REJECTED")

    print()
    print(f"  Admitted: {len(filtered_results)} / {len(asset_results)}")
    print(f"  Rejected: {len(rejected)} ({', '.join(rejected)})")
    print()

    if not filtered_results:
        print("✗ No assets passed the Sharpe Firewall. Loosen the threshold.")
        return

    asset_results = filtered_results

    # ── Phase 3: Portfolio Aggregation ────────────────────────────────────
    _sep(f"PHASE 3 — CONCENTRATED PORTFOLIO ({len(asset_results)} assets)")

    # Align returns by creating a DataFrame
    # Each asset has a different-length returns array; pad with 0
    max_len = max(len(v["returns"]) for v in asset_results.values())
    returns_df = pd.DataFrame()
    for ticker, data in asset_results.items():
        r = data["returns"]
        # Pad shorter arrays with 0
        if len(r) < max_len:
            r = np.concatenate([r, np.zeros(max_len - len(r))])
        returns_df[ticker] = r

    # Equal-weight portfolio: 1/N allocation
    N = len(asset_results)
    returns_df["PORTFOLIO"] = returns_df.sum(axis=1) / N

    port_returns = returns_df["PORTFOLIO"].values
    port_cum     = np.cumprod(1 + port_returns)

    # Portfolio statistics
    total_trades = sum(v["n_trades"] for v in asset_results.values())
    total_wins   = sum(
        int(v["n_trades"] * v["win_rate"] / 100)
        for v in asset_results.values()
    )
    port_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    ann_factor = 252
    port_mean   = np.mean(port_returns) * ann_factor
    port_std    = np.std(port_returns, ddof=1) * np.sqrt(ann_factor)
    port_sharpe = port_mean / port_std if port_std > 0 else 0.0

    port_cum_max = np.maximum.accumulate(port_cum)
    port_dd      = (port_cum / port_cum_max - 1) * 100
    port_mdd     = np.min(port_dd)

    years = len(port_returns) / ann_factor
    port_total_ret = (port_cum[-1] - 1) * 100
    port_cagr      = ((port_cum[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0

    final_equity = INITIAL_EQUITY * port_cum[-1]

    # Half-Kelly position sizing info
    risk_per_trade = HALF_KELLY_RISK
    max_simultaneous = N
    risk_per_trade_scaled = risk_per_trade / max_simultaneous

    print(f"  Assets in portfolio : {N}")
    print(f"  Total portfolio trades: {total_trades}")
    print(f"  Aggregate Win Rate : {port_win_rate:.1f}%")
    print(f"  Half-Kelly Risk    : {risk_per_trade*100:.1f}% per trade → {risk_per_trade_scaled*100:.2f}% scaled (1/{N})")
    print()

    # ── Phase 4: Portfolio Tear Sheet ────────────────────────────────────
    _sep("CONCENTRATED PORTFOLIO PERFORMANCE (Equal-Weight, Gross)")
    print(f"  {'Metric':<28} {'Value':>12}")
    print(f"  {'─'*28} {'─'*12}")
    print(f"  {'Total Return':<28} {port_total_ret:>+11.1f}%")
    print(f"  {'CAGR':<28} {port_cagr:>+11.1f}%")
    print(f"  {'Max Drawdown':<28} {port_mdd:>+11.1f}%")
    print(f"  {'Calmar Ratio':<28} {abs(port_cagr/port_mdd) if port_mdd != 0 else 0:>+11.2f}")
    print(f"  {'Sharpe Ratio':<28} {port_sharpe:>+11.2f}")
    print(f"  {'Final Equity':<28} ${final_equity:>10,.0f}")
    print()

    # ── Per-Asset Leaderboard ────────────────────────────────────────────
    _sep("PER-ASSET LEADERBOARD")
    print(f"  {'Ticker':<10} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Sharpe':>8} {'Return':>8}")
    print(f"  {'─'*10} {'─'*7} {'─'*8} {'─'*6} {'─'*8} {'─'*8}")
    for s in sorted(asset_summaries, key=lambda x: x["sharpe"], reverse=True):
        marker = "★" if s["sharpe"] > 0.5 else "─"
        print(f"  {marker} {s['ticker']:<8} {s['trades']:>6} {s['win_rate']:>7.1f}% {s['pf']:>5.2f} {s['sharpe']:>+7.2f} {s['return']:>+7.1f}%")
    print()

    # ── Phase 5: Layer 6 Statistical Validation ─────────────────────────
    _sep("PHASE 5 — LAYER 6 PORTFOLIO VALIDATION")

    passes = 0
    total_tests = 5

    # 6.1 Portfolio Sharpe
    sr_pass = port_sharpe > 0.5
    passes += int(sr_pass)
    print(f"  6.1 Portfolio Sharpe Ratio           SR={port_sharpe:>+.2f}  {'PASS ✓' if sr_pass else 'FAIL ✗'}")

    # 6.2 Monte Carlo Permutation on portfolio returns
    n_perms = 500
    actual_sr = port_mean / port_std if port_std > 0 else 0
    perm_count = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(port_returns)
        s_mean = np.mean(shuffled) * ann_factor
        s_std  = np.std(shuffled, ddof=1) * np.sqrt(ann_factor)
        if s_std > 0 and (s_mean / s_std) >= actual_sr:
            perm_count += 1
    mc_p = perm_count / n_perms
    mc_pass = mc_p < 0.05
    passes += int(mc_pass)
    print(f"  6.2 Monte Carlo Permutation          p={mc_p:.4f}  {'PASS ✓' if mc_pass else 'FAIL ✗'}")

    # 6.3 Deflated Sharpe Ratio
    from scipy import stats
    T = len(port_returns)
    dsr_z = (port_sharpe * np.sqrt(T)) / np.sqrt(1 + 0.5 * port_sharpe**2)
    dsr_p = 1 - stats.norm.cdf(dsr_z)
    dsr_pass = dsr_p < 0.05
    passes += int(dsr_pass)
    print(f"  6.3 Deflated Sharpe Ratio            z={dsr_z:>+.2f}  p={dsr_p:.4f}  {'PASS ✓' if dsr_pass else 'FAIL ✗'}")

    # 6.4 Trade count sufficiency (N >= 30)
    n_pass = total_trades >= 30
    passes += int(n_pass)
    print(f"  6.4 Trade Count Sufficiency           N={total_trades}  {'PASS ✓' if n_pass else 'FAIL ✗'}")

    # 6.5 Max Drawdown check (MDD < 20%)
    mdd_pass = abs(port_mdd) < 20
    passes += int(mdd_pass)
    print(f"  6.5 Max Drawdown Constraint          MDD={port_mdd:>+.1f}%  {'PASS ✓' if mdd_pass else 'FAIL ✗'}")

    print()
    print(f"  ══════════  {passes}/{total_tests} LAYER 6 TESTS PASSED  ══════════")
    print()

    # ── Verdict ──────────────────────────────────────────────────────────
    if passes >= 4:
        verdict = "✓ VERDICT: INSTITUTIONAL GRADE — DEPLOY TO PAPER TRADING"
    elif passes >= 2:
        verdict = "~ VERDICT: MARGINAL — NEEDS FURTHER REFINEMENT"
    else:
        verdict = "✗ VERDICT: WASTE — SCRAP THE STRATEGY"

    _sep(verdict)
    print(f"  Portfolio Sharpe {port_sharpe:>+.2f}  |  Total Trades {total_trades}  |  MDD {port_mdd:.1f}%  |  WinRate {port_win_rate:.1f}%")
    _sep()

    # ── Save ensemble results ────────────────────────────────────────────
    ensemble_payload = {
        "run_at": datetime.now().isoformat(),
        "n_assets": N,
        "total_trades": total_trades,
        "portfolio_sharpe": float(port_sharpe),
        "portfolio_total_return_pct": float(port_total_ret),
        "portfolio_cagr_pct": float(port_cagr),
        "portfolio_max_drawdown_pct": float(port_mdd),
        "portfolio_win_rate_pct": float(port_win_rate),
        "final_equity": float(final_equity),
        "layer6_passes": passes,
        "per_asset": asset_summaries,
    }

    ensemble_output = os.path.join(tmp_dir, "ensemble_results.json")
    with open(ensemble_output, "w") as f:
        json.dump(ensemble_payload, f, indent=2)

    try:
        import quantstats as qs
        # Generate pseudo-dates ending today for the portfolio returns
        dates = pd.date_range(end=datetime.now(), periods=len(port_returns), freq='B')
        qs_series = pd.Series(port_returns, index=dates)
        
        qs_path = os.path.join(tmp_dir, "ensemble_tearsheet.html")
        qs.reports.html(qs_series, output=qs_path, title="Ensemble Portfolio Tear Sheet")
    except Exception as e:
        pass

    print(f"  Total wall-clock time: {time.time()-wall_t0:.1f}s")
    print()

    # ── Automatically save the results ────────────────────────────────────
    final_dir = os.path.join(RESULTS_DIR, f"run_{run_id}")
    os.rename(tmp_dir, final_dir)
    print(f"\n  ✅ Results permanently saved to: {final_dir}/")
    print(f"  Run ID: {run_id}")


if __name__ == "__main__":
    main()
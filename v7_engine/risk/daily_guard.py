"""
Daily Guard — $125 absolute USD hard ceiling.
Pre-execution gate: if daily drawdown ≥ limit, halt ALL trading.
Reactive safeguard (complement to the pre-trade MC barrier).
"""
from __future__ import annotations
from v7_engine.config import BACKTEST_INITIAL_EQUITY, MAX_DAILY_DRAWDOWN_USD


class DailyGuard:
    """
    Tracks intraday P&L from the session high.
    Halts on a $125 drawdown (or custom limit).
    Resets at end-of-day to the closing equity.
    """

    def __init__(
        self,
        initial_equity:   float = BACKTEST_INITIAL_EQUITY,
        dd_limit_usd:     float = MAX_DAILY_DRAWDOWN_USD,
        max_total_dd_usd: float = 250.0,
        account_starting_balance: float = BACKTEST_INITIAL_EQUITY,
    ):
        self.initial_equity = initial_equity
        self.account_starting_balance = account_starting_balance
        self.day_start     = initial_equity
        self.dd_limit_usd  = dd_limit_usd
        self.max_total_dd_usd = max_total_dd_usd
        self.halted        = False
        self.halt_reason   = ""
        self._session_high = initial_equity
        self.equity = initial_equity
        self._current_day  = ""
        self.trading_days_count = 0

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, current_equity: float, day: str = "") -> bool:
        """
        Update equity. Returns True if trading is allowed, False if halted.

        Parameters
        ----------
        current_equity : current account equity in USD
        day            : optional label for logging (e.g. "2025-01-15")
        """
        if day and day != self._current_day:
            if self._current_day != "":
                self.reset_eod()
            self._current_day = day
            self.trading_days_count += 1

        if self.halted:
            return False

        self.equity = current_equity

        # Track session high for drawdown calculation
        if current_equity > self._session_high:
            self._session_high = current_equity

        # Profit lock
        from v7_engine.config import DAILY_PROFIT_LOCK_PCT
        profit_lock_amt = self.day_start * DAILY_PROFIT_LOCK_PCT
        if current_equity >= self.day_start + profit_lock_amt:
            self.halted = True
            self.halt_reason = f"Profit lock reached ({DAILY_PROFIT_LOCK_PCT*100}%) on {day}"
            return False

        dd_usd = self.day_start - current_equity
        if dd_usd >= self.dd_limit_usd:
            self.halted     = True
            self.halt_reason = (
                f"HARD STOP: Daily drawdown ${dd_usd:.2f} "
                f">= limit ${self.dd_limit_usd:.2f} on {day}"
            )
            return False
            
        total_dd_usd = self.account_starting_balance - current_equity
        if total_dd_usd >= self.max_total_dd_usd:
            self.halted = True
            self.halt_reason = (
                f"HARD STOP: Total drawdown ${total_dd_usd:.2f} "
                f">= limit ${self.max_total_dd_usd:.2f} on {day}"
            )
            return False

        return True

    def remaining_budget_usd(self) -> float:
        """How much more USD can be lost today before the hard stop fires."""
        if not hasattr(self, 'equity'):
            return self.dd_limit_usd
        dd_so_far = self.day_start - self.equity
        return max(0.0, self.dd_limit_usd - dd_so_far)

    def reset_eod(self) -> None:
        """Call at end of trading day. Resets for next session."""
        self.day_start      = self.equity
        self._session_high  = self.day_start
        self.halted         = False
        self.halt_reason    = ""

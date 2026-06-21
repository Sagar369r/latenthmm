from __future__ import annotations

from v7_engine.risk.daily_guard import DailyGuard


class PropGuardAdapter:
    """Overlays prop-firm drawdown limits on top of the internal DailyGuard."""

    def __init__(
        self,
        internal_guard:       DailyGuard,
        prop_daily_limit_usd: float,
        prop_max_loss_usd:    float,
    ) -> None:
        self.internal_guard       = internal_guard
        self.prop_daily_limit_usd = prop_daily_limit_usd
        self.prop_max_loss_usd    = prop_max_loss_usd
        self.prop_start_equity    = internal_guard.initial_equity

    def update(self, new_equity: float, day: str) -> bool:
        # Short-circuit: never allow when already halted
        if self.internal_guard.halted:
            return False

        internal_allowed = self.internal_guard.update(new_equity, day)
        if not internal_allowed:
            return False   # internal guard already handled halt

        prop_daily_loss = self.internal_guard.day_start - new_equity
        prop_total_loss = self.prop_start_equity - new_equity

        if prop_daily_loss >= self.prop_daily_limit_usd:
            self.internal_guard.halted      = True
            self.internal_guard.halt_reason = (
                f"PROP FIRM HARD STOP: Daily loss ${prop_daily_loss:.2f} "
                f">= limit ${self.prop_daily_limit_usd:.2f}"
            )
            return False

        if prop_total_loss >= self.prop_max_loss_usd:
            self.internal_guard.halted      = True
            self.internal_guard.halt_reason = (
                f"PROP FIRM ACCOUNT BLOWN: Total loss ${prop_total_loss:.2f} "
                f">= limit ${self.prop_max_loss_usd:.2f}"
            )
            return False

        return True

    def remaining_budget_usd(self) -> float:
        internal_budget  = self.internal_guard.remaining_budget_usd()
        prop_daily_loss  = max(0.0, self.internal_guard.day_start - self.internal_guard.equity)
        prop_budget      = max(0.0, self.prop_daily_limit_usd - prop_daily_loss)
        return min(internal_budget, prop_budget)

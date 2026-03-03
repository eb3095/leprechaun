"""Risk management for Leprechaun trading bot.

Implements position sizing, loss limits, and order validation to protect
capital and enforce disciplined risk management.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

from src.utils.config import TradingConfig


@dataclass
class HaltDecision:
    """Result of trading halt evaluation."""

    should_halt: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"should_halt": self.should_halt, "reason": self.reason}


@dataclass
class ValidationResult:
    """Result of order validation."""

    is_valid: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"is_valid": self.is_valid, "reason": self.reason}


class RiskManager:
    """Risk management for trading operations.

    Enforces position sizing rules, loss limits, and validates orders
    against risk parameters.
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        """Initialize risk manager with configuration.

        Args:
            config: Trading configuration. Uses defaults if not provided.
        """
        self.config = config or TradingConfig()
        self.position_risk_pct = self.config.position_risk_percent / 100
        self.daily_loss_limit = self.config.daily_loss_limit_percent / 100
        self.weekly_loss_limit = self.config.weekly_loss_limit_percent / 100
        self.stop_loss_pct = self.config.stop_loss_percent / 100
        self.profit_target_pct = self.config.profit_target_percent / 100

    def calculate_position_size(
        self,
        account_value: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
    ) -> int:
        """Calculate position size based on risk parameters.

        Uses the 1% rule: risk no more than position_risk_pct of account
        on any single trade.

        Args:
            account_value: Total account value in dollars.
            entry_price: Expected entry price per share.
            stop_loss_price: Stop loss price. If None, uses default stop loss %.

        Returns:
            Number of shares to buy (integer, rounded down).
        """
        if account_value <= 0 or entry_price <= 0:
            return 0

        if stop_loss_price is None:
            stop_loss_price = self.calculate_stop_loss_price(entry_price)

        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            return 0

        max_risk_amount = account_value * self.position_risk_pct
        shares = max_risk_amount / risk_per_share

        max_position_value = account_value * 0.10
        max_shares_by_value = max_position_value / entry_price

        return int(min(shares, max_shares_by_value))

    def check_daily_loss_limit(self, daily_pnl: float, account_value: float) -> bool:
        """Check if daily loss limit has been exceeded.

        Args:
            daily_pnl: Today's realized + unrealized P&L (negative for loss).
            account_value: Total account value at start of day.

        Returns:
            True if daily loss limit exceeded, False otherwise.
        """
        if account_value <= 0:
            return True

        daily_loss_pct = -daily_pnl / account_value
        return daily_loss_pct >= self.daily_loss_limit

    def check_weekly_loss_limit(self, weekly_pnl: float, account_value: float) -> bool:
        """Check if weekly loss limit has been exceeded.

        Args:
            weekly_pnl: This week's realized + unrealized P&L (negative for loss).
            account_value: Total account value at start of week.

        Returns:
            True if weekly loss limit exceeded, False otherwise.
        """
        if account_value <= 0:
            return True

        weekly_loss_pct = -weekly_pnl / account_value
        return weekly_loss_pct >= self.weekly_loss_limit

    def should_halt_trading(self, metrics: dict[str, Any]) -> HaltDecision:
        """Determine if trading should be halted based on current metrics.

        Args:
            metrics: Dictionary containing:
                - daily_pnl: Today's P&L
                - weekly_pnl: This week's P&L
                - account_value: Current account value
                - start_of_day_value: Account value at market open
                - start_of_week_value: Account value at week start
                - consecutive_losses: Number of consecutive losing trades
                - error_count: Number of execution errors today

        Returns:
            HaltDecision with decision and reason.
        """
        daily_pnl = metrics.get("daily_pnl", 0)
        weekly_pnl = metrics.get("weekly_pnl", 0)
        start_of_day_value = metrics.get("start_of_day_value", metrics.get("account_value", 0))
        start_of_week_value = metrics.get("start_of_week_value", metrics.get("account_value", 0))
        consecutive_losses = metrics.get("consecutive_losses", 0)
        error_count = metrics.get("error_count", 0)

        if self.check_daily_loss_limit(daily_pnl, start_of_day_value):
            loss_pct = abs(daily_pnl / start_of_day_value * 100) if start_of_day_value > 0 else 0
            return HaltDecision(
                should_halt=True,
                reason=f"Daily loss limit exceeded: {loss_pct:.2f}% "
                f"(limit: {self.daily_loss_limit*100:.1f}%)",
            )

        if self.check_weekly_loss_limit(weekly_pnl, start_of_week_value):
            loss_pct = abs(weekly_pnl / start_of_week_value * 100) if start_of_week_value > 0 else 0
            return HaltDecision(
                should_halt=True,
                reason=f"Weekly loss limit exceeded: {loss_pct:.2f}% "
                f"(limit: {self.weekly_loss_limit*100:.1f}%)",
            )

        if consecutive_losses >= 5:
            return HaltDecision(
                should_halt=True,
                reason=f"Consecutive losses: {consecutive_losses} (limit: 5)",
            )

        if error_count >= 10:
            return HaltDecision(
                should_halt=True,
                reason=f"Execution errors: {error_count} (limit: 10)",
            )

        return HaltDecision(should_halt=False, reason="Trading within limits")

    def calculate_stop_loss_price(self, entry_price: float) -> float:
        """Calculate stop loss price from entry price.

        Args:
            entry_price: Entry price per share.

        Returns:
            Stop loss price (entry - stop_loss_pct).
        """
        return entry_price * (1 - self.stop_loss_pct)

    def calculate_profit_target_price(self, entry_price: float) -> float:
        """Calculate profit target price from entry price.

        Args:
            entry_price: Entry price per share.

        Returns:
            Profit target price (entry + profit_target_pct).
        """
        return entry_price * (1 + self.profit_target_pct)

    def validate_order(
        self,
        order: dict[str, Any],
        account: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> ValidationResult:
        """Validate order against risk rules before execution.

        Args:
            order: Dictionary containing:
                - symbol: Stock ticker
                - side: "buy" or "sell"
                - qty: Number of shares
                - price: Expected price (for limit orders) or current price
            account: Dictionary containing:
                - buying_power: Available buying power
                - equity: Total account equity
            positions: List of current position dictionaries.

        Returns:
            ValidationResult with validation status and reason.
        """
        symbol = order.get("symbol")
        side = order.get("side", "").lower()
        qty = order.get("qty", 0)
        price = order.get("price", 0)

        basic_result = self._validate_basic_order_fields(symbol, side, qty, price)
        if not basic_result.is_valid:
            return basic_result

        if side == "buy":
            return self._validate_buy_order(symbol, qty, price, account, positions)

        return self._validate_sell_order(symbol, qty, positions)

    def _validate_basic_order_fields(
        self, symbol: Optional[str], side: str, qty: int, price: float
    ) -> ValidationResult:
        """Validate basic order field requirements."""
        if not symbol:
            return ValidationResult(is_valid=False, reason="Missing symbol")

        if side not in ("buy", "sell"):
            return ValidationResult(is_valid=False, reason=f"Invalid side: {side}")

        if qty <= 0:
            return ValidationResult(is_valid=False, reason="Quantity must be positive")

        if price <= 0:
            return ValidationResult(is_valid=False, reason="Price must be positive")

        return ValidationResult(is_valid=True, reason="Basic validation passed")

    def _validate_buy_order(
        self,
        symbol: str,
        qty: int,
        price: float,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> ValidationResult:
        """Validate buy order against account and position limits."""
        buying_power = account.get("buying_power", 0)
        order_value = qty * price

        if order_value > buying_power:
            return ValidationResult(
                is_valid=False,
                reason=f"Insufficient buying power: ${order_value:.2f} > ${buying_power:.2f}",
            )

        equity = account.get("equity", 0)
        if equity > 0 and (order_value / equity) > 0.10:
            position_pct = order_value / equity * 100
            return ValidationResult(
                is_valid=False,
                reason=f"Position too large: {position_pct:.1f}% of equity (max 10%)",
            )

        existing_symbols = {p.get("symbol") for p in positions if p.get("symbol")}
        if symbol in existing_symbols:
            return ValidationResult(
                is_valid=False,
                reason=f"Already have position in {symbol}",
            )

        return ValidationResult(is_valid=True, reason="Order validated")

    def _validate_sell_order(
        self,
        symbol: str,
        qty: int,
        positions: list[dict[str, Any]],
    ) -> ValidationResult:
        """Validate sell order against existing positions."""
        position = next(
            (p for p in positions if p.get("symbol") == symbol), None
        )
        if position is None:
            return ValidationResult(
                is_valid=False,
                reason=f"No position in {symbol} to sell",
            )

        position_qty = position.get("qty", 0)
        if qty > position_qty:
            return ValidationResult(
                is_valid=False,
                reason=f"Sell qty {qty} exceeds position {position_qty}",
            )

        return ValidationResult(is_valid=True, reason="Order validated")

    def calculate_max_positions(self, account_value: float, avg_position_size: float) -> int:
        """Calculate maximum number of simultaneous positions.

        Limits total position exposure to maintain diversification.

        Args:
            account_value: Total account value.
            avg_position_size: Average expected position size in dollars.

        Returns:
            Maximum number of positions (typically 5-10).
        """
        if account_value <= 0 or avg_position_size <= 0:
            return 0

        max_by_sizing = int(account_value * 0.10 / avg_position_size * 10)
        return min(max(max_by_sizing, 5), 10)

    def calculate_risk_metrics(
        self,
        positions: list[dict[str, Any]],
        account_value: float,
    ) -> dict[str, Any]:
        """Calculate current risk exposure metrics.

        Args:
            positions: List of position dictionaries with entry_price, qty, current_price.
            account_value: Total account value.

        Returns:
            Dictionary with risk metrics.
        """
        if not positions or account_value <= 0:
            return {
                "total_exposure": 0.0,
                "exposure_pct": 0.0,
                "largest_position_pct": 0.0,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "at_risk": 0.0,
            }

        total_exposure = 0.0
        total_unrealized_pnl = 0.0
        total_at_risk = 0.0
        largest_position = 0.0

        for pos in positions:
            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", entry_price)
            qty = pos.get("qty", 0)

            position_value = current_price * qty
            total_exposure += position_value
            largest_position = max(largest_position, position_value)

            unrealized = (current_price - entry_price) * qty
            total_unrealized_pnl += unrealized

            stop_price = self.calculate_stop_loss_price(entry_price)
            risk = (entry_price - stop_price) * qty
            total_at_risk += risk

        return {
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / account_value * 100,
            "largest_position_pct": largest_position / account_value * 100,
            "unrealized_pnl": total_unrealized_pnl,
            "unrealized_pnl_pct": total_unrealized_pnl / account_value * 100,
            "at_risk": total_at_risk,
            "at_risk_pct": total_at_risk / account_value * 100,
        }

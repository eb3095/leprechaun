"""Order execution for Leprechaun trading bot.

Provides both real Alpaca integration and a simulated executor for
backtesting and development.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class OrderStatus(Enum):
    """Order status values."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side values."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type values."""

    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    """Represents a trading order."""

    id: str
    symbol: str
    qty: int
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    filled_qty: int = 0
    filled_avg_price: Optional[float] = None
    limit_price: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "filled_avg_price": self.filled_avg_price,
            "limit_price": self.limit_price,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    qty: int
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    current_price: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_entry_price": self.avg_entry_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "current_price": self.current_price,
        }


@dataclass
class Account:
    """Represents account information."""

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "portfolio_value": self.portfolio_value,
            "currency": self.currency,
        }


class ExecutorBase(ABC):
    """Abstract base class for order executors."""

    @abstractmethod
    def place_market_order(self, symbol: str, qty: int, side: str) -> dict[str, Any]:
        """Place a market order."""
        pass

    @abstractmethod
    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float
    ) -> dict[str, Any]:
        """Place a limit order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get status of an order."""
        pass

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions."""
        pass

    @abstractmethod
    def get_account(self) -> dict[str, Any]:
        """Get account information."""
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> dict[str, Any]:
        """Close entire position for a symbol."""
        pass

    @abstractmethod
    def close_all_positions(self) -> list[dict[str, Any]]:
        """Close all open positions."""
        pass


class OrderExecutor(ExecutorBase):
    """Order executor with Alpaca client integration.

    Can operate with a real Alpaca client or in simulation mode
    when client is None.
    """

    def __init__(self, trading_client: Optional[Any] = None, paper_mode: bool = True):
        """Initialize executor.

        Args:
            trading_client: Alpaca TradingClient instance. If None, uses simulation.
            paper_mode: Whether using paper trading (for logging/safety checks).
        """
        self.client = trading_client
        self.paper_mode = paper_mode
        self._simulated = SimulatedExecutor() if trading_client is None else None

    @property
    def is_simulated(self) -> bool:
        """Check if running in simulated mode."""
        return self._simulated is not None

    def place_market_order(self, symbol: str, qty: int, side: str) -> dict[str, Any]:
        """Place a market order.

        Args:
            symbol: Stock ticker symbol.
            qty: Number of shares.
            side: "buy" or "sell".

        Returns:
            Order details dictionary.
        """
        if self._simulated:
            return self._simulated.place_market_order(symbol, qty, side)

        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

        alpaca_side = AlpacaSide.BUY if side.lower() == "buy" else AlpacaSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(request)
        return self._alpaca_order_to_dict(order)

    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float
    ) -> dict[str, Any]:
        """Place a limit order.

        Args:
            symbol: Stock ticker symbol.
            qty: Number of shares.
            side: "buy" or "sell".
            limit_price: Limit price per share.

        Returns:
            Order details dictionary.
        """
        if self._simulated:
            return self._simulated.place_limit_order(symbol, qty, side, limit_price)

        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

        alpaca_side = AlpacaSide.BUY if side.lower() == "buy" else AlpacaSide.SELL
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
        )
        order = self.client.submit_order(request)
        return self._alpaca_order_to_dict(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancelled successfully.
        """
        if self._simulated:
            return self._simulated.cancel_order(order_id)

        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get order status.

        Args:
            order_id: Order ID to query.

        Returns:
            Order details dictionary.
        """
        if self._simulated:
            return self._simulated.get_order_status(order_id)

        order = self.client.get_order_by_id(order_id)
        return self._alpaca_order_to_dict(order)

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions.

        Returns:
            List of position dictionaries.
        """
        if self._simulated:
            return self._simulated.get_positions()

        positions = self.client.get_all_positions()
        return [self._alpaca_position_to_dict(p) for p in positions]

    def get_account(self) -> dict[str, Any]:
        """Get account information.

        Returns:
            Account details dictionary.
        """
        if self._simulated:
            return self._simulated.get_account()

        account = self.client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "currency": account.currency,
        }

    def close_position(self, symbol: str) -> dict[str, Any]:
        """Close entire position for a symbol.

        Args:
            symbol: Stock ticker to close.

        Returns:
            Order details for the closing order.
        """
        if self._simulated:
            return self._simulated.close_position(symbol)

        order = self.client.close_position(symbol)
        return self._alpaca_order_to_dict(order)

    def close_all_positions(self) -> list[dict[str, Any]]:
        """Close all open positions.

        Returns:
            List of closing order details.
        """
        if self._simulated:
            return self._simulated.close_all_positions()

        responses = self.client.close_all_positions()
        results = []
        for response in responses:
            if hasattr(response, "body"):
                results.append(self._alpaca_order_to_dict(response.body))
        return results

    def _alpaca_order_to_dict(self, order: Any) -> dict[str, Any]:
        """Convert Alpaca order object to dictionary."""
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": int(order.qty) if order.qty else 0,
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "order_type": order.type.value if hasattr(order.type, "value") else str(order.type),
            "status": order.status.value if hasattr(order.status, "value") else str(order.status),
            "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
        }

    def _alpaca_position_to_dict(self, position: Any) -> dict[str, Any]:
        """Convert Alpaca position object to dictionary."""
        return {
            "symbol": position.symbol,
            "qty": int(position.qty),
            "avg_entry_price": float(position.avg_entry_price),
            "market_value": float(position.market_value),
            "unrealized_pnl": float(position.unrealized_pl),
            "unrealized_pnl_pct": float(position.unrealized_plpc) * 100,
            "current_price": float(position.current_price),
        }


class SimulatedExecutor(ExecutorBase):
    """Simulated executor for backtesting and development.

    Maintains virtual positions and cash for testing strategies
    without real market access.
    """

    def __init__(self, initial_cash: float = 100000.0):
        """Initialize simulated executor.

        Args:
            initial_cash: Starting cash balance.
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, Order] = {}
        self.prices: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """Set current price for a symbol (for simulation).

        Args:
            symbol: Stock ticker.
            price: Current price.
        """
        self.prices[symbol] = price

    def set_prices(self, prices: dict[str, float]) -> None:
        """Set prices for multiple symbols.

        Args:
            prices: Dictionary of symbol -> price.
        """
        self.prices.update(prices)

    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol.

        Args:
            symbol: Stock ticker.

        Returns:
            Current price or 0 if unknown.
        """
        return self.prices.get(symbol, 0.0)

    def place_market_order(self, symbol: str, qty: int, side: str) -> dict[str, Any]:
        """Place a simulated market order.

        Market orders fill immediately at current price.
        """
        price = self.get_price(symbol)
        if price <= 0:
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                status=OrderStatus.REJECTED,
            )
            self.orders[order.id] = order
            return order.to_dict()

        return self._execute_order(symbol, qty, side, price, OrderType.MARKET)

    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float
    ) -> dict[str, Any]:
        """Place a simulated limit order.

        For simulation, limit orders fill immediately if price is favorable.
        """
        current_price = self.get_price(symbol)
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        should_fill = (
            (side_enum == OrderSide.BUY and current_price <= limit_price)
            or (side_enum == OrderSide.SELL and current_price >= limit_price)
        )

        if should_fill and current_price > 0:
            return self._execute_order(symbol, qty, side, current_price, OrderType.LIMIT, limit_price)

        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            qty=qty,
            side=side_enum,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PENDING,
            limit_price=limit_price,
        )
        self.orders[order.id] = order
        return order.to_dict()

    def _execute_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        price: float,
        order_type: OrderType,
        limit_price: Optional[float] = None,
    ) -> dict[str, Any]:
        """Execute an order at the given price."""
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        order_value = qty * price

        if side_enum == OrderSide.BUY:
            if order_value > self.cash:
                order = Order(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    qty=qty,
                    side=side_enum,
                    order_type=order_type,
                    status=OrderStatus.REJECTED,
                    limit_price=limit_price,
                )
                self.orders[order.id] = order
                return order.to_dict()

            self.cash -= order_value

            if symbol in self.positions:
                pos = self.positions[symbol]
                total_qty = pos["qty"] + qty
                total_cost = pos["avg_entry_price"] * pos["qty"] + price * qty
                pos["avg_entry_price"] = total_cost / total_qty
                pos["qty"] = total_qty
            else:
                self.positions[symbol] = {
                    "qty": qty,
                    "avg_entry_price": price,
                }

        else:
            if symbol not in self.positions or self.positions[symbol]["qty"] < qty:
                order = Order(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    qty=qty,
                    side=side_enum,
                    order_type=order_type,
                    status=OrderStatus.REJECTED,
                    limit_price=limit_price,
                )
                self.orders[order.id] = order
                return order.to_dict()

            self.cash += order_value
            self.positions[symbol]["qty"] -= qty

            if self.positions[symbol]["qty"] == 0:
                del self.positions[symbol]

        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            qty=qty,
            side=side_enum,
            order_type=order_type,
            status=OrderStatus.FILLED,
            filled_qty=qty,
            filled_avg_price=price,
            limit_price=limit_price,
            filled_at=datetime.now(timezone.utc),
        )
        self.orders[order.id] = order
        return order.to_dict()

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False

        order.status = OrderStatus.CANCELLED
        return True

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get order status."""
        if order_id not in self.orders:
            return {"error": "Order not found", "id": order_id}
        return self.orders[order_id].to_dict()

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions."""
        result = []
        for symbol, pos in self.positions.items():
            current_price = self.get_price(symbol)
            if current_price <= 0:
                current_price = pos["avg_entry_price"]

            qty = pos["qty"]
            avg_entry = pos["avg_entry_price"]
            market_value = current_price * qty
            unrealized_pnl = (current_price - avg_entry) * qty
            unrealized_pnl_pct = ((current_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0

            result.append({
                "symbol": symbol,
                "qty": qty,
                "avg_entry_price": avg_entry,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "current_price": current_price,
            })
        return result

    def get_account(self) -> dict[str, Any]:
        """Get account information."""
        positions_value = sum(
            self.get_price(sym) * pos["qty"] if self.get_price(sym) > 0
            else pos["avg_entry_price"] * pos["qty"]
            for sym, pos in self.positions.items()
        )
        equity = self.cash + positions_value

        return {
            "equity": equity,
            "cash": self.cash,
            "buying_power": self.cash,
            "portfolio_value": equity,
            "currency": "USD",
        }

    def close_position(self, symbol: str) -> dict[str, Any]:
        """Close entire position for a symbol."""
        if symbol not in self.positions:
            return {"error": f"No position in {symbol}"}

        qty = self.positions[symbol]["qty"]
        return self.place_market_order(symbol, qty, "sell")

    def close_all_positions(self) -> list[dict[str, Any]]:
        """Close all open positions."""
        results = []
        symbols = list(self.positions.keys())
        for symbol in symbols:
            result = self.close_position(symbol)
            results.append(result)
        return results

    def reset(self) -> None:
        """Reset to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self.orders.clear()
        self.prices.clear()

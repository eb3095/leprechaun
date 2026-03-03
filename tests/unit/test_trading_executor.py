"""Unit tests for order execution module."""

import pytest

from src.core.trading.executor import (
    Account,
    Order,
    OrderExecutor,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    SimulatedExecutor,
)


class TestOrderDataclass:
    def test_order_creation(self):
        order = Order(
            id="test-123",
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
        )
        assert order.id == "test-123"
        assert order.symbol == "AAPL"
        assert order.qty == 10
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING

    def test_order_to_dict(self):
        order = Order(
            id="test-123",
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_qty=10,
            filled_avg_price=150.0,
        )
        d = order.to_dict()
        assert d["id"] == "test-123"
        assert d["symbol"] == "AAPL"
        assert d["side"] == "buy"
        assert d["order_type"] == "market"
        assert d["status"] == "filled"
        assert d["filled_avg_price"] == 150.0


class TestPositionDataclass:
    def test_position_creation(self):
        pos = Position(
            symbol="AAPL",
            qty=100,
            avg_entry_price=150.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            unrealized_pnl_pct=3.33,
            current_price=155.0,
        )
        assert pos.symbol == "AAPL"
        assert pos.qty == 100
        assert pos.avg_entry_price == 150.0

    def test_position_to_dict(self):
        pos = Position(
            symbol="AAPL",
            qty=100,
            avg_entry_price=150.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            unrealized_pnl_pct=3.33,
            current_price=155.0,
        )
        d = pos.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["qty"] == 100
        assert d["unrealized_pnl"] == 500.0


class TestAccountDataclass:
    def test_account_creation(self):
        acc = Account(
            equity=100000.0,
            cash=50000.0,
            buying_power=50000.0,
            portfolio_value=100000.0,
        )
        assert acc.equity == 100000.0
        assert acc.currency == "USD"

    def test_account_to_dict(self):
        acc = Account(
            equity=100000.0,
            cash=50000.0,
            buying_power=50000.0,
            portfolio_value=100000.0,
        )
        d = acc.to_dict()
        assert d["equity"] == 100000.0
        assert d["currency"] == "USD"


class TestSimulatedExecutor:
    @pytest.fixture
    def executor(self):
        """Create a fresh simulated executor."""
        return SimulatedExecutor(initial_cash=100000.0)

    def test_initial_state(self, executor):
        assert executor.cash == 100000.0
        assert executor.positions == {}
        assert executor.orders == {}
        account = executor.get_account()
        assert account["equity"] == 100000.0
        assert account["cash"] == 100000.0

    def test_set_price(self, executor):
        executor.set_price("AAPL", 150.0)
        assert executor.get_price("AAPL") == 150.0

    def test_set_prices(self, executor):
        executor.set_prices({"AAPL": 150.0, "MSFT": 300.0})
        assert executor.get_price("AAPL") == 150.0
        assert executor.get_price("MSFT") == 300.0

    def test_get_price_unknown(self, executor):
        assert executor.get_price("UNKNOWN") == 0.0


class TestSimulatedMarketOrders:
    @pytest.fixture
    def executor(self):
        exec = SimulatedExecutor(initial_cash=100000.0)
        exec.set_prices({"AAPL": 150.0, "MSFT": 300.0})
        return exec

    def test_buy_market_order(self, executor):
        result = executor.place_market_order("AAPL", 10, "buy")
        assert result["status"] == "filled"
        assert result["filled_qty"] == 10
        assert result["filled_avg_price"] == 150.0
        assert executor.cash == 100000 - (10 * 150)
        assert "AAPL" in executor.positions

    def test_sell_market_order(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        result = executor.place_market_order("AAPL", 5, "sell")
        assert result["status"] == "filled"
        assert result["filled_qty"] == 5
        assert executor.positions["AAPL"]["qty"] == 5

    def test_sell_all_removes_position(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        executor.place_market_order("AAPL", 10, "sell")
        assert "AAPL" not in executor.positions

    def test_buy_insufficient_funds(self, executor):
        result = executor.place_market_order("AAPL", 1000, "buy")
        assert result["status"] == "rejected"
        assert "AAPL" not in executor.positions

    def test_sell_no_position(self, executor):
        result = executor.place_market_order("AAPL", 10, "sell")
        assert result["status"] == "rejected"

    def test_sell_more_than_owned(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        result = executor.place_market_order("AAPL", 20, "sell")
        assert result["status"] == "rejected"

    def test_buy_unknown_price_rejected(self, executor):
        result = executor.place_market_order("UNKNOWN", 10, "buy")
        assert result["status"] == "rejected"

    def test_add_to_existing_position(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        executor.set_price("AAPL", 160.0)
        executor.place_market_order("AAPL", 10, "buy")
        pos = executor.positions["AAPL"]
        assert pos["qty"] == 20
        assert pos["avg_entry_price"] == 155.0


class TestSimulatedLimitOrders:
    @pytest.fixture
    def executor(self):
        exec = SimulatedExecutor(initial_cash=100000.0)
        exec.set_prices({"AAPL": 150.0})
        return exec

    def test_buy_limit_at_or_below_fills(self, executor):
        result = executor.place_limit_order("AAPL", 10, "buy", 155.0)
        assert result["status"] == "filled"
        assert result["filled_avg_price"] == 150.0

    def test_buy_limit_above_price_pending(self, executor):
        result = executor.place_limit_order("AAPL", 10, "buy", 140.0)
        assert result["status"] == "pending"
        assert "AAPL" not in executor.positions

    def test_sell_limit_at_or_above_fills(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        result = executor.place_limit_order("AAPL", 5, "sell", 145.0)
        assert result["status"] == "filled"

    def test_sell_limit_below_price_pending(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        result = executor.place_limit_order("AAPL", 5, "sell", 160.0)
        assert result["status"] == "pending"


class TestSimulatedOrderManagement:
    @pytest.fixture
    def executor(self):
        exec = SimulatedExecutor(initial_cash=100000.0)
        exec.set_prices({"AAPL": 150.0})
        return exec

    def test_cancel_pending_order(self, executor):
        result = executor.place_limit_order("AAPL", 10, "buy", 140.0)
        order_id = result["id"]
        assert executor.cancel_order(order_id) is True
        assert executor.orders[order_id].status == OrderStatus.CANCELLED

    def test_cancel_filled_order_fails(self, executor):
        result = executor.place_market_order("AAPL", 10, "buy")
        order_id = result["id"]
        assert executor.cancel_order(order_id) is False

    def test_cancel_nonexistent_order(self, executor):
        assert executor.cancel_order("fake-id") is False

    def test_get_order_status(self, executor):
        result = executor.place_market_order("AAPL", 10, "buy")
        order_id = result["id"]
        status = executor.get_order_status(order_id)
        assert status["id"] == order_id
        assert status["status"] == "filled"

    def test_get_order_status_not_found(self, executor):
        status = executor.get_order_status("fake-id")
        assert "error" in status


class TestSimulatedPositions:
    @pytest.fixture
    def executor(self):
        exec = SimulatedExecutor(initial_cash=100000.0)
        exec.set_prices({"AAPL": 150.0, "MSFT": 300.0})
        return exec

    def test_get_positions_empty(self, executor):
        positions = executor.get_positions()
        assert positions == []

    def test_get_positions_with_holdings(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        executor.place_market_order("MSFT", 5, "buy")
        positions = executor.get_positions()
        assert len(positions) == 2
        symbols = [p["symbol"] for p in positions]
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_position_pnl_calculation(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        executor.set_price("AAPL", 160.0)
        positions = executor.get_positions()
        aapl = next(p for p in positions if p["symbol"] == "AAPL")
        assert aapl["unrealized_pnl"] == 100.0
        assert aapl["market_value"] == 1600.0


class TestSimulatedClosePositions:
    @pytest.fixture
    def executor(self):
        exec = SimulatedExecutor(initial_cash=100000.0)
        exec.set_prices({"AAPL": 150.0, "MSFT": 300.0})
        exec.place_market_order("AAPL", 10, "buy")
        exec.place_market_order("MSFT", 5, "buy")
        return exec

    def test_close_position(self, executor):
        result = executor.close_position("AAPL")
        assert result["status"] == "filled"
        assert result["qty"] == 10
        assert "AAPL" not in executor.positions

    def test_close_position_not_found(self, executor):
        result = executor.close_position("GOOG")
        assert "error" in result

    def test_close_all_positions(self, executor):
        results = executor.close_all_positions()
        assert len(results) == 2
        assert executor.positions == {}
        assert executor.cash == 100000.0


class TestSimulatedAccount:
    @pytest.fixture
    def executor(self):
        exec = SimulatedExecutor(initial_cash=100000.0)
        exec.set_prices({"AAPL": 150.0})
        return exec

    def test_account_after_buy(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        account = executor.get_account()
        assert account["cash"] == 100000 - 1500
        assert account["equity"] == 100000
        assert account["portfolio_value"] == 100000

    def test_account_with_gain(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        executor.set_price("AAPL", 160.0)
        account = executor.get_account()
        assert account["equity"] == 100000 + 100
        assert account["cash"] == 100000 - 1500

    def test_account_with_loss(self, executor):
        executor.place_market_order("AAPL", 10, "buy")
        executor.set_price("AAPL", 140.0)
        account = executor.get_account()
        assert account["equity"] == 100000 - 100


class TestSimulatedReset:
    def test_reset(self):
        executor = SimulatedExecutor(initial_cash=100000.0)
        executor.set_prices({"AAPL": 150.0})
        executor.place_market_order("AAPL", 10, "buy")
        executor.reset()
        assert executor.cash == 100000.0
        assert executor.positions == {}
        assert executor.orders == {}
        assert executor.prices == {}


class TestOrderExecutorSimulatedMode:
    def test_without_client_uses_simulation(self):
        executor = OrderExecutor(trading_client=None, paper_mode=True)
        assert executor.is_simulated is True

    def test_simulated_operations(self):
        executor = OrderExecutor(trading_client=None, paper_mode=True)
        executor._simulated.set_price("AAPL", 150.0)
        result = executor.place_market_order("AAPL", 10, "buy")
        assert result["status"] == "filled"
        positions = executor.get_positions()
        assert len(positions) == 1
        account = executor.get_account()
        assert account["cash"] < 100000

    def test_limit_order_simulation(self):
        executor = OrderExecutor(trading_client=None)
        executor._simulated.set_price("AAPL", 150.0)
        result = executor.place_limit_order("AAPL", 10, "buy", 155.0)
        assert result["status"] == "filled"

    def test_cancel_order_simulation(self):
        executor = OrderExecutor(trading_client=None)
        executor._simulated.set_price("AAPL", 150.0)
        result = executor.place_limit_order("AAPL", 10, "buy", 140.0)
        assert executor.cancel_order(result["id"]) is True

    def test_close_position_simulation(self):
        executor = OrderExecutor(trading_client=None)
        executor._simulated.set_price("AAPL", 150.0)
        executor.place_market_order("AAPL", 10, "buy")
        result = executor.close_position("AAPL")
        assert result["status"] == "filled"

    def test_close_all_positions_simulation(self):
        executor = OrderExecutor(trading_client=None)
        executor._simulated.set_prices({"AAPL": 150.0, "MSFT": 300.0})
        executor.place_market_order("AAPL", 10, "buy")
        executor.place_market_order("MSFT", 5, "buy")
        results = executor.close_all_positions()
        assert len(results) == 2


class TestOrderStatusEnum:
    def test_all_statuses_exist(self):
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.ACCEPTED.value == "accepted"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"


class TestOrderSideEnum:
    def test_all_sides_exist(self):
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestOrderTypeEnum:
    def test_all_types_exist(self):
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"

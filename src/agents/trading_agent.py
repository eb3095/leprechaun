"""Trading decision agent for Leprechaun trading bot.

Makes final trading decisions by combining technical analysis,
sentiment analysis, and manipulation detection signals.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from src.agents.decision_log import Decision


@dataclass
class TradingStrategy:
    """Trading strategy parameters."""

    profit_target_pct: float = 2.5
    stop_loss_pct: float = 1.25
    min_manipulation_score: float = 0.5
    max_rsi_for_buy: float = 35.0
    min_sentiment_for_skip: float = 0.2
    max_positions: int = 10


@dataclass
class RiskManager:
    """Risk management parameters and state."""

    max_position_risk_pct: float = 1.0
    max_daily_loss_pct: float = 2.0
    max_weekly_loss_pct: float = 5.0
    daily_loss_pct: float = 0.0
    weekly_loss_pct: float = 0.0
    is_halted: bool = False
    halt_reason: Optional[str] = None


class TradingAgent:
    """Makes trading decisions combining all analysis inputs."""

    def __init__(
        self,
        strategy: Optional[TradingStrategy] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        """Initialize trading agent.

        Args:
            strategy: Trading strategy parameters.
            risk_manager: Risk management instance.
        """
        self.strategy = strategy or TradingStrategy()
        self.risk_manager = risk_manager or RiskManager()

    def make_decision(
        self,
        symbol: str,
        market_data: dict[str, Any],
        sentiment: dict[str, Any],
        manipulation: dict[str, Any],
        account: dict[str, Any],
    ) -> Decision:
        """Make trading decision combining all inputs.

        Args:
            symbol: Stock symbol.
            market_data: Dict with price, volume, technical indicators.
            sentiment: Sentiment analysis result.
            manipulation: Manipulation analysis result.
            account: Account info (cash, positions, equity).

        Returns:
            Decision object with full reasoning chain.
        """
        inputs = {
            "symbol": symbol,
            "market_data": self._sanitize_market_data(market_data),
            "sentiment": self._sanitize_dict(sentiment),
            "manipulation": self._sanitize_dict(manipulation),
            "account": self._sanitize_dict(account),
        }

        reasoning = []

        if self.risk_manager.is_halted:
            reasoning.append(f"Trading halted: {self.risk_manager.halt_reason}")
            return self._create_decision(
                symbol, "SKIP", 0.0, inputs, reasoning, executed=False
            )

        if self._check_risk_limits():
            reasoning.append("Risk limits exceeded, skipping")
            return self._create_decision(
                symbol, "SKIP", 0.0, inputs, reasoning, executed=False
            )

        position_count = len(account.get("positions", []))
        if position_count >= self.strategy.max_positions:
            reasoning.append(f"Max positions ({self.strategy.max_positions}) reached")
            return self._create_decision(
                symbol, "SKIP", 0.0, inputs, reasoning, executed=False
            )

        has_position = self._has_position(symbol, account)

        if has_position:
            return self._evaluate_exit(
                symbol, market_data, sentiment, manipulation, account, inputs
            )
        else:
            return self._evaluate_entry(
                symbol, market_data, sentiment, manipulation, account, inputs
            )

    def _evaluate_entry(
        self,
        symbol: str,
        market_data: dict[str, Any],
        sentiment: dict[str, Any],
        manipulation: dict[str, Any],
        account: dict[str, Any],
        inputs: dict[str, Any],
    ) -> Decision:
        """Evaluate potential entry into a new position."""
        reasoning = []
        confidence_factors = []

        manipulation_score = manipulation.get("manipulation_score", 0.0)
        reasoning.append(f"Manipulation score: {manipulation_score:.2f}")

        if manipulation_score < self.strategy.min_manipulation_score:
            reasoning.append(
                f"Below threshold ({self.strategy.min_manipulation_score}), skipping"
            )
            return self._create_decision(
                symbol, "SKIP", 0.0, inputs, reasoning, executed=False
            )
        confidence_factors.append(manipulation_score)

        technical = market_data.get("technical_indicators", {})
        rsi = technical.get("rsi_14", 50.0)
        if isinstance(rsi, Decimal):
            rsi = float(rsi)
        reasoning.append(f"RSI(14): {rsi:.1f}")

        if rsi > self.strategy.max_rsi_for_buy:
            reasoning.append(
                f"RSI above {self.strategy.max_rsi_for_buy}, not oversold enough"
            )
            return self._create_decision(
                symbol, "SKIP", 0.3, inputs, reasoning, executed=False
            )
        confidence_factors.append(
            (self.strategy.max_rsi_for_buy - rsi) / self.strategy.max_rsi_for_buy
        )

        sentiment_score = sentiment.get("composite_score", 0.0)
        reasoning.append(f"Sentiment score: {sentiment_score:.2f}")

        if sentiment_score > self.strategy.min_sentiment_for_skip:
            reasoning.append("Sentiment not negative enough for contrarian play")
            return self._create_decision(
                symbol, "SKIP", 0.4, inputs, reasoning, executed=False
            )
        confidence_factors.append(abs(sentiment_score))

        current_price = market_data.get("current_price", 0)
        if isinstance(current_price, Decimal):
            current_price = float(current_price)
        ema_21 = technical.get("ema_21", current_price)
        if isinstance(ema_21, Decimal):
            ema_21 = float(ema_21)

        if current_price > 0 and ema_21 > 0:
            price_vs_ema = (current_price - ema_21) / ema_21
            reasoning.append(f"Price vs EMA(21): {price_vs_ema*100:.1f}%")

            if price_vs_ema > 0:
                reasoning.append("Price above EMA, wait for pullback")
                return self._create_decision(
                    symbol, "HOLD", 0.4, inputs, reasoning, executed=False
                )
            confidence_factors.append(min(1.0, abs(price_vs_ema) * 10))

        evidence = manipulation.get("evidence", {})
        if evidence.get("no_news_catalyst"):
            reasoning.append("No news catalyst - supports manipulation thesis")
            confidence_factors.append(0.8)
        else:
            reasoning.append("News catalyst exists - sentiment may be justified")
            confidence_factors.append(0.3)

        confidence = (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.5
        )

        if confidence > 0.6:
            reasoning.append(f"BUY signal with {confidence:.0%} confidence")
            return self._create_decision(
                symbol, "BUY", confidence, inputs, reasoning, executed=False
            )
        else:
            reasoning.append(f"Confidence {confidence:.0%} below threshold, holding")
            return self._create_decision(
                symbol, "HOLD", confidence, inputs, reasoning, executed=False
            )

    def _evaluate_exit(
        self,
        symbol: str,
        market_data: dict[str, Any],
        sentiment: dict[str, Any],
        manipulation: dict[str, Any],
        account: dict[str, Any],
        inputs: dict[str, Any],
    ) -> Decision:
        """Evaluate exit conditions for existing position."""
        reasoning = []

        position = self._get_position(symbol, account)
        if position is None:
            reasoning.append("Position not found")
            return self._create_decision(
                symbol, "HOLD", 0.5, inputs, reasoning, executed=False
            )

        entry_price = position.get("entry_price", 0)
        if isinstance(entry_price, Decimal):
            entry_price = float(entry_price)
        current_price = market_data.get("current_price", entry_price)
        if isinstance(current_price, Decimal):
            current_price = float(current_price)

        if entry_price > 0:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = 0.0

        reasoning.append(f"Position P&L: {pnl_pct:.2f}%")

        if pnl_pct >= self.strategy.profit_target_pct:
            reasoning.append(
                f"Profit target ({self.strategy.profit_target_pct}%) reached"
            )
            return self._create_decision(
                symbol, "SELL", 0.95, inputs, reasoning, executed=False
            )

        if pnl_pct <= -self.strategy.stop_loss_pct:
            reasoning.append(f"Stop loss ({self.strategy.stop_loss_pct}%) triggered")
            return self._create_decision(
                symbol, "SELL", 0.95, inputs, reasoning, executed=False
            )

        sentiment_score = sentiment.get("composite_score", 0.0)
        if sentiment_score > 0.3:
            reasoning.append("Sentiment normalized, consider exit")
            return self._create_decision(
                symbol, "SELL", 0.7, inputs, reasoning, executed=False
            )

        technical = market_data.get("technical_indicators", {})
        rsi = technical.get("rsi_14", 50.0)
        if isinstance(rsi, Decimal):
            rsi = float(rsi)

        if rsi > 50:
            reasoning.append(f"RSI normalized ({rsi:.1f}), consider exit")
            return self._create_decision(
                symbol, "SELL", 0.6, inputs, reasoning, executed=False
            )

        reasoning.append("Holding position - no exit conditions met")
        return self._create_decision(
            symbol, "HOLD", 0.5, inputs, reasoning, executed=False
        )

    def _create_decision(
        self,
        symbol: str,
        decision: str,
        confidence: float,
        inputs: dict[str, Any],
        reasoning: list[str],
        executed: bool,
        execution_details: Optional[dict[str, Any]] = None,
    ) -> Decision:
        """Create a Decision object."""
        return Decision(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            decision=decision,
            confidence=confidence,
            inputs=inputs,
            reasoning=reasoning,
            executed=executed,
            execution_details=execution_details or {},
        )

    def generate_reasoning(
        self,
        inputs: dict[str, Any],
        decision: str,
    ) -> list[str]:
        """Generate human-readable reasoning chain.

        Args:
            inputs: All input signals.
            decision: The decision made (BUY, SELL, HOLD, SKIP).

        Returns:
            List of reasoning steps.
        """
        reasoning = []

        manipulation = inputs.get("manipulation", {})
        sentiment = inputs.get("sentiment", {})
        market = inputs.get("market_data", {})
        technical = market.get("technical_indicators", {})

        reasoning.append(
            f"Analyzed {inputs.get('symbol', 'UNKNOWN')} at "
            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        manip_score = manipulation.get("manipulation_score", 0.0)
        reasoning.append(f"Manipulation probability: {manip_score:.0%}")

        sent_score = sentiment.get("composite_score", 0.0)
        reasoning.append(f"Sentiment: {sent_score:.2f} (scale: -1 to 1)")

        rsi = technical.get("rsi_14", 50.0)
        if isinstance(rsi, Decimal):
            rsi = float(rsi)
        reasoning.append(f"RSI(14): {rsi:.1f}")

        evidence = manipulation.get("evidence", {})
        if evidence:
            triggered = [k for k, v in evidence.items() if v]
            if triggered:
                reasoning.append(f"Triggered signals: {', '.join(triggered)}")

        reasoning.append(f"Decision: {decision}")

        return reasoning

    def _check_risk_limits(self) -> bool:
        """Check if risk limits are exceeded."""
        if self.risk_manager.daily_loss_pct >= self.risk_manager.max_daily_loss_pct:
            return True
        if self.risk_manager.weekly_loss_pct >= self.risk_manager.max_weekly_loss_pct:
            return True
        return False

    def _has_position(self, symbol: str, account: dict[str, Any]) -> bool:
        """Check if we have an open position in the symbol."""
        positions = account.get("positions", [])
        for pos in positions:
            if pos.get("symbol") == symbol:
                return True
        return False

    def _get_position(
        self, symbol: str, account: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Get position details for a symbol."""
        positions = account.get("positions", [])
        for pos in positions:
            if pos.get("symbol") == symbol:
                return pos
        return None

    def _sanitize_market_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize market data for JSON serialization."""
        result = {}
        for key, value in data.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._sanitize_dict(value)
            else:
                result[key] = value
        return result

    def _sanitize_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize dictionary for JSON serialization."""
        result = {}
        for key, value in data.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._sanitize_dict(value)
            elif hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def calculate_position_size(
        self,
        account: dict[str, Any],
        current_price: float,
        stop_loss_price: float,
    ) -> int:
        """Calculate position size based on risk management.

        Uses fixed percentage risk per trade.

        Args:
            account: Account info with equity.
            current_price: Current stock price.
            stop_loss_price: Stop loss price level.

        Returns:
            Number of shares to buy.
        """
        equity = account.get("equity", 0)
        if isinstance(equity, Decimal):
            equity = float(equity)

        risk_amount = equity * (self.risk_manager.max_position_risk_pct / 100)

        risk_per_share = abs(current_price - stop_loss_price)
        if risk_per_share <= 0:
            risk_per_share = current_price * (self.strategy.stop_loss_pct / 100)

        if risk_per_share > 0:
            shares = int(risk_amount / risk_per_share)
        else:
            shares = 0

        max_shares = int((equity * 0.1) / current_price) if current_price > 0 else 0
        shares = min(shares, max_shares)

        return max(0, shares)

    def update_risk_state(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
    ) -> None:
        """Update risk manager state with current P&L.

        Args:
            daily_pnl_pct: Daily P&L percentage.
            weekly_pnl_pct: Weekly P&L percentage.
        """
        self.risk_manager.daily_loss_pct = abs(min(0, daily_pnl_pct))
        self.risk_manager.weekly_loss_pct = abs(min(0, weekly_pnl_pct))

        if self.risk_manager.daily_loss_pct >= self.risk_manager.max_daily_loss_pct:
            self.risk_manager.is_halted = True
            self.risk_manager.halt_reason = (
                f"Daily loss limit ({self.risk_manager.max_daily_loss_pct}%) exceeded"
            )
        elif self.risk_manager.weekly_loss_pct >= self.risk_manager.max_weekly_loss_pct:
            self.risk_manager.is_halted = True
            self.risk_manager.halt_reason = (
                f"Weekly loss limit ({self.risk_manager.max_weekly_loss_pct}%) exceeded"
            )

    def reset_daily_risk(self) -> None:
        """Reset daily risk tracking. Call at start of each trading day."""
        self.risk_manager.daily_loss_pct = 0.0

    def reset_weekly_risk(self) -> None:
        """Reset weekly risk tracking. Call at start of each trading week."""
        self.risk_manager.weekly_loss_pct = 0.0
        self.risk_manager.daily_loss_pct = 0.0

    def resume_trading(self) -> bool:
        """Resume trading after halt (requires manual intervention).

        Returns:
            True if trading was resumed.
        """
        if not self.risk_manager.is_halted:
            return False

        self.risk_manager.is_halted = False
        self.risk_manager.halt_reason = None
        return True

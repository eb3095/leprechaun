"""Trading strategy implementation for Leprechaun.

Implements the core contrarian trading strategy: buy oversold stocks with
high manipulation scores (negative sentiment without news catalyst) and
exit on profit target, stop loss, or Friday close.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.utils.config import TradingConfig


class EntryConfidence(Enum):
    """Entry signal confidence levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class EntrySignal:
    """Result of entry evaluation."""

    should_enter: bool
    confidence: EntryConfidence
    reasons: list[str]
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_enter": self.should_enter,
            "confidence": self.confidence.value,
            "reasons": self.reasons,
            "score": self.score,
        }


@dataclass
class ExitSignal:
    """Result of exit evaluation."""

    should_exit: bool
    reason: str
    exit_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_exit": self.should_exit,
            "reason": self.reason,
            "exit_type": self.exit_type,
        }


class TradingStrategy:
    """Core trading strategy for contrarian sentiment-based trades.

    Strategy:
    - Enter when RSI is oversold, sentiment is negative, manipulation score
      is high, and there's no material news catalyst
    - Exit on 2.5% profit, 1.25% stop loss, or Friday close
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        """Initialize strategy with configuration.

        Args:
            config: Trading configuration. Uses defaults if not provided.
        """
        self.config = config or TradingConfig()
        self.profit_target = self.config.profit_target_percent / 100
        self.stop_loss = self.config.stop_loss_percent / 100
        self.rsi_oversold = self.config.rsi_oversold
        self.manipulation_threshold = self.config.manipulation_threshold
        self.sentiment_threshold = self.config.sentiment_negative_threshold

    def evaluate_entry(self, stock_data: dict[str, Any]) -> EntrySignal:
        """Evaluate if we should enter a position.

        Args:
            stock_data: Dictionary containing:
                - symbol: Stock ticker
                - price: Current price
                - rsi: RSI-14 value
                - ema_21: 21-period EMA
                - sentiment_score: Aggregated sentiment (-1 to 1)
                - manipulation_score: Manipulation probability (0 to 1)
                - has_news: Whether material news exists

        Returns:
            EntrySignal with decision, confidence, and reasoning.
        """
        symbol = stock_data.get("symbol", "UNKNOWN")
        price = stock_data.get("price")
        rsi = stock_data.get("rsi")
        ema_21 = stock_data.get("ema_21")
        sentiment_score = stock_data.get("sentiment_score")
        manipulation_score = stock_data.get("manipulation_score")
        has_news = stock_data.get("has_news", False)

        reasons: list[str] = []
        score = 0.0

        if any(v is None for v in [price, rsi, sentiment_score, manipulation_score]):
            return EntrySignal(
                should_enter=False,
                confidence=EntryConfidence.LOW,
                reasons=["Insufficient data for analysis"],
                score=0.0,
            )

        if has_news:
            return EntrySignal(
                should_enter=False,
                confidence=EntryConfidence.HIGH,
                reasons=[f"{symbol}: Material news present - skip contrarian play"],
                score=0.0,
            )

        if rsi < self.rsi_oversold:
            reasons.append(f"RSI oversold at {rsi:.1f} (threshold: {self.rsi_oversold})")
            score += 0.25

        if ema_21 is not None and price < ema_21:
            pct_below = ((ema_21 - price) / ema_21) * 100
            reasons.append(f"Price {pct_below:.1f}% below 21-EMA")
            score += 0.15

        if sentiment_score < self.sentiment_threshold:
            reasons.append(
                f"Negative sentiment at {sentiment_score:.2f} "
                f"(threshold: {self.sentiment_threshold})"
            )
            score += 0.25

        if manipulation_score > self.manipulation_threshold:
            reasons.append(
                f"High manipulation score: {manipulation_score:.2f} "
                f"(threshold: {self.manipulation_threshold})"
            )
            score += 0.35

        should_enter = score >= 0.60
        confidence = self._score_to_confidence(score)

        if not should_enter and not reasons:
            reasons.append(f"{symbol}: Does not meet entry criteria")

        return EntrySignal(
            should_enter=should_enter,
            confidence=confidence,
            reasons=reasons,
            score=score,
        )

    def evaluate_exit(
        self, position: dict[str, Any], current_data: dict[str, Any]
    ) -> ExitSignal:
        """Evaluate if we should exit a position.

        Args:
            position: Dictionary containing:
                - symbol: Stock ticker
                - entry_price: Entry price
                - shares: Number of shares
            current_data: Dictionary containing:
                - price: Current price
                - rsi: Current RSI (optional)
                - is_friday_close: Whether it's Friday market close
                - has_breaking_news: Whether material negative news emerged

        Returns:
            ExitSignal with decision and reasoning.
        """
        entry_price = position.get("entry_price")
        current_price = current_data.get("price")
        is_friday_close = current_data.get("is_friday_close", False)
        has_breaking_news = current_data.get("has_breaking_news", False)
        rsi = current_data.get("rsi")

        if entry_price is None or current_price is None:
            return ExitSignal(
                should_exit=False,
                reason="Missing price data",
                exit_type="NONE",
            )

        pnl_percent = (current_price - entry_price) / entry_price

        if pnl_percent >= self.profit_target:
            return ExitSignal(
                should_exit=True,
                reason=f"Profit target reached: {pnl_percent*100:.2f}%",
                exit_type="TARGET",
            )

        if pnl_percent <= -self.stop_loss:
            return ExitSignal(
                should_exit=True,
                reason=f"Stop loss triggered: {pnl_percent*100:.2f}%",
                exit_type="STOP_LOSS",
            )

        if is_friday_close:
            return ExitSignal(
                should_exit=True,
                reason=f"Friday close - P&L: {pnl_percent*100:.2f}%",
                exit_type="FRIDAY_CLOSE",
            )

        if has_breaking_news:
            return ExitSignal(
                should_exit=True,
                reason="Material negative news emerged",
                exit_type="NEWS",
            )

        rsi_normalized = self.config.rsi_normalized
        if rsi is not None and rsi > rsi_normalized and pnl_percent > 0:
            return ExitSignal(
                should_exit=True,
                reason=f"RSI normalized at {rsi:.1f} with profit {pnl_percent*100:.2f}%",
                exit_type="TARGET",
            )

        return ExitSignal(
            should_exit=False,
            reason="Hold position",
            exit_type="NONE",
        )

    def rank_opportunities(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rank buy candidates by signal strength.

        Args:
            candidates: List of stock data dictionaries with entry signals.

        Returns:
            Sorted list with score field, highest first.
        """
        scored = []

        for candidate in candidates:
            signal = self.evaluate_entry(candidate)
            if signal.should_enter:
                scored.append(
                    {
                        **candidate,
                        "entry_signal": signal.to_dict(),
                        "rank_score": signal.score,
                    }
                )

        return sorted(scored, key=lambda x: x["rank_score"], reverse=True)

    def _score_to_confidence(self, score: float) -> EntryConfidence:
        """Convert numeric score to confidence level."""
        if score >= 0.85:
            return EntryConfidence.HIGH
        elif score >= 0.70:
            return EntryConfidence.MEDIUM
        return EntryConfidence.LOW

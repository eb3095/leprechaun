"""Agent orchestration for Leprechaun trading bot.

Coordinates all agents (sentiment, manipulation, trading) to run
analysis cycles and make trading decisions.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy.orm import Session

from src.agents.bayesian import BayesianManipulationDetector
from src.agents.decision_log import Decision, DecisionLogger
from src.agents.manipulation_agent import ManipulationAgent
from src.agents.sentiment_agent import SentimentAgent, SentimentResult
from src.agents.trading_agent import TradingAgent, TradingStrategy, RiskManager


class AgentOrchestrator:
    """Orchestrates agent coordination for trading decisions."""

    def __init__(
        self,
        db_session: Optional[Session] = None,
        trading_strategy: Optional[TradingStrategy] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        """Initialize orchestrator with all agents.

        Args:
            db_session: Database session for persistence.
            trading_strategy: Trading strategy parameters.
            risk_manager: Risk management instance.
        """
        self.db_session = db_session
        self.sentiment_agent = SentimentAgent()
        self.bayesian = BayesianManipulationDetector()
        self.manipulation_agent = ManipulationAgent(self.bayesian)
        self.trading_agent = TradingAgent(
            strategy=trading_strategy or TradingStrategy(),
            risk_manager=risk_manager or RiskManager(),
        )
        self.decision_logger = DecisionLogger(db_session)

    def run_analysis_cycle(
        self,
        symbols: list[str],
        market_data_provider: Optional[Any] = None,
        sentiment_data_provider: Optional[Any] = None,
        news_data_provider: Optional[Any] = None,
    ) -> list[Decision]:
        """Run full analysis cycle for given symbols.

        Pipeline:
        1. Gather sentiment for each symbol
        2. Run manipulation detection
        3. Get technical indicators
        4. Make trading decisions
        5. Log all decisions

        Args:
            symbols: List of stock symbols to analyze.
            market_data_provider: Service to get market data.
            sentiment_data_provider: Service to get sentiment data.
            news_data_provider: Service to get news data.

        Returns:
            List of decisions made.
        """
        decisions = []

        for symbol in symbols:
            try:
                decision = self._analyze_symbol(
                    symbol,
                    market_data_provider,
                    sentiment_data_provider,
                    news_data_provider,
                )
                if decision:
                    self.decision_logger.log_decision(decision)
                    decisions.append(decision)
            except Exception as e:
                error_decision = Decision(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    decision="SKIP",
                    confidence=0.0,
                    inputs={"error": str(e)},
                    reasoning=[f"Analysis failed: {str(e)}"],
                    executed=False,
                )
                self.decision_logger.log_decision(error_decision)
                decisions.append(error_decision)

        return decisions

    def _analyze_symbol(
        self,
        symbol: str,
        market_data_provider: Optional[Any],
        sentiment_data_provider: Optional[Any],
        news_data_provider: Optional[Any],
    ) -> Optional[Decision]:
        """Run full analysis pipeline for a single symbol."""
        sentiment_data = self._get_sentiment_data(symbol, sentiment_data_provider)
        sentiment_result = self.sentiment_agent.aggregate_sentiment(
            symbol, sentiment_data
        )

        news_data = self._get_news_data(symbol, news_data_provider)
        market_data = self._get_market_data(symbol, market_data_provider)
        posts = self._get_posts(symbol, sentiment_data_provider)

        manipulation_input = {
            "sentiment": sentiment_result.to_dict(),
            "price_history": market_data.get("price_history", []),
            "news": news_data,
            "technical_indicators": market_data.get("technical_indicators", {}),
            "posts": posts,
        }
        manipulation_result = self.manipulation_agent.analyze(
            symbol, manipulation_input
        )

        account = self._get_account_info()

        decision = self.trading_agent.make_decision(
            symbol=symbol,
            market_data=market_data,
            sentiment=sentiment_result.to_dict(),
            manipulation=manipulation_result,
            account=account,
        )

        return decision

    def _get_sentiment_data(
        self,
        symbol: str,
        provider: Optional[Any],
    ) -> list[dict]:
        """Get sentiment data for a symbol."""
        if provider is not None and hasattr(provider, "get_sentiment"):
            return provider.get_sentiment(symbol)
        return []

    def _get_news_data(
        self,
        symbol: str,
        provider: Optional[Any],
    ) -> list[dict]:
        """Get news data for a symbol."""
        if provider is not None and hasattr(provider, "get_news"):
            return provider.get_news(symbol)
        return []

    def _get_market_data(
        self,
        symbol: str,
        provider: Optional[Any],
    ) -> dict[str, Any]:
        """Get market data including price and technical indicators."""
        if provider is not None and hasattr(provider, "get_market_data"):
            return provider.get_market_data(symbol)
        return {
            "current_price": 0.0,
            "price_history": [],
            "technical_indicators": {},
        }

    def _get_posts(
        self,
        symbol: str,
        provider: Optional[Any],
    ) -> list[dict]:
        """Get social media posts for a symbol."""
        if provider is not None and hasattr(provider, "get_posts"):
            return provider.get_posts(symbol)
        return []

    def _get_account_info(self) -> dict[str, Any]:
        """Get current account information."""
        return {
            "cash": 0.0,
            "equity": 0.0,
            "positions": [],
        }

    def run_monday_cycle(
        self,
        symbols: list[str],
        account: dict[str, Any],
        market_data_provider: Optional[Any] = None,
        sentiment_data_provider: Optional[Any] = None,
        news_data_provider: Optional[Any] = None,
    ) -> list[dict]:
        """Monday buy cycle - evaluate and execute buys.

        Args:
            symbols: Candidate symbols to evaluate.
            account: Current account state.
            market_data_provider: Service for market data.
            sentiment_data_provider: Service for sentiment data.
            news_data_provider: Service for news data.

        Returns:
            List of execution results.
        """
        results = []

        decisions = self.run_analysis_cycle(
            symbols,
            market_data_provider,
            sentiment_data_provider,
            news_data_provider,
        )

        buy_decisions = [d for d in decisions if d.decision == "BUY"]

        buy_decisions.sort(key=lambda d: d.confidence, reverse=True)

        max_buys = min(
            self.trading_agent.strategy.max_positions,
            len(buy_decisions),
        )

        for decision in buy_decisions[:max_buys]:
            result = self._execute_buy(decision, account)
            results.append(result)

            if result.get("success"):
                self.decision_logger.update_execution(
                    decision.decision_id,
                    executed=True,
                    execution_details=result,
                )

        return results

    def _execute_buy(
        self,
        decision: Decision,
        account: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a buy decision.

        In a real implementation, this would call the trading executor.

        Args:
            decision: Buy decision to execute.
            account: Current account state.

        Returns:
            Execution result dict.
        """
        return {
            "success": False,
            "symbol": decision.symbol,
            "decision_id": decision.decision_id,
            "message": "Execution not implemented - requires trading executor",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def run_friday_cycle(
        self,
        positions: list[dict],
    ) -> list[dict]:
        """Friday sell cycle - close all positions.

        Args:
            positions: List of current open positions.

        Returns:
            List of execution results.
        """
        results = []

        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")

            decision = Decision(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                decision="SELL",
                confidence=1.0,
                inputs={
                    "position": self._sanitize_position(position),
                    "reason": "friday_close",
                },
                reasoning=["Friday close - exiting all positions per strategy"],
                executed=False,
            )

            self.decision_logger.log_decision(decision)

            result = self._execute_sell(decision, position)
            results.append(result)

            if result.get("success"):
                self.decision_logger.update_execution(
                    decision.decision_id,
                    executed=True,
                    execution_details=result,
                )

        return results

    def _execute_sell(
        self,
        decision: Decision,
        position: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a sell decision.

        Args:
            decision: Sell decision to execute.
            position: Position to close.

        Returns:
            Execution result dict.
        """
        return {
            "success": False,
            "symbol": decision.symbol,
            "decision_id": decision.decision_id,
            "position_id": position.get("id"),
            "message": "Execution not implemented - requires trading executor",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def run_signal_check(
        self,
        positions: list[dict],
        market_data_provider: Optional[Any] = None,
        sentiment_data_provider: Optional[Any] = None,
    ) -> list[dict]:
        """Regular signal check - evaluate exits for open positions.

        Called periodically during trading hours to check for exit signals
        (profit target, stop loss, or normalized conditions).

        Args:
            positions: List of current open positions.
            market_data_provider: Service for market data.
            sentiment_data_provider: Service for sentiment data.

        Returns:
            List of exit signals/results.
        """
        results = []

        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")

            market_data = self._get_market_data(symbol, market_data_provider)
            sentiment_data = self._get_sentiment_data(symbol, sentiment_data_provider)
            sentiment_result = self.sentiment_agent.aggregate_sentiment(
                symbol, sentiment_data
            )

            account = {"positions": positions}

            decision = self.trading_agent.make_decision(
                symbol=symbol,
                market_data=market_data,
                sentiment=sentiment_result.to_dict(),
                manipulation={},
                account=account,
            )

            self.decision_logger.log_decision(decision)

            if decision.decision == "SELL":
                result = self._execute_sell(decision, position)
                results.append(result)

                if result.get("success"):
                    self.decision_logger.update_execution(
                        decision.decision_id,
                        executed=True,
                        execution_details=result,
                    )
            else:
                results.append(
                    {
                        "symbol": symbol,
                        "decision": decision.decision,
                        "executed": False,
                        "message": "No exit signal",
                    }
                )

        return results

    def _sanitize_position(self, position: dict[str, Any]) -> dict[str, Any]:
        """Sanitize position data for JSON serialization."""
        result = {}
        for key, value in position.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, date):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def get_analysis_summary(
        self,
        symbol: str,
        market_data_provider: Optional[Any] = None,
        sentiment_data_provider: Optional[Any] = None,
        news_data_provider: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Get comprehensive analysis summary for a symbol.

        Useful for API endpoints and dashboards.

        Args:
            symbol: Stock symbol to analyze.
            market_data_provider: Service for market data.
            sentiment_data_provider: Service for sentiment data.
            news_data_provider: Service for news data.

        Returns:
            Dict with full analysis results.
        """
        sentiment_data = self._get_sentiment_data(symbol, sentiment_data_provider)
        sentiment_result = self.sentiment_agent.aggregate_sentiment(
            symbol, sentiment_data
        )

        news_data = self._get_news_data(symbol, news_data_provider)
        market_data = self._get_market_data(symbol, market_data_provider)
        posts = self._get_posts(symbol, sentiment_data_provider)

        manipulation_input = {
            "sentiment": sentiment_result.to_dict(),
            "price_history": market_data.get("price_history", []),
            "news": news_data,
            "technical_indicators": market_data.get("technical_indicators", {}),
            "posts": posts,
        }
        manipulation_result = self.manipulation_agent.analyze(
            symbol, manipulation_input
        )

        risk_assessment = self.manipulation_agent.get_risk_assessment(
            manipulation_result
        )

        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment": sentiment_result.to_dict(),
            "manipulation": manipulation_result,
            "risk_assessment": risk_assessment,
            "market_data": {
                "current_price": market_data.get("current_price"),
                "technical_indicators": market_data.get("technical_indicators", {}),
            },
        }

    def update_bayesian_priors(
        self,
        historical_outcomes: list[dict],
    ) -> dict[str, float]:
        """Update Bayesian priors from historical trade outcomes.

        Should be called periodically with trade results to improve
        manipulation detection accuracy.

        Args:
            historical_outcomes: List of dicts with:
                - was_manipulation: bool
                - evidence: dict of evidence that was present

        Returns:
            Updated parameters.
        """
        return self.bayesian.update_priors(historical_outcomes)

    def get_decision_statistics(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get statistics on trading decisions.

        Args:
            start: Start of time range.
            end: End of time range.

        Returns:
            Decision statistics.
        """
        return self.decision_logger.get_decision_statistics(start, end)

    def export_decisions(
        self,
        start: datetime,
        end: datetime,
        format: str = "json",
    ) -> str:
        """Export decisions for analysis.

        Args:
            start: Start of time range.
            end: End of time range.
            format: Export format ("json" or "csv").

        Returns:
            Exported data as string.
        """
        return self.decision_logger.export_decisions(start, end, format)

    def halt_trading(self, reason: str) -> None:
        """Manually halt trading.

        Args:
            reason: Reason for halt.
        """
        self.trading_agent.risk_manager.is_halted = True
        self.trading_agent.risk_manager.halt_reason = reason

    def resume_trading(self) -> bool:
        """Resume trading after halt.

        Returns:
            True if trading was resumed.
        """
        return self.trading_agent.resume_trading()

    def is_halted(self) -> tuple[bool, Optional[str]]:
        """Check if trading is halted.

        Returns:
            Tuple of (is_halted, reason).
        """
        return (
            self.trading_agent.risk_manager.is_halted,
            self.trading_agent.risk_manager.halt_reason,
        )

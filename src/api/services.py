"""Service container for lazy-initialized services in Leprechaun API.

Provides a centralized service container that lazily initializes services
on first access, handling cases where credentials may not be configured.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from flask import current_app
from sqlalchemy import text

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Lazy-initialized service container for API routes.

    Services are initialized on first access. Provides configuration
    validation methods to check if services are properly configured.
    """

    _instance: Optional["ServiceContainer"] = None

    def __init__(self):
        self._trading_strategy = None
        self._risk_manager = None
        self._order_executor = None
        self._orchestrator = None
        self._sentiment_agent = None
        self._manipulation_agent = None
        self._notification_service = None
        self._scheduler = None
        self._trading_config = None
        self._is_running = False
        self._last_cycle_time: Optional[datetime] = None

    @classmethod
    def get_instance(cls) -> "ServiceContainer":
        """Get the singleton service container instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def _get_config(self):
        """Get application configuration, handling missing config gracefully."""
        if self._trading_config is not None:
            return self._trading_config

        try:
            from src.utils.config import get_config

            config = get_config()
            self._trading_config = config.trading
            return self._trading_config
        except Exception as e:
            logger.warning("Could not load config: %s", e)
            from src.utils.config import TradingConfig

            self._trading_config = TradingConfig()
            return self._trading_config

    @property
    def trading_strategy(self):
        """Get or create the trading strategy instance."""
        if self._trading_strategy is None:
            from src.core.trading.strategy import TradingStrategy

            self._trading_strategy = TradingStrategy(self._get_config())
        return self._trading_strategy

    @property
    def risk_manager(self):
        """Get or create the risk manager instance."""
        if self._risk_manager is None:
            from src.core.trading.risk import RiskManager

            self._risk_manager = RiskManager(self._get_config())
        return self._risk_manager

    @property
    def order_executor(self):
        """Get or create the order executor instance."""
        if self._order_executor is None:
            from src.core.trading.executor import OrderExecutor

            trading_client = self._create_alpaca_client()
            paper_mode = os.getenv("ALPACA_PAPER", "true").lower() == "true"
            self._order_executor = OrderExecutor(trading_client, paper_mode=paper_mode)
        return self._order_executor

    def _create_alpaca_client(self):
        """Create Alpaca trading client if credentials are available."""
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")

        if not api_key or not api_secret:
            logger.info("Alpaca credentials not configured, using simulated executor")
            return None

        try:
            from alpaca.trading.client import TradingClient

            paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
            return TradingClient(api_key, api_secret, paper=paper)
        except Exception as e:
            logger.warning("Failed to create Alpaca client: %s", e)
            return None

    @property
    def orchestrator(self):
        """Get or create the agent orchestrator instance."""
        if self._orchestrator is None:
            from src.agents.orchestrator import AgentOrchestrator

            db_session = self._get_db_session()
            self._orchestrator = AgentOrchestrator(
                db_session=db_session,
                trading_strategy=self.trading_strategy,
                risk_manager=self.risk_manager,
            )
        return self._orchestrator

    @property
    def sentiment_agent(self):
        """Get or create the sentiment agent instance."""
        if self._sentiment_agent is None:
            from src.agents.sentiment_agent import SentimentAgent

            self._sentiment_agent = SentimentAgent()
        return self._sentiment_agent

    @property
    def manipulation_agent(self):
        """Get or create the manipulation agent instance."""
        if self._manipulation_agent is None:
            from src.agents.manipulation_agent import ManipulationAgent

            self._manipulation_agent = ManipulationAgent()
        return self._manipulation_agent

    @property
    def notification_service(self):
        """Get or create the notification service instance."""
        if self._notification_service is None:
            from src.utils.notifications import (
                NotificationService,
                DiscordNotifier,
                FirebaseNotifier,
            )

            discord = None
            firebase = None

            discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
            if discord_webhook:
                discord_enabled = os.getenv("DISCORD_ENABLED", "true").lower() == "true"
                discord = DiscordNotifier(discord_webhook, enabled=discord_enabled)

            firebase_creds = os.getenv("FIREBASE_CREDENTIALS_PATH")
            firebase_enabled = os.getenv("FIREBASE_ENABLED", "false").lower() == "true"
            if firebase_creds and firebase_enabled:
                firebase = FirebaseNotifier(firebase_creds, enabled=True)

            self._notification_service = NotificationService(
                discord=discord, firebase=firebase
            )
        return self._notification_service

    @property
    def scheduler(self):
        """Get or create the scheduler manager instance."""
        if self._scheduler is None:
            from src.core.scheduler.jobs import create_scheduler

            self._scheduler = create_scheduler(
                register_trading=False,
                register_maintenance=False,
            )
        return self._scheduler

    def _get_db_session(self):
        """Get database session from Flask app context or create one."""
        try:
            if hasattr(current_app, "db_session") and current_app.db_session:
                return current_app.db_session()
        except RuntimeError:
            pass

        try:
            from src.data.models import get_session_factory

            session_factory = get_session_factory()
            return session_factory()
        except Exception as e:
            logger.warning("Could not create database session: %s", e)
            return None

    def is_trading_configured(self) -> bool:
        """Check if trading services are properly configured."""
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        return bool(api_key and api_secret)

    def is_sentiment_configured(self) -> bool:
        """Check if sentiment services are properly configured."""
        reddit_id = os.getenv("REDDIT_CLIENT_ID")
        finnhub_key = os.getenv("FINNHUB_API_KEY")
        return bool(reddit_id or finnhub_key)

    def is_notifications_configured(self) -> bool:
        """Check if notification services are properly configured."""
        discord_url = os.getenv("DISCORD_WEBHOOK_URL")
        firebase_creds = os.getenv("FIREBASE_CREDENTIALS_PATH")
        return bool(discord_url or firebase_creds)

    def is_database_configured(self) -> bool:
        """Check if database is properly configured and accessible."""
        try:
            session = self._get_db_session()
            if session is None:
                return False
            session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @property
    def is_running(self) -> bool:
        """Check if trading bot is currently running."""
        return self._is_running

    @is_running.setter
    def is_running(self, value: bool) -> None:
        """Set trading bot running status."""
        self._is_running = value

    @property
    def last_cycle_time(self) -> Optional[datetime]:
        """Get the last signal check time."""
        return self._last_cycle_time

    @last_cycle_time.setter
    def last_cycle_time(self, value: datetime) -> None:
        """Set the last signal check time."""
        self._last_cycle_time = value

    def start_trading(self) -> dict[str, Any]:
        """Start the trading scheduler."""
        if self._is_running:
            return {"success": False, "message": "Trading is already running"}

        if not self.is_trading_configured():
            return {
                "success": False,
                "message": "Trading not configured - missing Alpaca credentials",
            }

        try:
            self.scheduler.start()
            self._is_running = True
            self._last_cycle_time = datetime.now(timezone.utc)
            return {"success": True, "message": "Trading started"}
        except Exception as e:
            logger.error("Failed to start trading: %s", e)
            return {"success": False, "message": f"Failed to start: {str(e)}"}

    def stop_trading(self) -> dict[str, Any]:
        """Stop the trading scheduler."""
        if not self._is_running:
            return {"success": False, "message": "Trading is not running"}

        try:
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            return {"success": True, "message": "Trading stopped"}
        except Exception as e:
            logger.error("Failed to stop trading: %s", e)
            return {"success": False, "message": f"Failed to stop: {str(e)}"}

    def get_trading_status(self) -> dict[str, Any]:
        """Get current trading status."""
        from src.core.scheduler.calendar import get_market_calendar

        calendar = get_market_calendar()
        is_halted, halt_reason = self.orchestrator.is_halted()

        if is_halted:
            status = "halted"
        elif self._is_running:
            status = "running"
        else:
            status = "stopped"

        positions = []
        try:
            positions = self.order_executor.get_positions()
        except Exception as e:
            logger.warning("Could not get positions: %s", e)

        return {
            "status": status,
            "is_market_open": calendar.is_market_open(),
            "is_trading_day": calendar.is_trading_day(),
            "active_positions": len(positions),
            "last_signal_check": (
                self._last_cycle_time.isoformat() if self._last_cycle_time else None
            ),
            "halt_reason": halt_reason,
            "is_simulated": self.order_executor.is_simulated,
        }


def get_services() -> ServiceContainer:
    """Get the service container instance."""
    return ServiceContainer.get_instance()

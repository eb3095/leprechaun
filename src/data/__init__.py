"""Data access layer module for Leprechaun trading bot.

This module provides:
- Stock universe management (S&P 500 and NASDAQ 100)
- Historical price data fetching from yfinance and Polygon
- Real-time market data from Alpaca
- Repository pattern for database access
- SQLAlchemy ORM models
"""

from src.data.historical import HistoricalDataProvider
from src.data.market import MarketDataProvider
from src.data.models import (
    Alert,
    AlertChannel,
    AlertType,
    Base,
    DatabaseSession,
    DecisionLog,
    DecisionType,
    ExitReason,
    IndexMembership,
    ManipulationScore,
    PortfolioMetric,
    Position,
    PositionStatus,
    PriceHistory,
    SentimentData,
    SentimentSource,
    SignalConfidence,
    SignalType,
    Stock,
    TechnicalIndicator,
    TradingHalt,
    TradingSignal,
    User,
    UserRole,
    create_db_engine,
    get_session_factory,
    init_db,
)
from src.data.repository import (
    AlertRepository,
    BaseRepository,
    PortfolioMetricsRepository,
    PositionRepository,
    PriceHistoryRepository,
    StockRepository,
    TradingSignalRepository,
)
from src.data.universe import (
    NASDAQ100_SYMBOLS,
    SP500_SYMBOLS,
    StockUniverse,
)

__all__ = [
    "Alert",
    "AlertChannel",
    "AlertRepository",
    "AlertType",
    "Base",
    "BaseRepository",
    "DatabaseSession",
    "DecisionLog",
    "DecisionType",
    "ExitReason",
    "HistoricalDataProvider",
    "IndexMembership",
    "ManipulationScore",
    "MarketDataProvider",
    "NASDAQ100_SYMBOLS",
    "PortfolioMetric",
    "PortfolioMetricsRepository",
    "Position",
    "PositionRepository",
    "PositionStatus",
    "PriceHistory",
    "PriceHistoryRepository",
    "SP500_SYMBOLS",
    "SentimentData",
    "SentimentSource",
    "SignalConfidence",
    "SignalType",
    "Stock",
    "StockRepository",
    "StockUniverse",
    "TechnicalIndicator",
    "TradingHalt",
    "TradingSignal",
    "TradingSignalRepository",
    "User",
    "UserRole",
    "create_db_engine",
    "get_session_factory",
    "init_db",
]

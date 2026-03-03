"""
SQLAlchemy ORM models for Leprechaun trading bot.

Uses SQLAlchemy 2.0 style with mapped_column and type annotations.
"""

import enum
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional
from urllib.parse import quote_plus

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


CASCADE_DELETE_ORPHAN = "all, delete-orphan"
FK_STOCKS_ID = "stocks.id"
FK_TRADING_SIGNALS_ID = "trading_signals.id"


class Base(DeclarativeBase):
    """Base class for all models with common serialization methods."""

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Convert model instance to dictionary for JSON serialization."""
        exclude = exclude or set()
        result = {}
        for column in self.__table__.columns:
            if column.name in exclude:
                continue
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                result[column.name] = value.isoformat()
            elif isinstance(value, date):
                result[column.name] = value.isoformat()
            elif isinstance(value, Decimal):
                result[column.name] = float(value)
            elif isinstance(value, enum.Enum):
                result[column.name] = value.value
            else:
                result[column.name] = value
        return result

    def __repr__(self) -> str:
        pk_cols = [col.name for col in self.__table__.primary_key.columns]
        pk_vals = ", ".join(f"{col}={getattr(self, col)!r}" for col in pk_cols)
        return f"<{self.__class__.__name__}({pk_vals})>"


class IndexMembership(enum.Enum):
    """Index membership for stocks."""

    SP500 = "SP500"
    NASDAQ100 = "NASDAQ100"
    BOTH = "BOTH"


class SentimentSource(enum.Enum):
    """Source of sentiment data."""

    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    NEWS = "news"


class SignalType(enum.Enum):
    """Trading signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    AVOID = "AVOID"


class SignalConfidence(enum.Enum):
    """Trading signal confidence levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ExitReason(enum.Enum):
    """Position exit reason."""

    TARGET = "TARGET"
    STOP_LOSS = "STOP_LOSS"
    FRIDAY_CLOSE = "FRIDAY_CLOSE"
    MANUAL = "MANUAL"
    HALT = "HALT"
    NEWS = "NEWS"


class PositionStatus(enum.Enum):
    """Position status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"


class AlertType(enum.Enum):
    """Alert types."""

    TRADE = "trade"
    HALT = "halt"
    RESUME = "resume"
    ERROR = "error"
    SIGNAL = "signal"
    DAILY_SUMMARY = "daily_summary"


class AlertChannel(enum.Enum):
    """Alert delivery channels."""

    DISCORD = "DISCORD"
    PUSH = "PUSH"
    BOTH = "BOTH"


class DecisionType(enum.Enum):
    """Agent decision types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SKIP = "SKIP"


class UserRole(enum.Enum):
    """User roles for API authentication."""

    ADMIN = "admin"
    VIEWER = "viewer"


class Stock(Base):
    """Stock universe (S&P 500 + NASDAQ 100)."""

    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    market_cap: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2))
    index_membership: Mapped[IndexMembership] = mapped_column(
        Enum(IndexMembership), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    price_history: Mapped[list["PriceHistory"]] = relationship(
        back_populates="stock", cascade=CASCADE_DELETE_ORPHAN
    )
    sentiment_data: Mapped[list["SentimentData"]] = relationship(
        back_populates="stock", cascade=CASCADE_DELETE_ORPHAN
    )
    manipulation_scores: Mapped[list["ManipulationScore"]] = relationship(
        back_populates="stock", cascade=CASCADE_DELETE_ORPHAN
    )
    technical_indicators: Mapped[list["TechnicalIndicator"]] = relationship(
        back_populates="stock", cascade=CASCADE_DELETE_ORPHAN
    )
    trading_signals: Mapped[list["TradingSignal"]] = relationship(
        back_populates="stock", cascade=CASCADE_DELETE_ORPHAN
    )
    positions: Mapped[list["Position"]] = relationship(
        back_populates="stock", cascade=CASCADE_DELETE_ORPHAN
    )

    __table_args__ = (Index("idx_active", "is_active"),)


class PriceHistory(Base):
    """Historical price data (cached from yfinance)."""

    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(FK_STOCKS_ID, ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    high: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    low: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    close: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    adj_close: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)

    stock: Mapped["Stock"] = relationship(back_populates="price_history")

    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uk_stock_date"),
        Index("idx_date", "date"),
    )


class SentimentData(Base):
    """Sentiment data aggregated by stock and time."""

    __tablename__ = "sentiment_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(FK_STOCKS_ID, ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    source: Mapped[SentimentSource] = mapped_column(
        Enum(SentimentSource), nullable=False
    )
    sentiment_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    sentiment_volume: Mapped[Optional[int]] = mapped_column(Integer)
    velocity: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    bot_fraction: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    coordination_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    raw_data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    stock: Mapped["Stock"] = relationship(back_populates="sentiment_data")

    __table_args__ = (
        Index("idx_sentiment_stock_time", "stock_id", "timestamp"),
        Index("idx_sentiment_source", "source"),
    )


class ManipulationScore(Base):
    """Manipulation detection scores."""

    __tablename__ = "manipulation_scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(FK_STOCKS_ID, ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    manipulation_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    bayesian_probability: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    divergence_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    has_news_catalyst: Mapped[Optional[bool]] = mapped_column(Boolean)
    triggered_signals: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    stock: Mapped["Stock"] = relationship(back_populates="manipulation_scores")

    __table_args__ = (
        Index("idx_manipulation_stock_time", "stock_id", "timestamp"),
        Index("idx_manipulation_score", "manipulation_score"),
    )


class TechnicalIndicator(Base):
    """Technical indicators calculated daily."""

    __tablename__ = "technical_indicators"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(FK_STOCKS_ID, ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    rsi_14: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    ema_9: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    ema_21: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    ema_50: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    macd_line: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    macd_signal: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    macd_histogram: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    bollinger_upper: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    bollinger_middle: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    bollinger_lower: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    atr_14: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    volume_sma_20: Mapped[Optional[int]] = mapped_column(BigInteger)

    stock: Mapped["Stock"] = relationship(back_populates="technical_indicators")

    __table_args__ = (UniqueConstraint("stock_id", "date", name="uk_tech_stock_date"),)


class TradingSignal(Base):
    """Trading signals generated by the system."""

    __tablename__ = "trading_signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(FK_STOCKS_ID, ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    signal_type: Mapped[SignalType] = mapped_column(Enum(SignalType), nullable=False)
    confidence: Mapped[SignalConfidence] = mapped_column(
        Enum(SignalConfidence), nullable=False
    )
    manipulation_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    rsi: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    sentiment_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    triggered_rules: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    is_executed: Mapped[bool] = mapped_column(Boolean, default=False)

    stock: Mapped["Stock"] = relationship(back_populates="trading_signals")
    entry_positions: Mapped[list["Position"]] = relationship(
        back_populates="entry_signal",
        foreign_keys="Position.entry_signal_id",
    )
    exit_positions: Mapped[list["Position"]] = relationship(
        back_populates="exit_signal",
        foreign_keys="Position.exit_signal_id",
    )

    __table_args__ = (
        Index("idx_signal_stock_time", "stock_id", "timestamp"),
        Index("idx_signal_type", "signal_type"),
        Index("idx_signal_executed", "is_executed", "signal_type"),
    )


class Position(Base):
    """Current and historical positions."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(FK_STOCKS_ID, ondelete="CASCADE"), nullable=False
    )
    alpaca_order_id: Mapped[Optional[str]] = mapped_column(String(64))
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    shares: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    entry_signal_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey(FK_TRADING_SIGNALS_ID, ondelete="SET NULL")
    )
    exit_date: Mapped[Optional[date]] = mapped_column(Date)
    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    exit_signal_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey(FK_TRADING_SIGNALS_ID, ondelete="SET NULL")
    )
    exit_reason: Mapped[Optional[ExitReason]] = mapped_column(Enum(ExitReason))
    realized_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    realized_pnl_percent: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    status: Mapped[PositionStatus] = mapped_column(Enum(PositionStatus), nullable=False)

    stock: Mapped["Stock"] = relationship(back_populates="positions")
    entry_signal: Mapped[Optional["TradingSignal"]] = relationship(
        back_populates="entry_positions",
        foreign_keys=[entry_signal_id],
    )
    exit_signal: Mapped[Optional["TradingSignal"]] = relationship(
        back_populates="exit_positions",
        foreign_keys=[exit_signal_id],
    )

    __table_args__ = (
        Index("idx_position_status", "status"),
        Index("idx_position_entry_date", "entry_date"),
        Index("idx_position_stock_status", "stock_id", "status"),
    )

    @property
    def is_open(self) -> bool:
        """Check if position is currently open."""
        return self.status == PositionStatus.OPEN

    def calculate_pnl(self, current_price: Decimal) -> tuple[Decimal, Decimal]:
        """Calculate unrealized P&L given current price."""
        pnl = (current_price - self.entry_price) * self.shares
        pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
        return pnl, pnl_percent


class PortfolioMetric(Base):
    """Daily portfolio snapshots."""

    __tablename__ = "portfolio_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    total_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2))
    cash: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2))
    positions_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2))
    daily_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    daily_pnl_percent: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    weekly_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    weekly_pnl_percent: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    total_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2))
    total_pnl_percent: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    win_count: Mapped[Optional[int]] = mapped_column(Integer)
    loss_count: Mapped[Optional[int]] = mapped_column(Integer)
    win_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    sharpe_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    max_drawdown: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))

    @property
    def total_trades(self) -> int:
        """Calculate total number of trades."""
        return (self.win_count or 0) + (self.loss_count or 0)


class TradingHalt(Base):
    """Trading halt status."""

    __tablename__ = "trading_halts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    reason: Mapped[Optional[str]] = mapped_column(String(255))
    daily_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    weekly_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    sandbox_test_required: Mapped[bool] = mapped_column(Boolean, default=True)
    sandbox_test_passed: Mapped[Optional[bool]] = mapped_column(Boolean)
    resumed_by: Mapped[Optional[str]] = mapped_column(String(100))

    __table_args__ = (Index("idx_halt_active", "is_active"),)


class Alert(Base):
    """Alerts sent via Discord and push notifications."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    alert_type: Mapped[AlertType] = mapped_column(Enum(AlertType), nullable=False)
    channel: Mapped[AlertChannel] = mapped_column(Enum(AlertChannel), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    message: Mapped[Optional[str]] = mapped_column(Text)
    alert_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column("metadata", JSON)
    sent_successfully: Mapped[Optional[bool]] = mapped_column(Boolean)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("idx_alert_timestamp", "timestamp"),
        Index("idx_alert_type", "alert_type"),
    )


class DecisionLog(Base):
    """Audit trail for agent decisions."""

    __tablename__ = "decision_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    decision: Mapped[DecisionType] = mapped_column(Enum(DecisionType), nullable=False)
    confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    inputs: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    reasoning: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    executed: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    __table_args__ = (
        Index("idx_decision_timestamp", "timestamp"),
        Index("idx_decision_symbol", "symbol"),
    )


class User(Base):
    """User accounts for API authentication."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole), nullable=False, default=UserRole.VIEWER
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Override to always exclude password_hash."""
        exclude = (exclude or set()) | {"password_hash"}
        return super().to_dict(exclude=exclude)


def _get_database_url() -> str:
    """Get database URL from config module or environment variables.

    Tries to use the config module first for consistency.
    Falls back to environment variables for testing or standalone use.
    """
    try:
        from src.utils.config import get_config

        config = get_config()
        return config.database.url
    except Exception:
        pass

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    user = os.getenv("DB_USER", "leprechaun")
    password = os.getenv("DB_PASSWORD")
    database = os.getenv("DB_NAME", "leprechaun")

    if password is None:
        env = os.getenv("ENV", "development")
        if env == "production":
            raise ValueError("DB_PASSWORD is required in production")
        password = ""

    return (
        f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}"
    )


def create_db_engine(echo: bool = False):
    """Create SQLAlchemy engine with connection pooling.

    Note: echo is forced to False in production to prevent credential leakage.
    """
    url = _get_database_url()
    if echo and os.getenv("ENV") == "production":
        echo = False
    return create_engine(
        url,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def get_session_factory(engine=None):
    """Create session factory for database operations."""
    if engine is None:
        engine = create_db_engine()
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine=None, drop_existing: bool = False):
    """Initialize database tables.

    Args:
        engine: SQLAlchemy engine (created if not provided).
        drop_existing: If True, drops all tables first. Requires LEPRECHAUN_DROP_DB=confirm
                       environment variable in production as a safety measure.

    Raises:
        ValueError: If drop_existing=True in production without confirmation env var.
    """
    if engine is None:
        engine = create_db_engine()

    if drop_existing:
        env = os.getenv("ENV", "development")
        if env == "production" and os.getenv("LEPRECHAUN_DROP_DB") != "confirm":
            raise ValueError(
                "drop_existing=True requires LEPRECHAUN_DROP_DB=confirm in production"
            )
        Base.metadata.drop_all(engine)

    Base.metadata.create_all(engine)


class DatabaseSession:
    """Context manager for database sessions."""

    def __init__(self, session_factory=None):
        self._session_factory = session_factory or get_session_factory()
        self._session: Optional[Session] = None

    def __enter__(self) -> Session:
        self._session = self._session_factory()
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._session is None:
            return False

        try:
            if exc_type is not None:
                self._session.rollback()
            else:
                self._session.commit()
        finally:
            self._session.close()
            self._session = None
        return False

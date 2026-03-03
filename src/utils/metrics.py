"""Prometheus metrics definitions for Leprechaun trading bot."""

from prometheus_client import Counter, Gauge, Histogram, Info

TRADE_ACTION_LABELS = ["action", "result"]
SYMBOL_LABELS = ["symbol"]
API_LABELS = ["endpoint", "method"]

trades_total = Counter(
    "leprechaun_trades_total",
    "Total number of trades executed",
    TRADE_ACTION_LABELS,
)

position_value = Gauge(
    "leprechaun_position_value",
    "Current value of a position in dollars",
    SYMBOL_LABELS,
)

portfolio_value = Gauge(
    "leprechaun_portfolio_value",
    "Total portfolio value including cash",
)

cash_balance = Gauge(
    "leprechaun_cash_balance",
    "Available cash balance",
)

sentiment_processing_seconds = Histogram(
    "leprechaun_sentiment_processing_seconds",
    "Time spent processing sentiment data",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

manipulation_score = Gauge(
    "leprechaun_manipulation_score",
    "Current manipulation score for a symbol (0-1)",
    SYMBOL_LABELS,
)

sentiment_score = Gauge(
    "leprechaun_sentiment_score",
    "Current sentiment score for a symbol (-1 to 1)",
    SYMBOL_LABELS,
)

api_request_duration = Histogram(
    "leprechaun_api_request_duration_seconds",
    "Duration of API requests",
    API_LABELS,
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

api_requests_total = Counter(
    "leprechaun_api_requests_total",
    "Total number of API requests",
    API_LABELS + ["status_code"],
)

active_positions = Gauge(
    "leprechaun_active_positions",
    "Number of currently open positions",
)

trading_halted = Gauge(
    "leprechaun_trading_halted",
    "Whether trading is currently halted (1=halted, 0=active)",
)

daily_pnl = Gauge(
    "leprechaun_daily_pnl",
    "Today's profit/loss in dollars",
)

daily_pnl_percent = Gauge(
    "leprechaun_daily_pnl_percent",
    "Today's profit/loss as percentage",
)

weekly_pnl = Gauge(
    "leprechaun_weekly_pnl",
    "This week's profit/loss in dollars",
)

weekly_pnl_percent = Gauge(
    "leprechaun_weekly_pnl_percent",
    "This week's profit/loss as percentage",
)

signals_generated = Counter(
    "leprechaun_signals_generated_total",
    "Total number of trading signals generated",
    ["signal_type", "confidence"],
)

data_source_errors = Counter(
    "leprechaun_data_source_errors_total",
    "Errors from external data sources",
    ["source"],
)

data_source_latency = Histogram(
    "leprechaun_data_source_latency_seconds",
    "Latency of external data source requests",
    ["source"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

alerts_sent = Counter(
    "leprechaun_alerts_sent_total",
    "Total number of alerts sent",
    ["channel", "alert_type"],
)

app_info = Info(
    "leprechaun_app",
    "Application information",
)


def record_trade(action: str, result: str) -> None:
    """Record a trade execution.

    Args:
        action: Trade action ('buy' or 'sell').
        result: Trade result ('success', 'failed', 'partial').
    """
    trades_total.labels(action=action, result=result).inc()


def update_position_value(symbol: str, value: float) -> None:
    """Update the current value of a position.

    Args:
        symbol: Stock symbol.
        value: Current position value in dollars.
    """
    position_value.labels(symbol=symbol).set(value)


def clear_position(symbol: str) -> None:
    """Remove position value metric when position is closed.

    Args:
        symbol: Stock symbol.
    """
    try:
        position_value.remove(symbol)
    except KeyError:
        pass


def update_portfolio(total_value: float, cash: float) -> None:
    """Update portfolio metrics.

    Args:
        total_value: Total portfolio value including positions.
        cash: Available cash balance.
    """
    portfolio_value.set(total_value)
    cash_balance.set(cash)


def update_manipulation_score(symbol: str, score: float) -> None:
    """Update manipulation score for a symbol.

    Args:
        symbol: Stock symbol.
        score: Manipulation score (0-1).
    """
    manipulation_score.labels(symbol=symbol).set(score)


def update_sentiment_score(symbol: str, score: float) -> None:
    """Update sentiment score for a symbol.

    Args:
        symbol: Stock symbol.
        score: Sentiment score (-1 to 1).
    """
    sentiment_score.labels(symbol=symbol).set(score)


def set_trading_halted(halted: bool) -> None:
    """Set the trading halted status.

    Args:
        halted: Whether trading is halted.
    """
    trading_halted.set(1 if halted else 0)


def update_pnl(
    daily: float,
    daily_percent: float,
    weekly: float,
    weekly_percent: float,
) -> None:
    """Update P&L metrics.

    Args:
        daily: Today's P&L in dollars.
        daily_percent: Today's P&L as percentage.
        weekly: This week's P&L in dollars.
        weekly_percent: This week's P&L as percentage.
    """
    daily_pnl.set(daily)
    daily_pnl_percent.set(daily_percent)
    weekly_pnl.set(weekly)
    weekly_pnl_percent.set(weekly_percent)


def record_signal(signal_type: str, confidence: str) -> None:
    """Record a generated trading signal.

    Args:
        signal_type: Type of signal ('buy', 'sell', 'hold').
        confidence: Confidence level ('high', 'medium', 'low').
    """
    signals_generated.labels(signal_type=signal_type, confidence=confidence).inc()


def record_data_source_error(source: str) -> None:
    """Record an error from a data source.

    Args:
        source: Name of the data source ('reddit', 'finnhub', 'stocktwits', etc).
    """
    data_source_errors.labels(source=source).inc()


def record_alert(channel: str, alert_type: str) -> None:
    """Record an alert that was sent.

    Args:
        channel: Notification channel ('discord', 'push').
        alert_type: Type of alert ('trade', 'halt', 'error', etc).
    """
    alerts_sent.labels(channel=channel, alert_type=alert_type).inc()


def set_app_info(version: str, env: str) -> None:
    """Set application information.

    Args:
        version: Application version string.
        env: Environment name ('development', 'production', etc).
    """
    app_info.info({"version": version, "environment": env})

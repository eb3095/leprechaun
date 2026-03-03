"""
Signal generation based on technical indicators for Leprechaun trading bot.

Combines technical analysis with sentiment data to generate entry/exit signals
for the contrarian trading strategy.
"""

from typing import Any

import pandas as pd


PROFIT_TARGET_PERCENT = 2.5
STOP_LOSS_PERCENT = 1.25


def is_oversold(rsi: float, threshold: float = 35.0) -> bool:
    """Check if RSI indicates oversold conditions.

    Per the Leprechaun strategy, RSI < 35 suggests the stock may be
    artificially depressed and could be a contrarian buy opportunity.

    Args:
        rsi: Current RSI value (0-100).
        threshold: RSI threshold for oversold (default 35).

    Returns:
        True if RSI is below the threshold.
    """
    if pd.isna(rsi):
        return False
    return float(rsi) < threshold


def is_below_ema(price: float, ema: float) -> bool:
    """Check if current price is below the EMA.

    Price below the 21-day EMA combined with other signals may indicate
    a buying opportunity in the contrarian strategy.

    Args:
        price: Current stock price.
        ema: Current EMA value.

    Returns:
        True if price is below EMA.
    """
    if pd.isna(price) or pd.isna(ema):
        return False
    return float(price) < float(ema)


def is_bollinger_squeeze(price: float, lower_band: float) -> bool:
    """Check if price is at or below the lower Bollinger Band.

    Price touching or falling below the lower band indicates the stock
    is trading at the lower end of its recent range - potential oversold.

    Args:
        price: Current stock price.
        lower_band: Current lower Bollinger Band value.

    Returns:
        True if price is at or below the lower band.
    """
    if pd.isna(price) or pd.isna(lower_band):
        return False
    return float(price) <= float(lower_band)


def detect_macd_crossover(
    macd_line: pd.Series,
    signal_line: pd.Series,
) -> pd.Series:
    """Detect MACD crossover signals.

    A bullish crossover occurs when the MACD line crosses above the signal line.
    A bearish crossover occurs when the MACD line crosses below the signal line.

    Args:
        macd_line: Series of MACD line values.
        signal_line: Series of signal line values.

    Returns:
        Series with values:
        - 1: Bullish crossover (MACD crosses above signal)
        - -1: Bearish crossover (MACD crosses below signal)
        - 0: No crossover
    """
    result = pd.Series(0, index=macd_line.index, dtype=int)

    if len(macd_line) < 2:
        return result

    macd_above = (macd_line > signal_line).astype(bool)
    macd_above_prev = macd_above.shift(1).fillna(False).astype(bool)

    bullish_cross = (~macd_above_prev) & macd_above
    bearish_cross = macd_above_prev & (~macd_above)

    result = result.where(~bullish_cross, 1)
    result = result.where(~bearish_cross, -1)
    result.iloc[0] = 0

    return result.astype(int)


def _evaluate_technical_signals(
    indicators: dict[str, Any],
) -> tuple[int, int, list[str]]:
    """Evaluate technical indicators for entry signals.

    Returns:
        Tuple of (buy_signals, sell_signals, reasons).
    """
    buy_signals = 0
    sell_signals = 0
    reasons: list[str] = []

    rsi = indicators.get("rsi")
    price = indicators.get("price")
    ema_21 = indicators.get("ema_21")
    bb_lower = indicators.get("bb_lower")
    macd_crossover = indicators.get("macd_crossover", 0)

    if rsi is not None and is_oversold(rsi, threshold=35.0):
        buy_signals += 2
        reasons.append(f"RSI oversold at {rsi:.1f}")

    if price is not None and ema_21 is not None and is_below_ema(price, ema_21):
        buy_signals += 1
        reasons.append(f"Price ({price:.2f}) below 21-EMA ({ema_21:.2f})")

    if price is not None and bb_lower is not None and is_bollinger_squeeze(price, bb_lower):
        buy_signals += 1
        reasons.append(f"Price at lower Bollinger Band ({bb_lower:.2f})")

    if macd_crossover == 1:
        buy_signals += 1
        reasons.append("Bullish MACD crossover")
    elif macd_crossover == -1:
        sell_signals += 1
        reasons.append("Bearish MACD crossover")

    return buy_signals, sell_signals, reasons


def _evaluate_sentiment_signals(
    sentiment_score: float,
    manipulation_score: float,
) -> tuple[int, list[str]]:
    """Evaluate sentiment and manipulation scores for entry signals.

    Returns:
        Tuple of (buy_signals, reasons).
    """
    buy_signals = 0
    reasons: list[str] = []

    if not pd.isna(sentiment_score) and sentiment_score < -0.4:
        buy_signals += 1
        reasons.append(f"Strong negative sentiment ({sentiment_score:.2f})")

    if not pd.isna(manipulation_score) and manipulation_score > 0.5:
        buy_signals += 2
        confidence_desc = "high" if manipulation_score > 0.7 else "elevated"
        reasons.append(f"Manipulation score {confidence_desc} ({manipulation_score:.2f})")

        if manipulation_score > 0.7:
            buy_signals += 1

    return buy_signals, reasons


def _determine_signal(
    buy_signals: int,
    sell_signals: int,
    reasons: list[str],
) -> dict[str, Any]:
    """Determine final signal based on accumulated scores.

    Returns:
        Dict with signal, confidence, and reasons.
    """
    if buy_signals >= 5:
        confidence = "HIGH" if buy_signals >= 7 else "MEDIUM"
        return {"signal": "BUY", "confidence": confidence, "reasons": reasons}

    if sell_signals >= 2:
        confidence = "MEDIUM" if sell_signals >= 3 else "LOW"
        if not reasons:
            reasons.append("Technical deterioration")
        return {"signal": "SELL", "confidence": confidence, "reasons": reasons}

    if not reasons:
        reasons.append("Insufficient signals for entry")
    return {"signal": "HOLD", "confidence": "LOW", "reasons": reasons}


def generate_entry_signal(
    indicators: dict[str, Any],
    sentiment_score: float,
    manipulation_score: float,
) -> dict[str, Any]:
    """Generate entry signal combining technical and sentiment analysis.

    The Leprechaun strategy looks for:
    - Oversold technical conditions (RSI < 35, price below EMA, Bollinger squeeze)
    - Negative sentiment without news catalyst (manipulation_score > 0.5)
    - Sentiment score < -0.4 indicating potential overreaction

    Args:
        indicators: Dict containing technical indicator values:
            - rsi: Current RSI value
            - price: Current price
            - ema_21: 21-period EMA
            - bb_lower: Lower Bollinger Band
            - macd_crossover: Latest MACD crossover signal (-1, 0, 1)
        sentiment_score: Aggregated sentiment score (-1.0 to 1.0).
        manipulation_score: Probability of manipulation (0.0 to 1.0).

    Returns:
        Dict with:
        - signal: "BUY", "SELL", or "HOLD"
        - confidence: "HIGH", "MEDIUM", or "LOW"
        - reasons: List of strings explaining the signal
    """
    tech_buy, tech_sell, tech_reasons = _evaluate_technical_signals(indicators)
    sent_buy, sent_reasons = _evaluate_sentiment_signals(sentiment_score, manipulation_score)

    total_buy = tech_buy + sent_buy
    all_reasons = tech_reasons + sent_reasons

    return _determine_signal(total_buy, tech_sell, all_reasons)


def generate_exit_signal(
    position: dict[str, Any],
    current_price: float,
    indicators: dict[str, Any],
) -> dict[str, Any]:
    """Generate exit signal for an open position.

    Exit conditions (in priority order):
    1. Stop loss hit: Price dropped 1.25% from entry
    2. Profit target hit: Price gained 2.5% from entry
    3. RSI normalized: RSI > 50 (momentum shifted)

    Note: Friday close exits are handled separately by the scheduler.

    Args:
        position: Dict containing position details:
            - entry_price: Price at which position was opened
            - entry_date: Date position was opened (optional)
        current_price: Current stock price.
        indicators: Dict containing current indicator values:
            - rsi: Current RSI value

    Returns:
        Dict with:
        - should_exit: Boolean indicating if position should be closed
        - reason: String explaining the exit decision
    """
    entry_price = position.get("entry_price")
    if entry_price is None or pd.isna(entry_price):
        return {
            "should_exit": False,
            "reason": "Invalid position: missing entry price",
        }

    entry_price = float(entry_price)
    current_price = float(current_price)

    pnl_percent = ((current_price - entry_price) / entry_price) * 100

    if pnl_percent <= -STOP_LOSS_PERCENT:
        return {
            "should_exit": True,
            "reason": f"Stop loss triggered at {pnl_percent:.2f}% loss",
        }

    if pnl_percent >= PROFIT_TARGET_PERCENT:
        return {
            "should_exit": True,
            "reason": f"Profit target reached at {pnl_percent:.2f}% gain",
        }

    rsi = indicators.get("rsi")
    if rsi is not None and not pd.isna(rsi) and float(rsi) > 50:
        if pnl_percent > 0:
            return {
                "should_exit": True,
                "reason": f"RSI normalized ({rsi:.1f}) with {pnl_percent:.2f}% profit",
            }

    return {
        "should_exit": False,
        "reason": f"Hold position (P&L: {pnl_percent:+.2f}%)",
    }


def calculate_position_size(
    account_value: float,
    entry_price: float,
    risk_per_trade: float = 0.01,
    stop_loss_percent: float = STOP_LOSS_PERCENT,
) -> int:
    """Calculate position size based on account risk management.

    Limits position size so that hitting the stop loss results in
    losing only the specified percentage of the account.

    Args:
        account_value: Total account value in dollars.
        entry_price: Expected entry price per share.
        risk_per_trade: Maximum account risk per trade (default 1%).
        stop_loss_percent: Stop loss percentage (default 1.25%).

    Returns:
        Number of shares to purchase (whole shares only).
    """
    max_loss_dollars = account_value * risk_per_trade

    loss_per_share = entry_price * (stop_loss_percent / 100)

    if loss_per_share <= 0:
        return 0

    shares = int(max_loss_dollars / loss_per_share)

    return max(0, shares)

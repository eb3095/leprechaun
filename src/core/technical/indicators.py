"""
Technical indicator calculations for Leprechaun trading bot.

Implements RSI, EMA, MACD, Bollinger Bands, ATR, and Volume SMA.
Uses pandas and numpy for efficient vectorized calculations.
"""

import numpy as np
import pandas as pd


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI measures the speed and magnitude of price changes to identify
    overbought (>70) or oversold (<30) conditions.

    Args:
        prices: Series of closing prices.
        period: Lookback period (default 14).

    Returns:
        Series of RSI values (0-100 scale).
    """
    if len(prices) < period + 1:
        return pd.Series(np.nan, index=prices.index)

    delta = prices.diff()

    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    avg_gain = gains.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(span=period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)

    return rsi


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    EMA gives more weight to recent prices, making it more responsive
    to new information than a simple moving average.

    Args:
        prices: Series of closing prices.
        period: EMA period (common values: 9, 21, 50).

    Returns:
        Series of EMA values.
    """
    if len(prices) < period:
        return pd.Series(np.nan, index=prices.index)

    return prices.ewm(span=period, min_periods=period, adjust=False).mean()


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD).

    MACD shows the relationship between two EMAs. Crossovers between
    the MACD line and signal line indicate momentum shifts.

    Args:
        prices: Series of closing prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    if len(prices) < slow:
        nan_series = pd.Series(np.nan, index=prices.index)
        return nan_series.copy(), nan_series.copy(), nan_series.copy()

    ema_fast = prices.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow

    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    set at a specified number of standard deviations above/below the middle.
    Price touching the lower band may indicate oversold conditions.

    Args:
        prices: Series of closing prices.
        period: SMA period (default 20).
        std_dev: Number of standard deviations (default 2.0).

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    if len(prices) < period:
        nan_series = pd.Series(np.nan, index=prices.index)
        return nan_series.copy(), nan_series.copy(), nan_series.copy()

    middle_band = prices.rolling(window=period, min_periods=period).mean()
    rolling_std = prices.rolling(window=period, min_periods=period).std()

    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)

    return upper_band, middle_band, lower_band


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range of
    an asset price for that period. Useful for position sizing and
    stop-loss placement.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: ATR period (default 14).

    Returns:
        Series of ATR values.
    """
    if len(close) < 2:
        return pd.Series(np.nan, index=close.index)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(span=period, min_periods=period, adjust=False).mean()

    return atr


def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average of volume.

    Volume SMA helps identify unusual trading activity. Volume significantly
    above the SMA may indicate strong conviction in price movement.

    Args:
        volume: Series of trading volume.
        period: SMA period (default 20).

    Returns:
        Series of volume SMA values.
    """
    if len(volume) < period:
        return pd.Series(np.nan, index=volume.index)

    return volume.rolling(window=period, min_periods=period).mean()


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for OHLCV data.

    Takes a DataFrame with OHLCV columns and adds all indicator columns.
    Expects columns: 'open', 'high', 'low', 'close', 'volume' (case-insensitive).

    Args:
        df: DataFrame with OHLCV data.

    Returns:
        DataFrame with original data plus indicator columns:
        - rsi_14: 14-period RSI
        - ema_9, ema_21, ema_50: Exponential moving averages
        - macd_line, macd_signal, macd_histogram: MACD components
        - bb_upper, bb_middle, bb_lower: Bollinger Bands
        - atr_14: 14-period ATR
        - volume_sma_20: 20-period volume SMA
    """
    result = df.copy()

    col_map = {col.lower(): col for col in df.columns}

    close = result[col_map.get("close", "close")]
    high = result[col_map.get("high", "high")]
    low = result[col_map.get("low", "low")]
    volume = result[col_map.get("volume", "volume")]

    result["rsi_14"] = calculate_rsi(close, period=14)

    result["ema_9"] = calculate_ema(close, period=9)
    result["ema_21"] = calculate_ema(close, period=21)
    result["ema_50"] = calculate_ema(close, period=50)

    macd_line, macd_signal, macd_histogram = calculate_macd(close)
    result["macd_line"] = macd_line
    result["macd_signal"] = macd_signal
    result["macd_histogram"] = macd_histogram

    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
    result["bb_upper"] = bb_upper
    result["bb_middle"] = bb_middle
    result["bb_lower"] = bb_lower

    result["atr_14"] = calculate_atr(high, low, close, period=14)

    result["volume_sma_20"] = calculate_volume_sma(volume, period=20)

    return result

"""NYSE market calendar utilities for Leprechaun trading bot.

Uses pandas_market_calendars for accurate NYSE trading calendar information
including holidays, early closes, and trading hours.
"""

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal
import pytz


class MarketCalendar:
    """NYSE market calendar utilities.

    Provides methods to check market hours, trading days, and get market
    open/close times. Caches calendar data for performance.
    """

    NYSE_TIMEZONE = pytz.timezone("US/Eastern")
    MARKET_OPEN_TIME = time(9, 30)
    MARKET_CLOSE_TIME = time(16, 0)
    CALENDAR_CACHE_YEARS = 2

    def __init__(self):
        self._calendar = mcal.get_calendar("NYSE")
        self._schedule_cache: dict[int, pd.DataFrame] = {}

    def _get_schedule(self, year: int) -> pd.DataFrame:
        """Get cached schedule for a given year."""
        if year not in self._schedule_cache:
            start = date(year, 1, 1)
            end = date(year, 12, 31)
            self._schedule_cache[year] = self._calendar.schedule(
                start_date=start.isoformat(), end_date=end.isoformat()
            )
        return self._schedule_cache[year]

    def _get_schedule_for_range(self, start: date, end: date) -> pd.DataFrame:
        """Get schedule covering a date range, potentially across years."""
        years = range(start.year, end.year + 1)
        schedules = [self._get_schedule(y) for y in years]
        combined = pd.concat(schedules)
        mask = (combined.index.date >= start) & (combined.index.date <= end)
        return combined[mask]

    def _now_eastern(self) -> datetime:
        """Get current time in US/Eastern timezone."""
        return datetime.now(self.NYSE_TIMEZONE)

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if the market is currently open.

        Args:
            dt: Datetime to check. If None, uses current time.

        Returns:
            True if market is open at the given time.
        """
        if dt is None:
            dt = self._now_eastern()
        elif dt.tzinfo is None:
            dt = self.NYSE_TIMEZONE.localize(dt)
        else:
            dt = dt.astimezone(self.NYSE_TIMEZONE)

        check_date = dt.date()
        if not self.is_trading_day(check_date):
            return False

        schedule = self._get_schedule(check_date.year)
        date_str = check_date.isoformat()

        if date_str not in schedule.index.strftime("%Y-%m-%d").values:
            return False

        day_schedule = schedule[schedule.index.strftime("%Y-%m-%d") == date_str]
        if day_schedule.empty:
            return False

        market_open = day_schedule["market_open"].iloc[0]
        market_close = day_schedule["market_close"].iloc[0]

        dt_utc = dt.astimezone(pytz.UTC)
        return market_open <= dt_utc <= market_close

    def is_trading_day(self, check_date: Optional[date] = None) -> bool:
        """Check if a given date is a trading day.

        Args:
            check_date: Date to check. If None, uses current date.

        Returns:
            True if the date is a trading day (not weekend, not holiday).
        """
        if check_date is None:
            check_date = self._now_eastern().date()

        if check_date.weekday() >= 5:
            return False

        schedule = self._get_schedule(check_date.year)
        trading_dates = schedule.index.date
        return check_date in trading_dates

    def get_next_trading_day(self, from_date: Optional[date] = None) -> date:
        """Get the next trading day from a given date.

        Args:
            from_date: Starting date. If None, uses current date.

        Returns:
            The next trading day after from_date.
        """
        if from_date is None:
            from_date = self._now_eastern().date()

        check_date = from_date + timedelta(days=1)
        max_days = 14

        for _ in range(max_days):
            if self.is_trading_day(check_date):
                return check_date
            check_date += timedelta(days=1)

        valid_days = self._calendar.valid_days(
            start_date=from_date.isoformat(),
            end_date=(from_date + timedelta(days=30)).isoformat(),
        )
        for day in valid_days:
            if day.date() > from_date:
                return day.date()

        raise ValueError(f"No trading day found within 30 days of {from_date}")

    def get_previous_trading_day(self, from_date: Optional[date] = None) -> date:
        """Get the previous trading day from a given date.

        Args:
            from_date: Starting date. If None, uses current date.

        Returns:
            The previous trading day before from_date.
        """
        if from_date is None:
            from_date = self._now_eastern().date()

        check_date = from_date - timedelta(days=1)
        max_days = 14

        for _ in range(max_days):
            if self.is_trading_day(check_date):
                return check_date
            check_date -= timedelta(days=1)

        raise ValueError(f"No trading day found within 14 days before {from_date}")

    def get_market_open_time(self, check_date: Optional[date] = None) -> datetime:
        """Get market open time for a given date.

        Args:
            check_date: Date to get open time for. If None, uses current date.

        Returns:
            Market open datetime in US/Eastern timezone.

        Raises:
            ValueError: If the date is not a trading day.
        """
        if check_date is None:
            check_date = self._now_eastern().date()

        if not self.is_trading_day(check_date):
            raise ValueError(f"{check_date} is not a trading day")

        schedule = self._get_schedule(check_date.year)
        date_str = check_date.isoformat()
        day_schedule = schedule[schedule.index.strftime("%Y-%m-%d") == date_str]

        if day_schedule.empty:
            raise ValueError(f"No schedule found for {check_date}")

        market_open_utc = day_schedule["market_open"].iloc[0]
        return market_open_utc.astimezone(self.NYSE_TIMEZONE)

    def get_market_close_time(self, check_date: Optional[date] = None) -> datetime:
        """Get market close time for a given date.

        Handles early close days (day before holidays, etc.).

        Args:
            check_date: Date to get close time for. If None, uses current date.

        Returns:
            Market close datetime in US/Eastern timezone.

        Raises:
            ValueError: If the date is not a trading day.
        """
        if check_date is None:
            check_date = self._now_eastern().date()

        if not self.is_trading_day(check_date):
            raise ValueError(f"{check_date} is not a trading day")

        schedule = self._get_schedule(check_date.year)
        date_str = check_date.isoformat()
        day_schedule = schedule[schedule.index.strftime("%Y-%m-%d") == date_str]

        if day_schedule.empty:
            raise ValueError(f"No schedule found for {check_date}")

        market_close_utc = day_schedule["market_close"].iloc[0]
        return market_close_utc.astimezone(self.NYSE_TIMEZONE)

    def get_trading_days_between(
        self, start: date, end: date, inclusive: bool = True
    ) -> list[date]:
        """Get all trading days in a date range.

        Args:
            start: Start date of range.
            end: End date of range.
            inclusive: If True, includes start and end if they are trading days.

        Returns:
            List of trading days in the range.
        """
        if start > end:
            return []

        valid_days = self._calendar.valid_days(
            start_date=start.isoformat(), end_date=end.isoformat()
        )
        result = [d.date() for d in valid_days]

        if not inclusive:
            result = [d for d in result if d != start and d != end]

        return result

    def is_monday(self, check_date: Optional[date] = None) -> bool:
        """Check if a trading day is the first trading day of the week.

        Returns True for Monday, or Tuesday if Monday is a holiday.

        Args:
            check_date: Date to check. If None, uses current date.

        Returns:
            True if this is the first trading day of the week.
        """
        if check_date is None:
            check_date = self._now_eastern().date()

        if not self.is_trading_day(check_date):
            return False

        if check_date.weekday() == 0:
            return True

        if check_date.weekday() == 1:
            monday = check_date - timedelta(days=1)
            return not self.is_trading_day(monday)

        return False

    def is_friday(self, check_date: Optional[date] = None) -> bool:
        """Check if a trading day is the last trading day of the week.

        Returns True for Friday, or Thursday if Friday is a holiday.

        Args:
            check_date: Date to check. If None, uses current date.

        Returns:
            True if this is the last trading day of the week.
        """
        if check_date is None:
            check_date = self._now_eastern().date()

        if not self.is_trading_day(check_date):
            return False

        if check_date.weekday() == 4:
            return True

        if check_date.weekday() == 3:
            friday = check_date + timedelta(days=1)
            return not self.is_trading_day(friday)

        return False

    def is_early_close(self, check_date: Optional[date] = None) -> bool:
        """Check if a trading day has an early close.

        Early close days typically close at 1:00 PM ET.

        Args:
            check_date: Date to check. If None, uses current date.

        Returns:
            True if this is an early close day.
        """
        if check_date is None:
            check_date = self._now_eastern().date()

        if not self.is_trading_day(check_date):
            return False

        close_time = self.get_market_close_time(check_date)
        return close_time.time() < self.MARKET_CLOSE_TIME

    def minutes_until_market_open(self, dt: Optional[datetime] = None) -> int:
        """Get minutes until market opens.

        Args:
            dt: Datetime to calculate from. If None, uses current time.

        Returns:
            Minutes until market opens. Negative if market is already open.
        """
        if dt is None:
            dt = self._now_eastern()
        elif dt.tzinfo is None:
            dt = self.NYSE_TIMEZONE.localize(dt)
        else:
            dt = dt.astimezone(self.NYSE_TIMEZONE)

        check_date = dt.date()
        if not self.is_trading_day(check_date):
            check_date = self.get_next_trading_day(check_date)

        open_time = self.get_market_open_time(check_date)
        delta = open_time - dt
        return int(delta.total_seconds() / 60)

    def minutes_until_market_close(self, dt: Optional[datetime] = None) -> int:
        """Get minutes until market closes.

        Args:
            dt: Datetime to calculate from. If None, uses current time.

        Returns:
            Minutes until market closes. Negative if market is closed.
        """
        if dt is None:
            dt = self._now_eastern()
        elif dt.tzinfo is None:
            dt = self.NYSE_TIMEZONE.localize(dt)
        else:
            dt = dt.astimezone(self.NYSE_TIMEZONE)

        check_date = dt.date()
        if not self.is_trading_day(check_date):
            return -1

        close_time = self.get_market_close_time(check_date)
        delta = close_time - dt
        return int(delta.total_seconds() / 60)

    def get_week_trading_days(self, week_date: Optional[date] = None) -> list[date]:
        """Get all trading days for the week containing the given date.

        Args:
            week_date: Any date within the week. If None, uses current date.

        Returns:
            List of trading days in that week (Monday to Friday).
        """
        if week_date is None:
            week_date = self._now_eastern().date()

        days_since_monday = week_date.weekday()
        week_start = week_date - timedelta(days=days_since_monday)
        week_end = week_start + timedelta(days=4)

        return self.get_trading_days_between(week_start, week_end)


@lru_cache(maxsize=1)
def get_market_calendar() -> MarketCalendar:
    """Get singleton market calendar instance."""
    return MarketCalendar()

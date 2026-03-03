"""Unit tests for the scheduler calendar module."""

from datetime import date, datetime, time, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from src.core.scheduler.calendar import MarketCalendar, get_market_calendar


@pytest.fixture
def calendar():
    """Create a MarketCalendar instance for testing."""
    return MarketCalendar()


@pytest.fixture
def et_tz():
    """Get US/Eastern timezone."""
    return pytz.timezone("US/Eastern")


class TestMarketCalendarInit:
    def test_initialization(self, calendar):
        assert calendar._calendar is not None
        assert calendar.NYSE_TIMEZONE == pytz.timezone("US/Eastern")
        assert calendar.MARKET_OPEN_TIME == time(9, 30)
        assert calendar.MARKET_CLOSE_TIME == time(16, 0)

    def test_schedule_caching(self, calendar):
        schedule_2026 = calendar._get_schedule(2026)
        assert 2026 in calendar._schedule_cache
        schedule_2026_again = calendar._get_schedule(2026)
        assert schedule_2026 is schedule_2026_again


class TestIsTradingDay:
    def test_regular_weekday_is_trading_day(self, calendar):
        wednesday = date(2026, 3, 4)
        assert wednesday.weekday() == 2
        assert calendar.is_trading_day(wednesday)

    def test_weekend_is_not_trading_day(self, calendar):
        saturday = date(2026, 3, 7)
        sunday = date(2026, 3, 8)
        assert saturday.weekday() == 5
        assert sunday.weekday() == 6
        assert not calendar.is_trading_day(saturday)
        assert not calendar.is_trading_day(sunday)

    def test_new_years_day_is_not_trading_day(self, calendar):
        new_years_2026 = date(2026, 1, 1)
        assert not calendar.is_trading_day(new_years_2026)

    def test_christmas_is_not_trading_day(self, calendar):
        christmas_2026 = date(2026, 12, 25)
        assert not calendar.is_trading_day(christmas_2026)

    def test_uses_current_date_when_none(self, calendar, et_tz):
        now = datetime(2026, 3, 4, 12, 0, 0, tzinfo=et_tz)
        with patch.object(calendar, "_now_eastern", return_value=now):
            result = calendar.is_trading_day(None)
            assert isinstance(result, bool)


class TestIsMarketOpen:
    def test_market_open_during_trading_hours(self, calendar, et_tz):
        trading_time = et_tz.localize(datetime(2026, 3, 4, 12, 0, 0))
        assert calendar.is_market_open(trading_time)

    def test_market_closed_before_open(self, calendar, et_tz):
        before_open = et_tz.localize(datetime(2026, 3, 4, 9, 0, 0))
        assert not calendar.is_market_open(before_open)

    def test_market_closed_after_close(self, calendar, et_tz):
        after_close = et_tz.localize(datetime(2026, 3, 4, 16, 30, 0))
        assert not calendar.is_market_open(after_close)

    def test_market_closed_on_weekend(self, calendar, et_tz):
        saturday_noon = et_tz.localize(datetime(2026, 3, 7, 12, 0, 0))
        assert not calendar.is_market_open(saturday_noon)

    def test_market_open_at_open_time(self, calendar, et_tz):
        at_open = et_tz.localize(datetime(2026, 3, 4, 9, 30, 0))
        assert calendar.is_market_open(at_open)

    def test_handles_naive_datetime(self, calendar):
        naive_dt = datetime(2026, 3, 4, 12, 0, 0)
        result = calendar.is_market_open(naive_dt)
        assert isinstance(result, bool)

    def test_handles_utc_datetime(self, calendar):
        utc_dt = datetime(2026, 3, 4, 17, 0, 0, tzinfo=pytz.UTC)
        result = calendar.is_market_open(utc_dt)
        assert isinstance(result, bool)


class TestGetNextTradingDay:
    def test_next_day_from_monday(self, calendar):
        monday = date(2026, 3, 2)
        next_day = calendar.get_next_trading_day(monday)
        assert next_day == date(2026, 3, 3)

    def test_next_day_from_friday(self, calendar):
        friday = date(2026, 3, 6)
        next_day = calendar.get_next_trading_day(friday)
        assert next_day == date(2026, 3, 9)

    def test_next_day_from_saturday(self, calendar):
        saturday = date(2026, 3, 7)
        next_day = calendar.get_next_trading_day(saturday)
        assert next_day == date(2026, 3, 9)

    def test_next_day_from_sunday(self, calendar):
        sunday = date(2026, 3, 8)
        next_day = calendar.get_next_trading_day(sunday)
        assert next_day == date(2026, 3, 9)

    def test_next_day_skips_holiday(self, calendar):
        day_before_christmas = date(2026, 12, 24)
        next_day = calendar.get_next_trading_day(day_before_christmas)
        assert next_day > date(2026, 12, 25)

    def test_uses_current_date_when_none(self, calendar, et_tz):
        now = datetime(2026, 3, 4, 12, 0, 0, tzinfo=et_tz)
        with patch.object(calendar, "_now_eastern", return_value=now):
            result = calendar.get_next_trading_day(None)
            assert result > date(2026, 3, 4)


class TestGetPreviousTradingDay:
    def test_previous_day_from_tuesday(self, calendar):
        tuesday = date(2026, 3, 3)
        prev_day = calendar.get_previous_trading_day(tuesday)
        assert prev_day == date(2026, 3, 2)

    def test_previous_day_from_monday(self, calendar):
        monday = date(2026, 3, 9)
        prev_day = calendar.get_previous_trading_day(monday)
        assert prev_day == date(2026, 3, 6)

    def test_previous_day_from_saturday(self, calendar):
        saturday = date(2026, 3, 7)
        prev_day = calendar.get_previous_trading_day(saturday)
        assert prev_day == date(2026, 3, 6)

    def test_previous_day_from_sunday(self, calendar):
        sunday = date(2026, 3, 8)
        prev_day = calendar.get_previous_trading_day(sunday)
        assert prev_day == date(2026, 3, 6)


class TestGetMarketOpenTime:
    def test_regular_day_open_time(self, calendar, et_tz):
        trading_day = date(2026, 3, 4)
        open_time = calendar.get_market_open_time(trading_day)
        assert open_time.tzinfo is not None
        assert open_time.hour == 9
        assert open_time.minute == 30

    def test_raises_for_non_trading_day(self, calendar):
        saturday = date(2026, 3, 7)
        with pytest.raises(ValueError, match="not a trading day"):
            calendar.get_market_open_time(saturday)

    def test_uses_current_date_when_none(self, calendar, et_tz):
        now = datetime(2026, 3, 4, 12, 0, 0, tzinfo=et_tz)
        with patch.object(calendar, "_now_eastern", return_value=now):
            result = calendar.get_market_open_time(None)
            assert result.hour == 9
            assert result.minute == 30


class TestGetMarketCloseTime:
    def test_regular_day_close_time(self, calendar, et_tz):
        trading_day = date(2026, 3, 4)
        close_time = calendar.get_market_close_time(trading_day)
        assert close_time.tzinfo is not None
        assert close_time.hour == 16
        assert close_time.minute == 0

    def test_raises_for_non_trading_day(self, calendar):
        saturday = date(2026, 3, 7)
        with pytest.raises(ValueError, match="not a trading day"):
            calendar.get_market_close_time(saturday)


class TestGetTradingDaysBetween:
    def test_full_week(self, calendar):
        monday = date(2026, 3, 2)
        friday = date(2026, 3, 6)
        days = calendar.get_trading_days_between(monday, friday)
        assert len(days) == 5
        assert days[0] == monday
        assert days[-1] == friday

    def test_includes_start_and_end(self, calendar):
        monday = date(2026, 3, 2)
        wednesday = date(2026, 3, 4)
        days = calendar.get_trading_days_between(monday, wednesday, inclusive=True)
        assert monday in days
        assert wednesday in days

    def test_excludes_start_and_end(self, calendar):
        monday = date(2026, 3, 2)
        wednesday = date(2026, 3, 4)
        days = calendar.get_trading_days_between(monday, wednesday, inclusive=False)
        assert monday not in days
        assert wednesday not in days

    def test_empty_for_reversed_range(self, calendar):
        start = date(2026, 3, 6)
        end = date(2026, 3, 2)
        days = calendar.get_trading_days_between(start, end)
        assert days == []

    def test_excludes_weekends(self, calendar):
        friday = date(2026, 3, 6)
        monday = date(2026, 3, 9)
        days = calendar.get_trading_days_between(friday, monday)
        assert len(days) == 2
        for d in days:
            assert d.weekday() < 5


class TestIsMonday:
    def test_actual_monday(self, calendar):
        monday = date(2026, 3, 2)
        assert monday.weekday() == 0
        assert calendar.is_monday(monday)

    def test_tuesday_is_not_monday(self, calendar):
        tuesday = date(2026, 3, 3)
        assert not calendar.is_monday(tuesday)

    def test_friday_is_not_monday(self, calendar):
        friday = date(2026, 3, 6)
        assert not calendar.is_monday(friday)

    def test_non_trading_day_is_not_monday(self, calendar):
        saturday = date(2026, 3, 7)
        assert not calendar.is_monday(saturday)

    def test_tuesday_after_monday_holiday_is_monday(self, calendar):
        mlk_day_2026 = date(2026, 1, 19)
        tuesday_after_mlk = date(2026, 1, 20)
        if not calendar.is_trading_day(mlk_day_2026):
            assert calendar.is_monday(tuesday_after_mlk)


class TestIsFriday:
    def test_actual_friday(self, calendar):
        friday = date(2026, 3, 6)
        assert friday.weekday() == 4
        assert calendar.is_friday(friday)

    def test_thursday_is_not_friday(self, calendar):
        thursday = date(2026, 3, 5)
        assert not calendar.is_friday(thursday)

    def test_monday_is_not_friday(self, calendar):
        monday = date(2026, 3, 2)
        assert not calendar.is_friday(monday)

    def test_non_trading_day_is_not_friday(self, calendar):
        saturday = date(2026, 3, 7)
        assert not calendar.is_friday(saturday)


class TestIsEarlyClose:
    def test_regular_day_not_early_close(self, calendar):
        regular_day = date(2026, 3, 4)
        assert not calendar.is_early_close(regular_day)

    def test_non_trading_day_not_early_close(self, calendar):
        saturday = date(2026, 3, 7)
        assert not calendar.is_early_close(saturday)


class TestMinutesUntilMarketOpen:
    def test_before_market_open(self, calendar, et_tz):
        before_open = et_tz.localize(datetime(2026, 3, 4, 8, 30, 0))
        minutes = calendar.minutes_until_market_open(before_open)
        assert minutes == 60

    def test_during_market_hours(self, calendar, et_tz):
        during_market = et_tz.localize(datetime(2026, 3, 4, 12, 0, 0))
        minutes = calendar.minutes_until_market_open(during_market)
        assert minutes < 0


class TestMinutesUntilMarketClose:
    def test_during_market_hours(self, calendar, et_tz):
        during_market = et_tz.localize(datetime(2026, 3, 4, 15, 0, 0))
        minutes = calendar.minutes_until_market_close(during_market)
        assert minutes == 60

    def test_on_non_trading_day(self, calendar, et_tz):
        saturday = et_tz.localize(datetime(2026, 3, 7, 12, 0, 0))
        minutes = calendar.minutes_until_market_close(saturday)
        assert minutes == -1


class TestGetWeekTradingDays:
    def test_regular_week(self, calendar):
        wednesday = date(2026, 3, 4)
        days = calendar.get_week_trading_days(wednesday)
        assert len(days) == 5
        assert days[0].weekday() == 0
        assert days[-1].weekday() == 4

    def test_uses_current_date_when_none(self, calendar, et_tz):
        now = datetime(2026, 3, 4, 12, 0, 0, tzinfo=et_tz)
        with patch.object(calendar, "_now_eastern", return_value=now):
            days = calendar.get_week_trading_days(None)
            assert len(days) <= 5


class TestGetMarketCalendarSingleton:
    def test_returns_same_instance(self):
        get_market_calendar.cache_clear()
        cal1 = get_market_calendar()
        cal2 = get_market_calendar()
        assert cal1 is cal2

    def test_returns_market_calendar_instance(self):
        get_market_calendar.cache_clear()
        cal = get_market_calendar()
        assert isinstance(cal, MarketCalendar)

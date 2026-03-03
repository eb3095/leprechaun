"""Unit tests for the scheduler jobs module."""

import time as time_module
from datetime import date, datetime
from unittest.mock import MagicMock, patch, call

import pytest
import pytz

from src.core.scheduler.calendar import MarketCalendar
from src.core.scheduler.jobs import (
    SchedulerManager,
    create_scheduler,
    first_trading_day_only,
    last_trading_day_only,
    market_hours_only,
    register_maintenance_jobs,
    register_trading_jobs,
    trading_day_only,
)


@pytest.fixture
def mock_calendar():
    """Create a mock market calendar."""
    calendar = MagicMock(spec=MarketCalendar)
    calendar.is_market_open.return_value = True
    calendar.is_trading_day.return_value = True
    calendar.is_monday.return_value = False
    calendar.is_friday.return_value = False
    return calendar


@pytest.fixture
def scheduler(mock_calendar):
    """Create a SchedulerManager with mock calendar."""
    return SchedulerManager(calendar=mock_calendar)


@pytest.fixture
def et_tz():
    """Get US/Eastern timezone."""
    return pytz.timezone("US/Eastern")


class TestSchedulerManagerInit:
    def test_initialization(self, mock_calendar):
        manager = SchedulerManager(calendar=mock_calendar)
        assert manager.calendar is mock_calendar
        assert not manager.is_running

    def test_default_calendar_when_none(self):
        with patch("src.core.scheduler.jobs.get_market_calendar") as mock_get:
            mock_get.return_value = MagicMock(spec=MarketCalendar)
            manager = SchedulerManager()
            mock_get.assert_called_once()

    def test_custom_max_workers(self, mock_calendar):
        manager = SchedulerManager(calendar=mock_calendar, max_workers=20)
        assert manager is not None

    def test_custom_job_defaults(self, mock_calendar):
        manager = SchedulerManager(
            calendar=mock_calendar,
            job_defaults={"max_instances": 2, "coalesce": False},
        )
        assert manager is not None


class TestSchedulerManagerStartStop:
    def test_start_scheduler(self, scheduler):
        scheduler.start()
        assert scheduler.is_running
        scheduler.shutdown()

    def test_start_already_running_logs_warning(self, scheduler, caplog):
        scheduler.start()
        scheduler.start()
        assert "already running" in caplog.text.lower()
        scheduler.shutdown()

    def test_shutdown_scheduler(self, scheduler):
        scheduler.start()
        scheduler.shutdown()
        assert not scheduler.is_running

    def test_shutdown_not_running_logs_warning(self, scheduler, caplog):
        scheduler.shutdown()
        assert "not running" in caplog.text.lower()

    def test_pause_resume(self, scheduler):
        scheduler.start()
        scheduler.pause()
        scheduler.resume()
        assert scheduler.is_running
        scheduler.shutdown()


class TestSchedulerManagerJobs:
    def test_add_job_with_cron_trigger(self, scheduler):
        from apscheduler.triggers.cron import CronTrigger

        def test_func():
            pass

        trigger = CronTrigger(hour=12, minute=0, timezone=scheduler.TIMEZONE)
        job_id = scheduler.add_job(test_func, trigger, job_id="test_job")
        assert job_id == "test_job"

    def test_add_job_with_interval_trigger(self, scheduler):
        from apscheduler.triggers.interval import IntervalTrigger

        def test_func():
            pass

        trigger = IntervalTrigger(minutes=5, timezone=scheduler.TIMEZONE)
        job_id = scheduler.add_job(test_func, trigger, job_id="interval_job")
        assert job_id == "interval_job"

    def test_add_job_auto_generates_id(self, scheduler):
        from apscheduler.triggers.interval import IntervalTrigger

        def my_custom_function():
            pass

        trigger = IntervalTrigger(minutes=5, timezone=scheduler.TIMEZONE)
        job_id = scheduler.add_job(my_custom_function, trigger)
        assert job_id is not None

    def test_remove_job(self, scheduler):
        from apscheduler.triggers.interval import IntervalTrigger

        def test_func():
            pass

        trigger = IntervalTrigger(minutes=5, timezone=scheduler.TIMEZONE)
        scheduler.add_job(test_func, trigger, job_id="to_remove")
        result = scheduler.remove_job("to_remove")
        assert result is True

    def test_remove_nonexistent_job(self, scheduler):
        result = scheduler.remove_job("nonexistent")
        assert result is False

    def test_get_jobs_empty(self, scheduler):
        jobs = scheduler.get_jobs()
        assert jobs == []

    def test_get_jobs_with_jobs(self, scheduler):
        from apscheduler.triggers.interval import IntervalTrigger

        def test_func():
            pass

        trigger = IntervalTrigger(minutes=5, timezone=scheduler.TIMEZONE)
        scheduler.add_job(test_func, trigger, job_id="job1", name="Job One")
        scheduler.add_job(test_func, trigger, job_id="job2", name="Job Two")

        jobs = scheduler.get_jobs()
        assert len(jobs) == 2
        job_ids = [j["id"] for j in jobs]
        assert "job1" in job_ids
        assert "job2" in job_ids

    def test_get_job(self, scheduler):
        from apscheduler.triggers.interval import IntervalTrigger

        def test_func():
            pass

        trigger = IntervalTrigger(minutes=5, timezone=scheduler.TIMEZONE)
        scheduler.add_job(test_func, trigger, job_id="my_job", name="My Job")

        job = scheduler.get_job("my_job")
        assert job is not None
        assert job["id"] == "my_job"
        assert job["name"] == "My Job"

    def test_get_nonexistent_job(self, scheduler):
        job = scheduler.get_job("nonexistent")
        assert job is None

    def test_run_job_now(self, scheduler):
        from apscheduler.triggers.cron import CronTrigger

        def test_func():
            pass

        trigger = CronTrigger(hour=23, minute=59, timezone=scheduler.TIMEZONE)
        scheduler.add_job(test_func, trigger, job_id="delayed_job")

        result = scheduler.run_job_now("delayed_job")
        assert result is True

    def test_run_nonexistent_job_now(self, scheduler):
        result = scheduler.run_job_now("nonexistent")
        assert result is False


class TestMarketHoursOnlyDecorator:
    def test_runs_during_market_hours(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_market_open.return_value = True

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = market_hours_only(mock_func)
            result = decorated()

        assert result == "executed"
        mock_func.assert_called_once()

    def test_skips_outside_market_hours(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_market_open.return_value = False

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = market_hours_only(mock_func)
            result = decorated()

        assert result is None
        mock_func.assert_not_called()

    def test_preserves_function_metadata(self):
        def my_documented_function():
            """This is my docstring."""
            pass

        decorated = market_hours_only(my_documented_function)
        assert decorated.__name__ == "my_documented_function"
        assert "docstring" in decorated.__doc__


class TestTradingDayOnlyDecorator:
    def test_runs_on_trading_day(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_trading_day.return_value = True

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = trading_day_only(mock_func)
            result = decorated()

        assert result == "executed"
        mock_func.assert_called_once()

    def test_skips_on_non_trading_day(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_trading_day.return_value = False

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = trading_day_only(mock_func)
            result = decorated()

        assert result is None
        mock_func.assert_not_called()


class TestFirstTradingDayOnlyDecorator:
    def test_runs_on_monday(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_monday.return_value = True

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = first_trading_day_only(mock_func)
            result = decorated()

        assert result == "executed"
        mock_func.assert_called_once()

    def test_skips_on_non_monday(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_monday.return_value = False

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = first_trading_day_only(mock_func)
            result = decorated()

        assert result is None
        mock_func.assert_not_called()


class TestLastTradingDayOnlyDecorator:
    def test_runs_on_friday(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_friday.return_value = True

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = last_trading_day_only(mock_func)
            result = decorated()

        assert result == "executed"
        mock_func.assert_called_once()

    def test_skips_on_non_friday(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_friday.return_value = False

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = last_trading_day_only(mock_func)
            result = decorated()

        assert result is None
        mock_func.assert_not_called()


class TestRegisterTradingJobs:
    def test_registers_all_trading_jobs(self, scheduler):
        job_ids = register_trading_jobs(scheduler)

        assert len(job_ids) == 5
        assert "monday_buy_check" in job_ids
        assert "friday_sell" in job_ids
        assert "signal_generation" in job_ids
        assert "daily_summary" in job_ids
        assert "sentiment_collection" in job_ids

    def test_uses_provided_functions(self, scheduler):
        monday_buy = MagicMock()
        friday_sell = MagicMock()

        register_trading_jobs(
            scheduler,
            monday_buy_func=monday_buy,
            friday_sell_func=friday_sell,
        )

        jobs = scheduler.get_jobs()
        assert len(jobs) == 5

    def test_placeholder_functions_work(self, scheduler):
        job_ids = register_trading_jobs(scheduler)
        assert len(job_ids) == 5


class TestRegisterMaintenanceJobs:
    def test_registers_all_maintenance_jobs(self, scheduler):
        job_ids = register_maintenance_jobs(scheduler)

        assert len(job_ids) == 4
        assert "database_cleanup" in job_ids
        assert "cache_refresh" in job_ids
        assert "health_check" in job_ids
        assert "universe_update" in job_ids

    def test_uses_provided_functions(self, scheduler):
        db_cleanup = MagicMock()
        cache_refresh = MagicMock()

        register_maintenance_jobs(
            scheduler,
            database_cleanup_func=db_cleanup,
            cache_refresh_func=cache_refresh,
        )

        jobs = scheduler.get_jobs()
        assert len(jobs) == 4


class TestCreateScheduler:
    def test_creates_scheduler_with_trading_jobs(self, mock_calendar):
        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            scheduler = create_scheduler(
                calendar=mock_calendar,
                register_trading=True,
                register_maintenance=False,
            )

        assert scheduler is not None
        jobs = scheduler.get_jobs()
        assert len(jobs) == 5

    def test_creates_scheduler_with_maintenance_jobs(self, mock_calendar):
        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            scheduler = create_scheduler(
                calendar=mock_calendar,
                register_trading=False,
                register_maintenance=True,
            )

        assert scheduler is not None
        jobs = scheduler.get_jobs()
        assert len(jobs) == 4

    def test_creates_scheduler_with_all_jobs(self, mock_calendar):
        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            scheduler = create_scheduler(
                calendar=mock_calendar,
                register_trading=True,
                register_maintenance=True,
            )

        jobs = scheduler.get_jobs()
        assert len(jobs) == 9

    def test_creates_scheduler_with_no_jobs(self, mock_calendar):
        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            scheduler = create_scheduler(
                calendar=mock_calendar,
                register_trading=False,
                register_maintenance=False,
            )

        jobs = scheduler.get_jobs()
        assert len(jobs) == 0

    def test_creates_scheduler_with_custom_funcs(self, mock_calendar):
        custom_monday = MagicMock()
        custom_cleanup = MagicMock()

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            scheduler = create_scheduler(
                calendar=mock_calendar,
                trading_funcs={"monday_buy": custom_monday},
                maintenance_funcs={"database_cleanup": custom_cleanup},
            )

        assert scheduler is not None
        jobs = scheduler.get_jobs()
        assert len(jobs) == 9


class TestJobListenerLogging:
    def test_logs_successful_execution(self, scheduler, caplog):
        from apscheduler.triggers.date import DateTrigger
        from datetime import datetime, timedelta

        call_count = {"value": 0}

        def quick_job():
            call_count["value"] += 1

        run_time = datetime.now(scheduler.TIMEZONE) + timedelta(milliseconds=100)
        trigger = DateTrigger(run_date=run_time, timezone=scheduler.TIMEZONE)
        scheduler.add_job(quick_job, trigger, job_id="quick_test")

        scheduler.start()
        time_module.sleep(0.5)
        scheduler.shutdown(wait=True)

        assert call_count["value"] == 1

    def test_logs_job_error(self, scheduler, caplog):
        from apscheduler.triggers.date import DateTrigger
        from datetime import datetime, timedelta

        def failing_job():
            raise ValueError("Test error")

        run_time = datetime.now(scheduler.TIMEZONE) + timedelta(milliseconds=100)
        trigger = DateTrigger(run_date=run_time, timezone=scheduler.TIMEZONE)
        scheduler.add_job(failing_job, trigger, job_id="failing_test")

        scheduler.start()
        time_module.sleep(0.5)
        scheduler.shutdown(wait=True)

        assert "failed" in caplog.text.lower() or "error" in caplog.text.lower()


class TestDecoratorChaining:
    def test_market_hours_and_trading_day(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_market_open.return_value = True
        mock_calendar.is_trading_day.return_value = True

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = market_hours_only(trading_day_only(mock_func))
            result = decorated()

        assert result == "executed"
        mock_func.assert_called_once()

    def test_chained_decorators_fail_first(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_market_open.return_value = False
        mock_calendar.is_trading_day.return_value = True

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = market_hours_only(trading_day_only(mock_func))
            result = decorated()

        assert result is None
        mock_func.assert_not_called()

    def test_chained_decorators_fail_second(self):
        mock_func = MagicMock(return_value="executed")
        mock_calendar = MagicMock(spec=MarketCalendar)
        mock_calendar.is_market_open.return_value = True
        mock_calendar.is_trading_day.return_value = False

        with patch(
            "src.core.scheduler.jobs.get_market_calendar", return_value=mock_calendar
        ):
            decorated = market_hours_only(trading_day_only(mock_func))
            result = decorated()

        assert result is None
        mock_func.assert_not_called()


class TestJobReplacement:
    def test_replace_existing_job(self, scheduler):
        from apscheduler.triggers.interval import IntervalTrigger

        def first_func():
            pass

        def second_func():
            pass

        scheduler.start()
        try:
            trigger = IntervalTrigger(minutes=5, timezone=scheduler.TIMEZONE)
            scheduler.add_job(
                first_func, trigger, job_id="replaceable", replace_existing=True
            )
            scheduler.add_job(
                second_func, trigger, job_id="replaceable", replace_existing=True
            )

            jobs = scheduler.get_jobs()
            job_ids = [j["id"] for j in jobs]
            assert job_ids.count("replaceable") == 1
        finally:
            scheduler.shutdown()

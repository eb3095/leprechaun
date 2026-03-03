"""Scheduled job definitions using APScheduler for Leprechaun trading bot.

Defines the scheduler manager and job registration for trading operations,
sentiment collection, and maintenance tasks.
"""

import functools
import logging
from datetime import datetime
from typing import Any, Callable, Optional

import pytz
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.core.scheduler.calendar import MarketCalendar, get_market_calendar

logger = logging.getLogger(__name__)


class SchedulerManager:
    """Manages APScheduler for Leprechaun trading bot.

    Handles job scheduling, execution monitoring, and graceful shutdown.
    All times are in US/Eastern timezone.
    """

    TIMEZONE = pytz.timezone("US/Eastern")

    def __init__(
        self,
        calendar: Optional[MarketCalendar] = None,
        max_workers: int = 10,
        job_defaults: Optional[dict[str, Any]] = None,
    ):
        """Initialize the scheduler manager.

        Args:
            calendar: Market calendar instance. Uses singleton if not provided.
            max_workers: Maximum number of threads for job execution.
            job_defaults: Default job options (coalesce, max_instances, etc.).
        """
        self.calendar = calendar or get_market_calendar()
        self._is_running = False

        default_job_config = {
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 60,
        }
        if job_defaults:
            default_job_config.update(job_defaults)

        self._scheduler = BackgroundScheduler(
            jobstores={"default": MemoryJobStore()},
            executors={"default": ThreadPoolExecutor(max_workers=max_workers)},
            job_defaults=default_job_config,
            timezone=self.TIMEZONE,
        )

        self._scheduler.add_listener(
            self._job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

    def _job_listener(self, event: JobExecutionEvent) -> None:
        """Handle job execution events for logging and monitoring."""
        if event.exception:
            logger.error(
                "Job %s failed with exception: %s",
                event.job_id,
                event.exception,
                exc_info=event.traceback,
            )
        else:
            logger.debug("Job %s executed successfully", event.job_id)

    @property
    def is_running(self) -> bool:
        """Check if scheduler is currently running."""
        return self._is_running and self._scheduler.running

    def start(self) -> None:
        """Start the scheduler.

        Begins executing scheduled jobs. Safe to call multiple times.
        """
        if self._is_running:
            logger.warning("Scheduler already running")
            return

        self._scheduler.start()
        self._is_running = True
        logger.info("Scheduler started")

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shutdown the scheduler.

        Args:
            wait: If True, waits for running jobs to complete.
        """
        if not self._is_running:
            logger.warning("Scheduler not running")
            return

        self._scheduler.shutdown(wait=wait)
        self._is_running = False
        logger.info("Scheduler stopped")

    def pause(self) -> None:
        """Pause all jobs without stopping the scheduler."""
        self._scheduler.pause()
        logger.info("Scheduler paused")

    def resume(self) -> None:
        """Resume all paused jobs."""
        self._scheduler.resume()
        logger.info("Scheduler resumed")

    def add_job(
        self,
        func: Callable,
        trigger: Any,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        replace_existing: bool = True,
        **kwargs: Any,
    ) -> str:
        """Add a job to the scheduler.

        Args:
            func: Function to execute.
            trigger: APScheduler trigger (CronTrigger, IntervalTrigger, etc.).
            job_id: Unique job identifier. Auto-generated if not provided.
            name: Human-readable job name.
            replace_existing: If True, replaces job with same ID.
            **kwargs: Additional arguments passed to func when executed.

        Returns:
            Job ID of the added job.
        """
        job = self._scheduler.add_job(
            func,
            trigger,
            id=job_id,
            name=name or (job_id or func.__name__),
            replace_existing=replace_existing,
            kwargs=kwargs,
        )
        logger.info("Added job %s: %s", job.id, job.name)
        return job.id

    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the scheduler.

        Args:
            job_id: ID of the job to remove.

        Returns:
            True if job was removed, False if job didn't exist.
        """
        try:
            self._scheduler.remove_job(job_id)
            logger.info("Removed job %s", job_id)
            return True
        except Exception:
            logger.warning("Job %s not found for removal", job_id)
            return False

    def get_jobs(self) -> list[dict[str, Any]]:
        """Get list of all scheduled jobs.

        Returns:
            List of job information dictionaries.
        """
        jobs = []
        for job in self._scheduler.get_jobs():
            next_run = getattr(job, "next_run_time", None)
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": next_run.isoformat() if next_run else None,
                    "trigger": str(job.trigger),
                    "pending": job.pending,
                }
            )
        return jobs

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get information about a specific job.

        Args:
            job_id: ID of the job to get.

        Returns:
            Job information dictionary or None if not found.
        """
        job = self._scheduler.get_job(job_id)
        if job is None:
            return None
        next_run = getattr(job, "next_run_time", None)
        return {
            "id": job.id,
            "name": job.name,
            "next_run_time": next_run.isoformat() if next_run else None,
            "trigger": str(job.trigger),
            "pending": job.pending,
        }

    def run_job_now(self, job_id: str) -> bool:
        """Immediately run a scheduled job.

        Args:
            job_id: ID of the job to run.

        Returns:
            True if job was triggered, False if job not found.
        """
        job = self._scheduler.get_job(job_id)
        if job is None:
            logger.warning("Job %s not found", job_id)
            return False
        job.modify(next_run_time=datetime.now(self.TIMEZONE))
        logger.info("Triggered immediate execution of job %s", job_id)
        return True


def market_hours_only(func: Callable) -> Callable:
    """Decorator that only runs function during market hours.

    If called outside market hours, logs a message and returns None.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that checks market hours before execution.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calendar = get_market_calendar()
        if not calendar.is_market_open():
            func_name = getattr(func, "__name__", repr(func))
            logger.debug("Skipping %s: market is closed", func_name)
            return None
        return func(*args, **kwargs)

    return wrapper


def trading_day_only(func: Callable) -> Callable:
    """Decorator that only runs function on trading days.

    If called on non-trading day, logs a message and returns None.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that checks trading day before execution.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calendar = get_market_calendar()
        if not calendar.is_trading_day():
            func_name = getattr(func, "__name__", repr(func))
            logger.debug("Skipping %s: not a trading day", func_name)
            return None
        return func(*args, **kwargs)

    return wrapper


def first_trading_day_only(func: Callable) -> Callable:
    """Decorator that only runs on the first trading day of the week (Monday/Tuesday).

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that checks if today is week's first trading day.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calendar = get_market_calendar()
        if not calendar.is_monday():
            func_name = getattr(func, "__name__", repr(func))
            logger.debug(
                "Skipping %s: not the first trading day of the week", func_name
            )
            return None
        return func(*args, **kwargs)

    return wrapper


def last_trading_day_only(func: Callable) -> Callable:
    """Decorator that only runs on the last trading day of the week (Friday/Thursday).

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that checks if today is week's last trading day.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calendar = get_market_calendar()
        if not calendar.is_friday():
            func_name = getattr(func, "__name__", repr(func))
            logger.debug("Skipping %s: not the last trading day of the week", func_name)
            return None
        return func(*args, **kwargs)

    return wrapper


def register_trading_jobs(
    scheduler: SchedulerManager,
    monday_buy_func: Optional[Callable] = None,
    friday_sell_func: Optional[Callable] = None,
    signal_generation_func: Optional[Callable] = None,
    daily_summary_func: Optional[Callable] = None,
    sentiment_collection_func: Optional[Callable] = None,
) -> list[str]:
    """Register all trading-related scheduled jobs.

    Jobs registered:
    - Monday buy check: 3:55 PM ET on weekdays (checks if Monday internally)
    - Friday sell: 3:50 PM ET on weekdays (checks if Friday internally)
    - Signal generation: Every 15 minutes during market hours (9:30 AM - 4:00 PM)
    - Daily summary: 5:00 PM ET on trading days
    - Sentiment collection: Every 5 minutes (always runs)

    Args:
        scheduler: SchedulerManager instance.
        monday_buy_func: Function for Monday buy check. Uses placeholder if None.
        friday_sell_func: Function for Friday sell. Uses placeholder if None.
        signal_generation_func: Function for signal generation. Uses placeholder if None.
        daily_summary_func: Function for daily summary. Uses placeholder if None.
        sentiment_collection_func: Function for sentiment collection. Uses placeholder if None.

    Returns:
        List of registered job IDs.
    """
    job_ids = []

    def _placeholder(name: str) -> Callable:
        def placeholder_func() -> None:
            logger.info("Placeholder job executed: %s", name)

        placeholder_func.__name__ = name
        return placeholder_func

    monday_buy = first_trading_day_only(
        monday_buy_func or _placeholder("monday_buy_check")
    )
    job_ids.append(
        scheduler.add_job(
            monday_buy,
            CronTrigger(
                day_of_week="mon-fri", hour=15, minute=55, timezone=scheduler.TIMEZONE
            ),
            job_id="monday_buy_check",
            name="Monday Buy Check (3:55 PM ET)",
        )
    )

    friday_sell = last_trading_day_only(friday_sell_func or _placeholder("friday_sell"))
    job_ids.append(
        scheduler.add_job(
            friday_sell,
            CronTrigger(
                day_of_week="mon-fri", hour=15, minute=50, timezone=scheduler.TIMEZONE
            ),
            job_id="friday_sell",
            name="Friday Sell (3:50 PM ET)",
        )
    )

    signal_gen = market_hours_only(
        trading_day_only(signal_generation_func or _placeholder("signal_generation"))
    )
    job_ids.append(
        scheduler.add_job(
            signal_gen,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute="*/15",
                timezone=scheduler.TIMEZONE,
            ),
            job_id="signal_generation",
            name="Signal Generation (Every 15 min during market hours)",
        )
    )

    daily_summary = trading_day_only(
        daily_summary_func or _placeholder("daily_summary")
    )
    job_ids.append(
        scheduler.add_job(
            daily_summary,
            CronTrigger(
                day_of_week="mon-fri", hour=17, minute=0, timezone=scheduler.TIMEZONE
            ),
            job_id="daily_summary",
            name="Daily Summary (5:00 PM ET)",
        )
    )

    sentiment_func = sentiment_collection_func or _placeholder("sentiment_collection")
    job_ids.append(
        scheduler.add_job(
            sentiment_func,
            IntervalTrigger(minutes=5, timezone=scheduler.TIMEZONE),
            job_id="sentiment_collection",
            name="Sentiment Collection (Every 5 min)",
        )
    )

    logger.info("Registered %d trading jobs", len(job_ids))
    return job_ids


def register_maintenance_jobs(
    scheduler: SchedulerManager,
    database_cleanup_func: Optional[Callable] = None,
    cache_refresh_func: Optional[Callable] = None,
    health_check_func: Optional[Callable] = None,
    universe_update_func: Optional[Callable] = None,
) -> list[str]:
    """Register maintenance-related scheduled jobs.

    Jobs registered:
    - Database cleanup: Daily at 2:00 AM ET
    - Cache refresh: Every 30 minutes
    - Health check: Every 1 minute
    - Universe update: Weekly on Sunday at 6:00 AM ET

    Args:
        scheduler: SchedulerManager instance.
        database_cleanup_func: Function for database cleanup. Uses placeholder if None.
        cache_refresh_func: Function for cache refresh. Uses placeholder if None.
        health_check_func: Function for health checks. Uses placeholder if None.
        universe_update_func: Function for stock universe updates. Uses placeholder if None.

    Returns:
        List of registered job IDs.
    """
    job_ids = []

    def _placeholder(name: str) -> Callable:
        def placeholder_func() -> None:
            logger.info("Placeholder job executed: %s", name)

        placeholder_func.__name__ = name
        return placeholder_func

    job_ids.append(
        scheduler.add_job(
            database_cleanup_func or _placeholder("database_cleanup"),
            CronTrigger(hour=2, minute=0, timezone=scheduler.TIMEZONE),
            job_id="database_cleanup",
            name="Database Cleanup (Daily 2:00 AM ET)",
        )
    )

    job_ids.append(
        scheduler.add_job(
            cache_refresh_func or _placeholder("cache_refresh"),
            IntervalTrigger(minutes=30, timezone=scheduler.TIMEZONE),
            job_id="cache_refresh",
            name="Cache Refresh (Every 30 min)",
        )
    )

    job_ids.append(
        scheduler.add_job(
            health_check_func or _placeholder("health_check"),
            IntervalTrigger(minutes=1, timezone=scheduler.TIMEZONE),
            job_id="health_check",
            name="Health Check (Every 1 min)",
        )
    )

    job_ids.append(
        scheduler.add_job(
            universe_update_func or _placeholder("universe_update"),
            CronTrigger(
                day_of_week="sun", hour=6, minute=0, timezone=scheduler.TIMEZONE
            ),
            job_id="universe_update",
            name="Stock Universe Update (Weekly Sunday 6:00 AM ET)",
        )
    )

    logger.info("Registered %d maintenance jobs", len(job_ids))
    return job_ids


def register_archival_jobs(
    scheduler: SchedulerManager,
    archive,
    sentiment_agent,
    universe,
) -> list[str]:
    """Register sentiment archival job.

    Runs hourly during market hours to capture sentiment snapshots
    for future backtesting.

    Args:
        scheduler: SchedulerManager instance.
        archive: SentimentArchive instance.
        sentiment_agent: SentimentAgent for getting current sentiment.
        universe: StockUniverse or list of symbols.

    Returns:
        List of registered job IDs.
    """
    from src.data.sentiment_archive import create_archival_job

    job_ids = []

    archival_func = market_hours_only(
        trading_day_only(create_archival_job(archive, sentiment_agent, universe))
    )

    job_ids.append(
        scheduler.add_job(
            archival_func,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-16",
                minute=0,
                timezone=scheduler.TIMEZONE,
            ),
            job_id="sentiment_archival",
            name="Sentiment Archival (Hourly during market hours)",
        )
    )

    logger.info("Registered %d archival jobs", len(job_ids))
    return job_ids


def create_scheduler(
    calendar: Optional[MarketCalendar] = None,
    register_trading: bool = True,
    register_maintenance: bool = True,
    trading_funcs: Optional[dict[str, Callable]] = None,
    maintenance_funcs: Optional[dict[str, Callable]] = None,
) -> SchedulerManager:
    """Create and configure a scheduler with standard jobs.

    Convenience function that creates a SchedulerManager and optionally
    registers trading and maintenance jobs.

    Args:
        calendar: Market calendar instance.
        register_trading: If True, registers trading jobs.
        register_maintenance: If True, registers maintenance jobs.
        trading_funcs: Dict mapping function names to implementations for trading jobs.
        maintenance_funcs: Dict mapping function names to implementations for maintenance jobs.

    Returns:
        Configured SchedulerManager instance (not started).
    """
    scheduler = SchedulerManager(calendar=calendar)

    if register_trading:
        funcs = trading_funcs or {}
        register_trading_jobs(
            scheduler,
            monday_buy_func=funcs.get("monday_buy"),
            friday_sell_func=funcs.get("friday_sell"),
            signal_generation_func=funcs.get("signal_generation"),
            daily_summary_func=funcs.get("daily_summary"),
            sentiment_collection_func=funcs.get("sentiment_collection"),
        )

    if register_maintenance:
        funcs = maintenance_funcs or {}
        register_maintenance_jobs(
            scheduler,
            database_cleanup_func=funcs.get("database_cleanup"),
            cache_refresh_func=funcs.get("cache_refresh"),
            health_check_func=funcs.get("health_check"),
            universe_update_func=funcs.get("universe_update"),
        )

    return scheduler

"""Unit tests for sentiment archive module."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.data.sentiment_archive import SentimentArchive, create_archival_job


class TestSentimentArchive:
    """Tests for SentimentArchive class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        return session

    @pytest.fixture
    def mock_stock(self):
        """Create a mock Stock object."""
        stock = MagicMock()
        stock.id = 1
        stock.symbol = "AAPL"
        return stock

    @pytest.fixture
    def archive(self, mock_session):
        """Create SentimentArchive with mock session."""
        return SentimentArchive(db_session=mock_session)

    @pytest.fixture
    def archive_no_session(self):
        """Create SentimentArchive without session."""
        return SentimentArchive(db_session=None)

    def test_init_with_session(self, archive, mock_session):
        """Test initialization with database session."""
        assert archive.session == mock_session

    def test_init_without_session(self, archive_no_session):
        """Test initialization without database session."""
        assert archive_no_session.session is None

    def test_archive_snapshot_requires_session(self, archive_no_session):
        """Test archive_snapshot raises error without session."""
        with pytest.raises(RuntimeError, match="Database session required"):
            archive_no_session.archive_snapshot("AAPL", {})

    def test_archive_snapshot_requires_valid_symbol(self, archive, mock_session):
        """Test archive_snapshot raises error for unknown symbol."""
        mock_session.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Stock AAPL not found"):
            archive.archive_snapshot("AAPL", {})

    def test_archive_snapshot_success(self, archive, mock_session, mock_stock):
        """Test successful sentiment archival."""
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock

        mock_record = MagicMock()
        mock_record.id = 123
        mock_session.add = MagicMock()
        mock_session.flush = MagicMock()

        with patch("src.data.sentiment_archive.SentimentData") as mock_sentiment_data:
            mock_sentiment_data.return_value = mock_record

            result = archive.archive_snapshot("AAPL", {
                "timestamp": datetime.utcnow(),
                "composite_score": -0.5,
                "volume": 100,
                "sources": {"reddit": {"score": -0.6}},
                "manipulation_score": 0.7,
            })

            assert result == 123
            mock_session.add.assert_called_once()
            mock_session.flush.assert_called_once()

    def test_archive_snapshot_with_string_timestamp(self, archive, mock_session, mock_stock):
        """Test archival with ISO string timestamp."""
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock

        mock_record = MagicMock()
        mock_record.id = 456
        mock_session.add = MagicMock()
        mock_session.flush = MagicMock()

        with patch("src.data.sentiment_archive.SentimentData") as mock_sentiment_data:
            mock_sentiment_data.return_value = mock_record

            result = archive.archive_snapshot("AAPL", {
                "timestamp": "2024-01-15T14:30:00Z",
                "composite_score": 0.3,
                "volume": 50,
            })

            assert result == 456

    def test_get_archived_sentiment_requires_session(self, archive_no_session):
        """Test get_archived_sentiment raises error without session."""
        with pytest.raises(RuntimeError, match="Database session required"):
            archive_no_session.get_archived_sentiment("AAPL", date.today(), date.today())

    def test_get_archived_sentiment_unknown_symbol(self, archive, mock_session):
        """Test get_archived_sentiment returns empty for unknown symbol."""
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = archive.get_archived_sentiment("UNKNOWN", date.today(), date.today())

        assert result == []

    def test_get_archived_sentiment_success(self, archive, mock_session, mock_stock):
        """Test successful sentiment retrieval."""
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock

        mock_record = MagicMock()
        mock_record.id = 1
        mock_record.timestamp = datetime(2024, 1, 15, 14, 30)
        mock_record.source = MagicMock()
        mock_record.source.value = "reddit"
        mock_record.sentiment_score = Decimal("0.5")
        mock_record.sentiment_volume = 100
        mock_record.velocity = Decimal("0.1")
        mock_record.bot_fraction = None
        mock_record.coordination_score = None
        mock_record.raw_data = {"manipulation_score": 0.3}

        mock_query = MagicMock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = [mock_record]
        mock_session.query.return_value = mock_query

        result = archive.get_archived_sentiment(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 31),
        )

        assert len(result) == 1
        assert result[0]["sentiment_score"] == 0.5
        assert result[0]["sentiment_volume"] == 100
        assert result[0]["is_synthetic"] is False

    def test_has_historical_data_requires_session(self, archive_no_session):
        """Test has_historical_data raises error without session."""
        with pytest.raises(RuntimeError, match="Database session required"):
            archive_no_session.has_historical_data("AAPL", date.today(), date.today())

    def test_has_historical_data_unknown_symbol(self, archive, mock_session):
        """Test has_historical_data returns False for unknown symbol."""
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = archive.has_historical_data("UNKNOWN", date.today(), date.today())

        assert result is False

    def test_has_historical_data_with_data(self, archive, mock_session, mock_stock):
        """Test has_historical_data returns True when data exists."""
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock
        mock_session.query.return_value.filter.return_value.scalar.return_value = 5

        result = archive.has_historical_data(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 31),
        )

        assert result is True

    def test_has_historical_data_no_data(self, archive, mock_session, mock_stock):
        """Test has_historical_data returns False when no data."""
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock
        mock_session.query.return_value.filter.return_value.scalar.return_value = 0

        result = archive.has_historical_data(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 31),
        )

        assert result is False

    def test_get_coverage_report_requires_session(self, archive_no_session):
        """Test get_coverage_report raises error without session."""
        with pytest.raises(RuntimeError, match="Database session required"):
            archive_no_session.get_coverage_report(["AAPL"], date.today(), date.today())

    def test_get_coverage_report_empty_symbols(self, archive, mock_session):
        """Test coverage report with empty symbol list."""
        result = archive.get_coverage_report([], date.today(), date.today())

        assert result["overall_coverage"] == 0.0
        assert result["symbol_coverage"] == {}

    def test_get_daily_sentiment_aggregation(self, archive, mock_session, mock_stock):
        """Test daily sentiment aggregation."""
        mock_session.query.return_value.filter.return_value.first.return_value = mock_stock

        mock_records = [
            MagicMock(
                id=1,
                timestamp=datetime(2024, 1, 15, 10, 0),
                source=MagicMock(value="reddit"),
                sentiment_score=Decimal("-0.5"),
                sentiment_volume=50,
                velocity=None,
                bot_fraction=None,
                coordination_score=None,
                raw_data={"manipulation_score": 0.6},
            ),
            MagicMock(
                id=2,
                timestamp=datetime(2024, 1, 15, 14, 0),
                source=MagicMock(value="stocktwits"),
                sentiment_score=Decimal("-0.3"),
                sentiment_volume=30,
                velocity=None,
                bot_fraction=None,
                coordination_score=None,
                raw_data={"manipulation_score": 0.4},
            ),
        ]

        mock_query = MagicMock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = mock_records
        mock_session.query.return_value = mock_query

        result = archive.get_daily_sentiment(
            "AAPL",
            date(2024, 1, 15),
            date(2024, 1, 15),
        )

        assert len(result) == 1
        assert result[0]["date"] == date(2024, 1, 15)
        assert result[0]["sentiment_volume"] == 80
        assert abs(result[0]["sentiment_score"] - (-0.4)) < 0.01
        assert result[0]["snapshot_count"] == 2


class TestCreateArchivalJob:
    """Tests for create_archival_job function."""

    def test_create_job_returns_callable(self):
        """Test that create_archival_job returns a callable."""
        mock_archive = MagicMock()
        mock_agent = MagicMock()
        mock_universe = ["AAPL", "MSFT"]

        job = create_archival_job(mock_archive, mock_agent, mock_universe)

        assert callable(job)
        assert job.__name__ == "archive_sentiment"

    def test_job_iterates_symbols(self):
        """Test job iterates through all symbols."""
        mock_archive = MagicMock()
        mock_agent = MagicMock()
        mock_agent.get_sentiment = MagicMock(return_value={"composite_score": 0.5})
        mock_universe = ["AAPL", "MSFT", "GOOGL"]

        job = create_archival_job(mock_archive, mock_agent, mock_universe)
        result = job()

        assert mock_agent.get_sentiment.call_count == 3
        assert mock_archive.archive_snapshot.call_count == 3
        assert result["archived"] == 3
        assert result["errors"] == 0

    def test_job_handles_universe_with_method(self):
        """Test job handles universe with get_all_symbols method."""
        mock_archive = MagicMock()
        mock_agent = MagicMock()
        mock_agent.get_sentiment = MagicMock(return_value={"composite_score": 0.5})

        mock_universe = MagicMock()
        mock_universe.get_all_symbols = MagicMock(return_value=["AAPL", "MSFT"])

        job = create_archival_job(mock_archive, mock_agent, mock_universe)
        result = job()

        mock_universe.get_all_symbols.assert_called_once()
        assert result["archived"] == 2

    def test_job_handles_errors_gracefully(self):
        """Test job continues on individual symbol errors."""
        mock_archive = MagicMock()
        mock_archive.archive_snapshot.side_effect = [None, Exception("DB error"), None]

        mock_agent = MagicMock()
        mock_agent.get_sentiment = MagicMock(return_value={"composite_score": 0.5})
        mock_universe = ["AAPL", "MSFT", "GOOGL"]

        job = create_archival_job(mock_archive, mock_agent, mock_universe)
        result = job()

        assert result["archived"] == 2
        assert result["errors"] == 1

    def test_job_uses_aggregate_sentiment_fallback(self):
        """Test job falls back to aggregate_sentiment if get_sentiment unavailable."""
        mock_archive = MagicMock()

        mock_result = MagicMock()
        mock_result.to_dict = MagicMock(return_value={"composite_score": 0.5})

        mock_agent = MagicMock(spec=["aggregate_sentiment"])
        mock_agent.aggregate_sentiment = MagicMock(return_value=mock_result)

        mock_universe = ["AAPL"]

        job = create_archival_job(mock_archive, mock_agent, mock_universe)
        result = job()

        mock_agent.aggregate_sentiment.assert_called_once_with("AAPL", [])
        assert result["archived"] == 1

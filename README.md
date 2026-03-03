# Leprechaun

A stock trading bot that uses sentiment analysis and manipulation detection to execute a contrarian trading strategy: "buy the bad press, sell on the correction."

## Overview

Leprechaun monitors social media sentiment for S&P 500 and NASDAQ 100 stocks, detects when negative sentiment artificially depresses prices without fundamental cause, and executes trades based on manipulation signals. The system enters positions on Mondays and exits on Fridays.

### Key Features

- **Sentiment Analysis** - Aggregates sentiment from Reddit, StockTwits, and financial news using VADER and FinBERT models
- **Manipulation Detection** - Bayesian inference to identify coordinated sentiment attacks without news catalysts
- **Technical Analysis** - RSI, EMA, Bollinger Bands, and MACD indicators to confirm oversold conditions
- **Risk Management** - Automatic position sizing, stop-losses, and trading halts on excessive losses
- **Monday/Friday Strategy** - Enters positions Monday (buy the dip), exits Friday (sell the correction)
- **Real-time Alerts** - Discord webhooks and Firebase push notifications

### Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.11+, Flask |
| Database | MariaDB 11 |
| Cache | Redis 7 |
| Trading | Alpaca API |
| Sentiment | PRAW, Finnhub, RapidAPI (StockTwits) |
| ML Models | VADER, FinBERT (transformers) |
| Deployment | Docker, Kubernetes (Kustomize) |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (For Kubernetes) kubectl, kustomize

### Local Development with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd leprechaun

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API credentials (see API Credentials section below)

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f leprechaun-api

# Stop services
docker-compose down
```

The API will be available at `http://localhost:8000`.

### Local Development without Docker

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration

# Start MariaDB and Redis (required)
# Option 1: Use Docker for just the databases
docker-compose up -d mariadb redis

# Option 2: Install locally and configure DB_HOST/REDIS_HOST in .env

# Run the application
flask run --host=0.0.0.0 --port=8000

# Or with gunicorn (production)
gunicorn -w 4 -b 0.0.0.0:8000 "src.api.app:create_app()"
```

## API Credentials Required

The following external services require API credentials:

### Alpaca (Trading)

- **Sign up**: https://alpaca.markets
- **Credential type**: API Key + Secret
- **Environment variables**: `ALPACA_API_KEY`, `ALPACA_API_SECRET`
- **Notes**: Paper trading is free, no deposit required. Set `ALPACA_PAPER=true` for testing.

### Reddit (Sentiment Data)

- **Sign up**: https://www.reddit.com/prefs/apps
- **Credential type**: OAuth Application (select "script" type)
- **Environment variables**: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
- **Notes**: Free tier provides 10K requests/month with streaming support.

### Finnhub (News & Sentiment)

- **Sign up**: https://finnhub.io
- **Credential type**: API Key
- **Environment variable**: `FINNHUB_API_KEY`
- **Notes**: Free tier allows 60 requests/minute.

### StockTwits via RapidAPI (Social Sentiment)

- **Sign up**: https://rapidapi.com
- **Credential type**: RapidAPI Key
- **Environment variables**: `STOCKTWITS_API_KEY`, `STOCKTWITS_API_HOST`
- **Notes**: Search for "StockTwits" on RapidAPI. Free tier provides 500K requests/month.

### Discord (Notifications)

- **Setup**: Server Settings → Integrations → Webhooks → New Webhook
- **Credential type**: Webhook URL
- **Environment variable**: `DISCORD_WEBHOOK_URL`
- **Notes**: Free, no account required beyond Discord server access.

### Firebase (Push Notifications)

- **Sign up**: https://console.firebase.google.com
- **Credential type**: Service Account JSON file
- **Environment variables**: `FIREBASE_CREDENTIALS_PATH`, `FIREBASE_ENABLED`
- **Notes**: Optional. Download credentials from Project Settings → Service Accounts.

## Project Structure

```
leprechaun/
├── src/
│   ├── api/                    # Flask REST API
│   │   ├── app.py              # Application factory
│   │   ├── routes/             # API endpoint handlers
│   │   │   ├── auth.py         # Authentication (login, logout, refresh)
│   │   │   ├── trading.py      # Trading operations
│   │   │   ├── sentiment.py    # Sentiment data
│   │   │   ├── metrics.py      # Portfolio metrics
│   │   │   └── alerts.py       # Alert management
│   │   └── middleware/         # Auth and rate limiting
│   ├── core/                   # Business logic
│   │   ├── trading/            # Order execution, strategy, risk
│   │   ├── sentiment/          # Sentiment analysis, manipulation detection
│   │   ├── technical/          # Technical indicators, signals
│   │   └── scheduler/          # Scheduled jobs, market calendar
│   ├── data/                   # Data access layer
│   │   └── models.py           # SQLAlchemy models
│   ├── agents/                 # Agentic AI components
│   └── utils/                  # Config, logging, metrics, notifications
├── web/                        # Flask web UI (templates, static)
├── k8s/                        # Kubernetes manifests
│   ├── base/                   # Base resources
│   ├── mariadb/                # MariaDB operator config
│   └── overlays/               # Environment-specific configs
│       ├── paper/              # Paper trading environment
│       └── live/               # Live trading environment
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── backtesting/            # Backtesting framework
├── docker-compose.yaml         # Local development setup
├── Dockerfile                  # Container image
├── requirements.txt            # Python dependencies
└── .env.example                # Environment template
```

## API Endpoints

All endpoints (except `/health` and `/api/v1/auth/login`) require JWT authentication.

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/login` | Authenticate and get JWT tokens |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| POST | `/api/v1/auth/logout` | Invalidate current token |
| GET | `/api/v1/auth/me` | Get current user info |

### Trading

| Method | Endpoint | Description | Role |
|--------|----------|-------------|------|
| GET | `/api/v1/trading/status` | Bot status (running/stopped/halted) | viewer |
| POST | `/api/v1/trading/start` | Start trading bot | admin |
| POST | `/api/v1/trading/stop` | Stop trading bot | admin |
| GET | `/api/v1/trading/positions` | List current positions | viewer |
| GET | `/api/v1/trading/orders` | List recent orders | viewer |
| POST | `/api/v1/trading/halt/resume` | Resume from halt | admin |

### Sentiment

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/sentiment/{symbol}` | Sentiment data for a stock |
| GET | `/api/v1/sentiment/trending` | Trending stocks by sentiment |
| GET | `/api/v1/manipulation/{symbol}` | Manipulation score for a stock |

### Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/metrics/portfolio` | Current portfolio metrics |
| GET | `/api/v1/metrics/performance` | Historical performance |
| GET | `/api/v1/metrics/trades` | Trade history |
| GET | `/api/v1/metrics/daily` | Daily P&L data |
| GET | `/api/v1/metrics/weekly` | Weekly P&L data |

### Alerts

| Method | Endpoint | Description | Role |
|--------|----------|-------------|------|
| GET | `/api/v1/alerts` | Alert history | viewer |
| POST | `/api/v1/alerts/test` | Send test alert | admin |
| GET | `/api/v1/alerts/settings` | Get alert settings | viewer |
| PUT | `/api/v1/alerts/settings` | Update alert settings | admin |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (no auth required) |

## Kubernetes Deployment

Leprechaun uses Kustomize for Kubernetes deployments with separate overlays for paper and live trading.

### Prerequisites

```bash
# Install MariaDB Operator
helm repo add mariadb-operator https://helm.mariadb.com/mariadb-operator
helm install mariadb-operator-crds mariadb-operator/mariadb-operator-crds
helm install mariadb-operator mariadb-operator/mariadb-operator
```

### Deploy to Paper Trading

```bash
# Create secrets (edit k8s/base/secret.yaml first)
kubectl create namespace leprechaun

# Apply paper trading overlay
kubectl apply -k k8s/overlays/paper

# Check deployment status
kubectl -n leprechaun get pods
kubectl -n leprechaun logs -f deployment/leprechaun-api
```

### Deploy to Live Trading

```bash
# Apply live trading overlay
kubectl apply -k k8s/overlays/live
```

### Directory Structure

- `k8s/base/` - Base deployment, service, configmap, secrets
- `k8s/mariadb/` - MariaDB operator resources
- `k8s/overlays/paper/` - Paper trading configuration
- `k8s/overlays/live/` - Live trading configuration (ALPACA_PAPER=false)

## Testing

### Run Unit Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all unit tests
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_api_routes_trading.py -v
```

### Run Integration Tests

```bash
# Requires running database and Redis
pytest tests/integration -v
```

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Check without modifying
black --check src/ tests/
```

## Backtesting

Leprechaun includes a backtesting framework that can use either archived historical sentiment or synthetic sentiment generated from price/volume patterns.

### Running Backtests

```bash
# Basic backtest with synthetic sentiment
python scripts/backtest.py --start 2024-01-01 --end 2024-12-31 --initial-cash 100000

# Force synthetic sentiment even if archive exists
python scripts/backtest.py --start 2024-01-01 --end 2024-12-31 --use-synthetic

# Technical indicators only (no sentiment)
python scripts/backtest.py --start 2024-01-01 --end 2024-12-31 --technicals-only

# Specific symbols with JSON output
python scripts/backtest.py --start 2024-01-01 --end 2024-12-31 --symbols AAPL,MSFT,GOOGL --output results.json
```

### Sentiment Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Archived** | Real historical sentiment data | Recent periods with collected data |
| **Synthetic** | Generated from price/volume | Any historical period |

Synthetic sentiment uses price drop patterns and volume anomalies to approximate what real sentiment might have been. This enables backtesting for periods before you started collecting data.

### Sentiment Archival

The system automatically archives sentiment during live trading (hourly during market hours). Over time, this builds a database of real sentiment for more accurate backtesting.

## Trading Parameters

Default parameters can be customized via environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PROFIT_TARGET_PERCENT` | 2.5 | Take profit target (%) |
| `STOP_LOSS_PERCENT` | 1.25 | Stop loss threshold (%) |
| `POSITION_RISK_PERCENT` | 1.0 | Max risk per position (% of account) |
| `DAILY_LOSS_LIMIT_PERCENT` | 2.0 | Daily loss limit before halt (%) |
| `WEEKLY_LOSS_LIMIT_PERCENT` | 5.0 | Weekly loss limit before halt (%) |
| `RSI_OVERSOLD` | 35 | RSI threshold for oversold condition |
| `MANIPULATION_THRESHOLD` | 0.5 | Minimum manipulation score for signals |

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

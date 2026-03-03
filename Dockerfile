# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libmariadb-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmariadb3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 leprechaun && \
    useradd --uid 1000 --gid leprechaun --shell /bin/bash --create-home leprechaun

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Copy source code
COPY --chown=leprechaun:leprechaun src/ ./src/
COPY --chown=leprechaun:leprechaun web/ ./web/

# Switch to non-root user
USER leprechaun

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--threads", "2", \
     "--worker-class", "gthread", "--timeout", "120", "--access-logfile", "-", \
     "--error-logfile", "-", "--factory", "src.api.app:create_app"]

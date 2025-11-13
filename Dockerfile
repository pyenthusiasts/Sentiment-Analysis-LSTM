# Multi-stage Dockerfile for Sentiment Analysis LSTM

# Stage 1: Builder
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY README.md .
COPY LICENSE .

# Install the package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

# Expose port for API (if running API)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sentiment_analysis; print('healthy')" || exit 1

# Default command
CMD ["python", "-m", "sentiment_analysis.cli"]

# Labels
LABEL maintainer="your.email@example.com" \
      version="1.0.0" \
      description="Sentiment Analysis with LSTM - Production Ready"

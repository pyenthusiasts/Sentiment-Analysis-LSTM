#!/bin/bash
# Start the FastAPI server

set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

# Configuration
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
WORKERS=${API_WORKERS:-4}
RELOAD=${API_RELOAD:-false}

echo -e "${GREEN}Starting Sentiment Analysis API...${NC}"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Reload: $RELOAD"

# Check if model exists
if [ ! -f "models/sentiment_lstm_model.h5" ]; then
    echo "Warning: Model file not found. Please train a model first."
    echo "Run: make train"
fi

# Start server
if [ "$RELOAD" = "true" ]; then
    uvicorn sentiment_analysis.api:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload
else
    uvicorn sentiment_analysis.api:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS"
fi

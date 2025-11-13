#!/bin/bash
# Training script with error handling and logging

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-128}
LOG_DIR="logs"
MODEL_DIR="models"

# Create directories
mkdir -p "$LOG_DIR" "$MODEL_DIR"

# Log file
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo -e "${GREEN}Starting model training...${NC}"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Log file: $LOG_FILE"

# Train model
python -m sentiment_analysis.cli train \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --verbose 2 \
    2>&1 | tee "$LOG_FILE"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo "Model saved to: $MODEL_DIR/sentiment_lstm_model.h5"
else
    echo -e "${RED}Training failed! Check log file: $LOG_FILE${NC}"
    exit 1
fi

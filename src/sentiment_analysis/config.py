"""
Configuration settings for the sentiment analysis model.
"""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model hyperparameters
class ModelConfig:
    """Configuration class for model hyperparameters."""

    # Data parameters
    VOCAB_SIZE = 10000
    MAX_LENGTH = 300

    # Model architecture parameters
    EMBEDDING_DIM = 128
    LSTM_UNITS_1 = 64
    LSTM_UNITS_2 = 32
    DROPOUT_RATE = 0.5

    # Training parameters
    BATCH_SIZE = 128
    EPOCHS = 5
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 0.001

    # Model file paths
    MODEL_PATH = MODELS_DIR / "sentiment_lstm_model.h5"
    TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
    HISTORY_PATH = MODELS_DIR / "training_history.json"

    @classmethod
    def to_dict(cls):
        """Convert configuration to dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and key.isupper()
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

"""Tests for configuration module."""

import pytest
from sentiment_analysis.config import ModelConfig, DATA_DIR, MODELS_DIR


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_config_attributes(self):
        """Test that config has required attributes."""
        assert hasattr(ModelConfig, 'VOCAB_SIZE')
        assert hasattr(ModelConfig, 'MAX_LENGTH')
        assert hasattr(ModelConfig, 'EMBEDDING_DIM')
        assert hasattr(ModelConfig, 'BATCH_SIZE')
        assert hasattr(ModelConfig, 'EPOCHS')

    def test_config_values(self):
        """Test that config values are correct types."""
        assert isinstance(ModelConfig.VOCAB_SIZE, int)
        assert isinstance(ModelConfig.MAX_LENGTH, int)
        assert isinstance(ModelConfig.BATCH_SIZE, int)
        assert isinstance(ModelConfig.EPOCHS, int)
        assert isinstance(ModelConfig.DROPOUT_RATE, float)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config_dict = ModelConfig.to_dict()
        assert isinstance(config_dict, dict)
        assert 'VOCAB_SIZE' in config_dict
        assert 'MAX_LENGTH' in config_dict

    def test_config_from_dict(self):
        """Test updating config from dictionary."""
        original_vocab = ModelConfig.VOCAB_SIZE
        ModelConfig.from_dict({'VOCAB_SIZE': 5000})
        assert ModelConfig.VOCAB_SIZE == 5000
        # Reset
        ModelConfig.VOCAB_SIZE = original_vocab

    def test_directories_exist(self):
        """Test that required directories are created."""
        assert DATA_DIR.exists()
        assert MODELS_DIR.exists()

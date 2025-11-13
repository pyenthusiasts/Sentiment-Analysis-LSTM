"""Tests for model module."""

import pytest
import numpy as np
from sentiment_analysis.model import SentimentLSTM
from sentiment_analysis.config import ModelConfig


class TestSentimentLSTM:
    """Test cases for SentimentLSTM class."""

    @pytest.fixture
    def model(self):
        """Create a SentimentLSTM instance."""
        return SentimentLSTM()

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.config is not None
        assert model.model is None

    def test_build_model_bidirectional(self, model):
        """Test building bidirectional LSTM model."""
        keras_model = model.build_model(bidirectional=True)

        assert keras_model is not None
        assert len(keras_model.layers) > 0
        assert keras_model.layers[-1].output_shape == (None, 1)

    def test_build_model_unidirectional(self, model):
        """Test building unidirectional LSTM model."""
        keras_model = model.build_model(bidirectional=False)

        assert keras_model is not None
        assert len(keras_model.layers) > 0

    def test_build_model_with_spatial_dropout(self, model):
        """Test building model with spatial dropout."""
        keras_model = model.build_model(spatial_dropout=True)

        assert keras_model is not None
        # Check that SpatialDropout is in the model
        layer_names = [layer.__class__.__name__ for layer in keras_model.layers]
        assert 'SpatialDropout1D' in layer_names

    def test_get_model_summary(self, model):
        """Test getting model summary."""
        model.build_model()
        summary = model.get_model_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'lstm' in summary.lower()

    def test_get_callbacks(self, model):
        """Test getting training callbacks."""
        callbacks = model.get_callbacks(patience=3)

        assert isinstance(callbacks, list)
        assert len(callbacks) > 0

    def test_predict_sentiment(self, model):
        """Test sentiment prediction."""
        model.build_model()

        # Create dummy input
        dummy_input = np.random.randint(0, 1000, (1, ModelConfig.MAX_LENGTH))

        score, sentiment = model.predict_sentiment(dummy_input)

        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert sentiment in ['Positive', 'Negative']

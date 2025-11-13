"""Tests for data loader module."""

import pytest
import numpy as np
from sentiment_analysis.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance."""
        return DataLoader(vocab_size=1000, max_length=100)

    def test_initialization(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader.vocab_size == 1000
        assert data_loader.max_length == 100
        assert data_loader.tokenizer is None

    def test_load_imdb_data(self, data_loader):
        """Test loading IMDB dataset."""
        (X_train, y_train), (X_test, y_test) = data_loader.load_imdb_data()

        # Check shapes
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert X_train.shape[1] == 100  # max_length
        assert X_test.shape[1] == 100

        # Check labels
        assert set(np.unique(y_train)) <= {0, 1}
        assert set(np.unique(y_test)) <= {0, 1}

    def test_preprocess_text(self, data_loader):
        """Test text preprocessing."""
        text = "This is a test review."
        preprocessed = data_loader.preprocess_text(text)

        assert isinstance(preprocessed, np.ndarray)
        assert preprocessed.shape[0] == 1
        assert preprocessed.shape[1] == data_loader.max_length

    def test_get_data_statistics(self, data_loader):
        """Test getting data statistics."""
        (X_train, y_train), (X_test, y_test) = data_loader.load_imdb_data()
        stats = data_loader.get_data_statistics(X_train, y_train, X_test, y_test)

        assert 'train_samples' in stats
        assert 'test_samples' in stats
        assert 'vocab_size' in stats
        assert stats['train_samples'] == len(X_train)
        assert stats['test_samples'] == len(X_test)

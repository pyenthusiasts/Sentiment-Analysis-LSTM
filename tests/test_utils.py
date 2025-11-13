"""Tests for utils module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from sentiment_analysis.utils import (
    setup_logger,
    save_json,
    load_json,
    ensure_dir,
    get_timestamp,
    calculate_metrics,
    format_time
)


class TestUtils:
    """Test cases for utility functions."""

    def test_setup_logger(self):
        """Test logger setup."""
        logger = setup_logger('test_logger')
        assert logger is not None
        assert logger.name == 'test_logger'

    def test_save_and_load_json(self):
        """Test JSON save and load."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save
            test_data = {'key': 'value', 'number': 42}
            save_json(test_data, temp_path)

            # Load
            loaded_data = load_json(temp_path)
            assert loaded_data == test_data

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_ensure_dir(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'test_subdir' / 'nested'
            ensure_dir(test_dir)
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_get_timestamp(self):
        """Test timestamp generation."""
        timestamp = get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        metrics = calculate_metrics(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

        # Check value ranges
        for metric_value in metrics.values():
            assert 0 <= metric_value <= 1

    def test_format_time(self):
        """Test time formatting."""
        assert format_time(30) == '30.0s'
        assert format_time(90) == '1.5m'
        assert format_time(3600) == '1.0h'

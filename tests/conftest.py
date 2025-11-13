"""Pytest configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This movie was fantastic!",
        "Terrible waste of time.",
        "It was okay, nothing special.",
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return np.array([1, 0, 0])


@pytest.fixture
def sample_sequences():
    """Sample sequences for testing."""
    return np.random.randint(0, 1000, (10, 100))

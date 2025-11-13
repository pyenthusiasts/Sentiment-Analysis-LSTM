"""
Utility functions for the sentiment analysis project.
"""

import logging
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime


def setup_logger(name, level=logging.INFO):
    """
    Set up a logger with the specified name and level.

    Args:
        name (str): Logger name
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Create handler
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def save_json(data, filepath):
    """
    Save data to a JSON file.

    Args:
        data (dict): Data to save
        filepath (str): Path to save the file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Load data from a JSON file.

    Args:
        filepath (str): Path to the JSON file

    Returns:
        dict: Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(directory):
    """
    Ensure that a directory exists.

    Args:
        directory (str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_timestamp():
    """
    Get current timestamp as string.

    Returns:
        str: Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

    return metrics


def format_time(seconds):
    """
    Format seconds into human-readable time string.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_section(title, width=70):
    """
    Print a formatted section title.

    Args:
        title (str): Section title
        width (int): Width of the section
    """
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width + "\n")


class EarlyStoppingMonitor:
    """Monitor for implementing custom early stopping logic."""

    def __init__(self, patience=5, min_delta=0.001):
        """
        Initialize the early stopping monitor.

        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        """
        Check if training should stop.

        Args:
            val_loss (float): Current validation loss

        Returns:
            bool: Whether to stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import random
        random.seed(seed)
    except ImportError:
        pass

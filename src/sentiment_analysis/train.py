"""
Training utilities for the sentiment analysis model.
"""

import json
import logging
from pathlib import Path
import numpy as np

from sentiment_analysis.model import SentimentLSTM
from sentiment_analysis.data_loader import DataLoader
from sentiment_analysis.config import ModelConfig
from sentiment_analysis.utils import setup_logger

logger = setup_logger(__name__)


class Trainer:
    """Class for training the sentiment analysis model."""

    def __init__(self, config=None):
        """
        Initialize the Trainer.

        Args:
            config (ModelConfig): Configuration object
        """
        self.config = config or ModelConfig()
        self.model = None
        self.data_loader = DataLoader(
            vocab_size=self.config.VOCAB_SIZE,
            max_length=self.config.MAX_LENGTH
        )
        self.history = None

    def prepare_data(self):
        """
        Prepare the dataset for training.

        Returns:
            tuple: Training and test data
        """
        logger.info("Preparing data...")
        (X_train, y_train), (X_test, y_test) = self.data_loader.load_imdb_data()

        # Log data statistics
        stats = self.data_loader.get_data_statistics(X_train, y_train, X_test, y_test)
        logger.info(f"Data statistics: {stats}")

        return (X_train, y_train), (X_test, y_test)

    def train(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        epochs=None,
        batch_size=None,
        validation_split=None,
        bidirectional=True,
        spatial_dropout=False,
        patience=3,
        verbose=2
    ):
        """
        Train the sentiment analysis model.

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test data (optional)
            y_test (np.ndarray): Test labels (optional)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            bidirectional (bool): Whether to use bidirectional LSTM
            spatial_dropout (bool): Whether to use spatial dropout
            patience (int): Patience for early stopping
            verbose (int): Verbosity mode

        Returns:
            dict: Training history
        """
        # Use config defaults if not specified
        epochs = epochs or self.config.EPOCHS
        batch_size = batch_size or self.config.BATCH_SIZE
        validation_split = validation_split or self.config.VALIDATION_SPLIT

        logger.info("Starting training...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Validation split: {validation_split}")

        # Build the model
        sentiment_model = SentimentLSTM(self.config)
        sentiment_model.build_model(
            bidirectional=bidirectional,
            spatial_dropout=spatial_dropout
        )
        self.model = sentiment_model

        # Get callbacks
        callbacks = sentiment_model.get_callbacks(patience=patience)

        # Train the model
        history = sentiment_model.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        self.history = history.history

        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            logger.info("Evaluating on test set...")
            test_loss, test_accuracy = sentiment_model.model.evaluate(
                X_test, y_test, verbose=0
            )
            logger.info(f"Test Loss: {test_loss:.4f}")
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")

            # Add test metrics to history
            self.history['test_loss'] = test_loss
            self.history['test_accuracy'] = test_accuracy

        # Save the model
        sentiment_model.save_model()

        # Save training history
        self.save_history()

        logger.info("Training completed successfully")

        return self.history

    def save_history(self, path=None):
        """
        Save training history to JSON file.

        Args:
            path (str): Path to save the history
        """
        if self.history is None:
            logger.warning("No training history to save")
            return

        save_path = path or self.config.HISTORY_PATH

        # Convert numpy types to native Python types for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, (list, np.ndarray)):
                history_serializable[key] = [float(v) for v in value]
            else:
                history_serializable[key] = float(value)

        with open(save_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)

        logger.info(f"Training history saved to {save_path}")

    def load_history(self, path=None):
        """
        Load training history from JSON file.

        Args:
            path (str): Path to the history file

        Returns:
            dict: Training history
        """
        load_path = path or self.config.HISTORY_PATH

        with open(load_path, 'r') as f:
            self.history = json.load(f)

        logger.info(f"Training history loaded from {load_path}")
        return self.history

    def get_best_epoch(self):
        """
        Get the epoch with the best validation accuracy.

        Returns:
            tuple: (best_epoch, best_val_accuracy)
        """
        if self.history is None or 'val_accuracy' not in self.history:
            return None, None

        val_accuracies = self.history['val_accuracy']
        best_epoch = np.argmax(val_accuracies)
        best_val_accuracy = val_accuracies[best_epoch]

        return best_epoch + 1, best_val_accuracy

"""
Data loading and preprocessing utilities.
"""

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import logging

from sentiment_analysis.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading and preprocessing data for sentiment analysis."""

    def __init__(self, vocab_size=None, max_length=None):
        """
        Initialize the DataLoader.

        Args:
            vocab_size (int): Maximum number of words to keep in vocabulary
            max_length (int): Maximum length of sequences
        """
        self.vocab_size = vocab_size or ModelConfig.VOCAB_SIZE
        self.max_length = max_length or ModelConfig.MAX_LENGTH
        self.tokenizer = None
        self.word_index = None

    def load_imdb_data(self):
        """
        Load and preprocess the IMDB dataset.

        Returns:
            tuple: (X_train, y_train), (X_test, y_test)
        """
        logger.info(f"Loading IMDB dataset with vocab_size={self.vocab_size}")

        # Load the IMDB dataset
        (X_train, y_train), (X_test, y_test) = imdb.load_data(
            num_words=self.vocab_size
        )

        logger.info(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")

        # Pad sequences to ensure uniform input size
        X_train = pad_sequences(X_train, maxlen=self.max_length)
        X_test = pad_sequences(X_test, maxlen=self.max_length)

        # Get word index for reference
        self.word_index = imdb.get_word_index()

        logger.info(f"Sequences padded to max_length={self.max_length}")

        return (X_train, y_train), (X_test, y_test)

    def preprocess_text(self, text, tokenizer=None):
        """
        Preprocess a text string for prediction.

        Args:
            text (str): Input text to preprocess
            tokenizer (Tokenizer): Keras tokenizer (optional)

        Returns:
            np.ndarray: Preprocessed and padded sequence
        """
        if tokenizer is None:
            # Create a new tokenizer if not provided
            tokenizer = Tokenizer(num_words=self.vocab_size)
            tokenizer.fit_on_texts([text])

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([text])

        # Pad sequence
        padded = pad_sequences(sequence, maxlen=self.max_length)

        return padded

    def save_tokenizer(self, tokenizer, path):
        """
        Save tokenizer to file.

        Args:
            tokenizer: Keras tokenizer object
            path (str): Path to save the tokenizer
        """
        with open(path, 'wb') as f:
            pickle.dump(tokenizer, f)
        logger.info(f"Tokenizer saved to {path}")

    def load_tokenizer(self, path):
        """
        Load tokenizer from file.

        Args:
            path (str): Path to the tokenizer file

        Returns:
            Tokenizer: Loaded tokenizer object
        """
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)
        logger.info(f"Tokenizer loaded from {path}")
        return tokenizer

    def decode_review(self, encoded_review):
        """
        Decode an encoded review back to text.

        Args:
            encoded_review (list): List of word indices

        Returns:
            str: Decoded text
        """
        if self.word_index is None:
            self.word_index = imdb.get_word_index()

        # Reverse word index
        reverse_word_index = {value: key for key, value in self.word_index.items()}

        # Decode the review (indices are offset by 3)
        decoded = ' '.join([
            reverse_word_index.get(i - 3, '?') for i in encoded_review
        ])

        return decoded

    def get_data_statistics(self, X_train, y_train, X_test, y_test):
        """
        Get statistics about the dataset.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            dict: Dictionary containing dataset statistics
        """
        stats = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_positive': np.sum(y_train == 1),
            'train_negative': np.sum(y_train == 0),
            'test_positive': np.sum(y_test == 1),
            'test_negative': np.sum(y_test == 0),
            'sequence_length': X_train.shape[1] if len(X_train.shape) > 1 else 0,
            'vocab_size': self.vocab_size
        }

        return stats

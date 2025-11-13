"""
Prediction utilities for the sentiment analysis model.
"""

import logging
import numpy as np
from pathlib import Path

from sentiment_analysis.model import SentimentLSTM
from sentiment_analysis.data_loader import DataLoader
from sentiment_analysis.config import ModelConfig
from sentiment_analysis.utils import setup_logger

logger = setup_logger(__name__)


class Predictor:
    """Class for making predictions with the sentiment analysis model."""

    def __init__(self, model_path=None, config=None):
        """
        Initialize the Predictor.

        Args:
            model_path (str): Path to the saved model
            config (ModelConfig): Configuration object
        """
        self.config = config or ModelConfig()
        self.model_path = model_path or self.config.MODEL_PATH

        # Initialize model and data loader
        self.sentiment_model = SentimentLSTM(self.config)
        self.data_loader = DataLoader(
            vocab_size=self.config.VOCAB_SIZE,
            max_length=self.config.MAX_LENGTH
        )

        # Load the model if it exists
        if Path(self.model_path).exists():
            self.load_model()
        else:
            logger.warning(f"Model not found at {self.model_path}")

    def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")
        self.sentiment_model.load_model(self.model_path)

    def predict_text(self, text):
        """
        Predict sentiment for a single text.

        Args:
            text (str): Input text

        Returns:
            dict: Prediction results with score and sentiment
        """
        if self.sentiment_model.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        # Preprocess the text
        preprocessed = self.data_loader.preprocess_text(text)

        # Make prediction
        score, sentiment = self.sentiment_model.predict_sentiment(preprocessed)

        result = {
            'text': text,
            'score': score,
            'sentiment': sentiment,
            'confidence': abs(score - 0.5) * 2  # Normalize confidence to 0-1
        }

        return result

    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts.

        Args:
            texts (list): List of input texts

        Returns:
            list: List of prediction results
        """
        results = []

        logger.info(f"Predicting sentiment for {len(texts)} texts...")

        for text in texts:
            result = self.predict_text(text)
            results.append(result)

        return results

    def predict_with_explanation(self, text, top_k=10):
        """
        Predict sentiment with word importance explanation.

        Args:
            text (str): Input text
            top_k (int): Number of top important words to return

        Returns:
            dict: Prediction results with word importance
        """
        # Get base prediction
        result = self.predict_text(text)

        # TODO: Implement gradient-based or attention-based word importance
        # This is a placeholder for future enhancement
        result['explanation'] = "Word importance analysis not yet implemented"

        return result

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on a test set.

        Args:
            X_test (np.ndarray): Test data
            y_test (np.ndarray): Test labels

        Returns:
            dict: Evaluation metrics
        """
        if self.sentiment_model.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        logger.info("Evaluating model...")

        # Evaluate
        test_loss, test_accuracy = self.sentiment_model.model.evaluate(
            X_test, y_test, verbose=0
        )

        # Get predictions for detailed metrics
        predictions = self.sentiment_model.model.predict(X_test, verbose=0)
        predictions_binary = (predictions > 0.5).astype(int).flatten()

        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix

        conf_matrix = confusion_matrix(y_test, predictions_binary)
        class_report = classification_report(
            y_test, predictions_binary,
            target_names=['Negative', 'Positive'],
            output_dict=True
        )

        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }

        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")

        return results

    def predict_examples(self):
        """
        Run predictions on example texts.

        Returns:
            list: List of prediction results
        """
        example_texts = [
            "This movie was fantastic! I really enjoyed the story and the acting was superb.",
            "Terrible movie. Waste of time and money. Do not watch.",
            "It was okay. Nothing special but not terrible either.",
            "Absolutely brilliant! One of the best films I've ever seen.",
            "Boring and predictable. The plot was full of holes.",
            "Great performances by the cast. Highly recommended!",
            "Disappointing. Had high expectations but it fell flat.",
            "A masterpiece of cinema. Stunning visuals and storytelling.",
        ]

        logger.info("Running predictions on example texts...")
        results = self.predict_batch(example_texts)

        # Print results
        for result in results:
            logger.info(f"\nText: {result['text'][:60]}...")
            logger.info(f"Sentiment: {result['sentiment']} (Score: {result['score']:.4f}, Confidence: {result['confidence']:.4f})")

        return results

"""
Sentiment Analysis with LSTM
A modular package for sentiment analysis using LSTM neural networks.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from sentiment_analysis.model import SentimentLSTM
from sentiment_analysis.data_loader import DataLoader
from sentiment_analysis.train import Trainer
from sentiment_analysis.predict import Predictor

__all__ = ["SentimentLSTM", "DataLoader", "Trainer", "Predictor"]

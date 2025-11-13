"""
Custom exceptions for the sentiment analysis package.
"""


class SentimentAnalysisError(Exception):
    """Base exception for all sentiment analysis errors."""

    pass


class ModelNotFoundError(SentimentAnalysisError):
    """Raised when a model file is not found."""

    def __init__(self, model_path):
        self.model_path = model_path
        super().__init__(f"Model not found at: {model_path}")


class ModelNotTrainedError(SentimentAnalysisError):
    """Raised when attempting to use an untrained model."""

    def __init__(self, message="Model has not been trained yet"):
        super().__init__(message)


class DataLoadError(SentimentAnalysisError):
    """Raised when data loading fails."""

    pass


class InvalidInputError(SentimentAnalysisError):
    """Raised when input data is invalid."""

    pass


class ConfigurationError(SentimentAnalysisError):
    """Raised when configuration is invalid."""

    pass


class PreprocessingError(SentimentAnalysisError):
    """Raised when preprocessing fails."""

    pass


class TrainingError(SentimentAnalysisError):
    """Raised when training fails."""

    pass


class PredictionError(SentimentAnalysisError):
    """Raised when prediction fails."""

    pass


class ValidationError(SentimentAnalysisError):
    """Raised when validation fails."""

    pass

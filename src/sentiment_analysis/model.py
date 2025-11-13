"""
LSTM model architecture for sentiment analysis.
"""

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import logging

from sentiment_analysis.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentLSTM:
    """LSTM-based model for sentiment analysis."""

    def __init__(self, config=None):
        """
        Initialize the sentiment analysis model.

        Args:
            config (ModelConfig): Configuration object with hyperparameters
        """
        self.config = config or ModelConfig()
        self.model = None

    def build_model(self, bidirectional=True, spatial_dropout=False):
        """
        Build the LSTM model architecture.

        Args:
            bidirectional (bool): Whether to use bidirectional LSTM
            spatial_dropout (bool): Whether to use spatial dropout

        Returns:
            Sequential: Compiled Keras model
        """
        logger.info("Building LSTM model...")

        model = Sequential(name='sentiment_lstm')

        # Embedding layer
        model.add(Embedding(
            input_dim=self.config.VOCAB_SIZE,
            output_dim=self.config.EMBEDDING_DIM,
            input_length=self.config.MAX_LENGTH,
            name='embedding'
        ))

        # Optional spatial dropout
        if spatial_dropout:
            model.add(SpatialDropout1D(0.2, name='spatial_dropout'))

        # First LSTM layer (bidirectional option)
        if bidirectional:
            model.add(Bidirectional(
                LSTM(self.config.LSTM_UNITS_1, return_sequences=True),
                name='bidirectional_lstm_1'
            ))
        else:
            model.add(LSTM(
                self.config.LSTM_UNITS_1,
                return_sequences=True,
                name='lstm_1'
            ))

        # Dropout for regularization
        model.add(Dropout(self.config.DROPOUT_RATE, name='dropout_1'))

        # Second LSTM layer
        model.add(LSTM(self.config.LSTM_UNITS_2, name='lstm_2'))

        # Output layer for binary classification
        model.add(Dense(1, activation='sigmoid', name='output'))

        # Compile the model
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        logger.info("Model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")

        return model

    def get_model_summary(self):
        """
        Get a summary of the model architecture.

        Returns:
            str: Model summary as string
        """
        if self.model is None:
            return "Model not built yet"

        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)

    def get_callbacks(self, patience=3):
        """
        Get training callbacks.

        Args:
            patience (int): Patience for early stopping

        Returns:
            list: List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.config.MODEL_PATH),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def save_model(self, path=None):
        """
        Save the trained model.

        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train the model first.")

        save_path = path or self.config.MODEL_PATH
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path=None):
        """
        Load a trained model.

        Args:
            path (str): Path to the saved model

        Returns:
            Sequential: Loaded Keras model
        """
        load_path = path or self.config.MODEL_PATH
        self.model = load_model(load_path)
        logger.info(f"Model loaded from {load_path}")
        return self.model

    def predict_sentiment(self, preprocessed_text):
        """
        Predict sentiment for preprocessed text.

        Args:
            preprocessed_text (np.ndarray): Preprocessed and padded text

        Returns:
            tuple: (prediction score, sentiment label)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load or build a model first.")

        prediction = self.model.predict(preprocessed_text, verbose=0)
        score = float(prediction[0][0])
        sentiment = "Positive" if score > 0.5 else "Negative"

        return score, sentiment

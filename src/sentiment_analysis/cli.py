"""
Command-line interface for sentiment analysis.
"""

import argparse
import sys
import json
from pathlib import Path

from sentiment_analysis.train import Trainer
from sentiment_analysis.predict import Predictor
from sentiment_analysis.config import ModelConfig
from sentiment_analysis.visualization import Visualizer
from sentiment_analysis.utils import setup_logger, print_section

logger = setup_logger(__name__)


def train_command():
    """CLI command for training the model."""
    parser = argparse.ArgumentParser(
        description='Train a sentiment analysis LSTM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  sentiment-train

  # Train with custom epochs and batch size
  sentiment-train --epochs 10 --batch-size 64

  # Train with spatial dropout
  sentiment-train --spatial-dropout --patience 5
        """
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=ModelConfig.EPOCHS,
        help=f'Number of training epochs (default: {ModelConfig.EPOCHS})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=ModelConfig.BATCH_SIZE,
        help=f'Batch size for training (default: {ModelConfig.BATCH_SIZE})'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=ModelConfig.VALIDATION_SPLIT,
        help=f'Validation split ratio (default: {ModelConfig.VALIDATION_SPLIT})'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=ModelConfig.LEARNING_RATE,
        help=f'Learning rate (default: {ModelConfig.LEARNING_RATE})'
    )

    # Model architecture
    parser.add_argument(
        '--no-bidirectional',
        action='store_true',
        help='Use unidirectional LSTM instead of bidirectional'
    )
    parser.add_argument(
        '--spatial-dropout',
        action='store_true',
        help='Use spatial dropout in the model'
    )

    # Early stopping
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Patience for early stopping (default: 3)'
    )

    # Output options
    parser.add_argument(
        '--verbose',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Verbosity mode (0=silent, 1=progress, 2=detailed)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots after training'
    )

    args = parser.parse_args()

    print_section("Sentiment Analysis LSTM - Training")

    # Update config with command-line arguments
    if args.learning_rate != ModelConfig.LEARNING_RATE:
        ModelConfig.LEARNING_RATE = args.learning_rate
        logger.info(f"Learning rate set to {args.learning_rate}")

    # Initialize trainer
    trainer = Trainer()

    # Prepare data
    (X_train, y_train), (X_test, y_test) = trainer.prepare_data()

    # Train model
    history = trainer.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        bidirectional=not args.no_bidirectional,
        spatial_dropout=args.spatial_dropout,
        patience=args.patience,
        verbose=args.verbose
    )

    # Get best epoch
    best_epoch, best_val_acc = trainer.get_best_epoch()
    if best_epoch:
        logger.info(f"\nBest epoch: {best_epoch} (Val Accuracy: {best_val_acc:.4f})")

    # Plot results
    if not args.no_plots:
        logger.info("Generating training plots...")
        visualizer = Visualizer()
        visualizer.plot_training_history(history)

    print_section("Training Complete!")
    logger.info(f"Model saved to: {ModelConfig.MODEL_PATH}")
    logger.info(f"History saved to: {ModelConfig.HISTORY_PATH}")


def predict_command():
    """CLI command for making predictions."""
    parser = argparse.ArgumentParser(
        description='Predict sentiment using trained LSTM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict sentiment for a single text
  sentiment-predict --text "This movie was amazing!"

  # Run predictions on example texts
  sentiment-predict --examples

  # Predict from a file (one text per line)
  sentiment-predict --file reviews.txt

  # Save predictions to JSON
  sentiment-predict --examples --output predictions.json
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text',
        type=str,
        help='Text to analyze'
    )
    input_group.add_argument(
        '--file',
        type=Path,
        help='File containing texts (one per line)'
    )
    input_group.add_argument(
        '--examples',
        action='store_true',
        help='Run predictions on example texts'
    )

    # Model options
    parser.add_argument(
        '--model',
        type=Path,
        help=f'Path to trained model (default: {ModelConfig.MODEL_PATH})'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=Path,
        help='Save predictions to JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed prediction information'
    )

    args = parser.parse_args()

    print_section("Sentiment Analysis LSTM - Prediction")

    # Initialize predictor
    model_path = args.model or ModelConfig.MODEL_PATH
    predictor = Predictor(model_path=model_path)

    results = []

    # Handle different input types
    if args.text:
        # Single text prediction
        result = predictor.predict_text(args.text)
        results = [result]

        # Display result
        logger.info(f"\nText: {result['text']}")
        logger.info(f"Sentiment: {result['sentiment']}")
        logger.info(f"Score: {result['score']:.4f}")
        logger.info(f"Confidence: {result['confidence']:.4f}")

    elif args.file:
        # Batch prediction from file
        if not args.file.exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)

        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(texts)} texts from {args.file}")
        results = predictor.predict_batch(texts)

        # Display results
        for i, result in enumerate(results, 1):
            logger.info(f"\n[{i}] {result['text'][:60]}...")
            logger.info(f"    Sentiment: {result['sentiment']} (Score: {result['score']:.4f})")

    elif args.examples:
        # Run example predictions
        results = predictor.predict_examples()

    # Save results if output file specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nPredictions saved to: {args.output}")

    print_section("Prediction Complete!")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis with LSTM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.set_defaults(func=train_command)

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.set_defaults(func=predict_command)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

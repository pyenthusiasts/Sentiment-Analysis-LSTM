"""
Basic usage example for the sentiment analysis package.

This script demonstrates how to:
1. Train a sentiment analysis model
2. Make predictions on new text
3. Visualize training results
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sentiment_analysis.train import Trainer
from sentiment_analysis.predict import Predictor
from sentiment_analysis.visualization import Visualizer


def main():
    """Run basic usage example."""
    print("=" * 70)
    print(" Sentiment Analysis LSTM - Basic Usage Example ".center(70, "="))
    print("=" * 70)

    # Step 1: Train the model
    print("\n[1/3] Training the model...")
    print("-" * 70)

    trainer = Trainer()
    (X_train, y_train), (X_test, y_test) = trainer.prepare_data()

    history = trainer.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=2,  # Use fewer epochs for quick demo
        verbose=1
    )

    # Step 2: Visualize training results
    print("\n[2/3] Visualizing training results...")
    print("-" * 70)

    visualizer = Visualizer(output_dir='outputs')
    visualizer.plot_training_history(history)

    # Step 3: Make predictions
    print("\n[3/3] Making predictions...")
    print("-" * 70)

    predictor = Predictor()

    # Example texts to analyze
    example_texts = [
        "This movie was absolutely fantastic! Best film I've seen this year.",
        "Terrible movie. Complete waste of time and money.",
        "It was okay. Nothing special, but not terrible either.",
        "Outstanding performance by the lead actor. Highly recommended!",
    ]

    print("\nPredictions:")
    for i, text in enumerate(example_texts, 1):
        result = predictor.predict_text(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Confidence: {result['confidence']:.4f}")

    print("\n" + "=" * 70)
    print(" Example Complete! ".center(70, "="))
    print("=" * 70)


if __name__ == '__main__':
    main()

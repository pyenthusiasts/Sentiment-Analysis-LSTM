"""
Prediction-only example using a pre-trained model.

This script demonstrates how to use a trained model for predictions
without retraining.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sentiment_analysis.predict import Predictor
from sentiment_analysis.config import ModelConfig


def main():
    """Run prediction example."""
    print("=" * 70)
    print(" Sentiment Prediction Example ".center(70, "="))
    print("=" * 70)

    # Check if model exists
    if not ModelConfig.MODEL_PATH.exists():
        print("\nError: No trained model found!")
        print(f"Expected model at: {ModelConfig.MODEL_PATH}")
        print("\nPlease train a model first by running:")
        print("  python examples/basic_usage.py")
        print("  or")
        print("  sentiment-train")
        return

    # Initialize predictor
    print("\nLoading trained model...")
    predictor = Predictor()

    # Interactive prediction loop
    print("\n" + "-" * 70)
    print("Enter movie reviews to analyze (or 'quit' to exit)")
    print("-" * 70)

    while True:
        print("\nEnter review text:")
        text = input("> ")

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if not text.strip():
            continue

        # Make prediction
        result = predictor.predict_text(text)

        # Display result
        print("\nPrediction:")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Confidence: {result['confidence']:.2%}")

        # Interpret confidence
        if result['confidence'] > 0.8:
            print("  (Very confident)")
        elif result['confidence'] > 0.5:
            print("  (Moderately confident)")
        else:
            print("  (Low confidence - neutral text)")

    print("\nGoodbye!")


if __name__ == '__main__':
    main()

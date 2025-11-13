"""
Custom training example with advanced options.

This script demonstrates:
1. Custom hyperparameter configuration
2. Different model architectures
3. Advanced training options
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sentiment_analysis.train import Trainer
from sentiment_analysis.config import ModelConfig
from sentiment_analysis.visualization import Visualizer


def main():
    """Run custom training example."""
    print("=" * 70)
    print(" Custom Training Example ".center(70, "="))
    print("=" * 70)

    # Customize configuration
    print("\n[1] Customizing model configuration...")
    ModelConfig.EPOCHS = 3
    ModelConfig.BATCH_SIZE = 64
    ModelConfig.LEARNING_RATE = 0.0005
    ModelConfig.DROPOUT_RATE = 0.3

    print(f"Epochs: {ModelConfig.EPOCHS}")
    print(f"Batch Size: {ModelConfig.BATCH_SIZE}")
    print(f"Learning Rate: {ModelConfig.LEARNING_RATE}")
    print(f"Dropout Rate: {ModelConfig.DROPOUT_RATE}")

    # Initialize trainer
    print("\n[2] Preparing data...")
    trainer = Trainer()
    (X_train, y_train), (X_test, y_test) = trainer.prepare_data()

    # Train with custom options
    print("\n[3] Training with custom options...")
    history = trainer.train(
        X_train,
        y_train,
        X_test,
        y_test,
        bidirectional=True,
        spatial_dropout=True,
        patience=2,
        verbose=1
    )

    # Get best epoch
    best_epoch, best_val_acc = trainer.get_best_epoch()
    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Visualize
    print("\n[4] Creating visualizations...")
    visualizer = Visualizer(output_dir='outputs')
    visualizer.plot_training_history(history)

    print("\n" + "=" * 70)
    print(" Training Complete! ".center(70, "="))
    print("=" * 70)


if __name__ == '__main__':
    main()

"""
Visualization utilities for model training and evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

from sentiment_analysis.utils import setup_logger

logger = setup_logger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Visualizer:
    """Class for creating visualizations of training and evaluation results."""

    def __init__(self, output_dir='outputs'):
        """
        Initialize the Visualizer.

        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history, save_path=None):
        """
        Plot training and validation accuracy and loss.

        Args:
            history (dict): Training history dictionary
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        if 'accuracy' in history:
            axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)

        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot loss
        if 'loss' in history:
            axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)

        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        else:
            save_path = self.output_dir / 'training_history.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()

        return fig

    def plot_confusion_matrix(self, confusion_matrix, labels=None, save_path=None):
        """
        Plot confusion matrix.

        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            labels (list): Class labels
            save_path (str): Path to save the plot
        """
        if labels is None:
            labels = ['Negative', 'Positive']

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )

        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            save_path = self.output_dir / 'confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_prediction_distribution(self, predictions, labels=None, save_path=None):
        """
        Plot distribution of prediction scores.

        Args:
            predictions (np.ndarray): Prediction scores
            labels (np.ndarray): True labels (optional)
            save_path (str): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if labels is not None:
            # Plot separate distributions for positive and negative samples
            pos_preds = predictions[labels == 1]
            neg_preds = predictions[labels == 0]

            ax.hist(neg_preds, bins=50, alpha=0.6, label='Negative Samples', color='red')
            ax.hist(pos_preds, bins=50, alpha=0.6, label='Positive Samples', color='green')
            ax.legend(fontsize=10)
        else:
            ax.hist(predictions, bins=50, alpha=0.7, color='blue')

        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
        ax.set_title('Prediction Score Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prediction Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution saved to {save_path}")
        else:
            save_path = self.output_dir / 'prediction_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution saved to {save_path}")

        plt.show()

    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve.

        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            save_path (str): Path to save the plot
        """
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        else:
            save_path = self.output_dir / 'roc_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")

        plt.show()

    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """
        Plot comparison of multiple metrics.

        Args:
            metrics_dict (dict): Dictionary of metrics
            save_path (str): Path to save the plot
        """
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color='steelblue', alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        else:
            save_path = self.output_dir / 'metrics_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")

        plt.show()

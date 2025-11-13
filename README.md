# Sentiment Analysis with LSTM

<div align="center">

A professional, modular sentiment analysis framework using LSTM neural networks for movie review classification.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Overview

This project provides a complete, production-ready implementation of sentiment analysis using Long Short-Term Memory (LSTM) neural networks. Built on TensorFlow/Keras, it offers a modular architecture, comprehensive testing, multiple interfaces (CLI, Python API, Jupyter notebooks), and extensive documentation.

### Key Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data loading, model building, training, prediction, and visualization
- **Bidirectional LSTM**: Captures dependencies from both forward and backward sequences for improved accuracy
- **Multiple Interfaces**:
  - Python API for programmatic access
  - CLI tools for command-line usage
  - Jupyter notebooks for interactive exploration
- **Comprehensive Testing**: Unit tests with pytest for reliability
- **Production Ready**: Model persistence, logging, configuration management
- **Rich Visualizations**: Training curves, confusion matrices, ROC curves, prediction distributions
- **Easy Installation**: Simple pip installation with all dependencies

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/pyenthusiasts/Sentiment-Analysis-LSTM.git
cd Sentiment-Analysis-LSTM

# Install the package
pip install -e .
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from sentiment_analysis.train import Trainer
from sentiment_analysis.predict import Predictor

# Train a model
trainer = Trainer()
(X_train, y_train), (X_test, y_test) = trainer.prepare_data()
history = trainer.train(X_train, y_train, X_test, y_test)

# Make predictions
predictor = Predictor()
result = predictor.predict_text("Amazing movie! Loved it!")
print(f"Sentiment: {result['sentiment']} (Score: {result['score']:.2f})")
```

## Project Structure

```
Sentiment-Analysis-LSTM/
├── src/sentiment_analysis/      # Main package
│   ├── config.py                # Configuration
│   ├── data_loader.py           # Data loading
│   ├── model.py                 # Model architecture
│   ├── train.py                 # Training logic
│   ├── predict.py               # Prediction logic
│   ├── utils.py                 # Utilities
│   └── visualization.py         # Visualizations
├── tests/                       # Unit tests
├── examples/                    # Example scripts
├── notebooks/                   # Jupyter notebooks
├── requirements.txt             # Dependencies
└── setup.py                     # Package setup
```

## Usage

### Training a Model

```python
from sentiment_analysis.train import Trainer

trainer = Trainer()
(X_train, y_train), (X_test, y_test) = trainer.prepare_data()
history = trainer.train(X_train, y_train, X_test, y_test, epochs=5)
```

### Making Predictions

```python
from sentiment_analysis.predict import Predictor

predictor = Predictor()
result = predictor.predict_text("This movie was amazing!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Model Architecture

1. **Embedding Layer**: 128-dimensional word embeddings
2. **Bidirectional LSTM**: 64 units processing in both directions
3. **Dropout**: 0.5 rate for regularization
4. **LSTM**: 32 units  
5. **Dense Output**: Sigmoid activation for binary classification

## Configuration

Edit `src/sentiment_analysis/config.py` to customize:

- `VOCAB_SIZE`: 10000 (vocabulary size)
- `MAX_LENGTH`: 300 (sequence length)
- `EMBEDDING_DIM`: 128
- `BATCH_SIZE`: 128
- `EPOCHS`: 5

## Examples

Run example scripts:

```bash
python examples/basic_usage.py
python examples/custom_training.py
python examples/prediction_only.py
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sentiment_analysis
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/)
- IMDB dataset from [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/)

---

<div align="center">
Made with passion for NLP and Deep Learning
</div>

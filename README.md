# Sentiment Analysis with LSTM - Movie Reviews

This project demonstrates how to build a deep learning model using an **LSTM (Long Short-Term Memory)** neural network for sentiment analysis of movie reviews. The model is trained on the IMDB dataset to classify movie reviews as positive or negative.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Examples](#examples)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

Sentiment analysis is a common natural language processing (NLP) task that involves classifying text into positive or negative sentiments. This project uses a deep learning approach with a **Bidirectional LSTM** neural network to perform sentiment analysis on movie reviews from the IMDB dataset. LSTMs are well-suited for this task because they are capable of learning long-term dependencies in sequential data.

## Features

- **LSTM-based Neural Network**: A deep learning model using LSTM layers to handle the sequential nature of text data.
- **Bidirectional LSTM**: Utilizes a bidirectional LSTM to capture dependencies from both forward and backward sequences.
- **Embedding Layer**: Converts words into dense vector representations to capture semantic relationships.
- **Dropout Regularization**: Prevents overfitting by randomly dropping neurons during training.
- **Text Preprocessing**: Automatic tokenization and padding to handle varying input lengths.
- **Sentiment Prediction**: Predicts whether a given movie review is positive or negative.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- Matplotlib

### Install Required Packages

If you haven't installed TensorFlow and Matplotlib yet, you can do so using pip:

```bash
pip install tensorflow matplotlib
```

## Usage

1. **Clone the Repository**:

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/sentiment-analysis-lstm.git
   ```

2. **Navigate to the Directory**:

   Go to the project directory:

   ```bash
   cd sentiment-analysis-lstm
   ```

3. **Run the Script**:

   Run the script using Python:

   ```bash
   python sentiment_analysis_lstm.py
   ```

### Running the Program

When you run the script, it will:

- Load the IMDB dataset.
- Preprocess the text data (tokenization and padding).
- Define and compile an LSTM-based neural network model.
- Train the model on the training set and validate it on a validation set.
- Evaluate the model on the test set.
- Plot the training and validation accuracy and loss over epochs.
- Predict the sentiment of a new movie review sample.

## Examples

### Output

The script will produce outputs similar to:

1. **Test Accuracy**: The accuracy of the model on the test dataset, for example:

   ```
   Test Accuracy: 0.86
   ```

2. **Training and Validation Curves**: Plots of accuracy and loss over the training epochs.

   ![Accuracy and Loss Plot](accuracy_loss_plot.png)

3. **Predicted Sentiment**: Displays the predicted sentiment (positive or negative) for a new review.

   ```
   Predicted Sentiment: Positive
   ```

### Predicting New Reviews

To predict the sentiment of a new movie review, you can modify the `new_review` variable in the script:

```python
new_review = "This movie was fantastic! I really enjoyed the story and the acting was superb."
```

Run the script again to see the predicted sentiment.

## Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please feel free to open an issue or create a pull request.

### Steps to Contribute

1. **Fork the Repository**: Click the 'Fork' button at the top right of this page.
2. **Clone Your Fork**: Clone your forked repository to your local machine.
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-lstm.git
   ```
3. **Create a Branch**: Create a new branch for your feature or bug fix.
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Changes**: Make your changes and commit them with a descriptive message.
   ```bash
   git commit -m "Add: feature description"
   ```
5. **Push Changes**: Push your changes to your forked repository.
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**: Go to the original repository on GitHub and create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for using the Sentiment Analysis with LSTM! If you have any questions or feedback, feel free to reach out. Happy coding! 😊

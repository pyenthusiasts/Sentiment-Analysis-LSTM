# Contributing to Sentiment Analysis LSTM

Thank you for your interest in contributing to this project! We welcome contributions from everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

By participating in this project, you are expected to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Types of Contributions

1. **Bug Fixes**: Fix identified bugs in the codebase
2. **New Features**: Implement new functionality
3. **Documentation**: Improve or expand documentation
4. **Tests**: Add or improve test coverage
5. **Examples**: Create helpful usage examples
6. **Performance**: Optimize existing code

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Sentiment-Analysis-LSTM.git
cd Sentiment-Analysis-LSTM
```

### 2. Install Development Dependencies

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings
- **Imports**: Organized using `isort`

### Code Formatting

We use the following tools:

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1, param2):
    """
    Brief description of function.

    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2

    Returns:
        type: Description of return value

    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

### Type Hints

Use type hints where appropriate:

```python
def predict_text(self, text: str) -> dict:
    """Predict sentiment for a single text."""
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sentiment_analysis --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestSentimentLSTM::test_build_model
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test classes `Test*`
- Name test functions `test_*`

Example test:

```python
import pytest
from sentiment_analysis.model import SentimentLSTM

class TestSentimentLSTM:
    """Test cases for SentimentLSTM class."""

    def test_build_model(self):
        """Test building the model."""
        model = SentimentLSTM()
        keras_model = model.build_model()
        assert keras_model is not None
        assert len(keras_model.layers) > 0
```

### Test Coverage

- Aim for at least 80% code coverage
- Write tests for all new features
- Write tests for bug fixes

## Pull Request Process

### Before Submitting

1. **Update Documentation**: Update README.md and docstrings
2. **Add Tests**: Ensure new code is tested
3. **Run Tests**: All tests must pass
4. **Format Code**: Run black and isort
5. **Check Linting**: Run flake8
6. **Update CHANGELOG**: Add entry for your changes

### Submitting

1. **Commit Your Changes**

```bash
git add .
git commit -m "Add: brief description of changes"
```

Commit message format:
- `Add: <description>` for new features
- `Fix: <description>` for bug fixes
- `Update: <description>` for updates
- `Refactor: <description>` for refactoring
- `Docs: <description>` for documentation

2. **Push to Your Fork**

```bash
git push origin feature/your-feature-name
```

3. **Open a Pull Request**

- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your fork and branch
- Fill out the PR template
- Link any related issues

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated documentation

## Related Issues
Fixes #(issue number)
```

### Review Process

- PRs require at least one approval
- Address reviewer feedback
- Keep PRs focused and small
- Be patient and respectful

## Reporting Bugs

### Before Reporting

1. **Search Existing Issues**: Check if already reported
2. **Update to Latest Version**: Verify bug exists in latest version
3. **Gather Information**: Collect error messages and logs

### Bug Report Template

```markdown
**Describe the Bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Run '...'
3. See error

**Expected Behavior**
What you expected to happen

**Screenshots/Logs**
If applicable, add screenshots or error logs

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.9]
- TensorFlow Version: [e.g., 2.10.0]

**Additional Context**
Any other relevant information
```

## Suggesting Enhancements

### Enhancement Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution you'd like**
Clear description of desired behavior

**Describe alternatives you've considered**
Alternative solutions or features

**Additional context**
Any other relevant information
```

## Project Structure

Understanding the structure:

```
src/sentiment_analysis/
├── config.py          # Configuration management
├── data_loader.py     # Data loading and preprocessing
├── model.py           # Model architecture
├── train.py           # Training logic
├── predict.py         # Prediction logic
├── utils.py           # Utility functions
├── visualization.py   # Visualization functions
└── cli.py             # Command-line interface
```

## Communication

- **Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions
- **Discussions**: For questions and ideas

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- A dedicated CONTRIBUTORS file

## Questions?

If you have questions:
1. Check the documentation
2. Search existing issues
3. Open a new issue with the "question" label

Thank you for contributing to Sentiment Analysis LSTM!

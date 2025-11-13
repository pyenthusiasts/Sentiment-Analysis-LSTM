# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Production-ready features and infrastructure

## [1.0.0] - 2024-01-XX

### Added
- Complete repository reorganization with modular architecture
- Modular package structure under `src/sentiment_analysis/`
- Separated concerns into dedicated modules:
  - `config.py` - Configuration management
  - `data_loader.py` - Data loading and preprocessing
  - `model.py` - LSTM model architecture with variants
  - `train.py` - Training logic with callbacks
  - `predict.py` - Prediction logic with batch support
  - `utils.py` - Utility functions and helpers
  - `visualization.py` - Comprehensive plotting functions
  - `cli.py` - Command-line interface
  - `exceptions.py` - Custom exception classes
  - `api.py` - REST API with FastAPI
- CLI tools (`sentiment-train`, `sentiment-predict`)
- Python API for programmatic access
- Comprehensive unit tests with pytest
- Multiple usage examples (basic, custom, interactive)
- Jupyter notebook tutorial
- Model persistence (save/load functionality)
- Training callbacks (early stopping, checkpointing, LR reduction)
- Rich visualizations (training curves, confusion matrix, ROC curves)
- Configuration management system
- Logging throughout the application
- Docker support (Dockerfile, docker-compose.yml)
- GitHub Actions CI/CD workflows
- Pre-commit hooks configuration
- Environment variable support (.env.example)
- Makefile for common development tasks
- Type checking configuration (mypy)
- Code formatting configuration (black, isort, flake8)
- Issue templates and PR template
- Comprehensive documentation (README.md, CONTRIBUTING.md)

### Changed
- Transformed from single monolithic script to professional package
- Enhanced README with detailed usage instructions
- Updated .gitignore for new project structure

### Fixed
- Various code quality improvements
- Better error handling with custom exceptions

## [0.1.0] - Initial Release

### Added
- Basic sentiment analysis with LSTM
- Single monolithic script implementation
- IMDB dataset support
- Basic model training and prediction
- Simple visualization of training results

---

## Legend

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

.PHONY: help install install-dev test lint format clean docker-build docker-run train predict

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
DOCKER := docker
DOCKER_COMPOSE := docker-compose

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package and dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

test: ## Run tests
	$(PYTEST) tests/ -v --cov=sentiment_analysis --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	$(PYTEST) tests/ -v

lint: ## Run linters
	$(FLAKE8) src/ tests/
	$(BLACK) --check src/ tests/
	$(ISORT) --check-only src/ tests/

format: ## Format code with black and isort
	$(BLACK) src/ tests/ examples/
	$(ISORT) src/ tests/ examples/

type-check: ## Run type checking with mypy
	mypy src/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-models: ## Clean saved models
	rm -rf models/*.h5
	rm -rf models/*.pkl
	rm -rf models/*.json

clean-all: clean clean-models ## Clean everything

# Docker targets
docker-build: ## Build Docker image
	$(DOCKER) build -t sentiment-analysis-lstm:latest .

docker-run: ## Run Docker container
	$(DOCKER) run -it --rm -v $(PWD)/models:/app/models sentiment-analysis-lstm:latest

docker-compose-up: ## Start all services with docker-compose
	$(DOCKER_COMPOSE) up -d

docker-compose-down: ## Stop all services
	$(DOCKER_COMPOSE) down

docker-compose-logs: ## View logs from docker-compose
	$(DOCKER_COMPOSE) logs -f

# Training and prediction
train: ## Train the model
	$(PYTHON) -m sentiment_analysis.cli train

train-custom: ## Train with custom parameters (epochs=10, batch-size=64)
	$(PYTHON) -m sentiment_analysis.cli train --epochs 10 --batch-size 64

predict: ## Run example predictions
	$(PYTHON) -m sentiment_analysis.cli predict --examples

predict-text: ## Predict sentiment for a text (use TEXT="your text")
	$(PYTHON) -m sentiment_analysis.cli predict --text "$(TEXT)"

# Development
dev-server: ## Run development server (if API is implemented)
	uvicorn sentiment_analysis.api:app --reload --host 0.0.0.0 --port 8000

notebook: ## Start Jupyter notebook
	jupyter notebook notebooks/

# Pre-commit
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# Build and distribution
build: clean ## Build package
	$(PYTHON) -m build

publish-test: build ## Publish to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	$(PYTHON) -m twine upload dist/*

# Documentation
docs: ## Build documentation (if using Sphinx)
	cd docs && make html

# CI/CD
ci: lint test ## Run CI checks locally

# Security
security-check: ## Run security checks
	safety check
	bandit -r src/

# Database migrations (if using databases)
# db-init: ## Initialize database
#	alembic init migrations

# db-migrate: ## Create new migration
#	alembic revision --autogenerate -m "$(MSG)"

# db-upgrade: ## Apply migrations
#	alembic upgrade head

# Monitoring and profiling
profile: ## Profile the training code
	$(PYTHON) -m cProfile -o profile.stats examples/basic_usage.py
	$(PYTHON) -m pstats profile.stats

# Version management
version: ## Show current version
	@$(PYTHON) -c "import sentiment_analysis; print(sentiment_analysis.__version__)"

bump-patch: ## Bump patch version
	bump2version patch

bump-minor: ## Bump minor version
	bump2version minor

bump-major: ## Bump major version
	bump2version major

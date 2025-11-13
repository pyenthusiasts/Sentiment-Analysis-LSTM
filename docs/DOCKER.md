# Docker Guide

This guide explains how to use Docker with the Sentiment Analysis LSTM project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building the Image](#building-the-image)
- [Running Containers](#running-containers)
- [Docker Compose](#docker-compose)
- [Environment Variables](#environment-variables)
- [Volumes](#volumes)
- [Production Deployment](#production-deployment)

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+ (optional)
- At least 4GB RAM available for Docker
- 10GB free disk space

## Building the Image

### Basic Build

```bash
docker build -t sentiment-analysis-lstm:latest .
```

### Build with specific platform

```bash
docker build --platform linux/amd64 -t sentiment-analysis-lstm:latest .
```

### Multi-stage build (already configured in Dockerfile)

The Dockerfile uses multi-stage builds to minimize image size:
- Stage 1: Builder (installs dependencies)
- Stage 2: Runtime (minimal image with only necessary files)

## Running Containers

### Training

```bash
# Basic training
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  sentiment-analysis-lstm:latest \
  python -m sentiment_analysis.cli train

# Training with custom parameters
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  sentiment-analysis-lstm:latest \
  python -m sentiment_analysis.cli train --epochs 10 --batch-size 64
```

### Prediction

```bash
# Example predictions
docker run --rm \
  -v $(pwd)/models:/app/models:ro \
  sentiment-analysis-lstm:latest \
  python -m sentiment_analysis.cli predict --examples

# Single prediction
docker run --rm \
  -v $(pwd)/models:/app/models:ro \
  sentiment-analysis-lstm:latest \
  python -m sentiment_analysis.cli predict --text "Great movie!"
```

### API Server

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  --name sentiment-api \
  sentiment-analysis-lstm:latest \
  uvicorn sentiment_analysis.api:app --host 0.0.0.0 --port 8000
```

Access the API at: http://localhost:8000/docs

## Docker Compose

Docker Compose provides an easy way to run multiple services.

### Start All Services

```bash
# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Available Services

1. **sentiment-train**: Training service
2. **sentiment-api**: API server (port 8000)
3. **sentiment-notebook**: Jupyter notebook (port 8888)

### Individual Service Management

```bash
# Start only API
docker-compose up -d sentiment-api

# Restart training
docker-compose restart sentiment-train

# View API logs
docker-compose logs -f sentiment-api
```

## Environment Variables

### Available Variables

```bash
# Model paths
MODEL_PATH=/app/models/sentiment_lstm_model.h5
TOKENIZER_PATH=/app/models/tokenizer.pkl

# Training
EPOCHS=5
BATCH_SIZE=128
LEARNING_RATE=0.001

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# TensorFlow
TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0
```

### Using .env File

```bash
# Create .env file
cp .env.example .env

# Edit with your values
vim .env

# Run with docker-compose (automatically loads .env)
docker-compose up -d
```

### Passing Environment Variables

```bash
# Single variable
docker run -e EPOCHS=10 sentiment-analysis-lstm:latest

# Multiple variables
docker run \
  -e EPOCHS=10 \
  -e BATCH_SIZE=64 \
  sentiment-analysis-lstm:latest

# From file
docker run --env-file .env sentiment-analysis-lstm:latest
```

## Volumes

### Recommended Volume Mounts

```bash
docker run \
  -v $(pwd)/data:/app/data \        # Training data
  -v $(pwd)/models:/app/models \    # Saved models
  -v $(pwd)/outputs:/app/outputs \  # Visualizations
  -v $(pwd)/logs:/app/logs \        # Log files
  sentiment-analysis-lstm:latest
```

### Read-only Mounts

For production API, mount models as read-only:

```bash
docker run \
  -v $(pwd)/models:/app/models:ro \
  sentiment-analysis-lstm:latest
```

## Production Deployment

### Best Practices

1. **Use specific image tags** (not `latest`)
```bash
docker build -t sentiment-analysis-lstm:1.0.0 .
```

2. **Set resource limits**
```bash
docker run \
  --cpus=2 \
  --memory=4g \
  sentiment-analysis-lstm:1.0.0
```

3. **Use health checks**
```bash
docker run \
  --health-cmd="python -c 'import sentiment_analysis'" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  sentiment-analysis-lstm:1.0.0
```

4. **Run as non-root user** (add to Dockerfile)
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

5. **Use secrets for sensitive data**
```bash
docker secret create model_path /path/to/model.h5
docker service create --secret model_path sentiment-analysis-lstm:1.0.0
```

### Docker Swarm Deployment

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml sentiment-stack

# Scale services
docker service scale sentiment-stack_sentiment-api=3

# Update service
docker service update --image sentiment-analysis-lstm:1.0.1 sentiment-stack_sentiment-api
```

### Kubernetes Deployment

Example deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-api
  template:
    metadata:
      labels:
        app: sentiment-api
    spec:
      containers:
      - name: sentiment-api
        image: sentiment-analysis-lstm:1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: sentiment-models-pvc
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs <container-id>

# Run interactively
docker run -it sentiment-analysis-lstm:latest /bin/bash
```

### Out of memory

```bash
# Increase Docker memory limit
# Or reduce batch size
docker run -e BATCH_SIZE=32 sentiment-analysis-lstm:latest
```

### Model not found

```bash
# Verify volume mount
docker run --rm -v $(pwd)/models:/app/models sentiment-analysis-lstm:latest ls -la /app/models

# Train model if missing
docker run --rm -v $(pwd)/models:/app/models sentiment-analysis-lstm:latest \
  python -m sentiment_analysis.cli train
```

### Permission issues

```bash
# Fix permissions on host
sudo chown -R $USER:$USER models/ data/ outputs/ logs/
```

## Advanced Usage

### Building with BuildKit

```bash
DOCKER_BUILDKIT=1 docker build -t sentiment-analysis-lstm:latest .
```

### Multi-platform builds

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t sentiment-analysis-lstm:latest .
```

### Inspect image

```bash
# View image layers
docker history sentiment-analysis-lstm:latest

# Inspect image
docker inspect sentiment-analysis-lstm:latest

# Scan for vulnerabilities
docker scan sentiment-analysis-lstm:latest
```

## Cleaning Up

```bash
# Stop all containers
docker-compose down

# Remove volumes
docker-compose down -v

# Remove images
docker rmi sentiment-analysis-lstm:latest

# Clean system
docker system prune -a
```

# API Documentation

REST API documentation for Sentiment Analysis LSTM.

## Table of Contents

- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Request/Response Examples](#requestresponse-examples)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Deployment](#deployment)

## Getting Started

### Installation

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Or install with extras
pip install -e ".[api]"
```

### Running the API

```bash
# Development mode
uvicorn sentiment_analysis.api:app --reload

# Production mode
uvicorn sentiment_analysis.api:app --host 0.0.0.0 --port 8000 --workers 4

# Using the script
bash scripts/start_api.sh

# Using Makefile
make dev-server
```

### Interactive Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Authentication

Currently, the API does not require authentication. For production use, implement authentication using:

- API Keys
- OAuth2
- JWT tokens

Example with API key (to be implemented):

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
```

## Endpoints

### Root

**GET /** - API Information

Response:
```json
{
  "name": "Sentiment Analysis API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

### Health Check

**GET /health** - Service Health Status

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Model Info

**GET /model/info** - Model Configuration

Response:
```json
{
  "vocab_size": 10000,
  "max_length": 300,
  "embedding_dim": 128,
  "model_path": "models/sentiment_lstm_model.h5",
  "model_exists": true
}
```

### Single Prediction

**POST /predict** - Predict Single Text

Request Body:
```json
{
  "text": "This movie was amazing!"
}
```

Response:
```json
{
  "text": "This movie was amazing!",
  "sentiment": "Positive",
  "score": 0.9234,
  "confidence": 0.8468,
  "timestamp": "2024-01-01T12:00:00"
}
```

### Batch Prediction

**POST /predict/batch** - Predict Multiple Texts

Request Body:
```json
{
  "texts": [
    "Great movie!",
    "Terrible film.",
    "It was okay."
  ]
}
```

Response:
```json
{
  "predictions": [
    {
      "text": "Great movie!",
      "sentiment": "Positive",
      "score": 0.95,
      "confidence": 0.90,
      "timestamp": "2024-01-01T12:00:00"
    },
    {
      "text": "Terrible film.",
      "sentiment": "Negative",
      "score": 0.12,
      "confidence": 0.76,
      "timestamp": "2024-01-01T12:00:00"
    },
    {
      "text": "It was okay.",
      "sentiment": "Positive",
      "score": 0.58,
      "confidence": 0.16,
      "timestamp": "2024-01-01T12:00:00"
    }
  ],
  "count": 3,
  "timestamp": "2024-01-01T12:00:00"
}
```

## Request/Response Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing movie!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Bad!", "Okay."]}'
```

### Using Python requests

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was fantastic!"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["Great movie!", "Terrible film.", "It was okay."]}
)
print(response.json())
```

### Using JavaScript/Fetch

```javascript
// Single prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'This movie was amazing!'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Error Handling

### HTTP Status Codes

- `200 OK` - Success
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service not ready

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

### Common Errors

**Empty text:**
```json
{
  "detail": "Text cannot be empty or only whitespace"
}
```

**Text too long:**
```json
{
  "detail": "ensure this value has at most 10000 characters"
}
```

**Model not loaded:**
```json
{
  "detail": "Predictor not initialized. Please check if model is available."
}
```

## Rate Limiting

Implement rate limiting for production:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict_sentiment(request: Request, ...):
    ...
```

## Deployment

### Using Docker

```bash
# Build
docker build -t sentiment-api .

# Run
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models:ro sentiment-api

# With environment variables
docker run -d \
  -p 8000:8000 \
  -e API_WORKERS=4 \
  -v $(pwd)/models:/app/models:ro \
  sentiment-api
```

### Using Docker Compose

```bash
docker-compose up -d sentiment-api
```

### Behind Nginx

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Using Gunicorn

```bash
gunicorn sentiment_analysis.api:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Environment Variables

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4
export MODEL_PATH=/path/to/model.h5
```

## Monitoring

### Prometheus Metrics (to be implemented)

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### Health Checks

```bash
# Simple health check
curl http://localhost:8000/health

# Detailed check with jq
curl -s http://localhost:8000/health | jq '.model_loaded'
```

## Performance Tips

1. **Use batch predictions** for multiple texts
2. **Enable caching** for frequently requested texts
3. **Scale horizontally** with multiple workers
4. **Use async workers** with Uvicorn
5. **Implement connection pooling** for databases
6. **Add Redis** for caching predictions

## Security

- Use HTTPS in production
- Implement authentication
- Add rate limiting
- Validate all inputs
- Configure CORS properly
- Don't expose stack traces
- Use environment variables for secrets

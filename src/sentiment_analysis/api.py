"""
REST API for sentiment analysis using FastAPI.

This module provides a production-ready REST API for sentiment analysis.
Install FastAPI dependencies: pip install fastapi uvicorn python-multipart
"""

import os
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
except ImportError:
    raise ImportError(
        "FastAPI dependencies not installed. "
        "Install with: pip install fastapi uvicorn python-multipart pydantic"
    )

from sentiment_analysis.predict import Predictor
from sentiment_analysis.config import ModelConfig
from sentiment_analysis.exceptions import (
    ModelNotFoundError,
    PredictionError,
    InvalidInputError,
)
from sentiment_analysis.utils import setup_logger

logger = setup_logger(__name__)

# API metadata
API_VERSION = "1.0.0"
API_TITLE = "Sentiment Analysis API"
API_DESCRIPTION = """
Production-ready sentiment analysis API using LSTM neural networks.

## Features

* **Predict** sentiment for single or batch texts
* **Health check** endpoint for monitoring
* **Model info** endpoint for model metadata
* **CORS enabled** for cross-origin requests

## Usage

Send a POST request to `/predict` with text to analyze.
"""

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = None


# Request/Response models
class PredictRequest(BaseModel):
    """Request model for prediction."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction."""

    texts: List[str] = Field(
        ..., min_items=1, max_items=100, description="List of texts to analyze"
    )

    @validator("texts")
    def validate_texts(cls, v):
        cleaned = [text.strip() for text in v if text.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty text is required")
        return cleaned


class PredictResponse(BaseModel):
    """Response model for prediction."""

    text: str = Field(..., description="Input text")
    sentiment: str = Field(..., description="Predicted sentiment (Positive/Negative)")
    score: float = Field(..., ge=0, le=1, description="Prediction score (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictResponse(BaseModel):
    """Response model for batch prediction."""

    predictions: List[PredictResponse]
    count: int = Field(..., description="Number of predictions")
    timestamp: str = Field(..., description="Batch prediction timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Health check timestamp")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    vocab_size: int
    max_length: int
    embedding_dim: int
    model_path: str
    model_exists: bool


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup."""
    global predictor

    try:
        logger.info("Initializing predictor...")
        model_path = os.getenv("MODEL_PATH", str(ModelConfig.MODEL_PATH))
        predictor = Predictor(model_path=model_path)
        logger.info("Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        # Continue without predictor - will return error on prediction requests


# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns service status and model availability.
    """
    model_loaded = predictor is not None and predictor.sentiment_model.model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=API_VERSION,
        timestamp=datetime.utcnow().isoformat(),
    )


# Model info endpoint
@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.

    Returns model configuration and status.
    """
    model_exists = Path(ModelConfig.MODEL_PATH).exists()

    return ModelInfoResponse(
        vocab_size=ModelConfig.VOCAB_SIZE,
        max_length=ModelConfig.MAX_LENGTH,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        model_path=str(ModelConfig.MODEL_PATH),
        model_exists=model_exists,
    )


# Prediction endpoint
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictRequest):
    """
    Predict sentiment for a single text.

    - **text**: Text to analyze (required, 1-10000 characters)

    Returns sentiment prediction with score and confidence.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialized. Please check if model is available.",
        )

    try:
        result = predictor.predict_text(request.text)
        return PredictResponse(
            text=result["text"],
            sentiment=result["sentiment"],
            score=result["score"],
            confidence=result["confidence"],
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Predict sentiment for multiple texts.

    - **texts**: List of texts to analyze (1-100 texts)

    Returns batch predictions with metadata.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialized. Please check if model is available.",
        )

    try:
        results = predictor.predict_batch(request.texts)

        predictions = [
            PredictResponse(
                text=r["text"],
                sentiment=r["sentiment"],
                score=r["score"],
                confidence=r["confidence"],
                timestamp=datetime.utcnow().isoformat(),
            )
            for r in results
        ]

        return BatchPredictResponse(
            predictions=predictions,
            count=len(predictions),
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


# Error handlers
@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


@app.exception_handler(PredictionError)
async def prediction_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


# Run with: uvicorn sentiment_analysis.api:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "sentiment_analysis.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )

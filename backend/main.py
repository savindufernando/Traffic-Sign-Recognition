"""
Traffic Sign Recognition - FastAPI Backend.
REST API for traffic sign prediction.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import base64
import io
from PIL import Image

from backend.predictor_service import predictor_service

# FastAPI App
app = FastAPI(
    title="Traffic Sign Recognition API",
    description="REST API for Sri Lankan Traffic Sign Recognition using Hybrid CNN-Transformer",
    version="1.0.0"
)

# CORS - Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Models
class PredictionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    is_confident: bool
    top_k: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class Base64ImageRequest(BaseModel):
    image: str  # Base64 encoded image


# Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Traffic Sign Recognition API",
        "docs": "/docs",
        "endpoints": {
            "predict": "/api/predict",
            "health": "/api/health"
        }
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model_loaded = predictor_service._predictor is not None
        return HealthResponse(status="healthy", model_loaded=model_loaded)
    except Exception as e:
        return HealthResponse(status="unhealthy", model_loaded=False)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict traffic sign from uploaded image.
    
    Accepts: image/jpeg, image/png
    Returns: Prediction with class name, confidence, and top-5 predictions
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        contents = await file.read()
        
        # Predict
        result = predictor_service.predict_from_bytes(contents)
        
        print(f"Prediction result: {result}")  # Debug log
        
        # Format response
        top_k = [
            PredictionItem(
                class_id=pred.get('class_id', i),
                class_name=pred.get('class_name', f'Class {i}'),
                confidence=pred.get('confidence', 0.0)
            )
            for i, pred in enumerate(result.get('top_k', []))
        ]
        
        return PredictionResponse(
            class_id=result['class_id'],
            class_name=result['class_name'],
            confidence=result['confidence'],
            is_confident=result['is_confident'],
            top_k=top_k
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Prediction error: {error_detail}")  # Log full traceback
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: Base64ImageRequest):
    """
    Predict traffic sign from base64-encoded image.
    Used for webcam capture from frontend.
    """
    try:
        # Decode base64
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[1]  # Remove data:image/...;base64,
        
        image_bytes = base64.b64decode(image_data)
        
        # Predict
        result = predictor_service.predict_from_bytes(image_bytes)
        
        # Format response
        top_k = [
            PredictionItem(
                class_id=pred.get('class_id', i),
                class_name=pred.get('class_name', f'Class {i}'),
                confidence=pred.get('confidence', 0.0)
            )
            for i, pred in enumerate(result.get('top_k', []))
        ]
        
        return PredictionResponse(
            class_id=result['class_id'],
            class_name=result['class_name'],
            confidence=result['confidence'],
            is_confident=result['is_confident'],
            top_k=top_k
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/classes")
async def get_classes():
    """Get list of all traffic sign classes."""
    try:
        classes = predictor_service.get_classes()
        return {"classes": classes, "count": len(classes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

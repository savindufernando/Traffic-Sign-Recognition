"""
Traffic Sign Recognition - FastAPI Backend.
REST API for traffic sign prediction.
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import base64
import io
from PIL import Image

from backend.security import apply_security

from backend.predictor_service import predictor_service

# FastAPI App
app = FastAPI(
    title="Traffic Sign Recognition API",
    description="REST API for Sri Lankan Traffic Sign Recognition using Hybrid CNN-Transformer",
    version="1.0.0"
)

# Security: API key auth, CORS, rate limiting, security headers
apply_security(app, module_name="tsr")


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
    
    For full dashcam frames, extracts multiple ROI (Region of Interest) crops
    where traffic signs typically appear (roadside areas), classifies each
    independently, and returns the best detection.
    
    This solves the core problem: the classifier was trained on tightly
    cropped sign images (224x224), but dashcam frames are full scenes
    (640x480) with tiny signs among cars, sky, and road.
    """
    try:
        # Decode base64
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[1]  # Remove data:image/...;base64,
        
        image_bytes = base64.b64decode(image_data)
        full_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        w, h = full_image.size
        
        # ─── ROI Extraction Strategy ─────────────────────────────────
        # Traffic signs in dashcam footage appear in predictable zones:
        #   - Left roadside:  left 40%, upper 60%
        #   - Right roadside: right 40%, upper 60%
        #   - Center-top:     center 50%, upper 40% (overhead signs)
        #   - Left quarter:   left 25%, middle band (close signs)
        #   - Right quarter:  right 25%, middle band (close signs)
        
        roi_regions = [
            # (left, upper, right, lower) — PIL crop box
            (0,           int(h*0.10), int(w*0.40), int(h*0.60)),   # Left roadside
            (int(w*0.60), int(h*0.10), w,           int(h*0.60)),   # Right roadside
            (int(w*0.20), int(h*0.05), int(w*0.80), int(h*0.45)),   # Center-top (overhead)
            (0,           int(h*0.20), int(w*0.30), int(h*0.70)),   # Left close
            (int(w*0.70), int(h*0.20), w,           int(h*0.70)),   # Right close
            (int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.65)),   # Wide center band
        ]
        
        best_result = None
        best_confidence = 0.0
        
        for roi_box in roi_regions:
            crop = full_image.crop(roi_box)
            result = predictor_service.predict(crop)
            
            if result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_result = result
        
        # Also try the full frame as a fallback (in case a sign fills the view)
        full_result = predictor_service.predict(full_image)
        if full_result['confidence'] > best_confidence:
            best_confidence = full_result['confidence']
            best_result = full_result
        
        if best_result is None:
            best_result = full_result
        
        print(f"[TSR] Best detection: {best_result['class_name']} "
              f"(conf={best_result['confidence']:.3f}, "
              f"confident={best_result['is_confident']})")
        
        # Get top-k from the best crop
        top_k = best_result.get('top_k', [])
        if not top_k:
            top_k = [{
                'class_id': best_result['class_id'],
                'class_name': best_result['class_name'],
                'confidence': best_result['confidence']
            }]
        
        top_k_items = [
            PredictionItem(
                class_id=pred.get('class_id', i),
                class_name=pred.get('class_name', f'Class {i}'),
                confidence=pred.get('confidence', 0.0)
            )
            for i, pred in enumerate(top_k[:5])
        ]
        
        return PredictionResponse(
            class_id=best_result['class_id'],
            class_name=best_result['class_name'],
            confidence=best_result['confidence'],
            is_confident=best_result['is_confident'],
            top_k=top_k_items
        )
        
    except Exception as e:
        import traceback
        print(f"[TSR] Prediction error: {traceback.format_exc()}")
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

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
import numpy as np

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
    bbox: Optional[List[int]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class Base64ImageRequest(BaseModel):
    image: str  # Base64 encoded image
    is_cropped: bool = False


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
        
        # If the image was already cropped by a YOLO layer on the frontend
        if request.is_cropped:
            best_result = predictor_service.predict(full_image)
            # Since it's already cropped, we don't need a bounding box relative to the original frame
            # The frontend is already tracking it.
            best_bbox = None
            
            # Out-of-Distribution Rejection
            if best_result['confidence'] < 0.15:
                best_result['is_confident'] = False
                best_result['class_name'] = "unknown"
                
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
                top_k=top_k_items,
                bbox=best_bbox
            )
        
        # ─── OpenCV Color & Shape Region Proposal ────────────────────
        import cv2
        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(full_image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Color masks for Sri Lankan Traffic Signs (Red, Blue, Yellow)
        # Highly relaxed HSV ranges to handle shadows, overcast sky, faded colors, and tiny signs
        mask_red1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([15, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 30, 30]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, np.array([80, 30, 30]), np.array([145, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([45, 255, 255]))
        
        # Combine masks
        mask = cv2.bitwise_or(mask_red, mask_blue)
        mask = cv2.bitwise_or(mask, mask_yellow)
        
        # Morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)  # Smaller kernel for tiny signs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Relaxed area constraint: allow tiny signs down to 50 pixels
            if 50 < area < (w * h * 0.5):
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                
                # Shape Analysis (Circularity for round signs, Polygons for triangles/octagons)
                circularity = 4 * np.pi * (area / (peri * peri))
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                vertices = len(approx)
                
                # For small regions (area < 1000), shape analysis can be unstable due to pixelation.
                is_candidate = False
                if area < 1000:
                    is_candidate = True
                elif circularity > 0.3 or (3 <= vertices <= 8):
                    is_candidate = True
                
                if is_candidate:
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    aspect_ratio = float(w_box) / float(h_box)
                    
                    # Traffic signs are generally square-ish (0.3 to 3.0 aspect ratio)
                    if 0.3 < aspect_ratio < 3.0:
                        # Add 40% margin for context
                        margin_x = int(w_box * 0.40)
                        margin_y = int(h_box * 0.40)
                        x1 = max(0, x - margin_x)
                        y1 = max(0, y - margin_y)
                        x2 = min(w, x + w_box + margin_x)
                        y2 = min(h, y + h_box + margin_y)
                        
                        roi_regions.append((x1, y1, x2, y2))
                            
        # Merge overlapping/very close boxes to minimize redundant classifier calls
        def get_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
            
        merged_regions = []
        for box in sorted(roi_regions, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True):
            overlap = False
            for m_box in merged_regions:
                if get_iou(box, m_box) > 0.2:
                    overlap = True
                    break
            if not overlap:
                merged_regions.append(box)
                
        roi_regions = merged_regions
        
        # Fallback to general regions if OpenCV misses the sign
        if not roi_regions:
            # Use a grid of overlapping crops along the top and sides, where signs usually are
            roi_regions = [
                (0,           int(h*0.10), int(w*0.35), int(h*0.50)),   # Top left
                (0,           int(h*0.30), int(w*0.35), int(h*0.70)),   # Mid left
                (int(w*0.65), int(h*0.10), w,           int(h*0.50)),   # Top right
                (int(w*0.65), int(h*0.30), w,           int(h*0.70)),   # Mid right
                (int(w*0.30), int(h*0.15), int(w*0.70), int(h*0.60)),   # Center top
                (int(w*0.10), int(h*0.10), int(w*0.50), int(h*0.50)),   # Left large
                (int(w*0.50), int(h*0.10), int(w*0.90), int(h*0.50)),   # Right large
            ]
        
        best_result = None
        best_confidence = 0.0
        best_bbox = None
        
        for roi_box in roi_regions:
            crop = full_image.crop(roi_box)
            result = predictor_service.predict(crop)
            
            if result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_result = result
                best_bbox = list(roi_box)
        
        # Also try the full frame as a fallback
        full_result = predictor_service.predict(full_image)
        if full_result['confidence'] > best_confidence:
            best_confidence = full_result['confidence']
            best_result = full_result
            best_bbox = [0, 0, w, h]
        
        # ─── Out-of-Distribution Rejection ───────────────────────────
        # Lowered to 0.15 to ensure we capture faint signs
        if best_result is not None and best_result['confidence'] < 0.15:
            best_result['is_confident'] = False
            best_result['class_name'] = "unknown"
        
        if best_result is None:
            best_result = full_result
            best_bbox = [0, 0, w, h]
        
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
            top_k=top_k_items,
            bbox=best_bbox
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

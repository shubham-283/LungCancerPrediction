import os
import base64
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from model import ModelManager
from image_model import ImageClassifierManager
from yolo_model import YOLOCancerDetector
import numpy as np

# -------------------------------
# Config
# -------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "pulmonary_model")
IMAGE_MODEL_PATH = r"D:\SEM 7\MajorProject-1\LungCancerPrediction\backend\densenet121_model\densenet121_final.pth"
YOLO_MODEL_PATH = r"D:\SEM 7\MajorProject-1\LungCancerPrediction\backend\yolo_model\best.pt"
CLASS_NAMES = ['Bengin cases', 'Malignant cases', 'Normal cases']

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="Comprehensive Lung Cancer Detection API",
    description="Multi-modal API for lung cancer detection using patient data, CT scans, and cancer cell detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Pydantic Schemas
# -------------------------------
class Sample(BaseModel):
    AGE: Optional[float] = None
    GENDER: Optional[int] = None
    SMOKING: Optional[int] = None
    FINGER_DISCOLORATION: Optional[int] = None
    MENTAL_STRESS: Optional[int] = None
    EXPOSURE_TO_POLLUTION: Optional[int] = None
    LONG_TERM_ILLNESS: Optional[int] = None
    ENERGY_LEVEL: Optional[float] = None
    IMMUNE_WEAKNESS: Optional[int] = None
    BREATHING_ISSUE: Optional[int] = None
    ALCOHOL_CONSUMPTION: Optional[int] = None
    THROAT_DISCOMFORT: Optional[int] = None
    OXYGEN_SATURATION: Optional[float] = None
    CHEST_TIGHTNESS: Optional[int] = None
    FAMILY_HISTORY: Optional[int] = None
    SMOKING_FAMILY_HISTORY: Optional[int] = None
    STRESS_IMMUNE: Optional[int] = None

    @validator(
        "GENDER", "SMOKING", "FINGER_DISCOLORATION", "MENTAL_STRESS",
        "EXPOSURE_TO_POLLUTION", "LONG_TERM_ILLNESS", "IMMUNE_WEAKNESS",
        "BREATHING_ISSUE", "ALCOHOL_CONSUMPTION", "THROAT_DISCOMFORT",
        "CHEST_TIGHTNESS", "FAMILY_HISTORY", "SMOKING_FAMILY_HISTORY",
        "STRESS_IMMUNE"
    )
    def bin_01(cls, v):
        if v is None:
            return v
        if v not in (0, 1):
            raise ValueError("must be 0 or 1")
        return v

class PredictRequest(BaseModel):
    sample: Sample
    explain: bool = Field(default=False)

class FeatureContribution(BaseModel):
    feature: str
    value: float
    direction: str  # "↑", "↓", "→"

class Prediction(BaseModel):
    predicted_class: Any
    predicted_proba: float
    shap_contributions: Optional[dict] = None
    shap_contribution_arrows: Optional[dict] = None
    sorted_top_contributions: Optional[List[FeatureContribution]] = None

class PredictResponse(BaseModel):
    prediction: Prediction

class ImagePredictionResponse(BaseModel):
    predicted_class: str
    confidence_scores: Dict[str, float]
    top_prediction_confidence: float

class YOLODetectionResponse(BaseModel):
    total_detections: int
    detections: List[Dict]
    confidence_distribution: Dict[str, int]
    statistics: Dict[str, float]
    risk_assessment: str
    recommendation: str
    annotated_image_base64: str

# -------------------------------
# Initialize Models
# -------------------------------
model_manager = ModelManager(MODEL_DIR)
image_classifier = ImageClassifierManager(IMAGE_MODEL_PATH, CLASS_NAMES)
yolo_detector = YOLOCancerDetector(YOLO_MODEL_PATH)

# -------------------------------
# Utility Functions
# -------------------------------
def create_prediction_response(pred, prob, shap_data) -> Prediction:
    if shap_data:
        contributions = [
            FeatureContribution(
                feature=item["feature"],
                value=item["value"],
                direction=item["direction"]
            )
            for item in shap_data["sorted"]
        ] if shap_data["sorted"] else None

        return Prediction(
            predicted_class=str(pred),
            predicted_proba=float(prob),
            shap_contributions=shap_data["contribs"],
            shap_contribution_arrows=shap_data["arrows"],
            sorted_top_contributions=contributions
        )
    else:
        return Prediction(predicted_class=str(pred), predicted_proba=float(prob))

def compute_risk(detection_count, avg_conf):
    if detection_count == 0:
        return "No Risk"
    elif detection_count <= 2 and avg_conf < 0.6:
        return "Low Risk"
    elif detection_count <= 5 and avg_conf < 0.8:
        return "Moderate Risk"
    else:
        return "High Risk"

def compute_recommendation(detection_count, avg_conf):
    if detection_count == 0:
        return "No cancer cells detected. Continue regular monitoring."
    elif detection_count <= 2 and avg_conf < 0.6:
        return "Few potential cancer cells detected with low confidence. Recommend follow-up examination."
    elif detection_count <= 5:
        return "Moderate number of cancer cells detected. Urgent medical consultation recommended."
    else:
        return "High number of cancer cells detected. Immediate oncological consultation required."

# -------------------------------
# Startup Event
# -------------------------------
@app.on_event("startup")
def startup_event():
    print("🚀 Starting API...")
    try:
        model_manager.load_model_artifacts()
        print("✅ Tabular model loaded")
    except Exception as e:
        print(f"❌ Error loading tabular model: {e}")

    try:
        image_classifier.load_model()
        print("✅ Image classifier loaded")
    except Exception as e:
        print(f"❌ Error loading image classifier: {e}")

    try:
        yolo_detector.load_model()
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")

    print("🎯 API startup completed!")

# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
def health():
    return {
        "tabular_model_loaded": model_manager.is_loaded(),
        "image_model_loaded": image_classifier.is_loaded(),
        "yolo_model_loaded": yolo_detector.is_loaded()
    }

# -------------------------------
# Tabular Prediction Endpoint
# -------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Tabular model not loaded")
    try:
        df = model_manager.to_dataframe([req.sample.dict(exclude_unset=True)])
        X = model_manager.preprocess(df)
        preds, probs, shap_struct = model_manager.predict(X, req.explain)
        shap_data = shap_struct[0] if shap_struct else None
        prediction = create_prediction_response(preds[0], probs[0], shap_data)
        return PredictResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# CT Scan Classification
# -------------------------------
@app.post("/predict-ct-scan", response_model=ImagePredictionResponse)
async def predict_ct_scan(file: UploadFile = File(...)):
    if not image_classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Image model not loaded")
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        image_bytes = await file.read()
        pred_class, conf_scores, top_conf = image_classifier.predict_from_bytes(image_bytes)
        return ImagePredictionResponse(
            predicted_class=pred_class,
            confidence_scores=conf_scores,
            top_prediction_confidence=top_conf
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# YOLO Cancer Detection
# -------------------------------
@app.post("/detect-cancer-cells", response_model=YOLODetectionResponse)
async def detect_cancer_cells(file: UploadFile = File(...)):
    if not yolo_detector.is_loaded():
        raise HTTPException(status_code=503, detail="YOLO model not loaded")
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        image_bytes = await file.read()
        results = yolo_detector.detect_image(image_bytes, show=False, save=False)
        detections = results.get("detections", [])
        total_detections = results.get("total_detections", 0)
        annotated_image_base64 = results.get("annotated_image_base64", "")

        # Compute confidence distribution
        high_conf = len([d for d in detections if d['confidence'] >= 0.8])
        medium_conf = len([d for d in detections if 0.5 <= d['confidence'] < 0.8])
        low_conf = len([d for d in detections if d['confidence'] < 0.5])
        confidence_distribution = {
            "high_confidence": high_conf,
            "medium_confidence": medium_conf,
            "low_confidence": low_conf
        }

        avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0.0
        total_area = sum([d['area'] for d in detections]) if detections else 0.0
        statistics = {
            "average_confidence": round(avg_conf, 3),
            "max_confidence": max([d['confidence'] for d in detections]) if detections else 0.0,
            "total_area": total_area
        }

        risk_assessment = compute_risk(total_detections, avg_conf)
        recommendation = compute_recommendation(total_detections, avg_conf)

        return YOLODetectionResponse(
            total_detections=total_detections,
            detections=detections,
            confidence_distribution=confidence_distribution,
            statistics=statistics,
            risk_assessment=risk_assessment,
            recommendation=recommendation,
            annotated_image_base64=annotated_image_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

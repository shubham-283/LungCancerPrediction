import os
import io
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import base64
from typing import List, Dict, Optional


class YOLOCancerDetector:
    """YOLOv8 Lung Cancer Detection Model Manager"""

    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[YOLO] = None
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.5

    def load_model(self):
        """Load YOLOv8 trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = YOLO(self.model_path)
        print(f"YOLO model loaded from: {self.model_path}")
        print(f"Using device: {self.device}")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV BGR format"""
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def detect_image(self, image_bytes: bytes, show: bool = False, save: bool = False) -> Dict:
        """Detect cancer cells in a single image and return results + annotated image"""
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")

        img = self.preprocess_image(image_bytes)

        results = self.model.predict(
            source=img,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            show=show,
            save=save,
            verbose=False
        )

        detections = []
        annotated_img = img.copy()

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf_score = float(box.conf[0])
                    color = (0, 0, 255) if conf_score >= 0.8 else (0, 165, 255) if conf_score >= 0.5 else (0, 255, 255)
                    thickness = 3 if conf_score >= 0.8 else 2
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
                    label = f"Cancer: {conf_score:.2f}"
                    cv2.putText(annotated_img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf_score,
                        'class': 'cancer_cell',
                        'area': (x2 - x1) * (y2 - y1)
                    })

        # Encode annotated image as base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'total_detections': len(detections),
            'detections': detections,
            'annotated_image_base64': annotated_base64
        }

    def detect_folder(self, folder_path: str, show: bool = False, save: bool = False) -> List[Dict]:
        """Detect cancer cells in all images of a folder"""
        results = []
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        for img_path in image_files:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            res = self.detect_image(img_bytes, show=show, save=save)
            res['image_path'] = img_path
            results.append(res)
        return results

    def set_conf_threshold(self, conf: float):
        """Set confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, conf))
        print(f"Confidence threshold updated to: {self.confidence_threshold}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_info(self) -> Dict:
        return {
            'model_type': 'YOLOv8',
            'task': 'Lung Cancer Detection',
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'is_loaded': self.is_loaded()
        }

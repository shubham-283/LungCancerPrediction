import io
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
import torchvision.models as models


class ImageClassifierManager:
    """Handles PyTorch DenseNet121 lung cancer image classification model loading, preprocessing, and predictions"""
    
    def __init__(self, model_path: str, class_names: List[str] = None, device: str = None):
        self.model_path = model_path
        
        # Default class names based on your training data
        if class_names is None:
            self.class_names = ['Bengin cases', 'Malignant cases', 'Normal cases']
        else:
            self.class_names = class_names
            
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Model artifact
        self.model = None
        
        # Data transforms - exactly matching your training setup
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load the trained DenseNet121 model"""
        try:
            # Initialize DenseNet121 with the same architecture as training
            self.model = models.densenet121(weights=None)  # No pretrained weights
            
            # Modify classifier to match your number of classes
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(num_ftrs, len(self.class_names))
            
            # Load the saved weights
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"DenseNet121 lung cancer model loaded successfully on {self.device}")
            print(f"Model loaded from: {self.model_path}")
            print(f"Classes: {self.class_names}")
            
        except Exception as e:
            print(f"Error loading DenseNet121 model: {e}")
            raise e
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess image bytes to tensor"""
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if not already (important for medical images)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms and add batch dimension
            image_tensor = self.data_transforms(image).unsqueeze(0).to(self.device)
            return image_tensor
            
        except Exception as e:
            raise ValueError(f"Error preprocessing lung image: {str(e)}")
    
    def preprocess_pil_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL Image to tensor"""
        try:
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms and add batch dimension
            image_tensor = self.data_transforms(image).unsqueeze(0).to(self.device)
            return image_tensor
            
        except Exception as e:
            raise ValueError(f"Error preprocessing PIL lung image: {str(e)}")
    
    def predict(self, image_tensor: torch.Tensor) -> Tuple[str, Dict[str, float], float]:
        """Make prediction on preprocessed image tensor"""
        if self.model is None:
            raise RuntimeError("DenseNet121 model not loaded")
        
        try:
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                predicted_class = self.class_names[predicted.item()]
                
                # Convert probabilities to percentage and create confidence dict
                confidence_scores = {}
                for i, class_name in enumerate(self.class_names):
                    confidence_scores[class_name] = round(probabilities[0][i].item() * 100, 2)
                
                top_confidence = confidence_scores[predicted_class]
                
            return predicted_class, confidence_scores, top_confidence
            
        except Exception as e:
            raise RuntimeError(f"Error during lung cancer prediction: {str(e)}")
    
    def predict_from_bytes(self, image_bytes: bytes) -> Tuple[str, Dict[str, float], float]:
        """Complete prediction pipeline from image bytes"""
        image_tensor = self.preprocess_image(image_bytes)
        return self.predict(image_tensor)
    
    def predict_from_pil(self, image: Image.Image) -> Tuple[str, Dict[str, float], float]:
        """Complete prediction pipeline from PIL Image"""
        image_tensor = self.preprocess_pil_image(image)
        return self.predict(image_tensor)
    
    def predict_with_detailed_info(self, image_bytes: bytes) -> Dict:
        """Predict with additional diagnostic information"""
        try:
            predicted_class, confidence_scores, top_confidence = self.predict_from_bytes(image_bytes)
            
            # Determine risk level based on prediction
            risk_level = "High Risk" if predicted_class == "Malignant cases" else \
                        "Low Risk" if predicted_class == "Normal cases" else \
                        "Moderate Risk"
            
            # Create recommendation based on prediction
            if predicted_class == "Malignant cases":
                recommendation = "Immediate medical consultation recommended. Further diagnostic tests may be required."
            elif predicted_class == "Bengin cases":
                recommendation = "Benign finding detected. Regular monitoring advised."
            else:
                recommendation = "Normal findings. Continue regular health check-ups."
            
            return {
                "predicted_class": predicted_class,
                "confidence_scores": confidence_scores,
                "top_prediction_confidence": top_confidence,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "model_info": {
                    "model_type": "DenseNet121",
                    "trained_on": "IQ-OTHNCCD lung cancer dataset",
                    "classes": self.class_names
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Error in detailed prediction: {str(e)}")
    
    def batch_predict_from_bytes(self, image_bytes_list: List[bytes]) -> List[Tuple[str, Dict[str, float], float]]:
        """Predict on multiple images"""
        if self.model is None:
            raise RuntimeError("DenseNet121 model not loaded")
        
        results = []
        for image_bytes in image_bytes_list:
            try:
                result = self.predict_from_bytes(image_bytes)
                results.append(result)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append((None, {}, 0.0))
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            "model_type": "DenseNet121",
            "architecture": "CNN - Convolutional Neural Network",
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "device": str(self.device),
            "input_size": "224x224",
            "model_path": self.model_path,
            "is_loaded": self.is_loaded(),
            "dataset": "IQ-OTHNCCD lung cancer dataset",
            "training_info": {
                "epochs": 10,
                "optimizer": "Adam",
                "learning_rate": 0.0001,
                "loss_function": "CrossEntropyLoss",
                "image_augmentation": True,
                "target_images_per_class": 600
            },
            "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"],
            "preprocessing": {
                "resize": "224x224",
                "normalization": "ImageNet mean and std",
                "transforms": ["Resize", "ToTensor", "Normalize"]
            }
        }
    
    def get_class_distribution_info(self) -> Dict:
        """Get information about the classes"""
        return {
            "classes": {
                "Bengin cases": {
                    "description": "Benign lung conditions",
                    "risk_level": "Low",
                    "action": "Regular monitoring"
                },
                "Malignant cases": {
                    "description": "Malignant lung tumors/cancer",
                    "risk_level": "High",
                    "action": "Immediate medical consultation"
                },
                "Normal cases": {
                    "description": "Normal healthy lung tissue",
                    "risk_level": "None",
                    "action": "Continue regular check-ups"
                }
            },
            "total_classes": len(self.class_names),
            "balanced_dataset": True,
            "images_per_class": 600
        }
    
    def validate_image(self, image_bytes: bytes) -> bool:
        """Validate if the image can be processed"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Check if image can be converted to RGB
            image.convert('RGB')
            return True
        except Exception:
            return False
    
    def get_prediction_confidence_interpretation(self, confidence_scores: Dict[str, float]) -> Dict:
        """Interpret confidence scores"""
        max_confidence = max(confidence_scores.values())
        
        if max_confidence >= 90:
            interpretation = "Very High Confidence"
            reliability = "Highly Reliable"
        elif max_confidence >= 75:
            interpretation = "High Confidence"
            reliability = "Reliable"
        elif max_confidence >= 60:
            interpretation = "Moderate Confidence"
            reliability = "Moderately Reliable"
        else:
            interpretation = "Low Confidence"
            reliability = "Consider Additional Testing"
        
        return {
            "confidence_interpretation": interpretation,
            "reliability": reliability,
            "max_confidence": max_confidence,
            "recommendation": "Consult healthcare professional for diagnosis confirmation" if max_confidence < 75 else "Strong prediction, but medical confirmation advised"
        }

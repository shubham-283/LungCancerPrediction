import os
import pickle
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import shap


class ModelManager:
    """Handles model loading, preprocessing, and predictions with SHAP explanations"""
    
    def __init__(self, model_dir: str = "pulmonary_model"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.le_path = os.path.join(model_dir, "label_encoder.pkl")
        self.features_path = os.path.join(model_dir, "feature_names.pkl")
        
        # Model artifacts
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names: Optional[List[str]] = None
        self.explainer = None
        
        # Constants
        self.numeric_cols = ["AGE", "ENERGY_LEVEL", "OXYGEN_SATURATION"]
        self.top_k = int(os.getenv("TOP_K", "10"))
    
    def load_pickle(self, path: str):
        """Load a pickle file"""
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def ensure_artifacts_exist(self):
        """Check if required model artifacts exist"""
        missing = []
        for path in [self.model_path, self.le_path, self.features_path]:
            if not os.path.exists(path):
                missing.append(path)
        if missing:
            raise RuntimeError(f"Missing required artifact(s): {missing}")
    
    def load_model_artifacts(self):
        """Load all model artifacts"""
        self.ensure_artifacts_exist()
        
        # Load required artifacts
        self.model = self.load_pickle(self.model_path)
        self.label_encoder = self.load_pickle(self.le_path)
        self.feature_names = self.load_pickle(self.features_path)
        
        # Load optional scaler
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = self.load_pickle(self.scaler_path)
            except Exception:
                self.scaler = None
        
        # Initialize SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = None
    
    def to_dataframe(self, samples: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert samples to DataFrame with proper feature ordering"""
        df = pd.DataFrame(samples)
        # Add missing columns and order by saved feature_names
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_names]
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input DataFrame"""
        X = df.copy()
        
        # Scale numeric columns if scaler is available
        if self.scaler is not None:
            cols = [c for c in self.numeric_cols if c in X.columns]
            if cols:
                # Fill missing numerics with median before scaling
                X[cols] = X[cols].fillna(X[cols].median())
                X[cols] = self.scaler.transform(X[cols])
        
        # Fill remaining NaN (categorical/binary) with 0
        X = X.fillna(0)
        return X
    
    def shap_to_arrows_and_sorted(self, contribs: Dict[str, float]) -> Tuple[Dict[str, str], List[Dict]]:
        """Convert SHAP contributions to arrows and sorted list"""
        arrows: Dict[str, str] = {}
        items = []
        
        for k, v in contribs.items():
            if v > 0:
                arrows[k] = "↑"
            elif v < 0:
                arrows[k] = "↓"
            else:
                arrows[k] = "→"
            items.append((k, v))
        
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        sorted_list = [
            {
                "feature": k,
                "value": float(v),
                "direction": arrows[k]
            }
            for k, v in items[:self.top_k]
        ]
        
        return arrows, sorted_list
    
    def extract_shap_contributions(self, X: pd.DataFrame, class_idx, shap_values):
        """Extract SHAP contributions for each row"""
        shap_struct = []
        
        for i in range(X.shape[0]):
            ci = class_idx[i] if isinstance(class_idx, np.ndarray) else class_idx.iloc[i]
            
            # Handle different SHAP value structures
            if isinstance(shap_values, list):
                vals = shap_values[ci][i]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                vals = shap_values[i, :, ci]
            else:
                vals = shap_values[i]
            
            contribs = {
                self.feature_names[j]: float(vals[j]) 
                for j in range(len(self.feature_names))
            }
            arrows, sorted_list = self.shap_to_arrows_and_sorted(contribs)
            
            shap_struct.append({
                "contribs": contribs,
                "arrows": arrows,
                "sorted": sorted_list
            })
        
        return shap_struct
    
    def predict(self, X: pd.DataFrame, explain: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
        """Make predictions with optional SHAP explanations"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Get predictions and probabilities
        probs = self.model.predict_proba(X)[:, 1]
        class_idx = self.model.predict(X)
        
        # Convert class indices to labels if label encoder has classes
        if hasattr(self.label_encoder, "classes_"):
            preds = self.label_encoder.classes_[class_idx]
        else:
            preds = class_idx
        
        # Generate SHAP explanations if requested
        shap_struct = None
        if explain:
            if self.explainer is not None:
                shap_values = self.explainer.shap_values(X)
                shap_struct = self.extract_shap_contributions(X, class_idx, shap_values)
            else:
                # Return empty structure if explainer not available
                shap_struct = [
                    {"contribs": None, "arrows": None, "sorted": None}
                    for _ in range(X.shape[0])
                ]
        
        return preds, probs, shap_struct
    
    def is_loaded(self) -> bool:
        """Check if all required model artifacts are loaded"""
        return (
            self.model is not None and 
            self.label_encoder is not None and 
            self.feature_names is not None
        )

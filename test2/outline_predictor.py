import pandas as pd
import joblib
from pdf_feature_extractor import PDFFeatureExtractor

class OutlinePredictor:
    """
    Uses a trained model to predict outline items from PDF features.
    """
    
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.feature_extractor = PDFFeatureExtractor()
    
    def predict_from_pdf(self, pdf_path: str, threshold: float = 0.5) -> list:
        """
        Predicts outline items from a PDF file.
        
        Args:
            pdf_path: Path to input PDF
            threshold: Probability threshold for considering an item an outline
            
        Returns:
            List of outline items with predictions
        """
        # Extract features
        self.feature_extractor.pdf_path = pdf_path
        df = self.feature_extractor.extract_features()
        
        # Make predictions
        X = df[self.feature_extractor.feature_names]
        df["is_outline"] = self.model.predict(X)
        df["outline_prob"] = self.model.predict_proba(X)[:, 1]
        
        # Filter and sort results
        outlines = df[df["outline_prob"] >= threshold].sort_values(["page", "relative_y"])
        outlines = self.feature_extractor.detect_hierarchy(outlines)
        
        return outlines.to_dict("records")
    
    def predict_from_features(self, features: pd.DataFrame, threshold: float = 0.5) -> list:
        """
        Predicts outline items from pre-extracted features.
        """
        X = features[self.feature_extractor.feature_names]
        features["is_outline"] = self.model.predict(X)
        features["outline_prob"] = self.model.predict_proba(X)[:, 1]
        
        outlines = features[features["outline_prob"] >= threshold].sort_values(["page", "relative_y"])
        outlines = self.feature_extractor.detect_hierarchy(outlines)
        
        return outlines.to_dict("records")
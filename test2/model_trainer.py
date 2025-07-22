from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib

class OutlineModelTrainer:
    """
    Handles training and evaluation of the outline detection model.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            'relative_font_size', 'is_bold', 'is_title_case', 'ends_with_punct',
            'starts_with_number', 'has_colon', 'capital_ratio', 'relative_y',
            'length_norm', 'line_density', 'prev_spacing', 'next_spacing'
        ]
    
    def train(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Trains a Random Forest model on annotated data.
        
        Args:
            data_path: Path to CSV with features and 'is_outline' label
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        df = pd.read_csv(data_path)
        X = df[self.feature_names]
        y = df["is_outline"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def save_model(self, path: str):
        """Saves the trained model to disk."""
        if not self.model:
            raise ValueError("No model trained yet")
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Loads a pre-trained model from disk."""
        self.model = joblib.load(path)
        return self.model
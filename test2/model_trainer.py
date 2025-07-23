import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib
import sys
import os

class OutlineModelTrainer:
    def __init__(self, feature_names):
        """
        Initializes the trainer with a given list of feature names.
        Args:
            feature_names (List[str]): Features to use for training the model.
        """
        self.feature_names = feature_names
        self.model = None

    def train(self, df: pd.DataFrame, label_column: str = "is_outline"):
        """
        Trains a Random Forest classifier on the given DataFrame.

        Args:
            df (pd.DataFrame): The feature dataframe.
            label_column (str): The target column.
        """
        X = df[self.feature_names]
        y = df[label_column]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2', None]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='f1')
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_

        print("✅ Best Parameters:", grid_search.best_params_)

        y_pred = self.model.predict(X_val)
        print("\n📋 Classification Report:\n", classification_report(y_val, y_pred))

        return self.model

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        joblib.dump(self.model, path)
        print(f"💾 Model saved to: {path}")

    def load_model(self, path: str):
        self.model = joblib.load(path)
        print(f"✅ Model loaded from: {path}")
        return self.model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_trainer.py <path_to_xlsx>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    # Define feature list manually
    selected_features = [
        'relative_font_size', 'is_bold', 'is_title_case',
        'ends_with_punct', 'starts_with_number', 'has_colon',
        'capital_ratio', 'relative_y', 'length_norm',
        'line_density', 'prev_spacing', 'next_spacing'
    ]

    # Load data
    df = pd.read_excel(file_path)

    trainer = OutlineModelTrainer(feature_names=selected_features)
    model = trainer.train(df)

    # Save model
    model_path = os.path.splitext(file_path)[0] + "_rf_model.pkl"
    trainer.save_model(model_path)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib
import sys
import os
class OutlineModelTrainer:
    def __init__(self, feature_names):
        """
        Initializes the trainer with a list of feature names.
        """
        self.feature_names = feature_names
        self.model = None
        self.label_map = {
            0: "title",
            1: "h1",
            2: "h2",
            3: "h3",
            4: "h4",
            5: "paragraph"
        }

    def balance_classes(self, df: pd.DataFrame, label_column: str):
        """
        Balances classes by downsampling the majority class.
        """
        df_majority = df[df[label_column] == 5]
        df_minority = df[df[label_column] != 5]

        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority)+10,
            random_state=42
        )

        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle
        return df_balanced

    def train(self, df: pd.DataFrame, label_column: str = "label", balance: bool = True):
        """
        Trains a Random Forest classifier.
        """
        if balance:
            df = self.balance_classes(df, label_column)

        X = df[self.feature_names]
        y = df[label_column]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        param_grid = { 
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2', None]
        }

        rf = RandomForestClassifier(
            class_weight='balanced',  # Important!
            random_state=42
        )

        grid_search = GridSearchCV(
            rf, param_grid, cv=3, n_jobs=-1, scoring='f1_macro'
        )
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        print("✅ Best Parameters:", grid_search.best_params_)

        y_pred = self.model.predict(X_val)
        all_labels = sorted(self.label_map.keys())
        print("\n📋 Classification Report:\n", classification_report(
            y_val, y_pred,
            labels=all_labels,
            target_names=[self.label_map[i] for i in all_labels],
            zero_division=0
        ))


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
    
    def predict(self, df: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model is not loaded or trained yet.")
        
        X = df[self.feature_names]
        preds = self.model.predict(X)
        pred_labels = [self.label_map[p] for p in preds]
        
        return preds, pred_labels



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python outline_model_trainer.py <path_to_xlsx>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    # Feature columns to use
    selected_features = [
        'relative_font_size', 'is_bold', 'is_title_case',
        'ends_with_punct', 'starts_with_number', 
         'relative_y', 'length_norm',
        'line_density', 'prev_spacing', 'next_spacing','outline_score',	
        'heading_score',	'title_score',	'norm_font_size'	,'norm_indent'

    ]

    # Load data
    df = pd.read_excel(file_path)

    trainer = OutlineModelTrainer(feature_names=selected_features)
    model = trainer.train(df, label_column="label", balance=True)

    model_path = os.path.splitext(file_path)[0] + "_rf_model.pkl"
    trainer.save_model(model_path)

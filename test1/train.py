import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import sys
import os
import numpy as np

class OutlineModelTrainer:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.model = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, class_weight='balanced'
        )
        self.label_map = {0: "title", 1: "H1", 2: "H2", 3: "H3", 4: "H4", 5: "paragraph"}

    def sample_paragraphs(self, df, label_column="label"):
        """Sample paragraph examples to reduce class imbalance."""
        # Separate heading/title and paragraph examples
        df_headings = df[df[label_column] != 5]
        df_paragraphs = df[df[label_column] == 5]
        
        # Sample paragraphs (max 2x number of headings)
        max_paragraphs = len(df_headings) * 2 if len(df_headings) > 0 else len(df_paragraphs)
        if len(df_paragraphs) > max_paragraphs:
            df_paragraphs = df_paragraphs.sample(n=max_paragraphs, random_state=42)
        
        # Combine and shuffle
        df_balanced = pd.concat([df_headings, df_paragraphs]).sample(frac=1, random_state=42)
        print(f"Sampled dataset distribution:\n{df_balanced[label_column].value_counts()}")
        return df_balanced

    def train(self, df, label_column="label"):
        # Sample paragraphs to balance dataset
        df = self.sample_paragraphs(df, label_column)
        
        X = df[self.feature_names]
        y = df[label_column]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Apply SMOTE to oversample minority classes
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        all_labels = sorted(self.label_map.keys())
        print("\n📋 Classification Report:\n", classification_report(
            y_val, y_pred,
            labels=all_labels,
            target_names=[self.label_map[i] for i in all_labels],
            zero_division=0
        ))
        return self.model

    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"💾 Model saved to: {path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_xlsx>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    feature_names = [
        'relative_font_size', 'is_bold', 'is_title_case', 'ends_with_punct',
        'starts_with_number', 'has_colon', 'relative_y', 'length_norm',
        'line_density', 'prev_spacing', 'next_spacing', 'number_depth'
    ]
    df = pd.read_excel(file_path)
    trainer = OutlineModelTrainer(feature_names)
    model = trainer.train(df)
    trainer.save_model("rf_model.pkl")
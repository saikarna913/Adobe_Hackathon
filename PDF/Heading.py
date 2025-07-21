import pandas as pd
import joblib
from typing import List, Dict


class HeadingPredictor:
    """
    Loads a trained model and predicts heading levels (H1, H2, etc.) for PDF text lines.
    """

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, df: pd.DataFrame) -> List[Dict]:
        """
        Predict heading levels using a trained model.

        Args:
            df (pd.DataFrame): DataFrame of extracted features.

        Returns:
            List[Dict]: Predicted headings in structured format.
        """
        required_columns = [
            "length", "is_bold", "ends_with_punct", "is_upper", "is_title_case",
            "starts_with_number", "relative_font_size", "has_colon", "capital_ratio"
        ]

        X = df[required_columns]
        preds = self.model.predict(X)

        headings = []
        for i, label in enumerate(preds):
            if label != "O":  # 'O' means not a heading
                headings.append({
                    "level": label,
                    "text": df.iloc[i]["text"],
                    "page": int(df.iloc[i]["page"])
                })

        return headings

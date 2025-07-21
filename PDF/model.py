import pandas as pd
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_model(csv_path: str, model_output: str):
    df = pd.read_csv(csv_path)

    required_columns = [
        "length", "is_bold", "ends_with_punct", "is_upper", "is_title_case",
        "starts_with_number", "relative_font_size", "has_colon", "capital_ratio"
    ]

    if "label" not in df.columns:
        raise ValueError("Training CSV must contain a 'label' column with heading levels (e.g. H1, H2, O)")

    X = df[required_columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(model, model_output)
    print(f"[+] Model saved to {model_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a heading detection model.")
    parser.add_argument("csv_path", help="Path to feature CSV with labels")
    parser.add_argument("--model_output", default="heading_model.pkl", help="Output path for the model")
    args = parser.parse_args()

    train_model(args.csv_path, args.model_output)

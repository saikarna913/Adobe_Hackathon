import pandas as pd
import joblib
import sys
import os
import json
from feature import PDFFeatureExtractor

# Label map must match the training
label_map = {
    0: "title",
    1: "H1",
    2: "H2",
    3: "H3",
    4: "H4",
    5: "paragraph"
}

feature_names = [
    'relative_font_size', 'is_bold', 'is_title_case', 'ends_with_punct',
    'starts_with_number', 'has_colon', 'relative_y', 'length_norm',
    'line_density', 'prev_spacing', 'next_spacing', 'number_depth'
]

def predict_outline(df, model):
    missing = [feat for feat in feature_names if feat not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = df[feature_names]
    y_pred = model.predict(X)

    df["predicted_label"] = y_pred
    df["predicted_level"] = df["predicted_label"].map(label_map)

    outline = []
    title_text = ""

    for _, row in df.iterrows():
        level = row["predicted_level"]
        text = row.get("text", "").strip()
        page = int(row.get("page", 1))  # default to 1 if missing

        if level == "title" and not title_text:
            title_text = text

        if level != "paragraph" and level != "title":
            outline.append({
                "level": level,
                "text": text,
                "page": page
            })

    return {
        "title": title_text,
        "outline": outline
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    # Load model
    model = joblib.load(os.path.join(os.path.dirname(__file__), "rf_model.pkl"))

    # Extract features directly from PDF
    extractor = PDFFeatureExtractor(pdf_path)
    df, _ = extractor.extract_features()

    # Predict and convert to JSON
    result = predict_outline(df, model)

    # Output file name
    output_file = "output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"✅ Prediction complete. Output written to {output_file}")

if __name__ == "__main__":
    main()
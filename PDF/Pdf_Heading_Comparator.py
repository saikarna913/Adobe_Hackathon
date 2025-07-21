import json
import argparse
import pandas as pd
from Pdf_Heading_Model_Predictor import HeadingPredictor
from Pdf_Heading_Feature_Extractor import PDFHeadingFeatureExtractor
from typing import Dict


def compare_headings(pdf_path: str, model_path: str) -> Dict:
    extractor = PDFHeadingFeatureExtractor(pdf_path)
    df = extractor.extract()

    predictor = HeadingPredictor(model_path)
    model_headings = predictor.predict(df)

    # Rule-based method: simple threshold on relative_font_size for H1 vs H2
    rule_based_headings = []
    for _, row in df.iterrows():
        if row["relative_font_size"] > 1.5 and row["is_bold"]:
            rule_based_headings.append({"level": "H1", "text": row["text"], "page": int(row["page"])})
        elif row["relative_font_size"] > 1.2:
            rule_based_headings.append({"level": "H2", "text": row["text"], "page": int(row["page"])})

    # Extract best title candidate
    df["title_score"] = df.apply(extractor.calculate_title_score, axis=1)
    title_row = df.loc[df["title_score"].idxmax()]
    title = title_row["text"]

    final_result = {
        "title": title,
        "outline": model_headings or rule_based_headings  # fallback to rule-based if model fails
    }

    return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model and rule-based heading extraction.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("--output", default="output.json", help="Path to save the final result")
    args = parser.parse_args()

    result = compare_headings(args.pdf_path, args.model_path)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)

    print(f"[+] Output written to {args.output}")

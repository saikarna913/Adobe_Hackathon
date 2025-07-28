import sys
import os
import json
from predict import predict_outline
from feature import PDFFeatureExtractor
from Rule_Based import ImprovedPDFExtractor
import joblib

def merge_outlines(json1, json2):
    """Merge outlines from two JSONs, removing duplicates (by text, level, and page)."""
    seen = set()
    merged_outline = []
    for item in json1.get("outline", []) + json2.get("outline", []):
        key = (item.get("text", "").strip(), item.get("level", ""), item.get("page", None))
        if key not in seen:
            seen.add(key)
            merged_outline.append(item)
    return merged_outline

def main():
    pdf_dir = os.path.join('.', 'pdf')
    output_dir = os.path.join('.', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        sys.exit(1)

    # Load model once
    model_path = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_model.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Model file rf_model.pkl not found!")
        sys.exit(1)
    model = joblib.load(model_path)

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n[INFO] Processing {pdf_path} ...")
        try:
            # ML extractor
            extractor = PDFFeatureExtractor(pdf_path)
            df, _ = extractor.extract_features()
            pred_json = predict_outline(df, model)
        except Exception as e:
            print(f"❌ Error running ML extractor on {pdf_file}: {e}")
            continue

        try:
            # Rule-based extractor
            rule_extractor = ImprovedPDFExtractor()
            rule_json = rule_extractor.extract_outline_json(pdf_path)
        except Exception as e:
            print(f"❌ Error running Rule-Based extractor on {pdf_file}: {e}")
            continue

        # Merge outlines
        merged_outline = merge_outlines(pred_json, rule_json)
        final_json = {
            "title": pred_json.get("title") or rule_json.get("title", ""),
            "outline": merged_outline
        }

        pdf_base = os.path.splitext(os.path.basename(pdf_file))[0]
        output_file = os.path.join(output_dir, f"{pdf_base}_output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)
        print(f"✅ Combined output written to {output_file}")

if __name__ == "__main__":
    main()

from pdf_parser import extract_features_from_pdf
from model import train_classifier, predict_headings
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--mode", choices=["train", "predict"], default="predict")
    parser.add_argument("--csv", help="Labeled CSV for training (required for training mode)")
    args = parser.parse_args()

    print("[*] Extracting features...")
    df = extract_features_from_pdf(args.pdf_path)

    if args.mode == "train":
        if not args.csv:
            print("Please provide --csv for training.")
            return
        labeled_df = pd.read_csv(args.csv)
        clf = train_classifier(labeled_df)
        print("[+] Model trained and saved as model.pkl")
    else:
        print("[*] Predicting headings...")
        df = predict_headings(df)
        headings = df[df['pred'] == 1]['text'].tolist()
        print("\n--- Extracted Headings ---")
        for h in headings:
            print("•", h)

if __name__ == "__main__":
    main()

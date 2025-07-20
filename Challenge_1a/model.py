import pandas as pd
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    "size", "spacing", "length", "word_count",
    "is_bold", "ends_with_punct", "is_upper", "is_title_case",
    "starts_with_number", "relative_y", "line_number"
]

def train_classifier(df):
    df = df.dropna()
    df["is_bold"] = df["is_bold"].astype(int)
    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df["label"])  # label = title/h1/h2/paragraph

    clf = RandomForestClassifier(n_estimators=100, max_depth=8)
    clf.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump((clf, le), f)

    print("[+] Model trained and saved as 'model.pkl'")
    return clf

def predict_headings(df):
    df["is_bold"] = df["is_bold"].astype(int)
    with open("model.pkl", "rb") as f:
        clf, le = pickle.load(f)

    X = df[FEATURES]
    df["pred"] = le.inverse_transform(clf.predict(X))
    return df

# ================= MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heading classifier and optionally predict on new data.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to Excel/CSV file with labeled training data")
    parser.add_argument("--predict_file", type=str, help="Optional: path to CSV file to predict headings")
    parser.add_argument("--output_file", type=str, default="predicted_output.csv", help="Output file for predictions (CSV)")

    args = parser.parse_args()

    # Read training file
    if args.train_file.endswith(".xlsx"):
        train_df = pd.read_excel(args.train_file)
    else:
        train_df = pd.read_csv(args.train_file)

    # Train model
    train_classifier(train_df)

    # Optional prediction
    if args.predict_file:
        test_df = pd.read_csv(args.predict_file)
        pred_df = predict_headings(test_df)
        pred_df.to_csv(args.output_file, index=False)
        print(f"[+] Predictions saved to '{args.output_file}'")

import fitz  # PyMuPDF
import pandas as pd
import re
from statistics import median
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import joblib

class PDFOutlineExtractor:
    """
    Extracts and engineers features from a PDF document for outline detection,
    with optional machine learning capabilities.
    """

    def __init__(self, pdf_path: str = None, model_path: str = None):
        self.pdf_path = pdf_path
        self.model_path = model_path
        self.data = []
        self.font_sizes = []
        self.model = None
        self.feature_names = [
            'relative_font_size', 'is_bold', 'is_title_case', 'ends_with_punct',
            'starts_with_number', 'has_colon', 'capital_ratio', 'relative_y',
            'length_norm', 'line_density', 'prev_spacing', 'next_spacing'
        ]

    def extract_features(self) -> pd.DataFrame:
        """Main method to extract and return features from the PDF as a DataFrame."""
        if not self.pdf_path:
            raise ValueError("PDF path not provided")

        doc = fitz.open(self.pdf_path)
        all_lines = []
        prev_y = 0

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            page_width = page.rect.width
            line_number = 0

            # First pass to collect all lines with basic features
            page_lines = []
            for block in blocks:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    full_text = ""
                    bold_count = 0
                    font_sizes = []
                    fonts = []
                    bbox = line["bbox"]

                    for span in spans:
                        span_text = span.get("text", "").strip()
                        if not span_text:
                            continue
                        full_text += span_text + " "
                        if "bold" in span["font"].lower():
                            bold_count += 1
                        font_sizes.append(span["size"])
                        fonts.append(span["font"])
                        self.font_sizes.append(span["size"])

                    full_text = full_text.strip()
                    if not full_text:
                        continue

                    avg_font_size = sum(font_sizes) / len(font_sizes)
                    is_bold = int(bold_count > len(spans) / 2)
                    y0 = bbox[1]
                    spacing = y0 - prev_y if prev_y else 0
                    relative_y = y0 / page_height
                    line_width = bbox[2] - bbox[0]
                    relative_x = bbox[0] / page_width

                    line_data = {
                        "text": full_text,
                        "length": len(full_text),
                        "is_bold": is_bold,
                        "ends_with_punct": int(bool(re.search(r"[.!?:;]$", full_text))),
                        "is_upper": int(full_text.isupper()),
                        "is_title_case": int(full_text.istitle()),
                        "starts_with_number": int(bool(re.match(r"^\d+(\.\d+)*", full_text))),
                        "has_colon": int(":" in full_text),
                        "capital_ratio": sum(1 for c in full_text if c.isupper()) / len(full_text) if len(full_text) > 0 else 0,
                        "font_size": avg_font_size,
                        "line_number": line_number,
                        "page": page_num,
                        "relative_y": relative_y,
                        "relative_x": relative_x,
                        "line_width": line_width,
                        "spacing": spacing,
                        "bbox": bbox
                    }
                    page_lines.append(line_data)
                    prev_y = y0
                    line_number += 1

            # Second pass to calculate context-aware features
            for i, line in enumerate(page_lines):
                # Calculate spacing features
                prev_spacing = page_lines[i-1]["spacing"] if i > 0 else 0
                next_spacing = page_lines[i+1]["spacing"] if i < len(page_lines)-1 else 0
                
                # Calculate line density (characters per unit width)
                line_density = line["length"] / line["line_width"] if line["line_width"] > 0 else 0
                
                line.update({
                    "prev_spacing": prev_spacing,
                    "next_spacing": next_spacing,
                    "line_density": line_density
                })
            
            all_lines.extend(page_lines)

        df = pd.DataFrame(all_lines)
        
        # Calculate normalized features
        median_font_size = median(self.font_sizes) if self.font_sizes else 1.0
        df["relative_font_size"] = df["font_size"] / median_font_size
        max_length = df["length"].max() if not df["length"].empty else 1
        df["length_norm"] = df["length"] / max_length
        
        # Add composite scores
        df["outline_score"] = df.apply(self.calculate_outline_score, axis=1)
        df["heading_score"] = df.apply(self.calculate_heading_score, axis=1)
        df["title_score"] = df.apply(self.calculate_title_score, axis=1)
        
        return df

    def calculate_outline_score(self, row: pd.Series) -> float:
        """Calculates a comprehensive outline score combining multiple features."""
        score = 0.0
        
        # Font characteristics
        score += row["relative_font_size"] * 3.0
        score += 15 if row["is_bold"] else 0
        
        # Text characteristics
        score += 10 if row["is_title_case"] else 0
        score += -15 if row["ends_with_punct"] else 5
        score += 8 if row["has_colon"] else 0
        score += min(row["capital_ratio"] * 20, 10)  # Cap at 10
        
        # Position characteristics
        score += (1 - row["relative_y"]) * 15  # Higher on page is better
        score += min(row["prev_spacing"] * 0.5, 10)  # More space before is better
        
        # Structural characteristics
        score += -10 if row["length_norm"] > 0.5 else 5  # Shorter is better
        score += 5 if row["starts_with_number"] else 0
        
        return score

    def calculate_heading_score(self, row: pd.Series) -> float:
        """Calculates a score for section headings."""
        score = 0.0
        
        score += row["relative_font_size"] * 2.5
        score += 10 if row["is_bold"] else 0
        score += 8 if row["is_title_case"] else 0
        score += 5 if row["has_colon"] else 0
        score += (1 - row["relative_y"]) * 10
        score += 5 if row["starts_with_number"] else 0
        score += min(row["prev_spacing"] * 0.3, 5)  # Some space before is good
        
        # Penalize very long lines
        score += -10 if row["length_norm"] > 0.7 else 0
        
        return score

    def calculate_title_score(self, row: pd.Series) -> float:
        """Calculates a score for document title."""
        score = 0.0
        
        # Title should be at the top
        if row["relative_y"] > 0.2:
            return 0
            
        score += row["relative_font_size"] * 4.0
        score += 20 if row["is_bold"] else 0
        score += 15 if row["is_title_case"] else 0
        score += -20 if row["ends_with_punct"] else 10
        score += -10 if row["starts_with_number"] else 5
        score += (1 - row["relative_y"]) * 25
        
        return score

    def train_model(self, annotated_data_path: str, save_path: str = None):
        """
        Trains a Random Forest model on annotated data.
        annotated_data_path: CSV with features and 'is_outline' label
        save_path: Optional path to save the trained model
        """
        df = pd.read_csv(annotated_data_path)
        X = df[self.feature_names]
        y = df["is_outline"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        if save_path:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")

    def load_model(self, model_path: str):
        """Loads a pre-trained model."""
        self.model = joblib.load(model_path)
        self.model_path = model_path

    def predict_outlines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicts outline items using the trained model."""
        if not self.model:
            raise ValueError("No model loaded or trained")
        
        X = df[self.feature_names]
        df["is_outline"] = self.model.predict(X)
        df["outline_prob"] = self.model.predict_proba(X)[:, 1]
        return df

    def extract_outlines(self, use_model: bool = False, threshold: float = 0.5) -> list:
        """
        Extracts outlines from the PDF either using heuristic scores or ML model.
        Returns a list of outline items with hierarchy detection.
        """
        df = self.extract_features()
        
        if use_model and self.model:
            df = self.predict_outlines(df)
            outlines = df[df["outline_prob"] >= threshold].sort_values(
                ["page", "relative_y"]
            )
        else:
            outlines = df[df["outline_score"] >= 50].sort_values(
                ["page", "relative_y"]
            )
        
        # Add simple hierarchy detection based on indentation and font size
        outlines = self._detect_hierarchy(outlines)
        
        return outlines.to_dict("records")

    def _detect_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds hierarchy level based on indentation and font size."""
        if len(df) == 0:
            return df
        
        # Normalize features for hierarchy detection
        df["norm_font_size"] = (df["font_size"] - df["font_size"].min()) / \
                              (df["font_size"].max() - df["font_size"].min())
        df["norm_indent"] = df["relative_x"]
        
        # Simple clustering for hierarchy levels
        bins = np.linspace(0, 1, 4)  # 3 levels of hierarchy
        df["level"] = np.digitize(df["norm_font_size"] * 0.7 + df["norm_indent"] * 0.3, bins)
        df["level"] = df["level"].max() - df["level"]  # Reverse so bigger fonts are higher level
        
        return df

    def save_outlines_to_csv(self, outlines: list, output_path: str):
        """Saves extracted outlines to CSV."""
        df = pd.DataFrame(outlines)
        # Select only the most important columns
        df = df[["text", "page", "level", "font_size", "relative_font_size", 
                "outline_score", "is_bold", "relative_y"]]
        df.to_csv(output_path, index=False)
        print(f"Outlines saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract outlines from a PDF.")
    parser.add_argument("pdf_path", help="Path to input PDF file")
    parser.add_argument("--output", help="Path to save outlines as CSV")
    parser.add_argument("--model", help="Path to trained model (optional)")
    parser.add_argument("--train", help="Path to annotated data for training")
    parser.add_argument("--save_model", help="Path to save trained model")
    
    args = parser.parse_args()

    extractor = PDFOutlineExtractor(args.pdf_path, args.model)
    
    if args.train:
        print("Training model...")
        extractor.train_model(args.train, args.save_model)
    else:
        print("Extracting outlines...")
        outlines = extractor.extract_outlines(use_model=bool(args.model))
        
        print("\nTop outline candidates:")
        for item in outlines[:10]:
            print(f"Page {item['page']+1} (Level {item.get('level', 0)}): {item['text']}")
        
        if args.output:
            extractor.save_outlines_to_csv(outlines, args.output)
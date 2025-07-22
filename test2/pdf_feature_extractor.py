import fitz  # PyMuPDF
import pandas as pd
import re
from statistics import median
import numpy as np

class PDFFeatureExtractor:
    """
    Extracts and engineers features from PDF documents for outline detection.
    """
    
    def __init__(self, pdf_path: str = None):
        self.pdf_path = pdf_path
        self.font_sizes = []
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

            # Add context-aware features
            for i, line in enumerate(page_lines):
                prev_spacing = page_lines[i-1]["spacing"] if i > 0 else 0
                next_spacing = page_lines[i+1]["spacing"] if i < len(page_lines)-1 else 0
                line_density = line["length"] / line["line_width"] if line["line_width"] > 0 else 0
                
                line.update({
                    "prev_spacing": prev_spacing,
                    "next_spacing": next_spacing,
                    "line_density": line_density
                })
            
            all_lines.extend(page_lines)

        df = pd.DataFrame(all_lines)
        
        # Normalize features
        median_font_size = median(self.font_sizes) if self.font_sizes else 1.0
        df["relative_font_size"] = df["font_size"] / median_font_size
        max_length = df["length"].max() if not df["length"].empty else 1
        df["length_norm"] = df["length"] / max_length
        
        # Add rule-based scores
        df["outline_score"] = df.apply(self.calculate_outline_score, axis=1)
        df["heading_score"] = df.apply(self.calculate_heading_score, axis=1)
        df["title_score"] = df.apply(self.calculate_title_score, axis=1)
        
        return df

    # Rule-based scoring methods (same as original)
    def calculate_outline_score(self, row: pd.Series) -> float: ...
    def calculate_heading_score(self, row: pd.Series) -> float: ...
    def calculate_title_score(self, row: pd.Series) -> float: ...

    def detect_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds hierarchy level based on indentation and font size."""
        if len(df) == 0:
            return df
        
        df["norm_font_size"] = (df["font_size"] - df["font_size"].min()) / \
                              (df["font_size"].max() - df["font_size"].min())
        df["norm_indent"] = df["relative_x"]
        
        bins = np.linspace(0, 1, 4)
        df["level"] = np.digitize(df["norm_font_size"] * 0.7 + df["norm_indent"] * 0.3, bins)
        df["level"] = df["level"].max() - df["level"]
        
        return df
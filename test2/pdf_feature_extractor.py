import fitz  # PyMuPDF
import pandas as pd
import re
from statistics import median
import numpy as np
import sys
import os

class PDFFeatureExtractor:
    """
    Extracts and engineers features from PDF documents for outline detection.
    """
    
    def __init__(self, pdf_path: str = None):
        self.pdf_path = pdf_path
        self.feature_names = [
            'relative_font_size', 'is_bold', 'is_title_case', 'ends_with_punct',
            'starts_with_number', 'has_colon', 'capital_ratio', 'relative_y',
            'length_norm', 'line_density', 'prev_spacing', 'next_spacing',
            'is_all_caps', 'word_count', 'is_centered',
            'font_variation', 'position_score', 'structure_score', 'semantic_score'
        ]
    
    def extract_features(self) -> pd.DataFrame:
        """Main method to extract and return features from the PDF as a DataFrame."""
        if not self.pdf_path:
            raise ValueError("PDF path not provided")

        doc = fitz.open(self.pdf_path)
        all_lines = []
        
        # First pass to gather font stats
        font_sizes = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.append(span["size"])

        if not font_sizes:
            return pd.DataFrame(columns=self.feature_names + ['text', 'page'])

        median_font_size = median(font_sizes) if font_sizes else 12.0
        font_std = np.std(font_sizes) if len(font_sizes) > 1 else 0

        # Second pass to extract features
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            page_width = page.rect.width
            
            page_lines = []
            for block in blocks:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    # Reassemble line
                    full_text = " ".join([s["text"].strip() for s in spans]).strip()
                    if not full_text:
                        continue

                    # Line-level properties
                    bbox = line["bbox"]
                    avg_font_size = sum(s["size"] for s in spans) / len(spans)
                    is_bold = int(any("bold" in s["font"].lower() for s in spans))
                    
                    # Feature calculation
                    relative_y = bbox[1] / page_height if page_height > 0 else 0
                    line_width = bbox[2] - bbox[0]
                    relative_x = bbox[0] / page_width if page_width > 0 else 0
                    
                    # Position score
                    position_score = (1 - relative_y) * 0.5
                    if relative_x < 0.2:
                        position_score += 0.3
                    
                    # Structure score
                    structure_score = 0
                    if re.match(r'^\d+\.?\s+', full_text):
                        structure_score += 0.5
                    if full_text.endswith(':'):
                        structure_score += 0.3
                    if full_text.isupper() and 3 < len(full_text) < 50:
                        structure_score += 0.2

                    # Semantic score
                    semantic_score = 0
                    semantic_patterns = [
                        r'\b(introduction|overview|summary|conclusion|background)\b',
                        r'\b(chapter|section|part|appendix)\s+\w+',
                        r'\b(table\s+of\s+contents|references|bibliography)\b',
                        r'\b(objectives?|methodology|results?|discussion)\b'
                    ]
                    if any(re.search(p, full_text.lower()) for p in semantic_patterns):
                        semantic_score = 0.2

                    line_data = {
                        "text": full_text,
                        "is_bold": is_bold,
                        "ends_with_punct": int(bool(re.search(r"[.!?:;]$", full_text))),
                        "is_all_caps": int(full_text.isupper() and len(full_text) > 1),
                        "word_count": len(full_text.split()),
                        "is_centered": int(abs((relative_x + (line_width / page_width) / 2) - 0.5) < 0.1 if page_width > 0 else 0),
                        "is_title_case": int(full_text.istitle()),
                        "starts_with_number": int(bool(re.match(r"^\d+(\.\d+)*", full_text))),
                        "has_colon": int(":" in full_text),
                        "capital_ratio": sum(1 for c in full_text if c.isupper()) / len(full_text) if len(full_text) > 0 else 0,
                        "font_size": avg_font_size,
                        "page": page_num,
                        "relative_y": relative_y,
                        "bbox": bbox,
                        "font_variation": font_std / median_font_size if median_font_size > 0 else 0,
                        "position_score": position_score,
                        "structure_score": structure_score,
                        "semantic_score": semantic_score,
                        "length": len(full_text)
                    }
                    page_lines.append(line_data)
            
            # Add context-aware features (spacing, density)
            for i, line in enumerate(page_lines):
                prev_bbox_bottom = page_lines[i-1]["bbox"][3] if i > 0 else 0
                next_bbox_top = page_lines[i+1]["bbox"][1] if i < len(page_lines)-1 else page_height
                
                line["prev_spacing"] = (line["bbox"][1] - prev_bbox_bottom) / page_height if page_height > 0 else 0
                line["next_spacing"] = (next_bbox_top - line["bbox"][3]) / page_height if page_height > 0 else 0
                line["line_density"] = line["length"] / (line["bbox"][2] - line["bbox"][0]) if (line["bbox"][2] - line["bbox"][0]) > 0 else 0
            
            all_lines.extend(page_lines)

        if not all_lines:
            return pd.DataFrame(columns=self.feature_names + ['text', 'page'])

        df = pd.DataFrame(all_lines)
        
        # Normalize features
        df["relative_font_size"] = df["font_size"] / median_font_size
        max_len = df["length"].max()
        df["length_norm"] = df["length"] / max_len if max_len > 0 else 0
        
        # Ensure all feature columns exist, even if empty
        for col in self.feature_names:
            if col not in df:
                df[col] = 0
        
        # Drop temporary columns and ensure correct order
        df_final = df[self.feature_names + ['text', 'page']]
        
        return df_final

# ✅ CLI Entry Point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    extractor = PDFFeatureExtractor(pdf_path)
    df = extractor.extract_features()
    
    # For CLI execution, print the DataFrame
    print(df.head().to_string())
    output_path = os.path.splitext(pdf_path)[0] + "_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Features extracted and saved to {output_path}")

    output_path = os.path.splitext(pdf_path)[0] + "_features.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Feature extraction complete. Output saved to: {output_path}")
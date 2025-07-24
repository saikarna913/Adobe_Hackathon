import json
import os
from pathlib import Path
import re
import fitz  # PyMuPDF
import pandas as pd
import numpy as np

class PDFFeatureExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.font_sizes = []
        self.feature_names = [
            'relative_font_size', 'is_bold', 'is_title_case', 'ends_with_punct',
            'starts_with_number', 'has_colon', 'relative_y', 'length_norm',
            'line_density', 'prev_spacing', 'next_spacing', 'number_depth'
        ]

    def extract_features(self):
        doc = fitz.open(self.pdf_path)
        all_lines = []
        prev_y = {page_num: 0 for page_num in range(len(doc))}

        # Get title from metadata or first page
        title = doc.metadata.get("title", "").strip()
        if not title:
            first_page = doc[0]
            text_blocks = first_page.get_text("dict")["blocks"]
            font_sizes = []
            for block in text_blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.append((span["size"], span["text"]))
            if font_sizes:
                title = max(font_sizes, key=lambda x: x[0])[1].strip()

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

                    full_text = " ".join(span["text"].strip() for span in spans if span["text"].strip())
                    if not full_text:
                        continue

                    avg_font_size = sum(span["size"] for span in spans) / len(spans)
                    is_bold = int(sum("bold" in span["font"].lower() for span in spans) > len(spans) / 2)
                    y0 = line["bbox"][1]
                    spacing = y0 - prev_y[page_num] if prev_y[page_num] else 0
                    relative_y = y0 / page_height
                    line_width = line["bbox"][2] - line["bbox"][0]
                    relative_x = line["bbox"][0] / page_width

                    # Detect section numbering depth (e.g., "1.", "1.1", "1.1.1")
                    number_match = re.match(r"^(\d+(\.\d+)*)", full_text)
                    number_depth = len(number_match.group(1).split(".")) if number_match else 0

                    # Multilingual support: Detect Japanese chapter headings (e.g., "第1章")
                    if re.match(r"^第\d+章", full_text):
                        number_depth = max(number_depth, 1)  # Treat as H1

                    line_data = {
                        "text": full_text,
                        "length": len(full_text),
                        "is_bold": is_bold,
                        "ends_with_punct": int(bool(re.search(r"[.!?:;]$", full_text))),
                        "is_title_case": int(full_text.istitle()),
                        "starts_with_number": int(bool(number_match)),
                        "has_colon": int(":" in full_text),
                        "font_size": avg_font_size,
                        "page": page_num + 1,
                        "relative_y": relative_y,
                        "relative_x": relative_x,
                        "line_width": line_width,
                        "spacing": spacing,
                        "number_depth": number_depth
                    }
                    page_lines.append(line_data)
                    prev_y[page_num] = y0
                    self.font_sizes.append(avg_font_size)

            # Add context-aware features
            for i, line in enumerate(page_lines):
                line["prev_spacing"] = page_lines[i-1]["spacing"] if i > 0 else 0
                line["next_spacing"] = page_lines[i+1]["spacing"] if i < len(page_lines)-1 else 0
                line["line_density"] = line["length"] / line["line_width"] if line["line_width"] > 0 else 0
                page_lines[i] = line

            all_lines.extend(page_lines)

        doc.close()
        df = pd.DataFrame(all_lines)
        if df.empty:
            return df, title

        # Normalize features
        median_font_size = np.median(self.font_sizes) if self.font_sizes else 1.0
        df["relative_font_size"] = df["font_size"] / median_font_size
        df["length_norm"] = df["length"] / df["length"].max() if not df["length"].empty else 0

        # Add a placeholder label column (to be filled manually)
        df["label"] = -1  # -1 indicates unlabeled
        return df, title

def generate_training_data(input_dir, output_dir):
    Path(output_dir).mkdir(exist_ok=True)
    extractor = PDFFeatureExtractor(None)
    for pdf_file in Path(input_dir).glob("*.pdf"):
        extractor.pdf_path = str(pdf_file)
        df, title = extractor.extract_features()
        if not df.empty:
            output_file = Path(output_dir) / f"{pdf_file.stem}_features.xlsx"
            df.to_excel(output_file, index=False)
            print(f"Features extracted for {pdf_file.name}, saved to {output_file}")

if __name__ == "__main__":
    input_dir = "input_pdfs"  # Directory containing PDFs
    output_dir = "training_data"  # Directory to save Excel files
    generate_training_data(input_dir, output_dir)
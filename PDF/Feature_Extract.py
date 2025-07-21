import fitz  # PyMuPDF
import pandas as pd
import re
from statistics import median


class PDFHeadingFeatureExtractor:
    """
    Extracts and engineers features from a PDF document for heading/title detection.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.data = []
        self.max_font_size = 0
        self.font_sizes = []

    def extract(self) -> pd.DataFrame:
        """Main method to extract and return features from the PDF as a DataFrame."""
        doc = fitz.open(self.pdf_path)

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            prev_y = 0
            page_height = page.rect.height
            line_number = 0

            for block in blocks:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    full_text = ""
                    bold_count = 0
                    font_sizes = []
                    fonts = []

                    for span in spans:
                        span_text = span.get("text", "").strip()
                        if not span_text:
                            continue
                        full_text += span_text + " "
                        if "Bold" in span["font"]:
                            bold_count += 1
                        font_sizes.append(span["size"])
                        fonts.append(span["font"])
                        self.font_sizes.append(span["size"])

                    full_text = full_text.strip()
                    if not full_text:
                        continue

                    font = max(set(fonts), key=fonts.count)
                    size = sum(font_sizes) / len(font_sizes)
                    is_bold = int(bold_count > len(spans) // 2)

                    y0 = line["bbox"][1]
                    spacing = y0 - prev_y if prev_y else 0
                    relative_y = y0 / page_height

                    self.max_font_size = max(self.max_font_size, size)

                    features = {
                        "text": full_text,
                        "length": len(full_text),
                        "is_bold": is_bold,
                        "ends_with_punct": int(bool(re.search(r"[.!?:;]$", full_text))),
                        "is_upper": int(full_text.isupper()),
                        "is_title_case": int(full_text.istitle()),
                        "starts_with_number": int(bool(re.match(r"^\d+(\.\d+)*", full_text))),
                        "has_colon": int(":" in full_text),
                        "capital_ratio": sum(c.isupper() for c in full_text) / len(full_text)
                        if len(full_text) > 0 else 0,
                        "font_size": size,
                        "line_number": line_number,
                        "page": page_num,
                        "relative_y": relative_y,
                    }
                    self.data.append(features)
                    prev_y = y0
                    line_number += 1

        df = pd.DataFrame(self.data)
        median_font_size = median(self.font_sizes) if self.font_sizes else 1.0
        df["relative_font_size"] = df["font_size"] / median_font_size

        return df

    def calculate_title_score(self, row: pd.Series) -> float:
        """Calculates a score to estimate if a line is the document title."""
        score = 0.0

        score += row["relative_font_size"] * 3  # size
        score += 20 if row["is_bold"] else 0
        score += 15 if row["is_title_case"] else 0
        score += 10 if not row["ends_with_punct"] else -10
        score += 5 if row["starts_with_number"] == 0 else -5
        score += (1 - row["relative_y"]) * 20  # higher is better

        return score

    def calculate_heading_score(self, row: pd.Series) -> float:
        """Calculates a score to estimate if a line is a heading."""
        score = 0.0

        score += row["relative_font_size"] * 2
        score += 10 if row["is_bold"] else 0
        score += 10 if row["is_title_case"] else 0
        score += 5 if row["has_colon"] else 0
        score += (1 - row["relative_y"]) * 10
        score += 5 if row["starts_with_number"] else 0

        return score
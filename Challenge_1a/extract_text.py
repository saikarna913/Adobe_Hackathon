import fitz  # PyMuPDF
import pandas as pd
import re
import argparse


import fitz
import pandas as pd
import re

def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    data = []
    max_font_size = 0

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        prev_y = 0
        line_number = 0
        page_height = page.rect.height

        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                full_text = ""
                bold_count = 0
                font_sizes = []
                fonts = []
                x0_values = []

                for span in spans:
                    span_text = span.get("text", "").strip()
                    if not span_text:
                        continue
                    full_text += span_text + " "
                    if "Bold" in span["font"]:
                        bold_count += 1
                    font_sizes.append(span["size"])
                    fonts.append(span["font"])
                    x0_values.append(span["bbox"][0])

                full_text = full_text.strip()
                if not full_text:
                    continue

                # Choose the most common font and average size
                font = max(set(fonts), key=fonts.count)
                size = sum(font_sizes) / len(font_sizes)
                is_bold = int(bold_count > len(spans) // 2)
                x0 = sum(x0_values) / len(x0_values)

                y0 = line["bbox"][1]
                spacing = y0 - prev_y if prev_y else 0
                relative_y = y0 / page_height
                word_count = len(full_text.split())

                max_font_size = max(max_font_size, size)

                features = {
                    "text": full_text,
                    "font": font,
                    "size": size,
                    "spacing": spacing,
                    "length": len(full_text),
                    "word_count": word_count,
                    "is_bold": is_bold,
                    "ends_with_punct": int(bool(re.search(r"[.!?:;]$", full_text))),
                    "is_upper": int(full_text.isupper()),
                    "is_title_case": int(full_text.istitle()),
                    "starts_with_number": int(bool(re.match(r"^\d+(\.\d+)*", full_text))),
                    "relative_y": relative_y,
                    "line_number": line_number,
                    "page": page_num,
                    "x0": x0  # used for center/side align detection
                }

                data.append(features)
                prev_y = y0
                line_number += 1

    df = pd.DataFrame(data)
    df["relative_font_size"] = df["size"] / max_font_size if max_font_size else 1.0
    df["has_colon"] = df["text"].str.contains(":").astype(int)
    df["capital_ratio"] = df["text"].apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0)
    return df


def add_text_features(df):
    """Adds extra text-based features to the dataframe."""
    df["has_colon"] = df["text"].str.contains(":").astype(int)
    df["capital_ratio"] = df["text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    return df


def merge_title_lines_safe(df):
    """Merges multi-line titles heuristically based on similarity."""
    merged = []
    skip_next = False
    for i in range(len(df) - 1):
        if skip_next:
            skip_next = False
            continue

        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        is_same_page = row["page"] == next_row["page"]
        is_same_font = row["font"] == next_row["font"]
        is_similar_size = abs(row["size"] - next_row["size"]) < 0.1
        is_close = abs(row["spacing"]) < 40
        both_bold = row["is_bold"] and next_row["is_bold"]
        no_punctuation_end = not re.search(r"[.:;!?]$", row["text"])
        short_texts = len(row["text"]) < 50 and len(next_row["text"]) < 50
        both_title_like = row["is_title_case"] or row["is_upper"]
        center_aligned = abs(row.get("x0", 0) - next_row.get("x0", 0)) < 10

        if (
            is_same_page
            and is_same_font
            and is_similar_size
            and is_close
            and both_bold
            and no_punctuation_end
            and short_texts
            and both_title_like
            and center_aligned
        ):
            combined_row = row.copy()
            combined_row["text"] = f"{row['text']} {next_row['text']}"
            skip_next = True
            merged.append(combined_row)
        else:
            merged.append(row)

    if not skip_next:
        merged.append(df.iloc[-1])

    return pd.DataFrame(merged)


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from a PDF and save to CSV.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--save", action="store_true", help="Save output as CSV file")
    args = parser.parse_args()

    df = extract_features_from_pdf(args.pdf_path)
    df = merge_title_lines_safe(df)
    df = add_text_features(df)

    df.to_csv("test_data.csv", index=False)
    print("[+] Saved to test_data.csv")

    print(df.head(10))

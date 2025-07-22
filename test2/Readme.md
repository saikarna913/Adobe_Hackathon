# PDF Outline Extractor

## Overview

This tool extracts structural outlines from PDF documents using a combination of heuristic rules and machine learning. It identifies headings, titles, and section headers while detecting hierarchical relationships between them.

## Features

- **Feature Extraction**: Extracts typographic and structural features from PDF text elements
- **Rule-based Scoring**: Uses configurable scoring rules to identify outline candidates
- **Machine Learning**: Optional Random Forest classifier for improved accuracy
- **Hierarchy Detection**: Automatically detects heading levels based on font size and indentation
- **Multiple Output Formats**: Results can be exported to CSV for further analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-outline-extractor.git
   cd pdf-outline-extractor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install pymupdf pandas scikit-learn numpy
   ```

## Usage

### Basic Outline Extraction

```bash
python pdf_outline.py input.pdf --output outlines.csv
```

### ML-based Extraction (with trained model)

```bash
python pdf_outline.py input.pdf --model outline_model.joblib --output outlines.csv
```

### Model Training

1. Prepare a CSV file with annotated data (containing features and `is_outline` labels)
2. Train the model:
   ```bash
   python pdf_outline.py --train annotated_data.csv --save_model outline_model.joblib
   ```

### Command Line Options

| Option       | Description                                      |
|--------------|--------------------------------------------------|
| pdf_path     | Path to input PDF file (required)                |
| --output     | Path to save outlines as CSV                    |
| --model      | Path to trained model for ML-based extraction   |
| --train      | Path to annotated data for training             |
| --save_model | Path to save trained model                      |

## Output Format

The output CSV contains these columns:

| Column               | Description                                      |
|----------------------|--------------------------------------------------|
| text                 | The extracted text                              |
| page                 | Page number (0-indexed)                         |
| level                | Hierarchy level (0=top level)                   |
| font_size            | Absolute font size                              |
| relative_font_size   | Font size relative to document median           |
| outline_score        | Heuristic outline score (0-100)                 |
| is_bold              | Whether text is bold (1) or not (0)             |
| relative_y           | Vertical position on page (0=top, 1=bottom)     |

## Examples

### Extract outlines from a document
```bash
python pdf_outline.py research_paper.pdf --output paper_outlines.csv
```

### Train a model on annotated data
```bash
python pdf_outline.py --train training_data.csv --save_model my_model.joblib
```

### Use trained model for extraction
```bash
python pdf_outline.py contract.pdf --model my_model.joblib --output contract_outlines.csv
```

## Configuration

You can modify these aspects of the extraction:

1. **Score thresholds**: Adjust in `calculate_outline_score()` method
2. **Feature selection**: Modify `self.feature_names` list
3. **Hierarchy levels**: Adjust bins in `_detect_hierarchy()` method

## Requirements

- Python 3.7+
- PyMuPDF (fitz)
- pandas
- scikit-learn
- numpy

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please open an issue or pull request for any improvements.

## Support

For questions or issues, please open an issue on GitHub.
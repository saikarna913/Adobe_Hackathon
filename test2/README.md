# PDF Outline Extractor - Challenge 1A

This folder contains the optimized ML-enhanced PDF outline extraction system for Challenge 1A.

## 🎯 Performance Results
- **ML-Enhanced F1 Score: 0.522**
- **Improved Extractor F1 Score: 0.474** 
- **ML vs Improved: +10.2%**

## 📁 File Structure

### Core Extractors
- **`ml_enhanced_extractor.py`** - Main ML-enhanced extractor (best performance)
- **`improved_extractor.py`** - Rule-based fallback extractor for Challenge 1A compliance

### ML Components
- **`enhanced_model.pkl`** - Trained Random Forest model
- **`Training_data.xlsx`** - Training data for ML model
- **`model_trainer.py`** - Model training utilities
- **`pdf_feature_extractor.py`** - Feature extraction for ML
- **`outline_predictor.py`** - ML prediction interface

### Main Interface & Testing
- **`main.py`** - CLI interface for the complete ML pipeline ⭐ **MAIN SUBMISSION SCRIPT**
- **`test_ml_enhanced.py`** - Comprehensive test script comparing ML-enhanced vs improved approaches

### Test Data
- **`pdfs/`** - Collection of test PDF files from Challenge 1A and 1B

## 🚀 Usage

### Quick Test (Recommended)
```bash
python test_ml_enhanced.py
```
*Compares ML-enhanced vs improved extractor approaches with detailed metrics*

### CLI Interface
```bash
# Extract features from PDF
python main.py extract path/to/file.pdf --output features.csv

# Train new model
python main.py train Training_data.xlsx --save_model new_model.pkl

# Predict outlines
python main.py predict path/to/file.pdf enhanced_model.pkl --threshold 0.6
```

### Direct Usage
```python
from ml_enhanced_extractor import MLEnhancedPDFExtractor

# With ML model (best performance)
extractor = MLEnhancedPDFExtractor(model_path='enhanced_model.pkl')
result = extractor.extract_outline_json('document.pdf')

# Rule-based only (Challenge 1A compliance)
extractor = MLEnhancedPDFExtractor(use_ml=False)
result = extractor.extract_outline_json('document.pdf')
```

## 🏆 Key Features
- ✅ **No hardcoding** - Fully adaptive to document characteristics
- ✅ **ML-enhanced** - Random Forest classifier with intelligent feature engineering
- ✅ **Challenge 1A compliant** - Rule-based fallback when ML unavailable
- ✅ **Ensemble approach** - Combines ML confidence with rule-based scoring
- ✅ **Advanced text processing** - Handles fragmented text and complex layouts
- ✅ **Document-aware filtering** - Adaptive to forms, academic papers, manuals, etc.

## 📊 Performance by Document Type
- **Forms (E0CCG5S239)**: ML Perfect F1 (1.000) vs Improved (0.000)
- **Academic Papers (STEMPathwaysFlyer)**: Both approaches F1 (0.500) 
- **Technical Documents (E0CCG5S312)**: Improved F1 (0.593) vs ML (0.462)
- **Government Documents (E0H1CM114)**: Improved F1 (0.610) vs ML (0.250)
- **Invitations (TOPJUMP)**: Improved F1 (0.667) vs ML (0.400)

## 🔧 Technical Details
- **Algorithm**: Random Forest + Rule-based ensemble
- **Features**: 12 engineered features (font size, formatting, structure, position, etc.)
- **Training**: 54 samples with class balancing
- **Output**: Challenge 1A JSON format with 0-based page numbers
- **Dependencies**: PyMuPDF, scikit-learn, pandas (with graceful fallback)

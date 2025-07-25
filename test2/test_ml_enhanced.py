#!/usr/bin/env python3
"""
Test script for PDF Outline Extractors
Compares ML-enhanced vs improved approaches
"""

import os
import json
import re
import time
import difflib
from ml_enhanced_extractor import MLEnhancedPDFExtractor
from improved_extractor import ImprovedPDFExtractor
from datetime import datetime
from collections import defaultdict

def load_ground_truths(gt_dir):
    """Loads all ground truth JSON files from a directory."""
    ground_truths = {}
    if not os.path.isdir(gt_dir):
        return {}
        
    for filename in os.listdir(gt_dir):
        if filename.endswith('.json'):
            pdf_name = os.path.splitext(filename)[0]
            with open(os.path.join(gt_dir, filename), 'r') as f:
                ground_truths[pdf_name] = json.load(f)
    return ground_truths

def normalize_text(text):
    """Normalize text for comparison"""
    normalized = ' '.join(text.strip().split())
    normalized = re.sub(r'\b([A-Z])\s+([a-z]+)\b', r'\1\2', normalized)
    normalized = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', normalized)
    normalized = re.sub(r'\b([A-Z])\s+([A-Z]+)\s+([A-Z])\b', r'\1\2\3', normalized)
    normalized = re.sub(r'\b([A-Z])\s+([A-Z][A-Z]+)\b', r'\1\2', normalized)
    normalized = re.sub(r'\s+([!?.,;:])', r'\1', normalized)
    normalized = re.sub(r'([!?.,;:])\s+$', r'\1', normalized)
    return normalized

def calculate_metrics(predicted, ground_truth):
    """Calculate precision, recall, and F1 score using fuzzy matching"""
    pred_items = {normalize_text(item['text']): item['page'] for item in predicted.get('outline', [])}
    gt_items = {normalize_text(item['text']): item['page'] for item in ground_truth.get('outline', [])}
    
    if not gt_items and not pred_items:
        return 1.0, 1.0, 1.0
    
    if not pred_items:
        return 0.0, 0.0, 0.0
    
    if not gt_items:
        return 0.0, 0.0, 0.0
    
    # Use fuzzy matching for better comparison
    true_positives = 0
    
    # Create a set of predicted items for faster lookup
    pred_set = set(pred_items.keys())
    
    for gt_text, gt_page in gt_items.items():
        best_match_ratio = 0
        best_match_text = None
        
        # Find the best matching predicted text
        for pred_text in pred_set:
            ratio = difflib.SequenceMatcher(None, gt_text.lower(), pred_text.lower()).ratio()
            if ratio > best_match_ratio:
                best_match_ratio = ratio
                best_match_text = pred_text
        
        # If a good match is found, check if the page number is also correct
        if best_match_ratio >= 0.8:  # Stricter 80% similarity threshold
            if abs(pred_items[best_match_text] - gt_page) <= 1: # Allow for off-by-one page errors
                true_positives += 1
                # Remove the matched item to prevent it from being matched again
                pred_set.remove(best_match_text)

    precision = true_positives / len(pred_items) if pred_items else 0
    recall = true_positives / len(gt_items) if gt_items else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def save_output(data, output_dir, pdf_name, approach):
    """Saves the output of an extractor to a JSON file."""
    output_path = os.path.join(output_dir, f"{pdf_name}_{approach}.json")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_test(name, extractor, pdf_path, gt_outline, output_dir, pdf_name, approach_key, results):
    """Runs a test for a single extractor and PDF."""
    print(f"\n🔧 {name}:")
    start_time = time.time()
    try:
        predicted = extractor.extract_outline_json(pdf_path)
        duration = time.time() - start_time
        
        save_output(predicted, output_dir, pdf_name, approach_key)
        
        count = len(predicted.get('outline', []))
        print(f"   ⏱️  Time: {duration:.2f}s")
        print(f"   🎯 Found: {count} headings")
        print(f"   💾 Saved to: {os.path.join(output_dir, f'{pdf_name}_{approach_key}.json')}")
        
        precision, recall, f1 = calculate_metrics(predicted, gt_outline)
        print(f"   📊 P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")
        
        results[pdf_name][approach_key] = {
            'f1': f1, 'precision': precision, 'recall': recall,
            'count': count, 'time': duration
        }
        results[pdf_name]['gt_count'] = len(gt_outline.get('outline', []))
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

def generate_summary_report(results, output_dir):
    """Generates a summary report of the test results."""
    report_lines = []
    
    # Header
    report_lines.append(f"{'PDF Name':<35} {'Improved F1':<10} {'ML-Enhanced F1':<15} {'GT Count':<10} {'Winner':<10}")
    report_lines.append("-" * 80)
    
    improved_scores = []
    ml_scores = []
    
    for pdf_name, result in results.items():
        improved_f1 = result.get('improved', {}).get('f1', 0.0)
        ml_f1 = result.get('ml_enhanced', {}).get('f1', 0.0)
        gt_count = result.get('gt_count', 0)
        
        # Determine winner
        winner = "Tie"
        if improved_f1 > ml_f1:
            winner = "Improved"
        elif ml_f1 > improved_f1:
            winner = "ML-Enhanced"
        
        report_lines.append(f"{pdf_name:<35} {improved_f1:<10.3f} {ml_f1:<15.3f} {gt_count:<10} {winner:<10}")
        
        improved_scores.append(improved_f1)
        if ml_f1 > 0:
            ml_scores.append(ml_f1)
    
    # Calculate averages
    avg_improved = sum(improved_scores) / len(improved_scores) if improved_scores else 0
    avg_ml = sum(ml_scores) / len(ml_scores) if ml_scores else 0
    
    report_lines.append("-" * 80)
    ml_avg_str = f"{avg_ml:.3f}" if avg_ml > 0 else "N/A"
    report_lines.append(f"{'AVERAGE':<35} {avg_improved:<10.3f} {ml_avg_str:<15}")
    
    perc_diff = (avg_ml - avg_improved) / avg_improved * 100 if avg_improved > 0 else 0
    report_lines.append(f"   📊 ML vs Improved: {avg_ml - avg_improved:+.3f} ({perc_diff:+.1f}%)")
    
    # Print to console
    print("\n".join(report_lines))

def create_file_index(output_dir):
    """Creates an index of all generated files in the output directory."""
    index_data = {
        'output_directory': output_dir,
        'created_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'files_generated': []
    }
    
    for file in os.listdir(output_dir):
        if file.endswith('.json'):
            file_path = os.path.join(output_dir, file)
            index_data['files_generated'].append({
                'filename': file,
                'path': file_path
            })
    
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved file index to: {index_path}")

def main():
    """Main function to run the comprehensive test suite."""
    print("🚀 COMPREHENSIVE PDF OUTLINE EXTRACTOR PERFORMANCE TEST")
    print("=" * 80)
    
    # Create a unique output directory for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), f"test_outputs_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Creating output directory: {output_dir}")
    
    # Find the latest model file
    model_files = [f for f in os.listdir(os.path.dirname(__file__)) if f.endswith('_rf_model.pkl')]
    if model_files:
        latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(os.path.dirname(__file__), f)))
        model_path = os.path.join(os.path.dirname(__file__), latest_model)
        print(f"✅ Using latest model: {model_path}")
    else:
        model_path = "enhanced_model.pkl" # Fallback to default
        print(f"⚠️  No new model file found, using default: {model_path}")

    # Load extractors
    if model_path and os.path.exists(model_path):
        try:
            ml_extractor = MLEnhancedPDFExtractor(model_path=model_path)
            print("✅ ML-Enhanced extractor loaded")
        except Exception as e:
            print(f"⚠️  Could not load ML-Enhanced extractor: {e}")
            ml_extractor = None
    else:
        ml_extractor = None

    try:
        improved_extractor = ImprovedPDFExtractor()
        print("✅ Improved extractor loaded")
    except Exception as e:
        print(f"⚠️  Could not load Improved extractor: {e}")
        improved_extractor = None

    # Load ground truth data
    ground_truth_dir = os.path.join(os.path.dirname(__file__), '..', 'Challenge_1a', 'Datasets', 'Output.json')
    ground_truths = load_ground_truths(ground_truth_dir)
    
    # PDF directory
    pdf_dir = os.path.join(os.path.dirname(__file__), 'pdfs')
    
    # Results
    results = defaultdict(dict)
    
    print("\n📊 TESTING PDFs WITH GROUND TRUTH")
    print("=" * 80)
    
    # Test PDFs with ground truth
    for pdf_name, gt_outline in ground_truths.items():
        pdf_path = os.path.join(pdf_dir, f"{pdf_name}.pdf")
        if not os.path.exists(pdf_path):
            continue
            
        print(f"\n📄 TESTING: {pdf_name}")
        print("-" * 60)
        print(f"📋 Ground Truth: {len(gt_outline.get('outline', []))} items")

        # Test Improved Extractor
        if improved_extractor:
            run_test("Improved Extractor (Advanced Rule-based)", improved_extractor, pdf_path, gt_outline, output_dir, pdf_name, "improved", results)

        # Test ML-Enhanced Extractor
        if ml_extractor:
            run_test("ML-Enhanced Approach", ml_extractor, pdf_path, gt_outline, output_dir, pdf_name, "ml_enhanced", results)
            
        # Determine winner for this PDF
        f1_improved = results[pdf_name].get('improved', {}).get('f1', 0)
        f1_ml = results[pdf_name].get('ml_enhanced', {}).get('f1', 0)
        
        if f1_improved > f1_ml:
            results[pdf_name]['winner'] = 'Improved'
            print(f"\n🏆 Best performer: Improved (F1: {f1_improved:.3f})")
        elif f1_ml > f1_improved:
            results[pdf_name]['winner'] = 'ML-Enhanced'
            print(f"\n🏆 Best performer: ML-Enhanced (F1: {f1_ml:.3f})")
        else:
            results[pdf_name]['winner'] = 'Tie'
            print(f"\n🏆 It's a tie (F1: {f1_ml:.3f})")

    # Test additional PDFs without ground truth
    print("\n📁 TESTING ADDITIONAL PDFs (No Ground Truth)")
    print("=" * 80)
    
    additional_pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf') and os.path.splitext(f)[0] not in ground_truths]
    
    for pdf_name in additional_pdfs:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        pdf_base_name = os.path.splitext(pdf_name)[0]
        
        print(f"\n📄 {pdf_base_name}:")
        
        # Run Improved Extractor
        if improved_extractor:
            start_time = time.time()
            extracted_obj = improved_extractor.extract_outline_json(pdf_path)
            duration = time.time() - start_time
            title = extracted_obj.get('title', 'N/A')
            num_headings = len(extracted_obj.get('outline', []))
            print(f"   Improved: ⏱️  {duration:.2f}s | 📋 '{title[:30]}...' | 🎯 {num_headings} headings")
            save_output(extracted_obj, output_dir, pdf_base_name, "improved")

        # Run ML-Enhanced Extractor
        if ml_extractor:
            start_time = time.time()
            extracted_obj = ml_extractor.extract_outline_json(pdf_path)
            duration = time.time() - start_time
            title = extracted_obj.get('title', 'N/A')
            num_headings = len(extracted_obj.get('outline', []))
            print(f"   ML-Enhanced: ⏱️  {duration:.2f}s | 📋 '{title[:30]}...' | 🎯 {num_headings} headings")
            save_output(extracted_obj, output_dir, pdf_base_name, "ml_enhanced")

    # Save comprehensive comparison
    comparison_path = os.path.join(output_dir, "comprehensive_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Saved comprehensive comparison to: {comparison_path}")

    # Generate and print summary report
    generate_summary_report(results, output_dir)

    # Create an index of all generated files
    create_file_index(output_dir)

if __name__ == "__main__":
    main()

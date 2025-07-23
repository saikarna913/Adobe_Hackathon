#!/usr/bin/env python3
"""
Test script for ML-Enhanced PDF Outline Extractor
Compares ML-enhanced vs rule-based approaches
"""

import os
import json
import re
from ml_enhanced_extractor import MLEnhancedPDFExtractor

def load_ground_truth(pdf_name):
    """Load ground truth for a PDF"""
    ground_truth_files = {
        'STEMPathwaysFlyer': '/home/pakambo/Documents/branch1/Adobe_Hackathon/Challenge_1a/Datasets/Output.json/STEMPathwaysFlyer.json',
        'E0CCG5S239': '/home/pakambo/Documents/branch1/Adobe_Hackathon/Challenge_1a/Datasets/Output.json/E0CCG5S239.json',
        'E0CCG5S312': '/home/pakambo/Documents/branch1/Adobe_Hackathon/Challenge_1a/Datasets/Output.json/E0CCG5S312.json',
        'E0H1CM114': '/home/pakambo/Documents/branch1/Adobe_Hackathon/Challenge_1a/Datasets/Output.json/E0H1CM114.json',
        'TOPJUMP-PARTY-INVITATION-20161003-V01': '/home/pakambo/Documents/branch1/Adobe_Hackathon/Challenge_1a/Datasets/Output.json/TOPJUMP-PARTY-INVITATION-20161003-V01.json'
    }
    
    gt_file = ground_truth_files.get(pdf_name)
    if gt_file and os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            return json.load(f)
    return {"outline": []}

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
    """Calculate precision, recall, and F1 score"""
    pred_items = [(normalize_text(item['text']), item['page']) for item in predicted.get('outline', [])]
    gt_items = [(normalize_text(item['text']), item['page']) for item in ground_truth.get('outline', [])]
    
    pred_set = set(pred_items)
    gt_set = set(gt_items)
    
    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0
    
    if not pred_set:
        return 0.0, 0.0, 0.0
    
    if not gt_set:
        return 0.0, 0.0, 0.0
    
    true_positives = len(pred_set & gt_set)
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gt_set) if gt_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def test_both_approaches():
    """Test ML-enhanced vs rule-based approaches"""
    
    print("🚀 TESTING ML-ENHANCED vs RULE-BASED PDF OUTLINE EXTRACTOR")
    print("=" * 80)
    
    # Test PDFs
    test_pdfs = [
        ('STEMPathwaysFlyer', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/STEMPathwaysFlyer.pdf'),
        ('E0CCG5S239', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/E0CCG5S239.pdf'),
        ('E0CCG5S312', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/E0CCG5S312.pdf'),
        ('E0H1CM114', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/E0H1CM114.pdf'),
        ('TOPJUMP-PARTY-INVITATION-20161003-V01', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/TOPJUMP-PARTY-INVITATION-20161003-V01.pdf')
    ]
    
    # Initialize extractors
    ml_extractor = MLEnhancedPDFExtractor(model_path='enhanced_model.pkl', use_ml=True)
    rule_extractor = MLEnhancedPDFExtractor(use_ml=False)
    
    ml_results = []
    rule_results = []
    
    for pdf_name, pdf_path in test_pdfs:
        if not os.path.exists(pdf_path):
            print(f"⚠️  PDF not found: {pdf_path}")
            continue
        
        print(f"\n📄 TESTING: {pdf_name}")
        print("-" * 60)
        
        try:
            # Get ground truth
            ground_truth = load_ground_truth(pdf_name)
            gt_count = len(ground_truth.get('outline', []))
            
            # Test ML approach
            print("🤖 ML-Enhanced Approach:")
            ml_predicted = ml_extractor.extract_outline_json(pdf_path)
            ml_precision, ml_recall, ml_f1 = calculate_metrics(ml_predicted, ground_truth)
            ml_count = len(ml_predicted.get('outline', []))
            
            print(f"   📊 P: {ml_precision:.3f}, R: {ml_recall:.3f}, F1: {ml_f1:.3f} | GT: {gt_count}, Pred: {ml_count}")
            
            # Test rule-based approach
            print("📐 Rule-Based Approach:")
            rule_predicted = rule_extractor.extract_outline_json(pdf_path)
            rule_precision, rule_recall, rule_f1 = calculate_metrics(rule_predicted, ground_truth)
            rule_count = len(rule_predicted.get('outline', []))
            
            print(f"   📊 P: {rule_precision:.3f}, R: {rule_recall:.3f}, F1: {rule_f1:.3f} | GT: {gt_count}, Pred: {rule_count}")
            
            # Show improvement
            f1_improvement = ml_f1 - rule_f1
            if f1_improvement > 0:
                print(f"   ✅ ML improvement: +{f1_improvement:.3f} F1")
            elif f1_improvement < 0:
                print(f"   ⬇️  ML degradation: {f1_improvement:.3f} F1")
            else:
                print("   ➡️  No change in F1")
            
            ml_results.append((pdf_name, ml_f1, ml_precision, ml_recall, gt_count, ml_count))
            rule_results.append((pdf_name, rule_f1, rule_precision, rule_recall, gt_count, rule_count))
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            ml_results.append((pdf_name, 0.0, 0.0, 0.0, 0, 0))
            rule_results.append((pdf_name, 0.0, 0.0, 0.0, 0, 0))
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'PDF':<40} {'ML F1':<8} {'Rule F1':<8} {'Diff':<8} {'GT':<5} {'ML':<5} {'Rule':<5}")
    print("-" * 80)
    
    total_ml_f1 = 0
    total_rule_f1 = 0
    valid_results = 0
    
    for i, (pdf_name, _, _, _, _, _) in enumerate(ml_results):
        ml_f1 = ml_results[i][1]
        rule_f1 = rule_results[i][1]
        gt_count = ml_results[i][4]
        ml_count = ml_results[i][5]
        rule_count = rule_results[i][5]
        
        diff = ml_f1 - rule_f1
        diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
        
        print(f"{pdf_name:<40} {ml_f1:<8.3f} {rule_f1:<8.3f} {diff_str:<8} {gt_count:<5} {ml_count:<5} {rule_count:<5}")
        
        total_ml_f1 += ml_f1
        total_rule_f1 += rule_f1
        valid_results += 1
    
    if valid_results > 0:
        avg_ml_f1 = total_ml_f1 / valid_results
        avg_rule_f1 = total_rule_f1 / valid_results
        avg_improvement = avg_ml_f1 - avg_rule_f1
        
        print("-" * 80)
        print(f"{'AVERAGE':<40} {avg_ml_f1:<8.3f} {avg_rule_f1:<8.3f} {avg_improvement:+.3f}")
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   🤖 ML-Enhanced F1: {avg_ml_f1:.3f}")
        print(f"   📐 Rule-Based F1:  {avg_rule_f1:.3f}")
        if avg_improvement > 0:
            print(f"   ✅ ML Improvement: +{avg_improvement:.3f} ({(avg_improvement/avg_rule_f1*100):+.1f}%)")
        else:
            print(f"   ⬇️  ML Performance: {avg_improvement:.3f} ({(avg_improvement/avg_rule_f1*100):+.1f}%)")

if __name__ == "__main__":
    test_both_approaches()

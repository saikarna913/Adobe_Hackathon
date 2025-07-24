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
    return {"title": "", "outline": []}

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
    pred_items = [normalize_text(item['text']) for item in predicted.get('outline', [])]
    gt_items = [normalize_text(item['text']) for item in ground_truth.get('outline', [])]
    
    if not gt_items and not pred_items:
        return 1.0, 1.0, 1.0
    
    if not pred_items:
        return 0.0, 0.0, 0.0
    
    if not gt_items:
        return 0.0, 0.0, 0.0
    
    # Use fuzzy matching for better comparison
    matches = 0
    for gt_text in gt_items:
        best_match_ratio = 0
        for pred_text in pred_items:
            ratio = difflib.SequenceMatcher(None, gt_text.lower(), pred_text.lower()).ratio()
            if ratio > best_match_ratio:
                best_match_ratio = ratio
        
        if best_match_ratio >= 0.6:  # 60% similarity threshold
            matches += 1
    
    precision = matches / len(pred_items) if pred_items else 0
    recall = matches / len(gt_items) if gt_items else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def test_all_approaches():
    """Test ML-enhanced vs improved approaches"""
    
    print("🚀 COMPREHENSIVE PDF OUTLINE EXTRACTOR PERFORMANCE TEST")
    print("=" * 80)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_outputs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Creating output directory: {output_dir}")
    
    # Test PDFs
    test_pdfs = [
        ('STEMPathwaysFlyer', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/STEMPathwaysFlyer.pdf'),
        ('E0CCG5S239', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/E0CCG5S239.pdf'),
        ('E0CCG5S312', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/E0CCG5S312.pdf'),
        ('E0H1CM114', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/E0H1CM114.pdf'),
        ('TOPJUMP-PARTY-INVITATION-20161003-V01', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/TOPJUMP-PARTY-INVITATION-20161003-V01.pdf')
    ]
    
    # Additional test PDFs without ground truth
    additional_pdfs = [
        ('sample', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/sample.pdf'),
        ('Breakfast Ideas', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/Breakfast Ideas.pdf'),
        ('South of France - Cities', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/South of France - Cities.pdf'),
        ('Learn Acrobat - Create and Convert_1', '/home/pakambo/Documents/branch1/Adobe_Hackathon/test2/pdfs/Learn Acrobat - Create and Convert_1.pdf'),
    ]
    
    # Initialize extractors
    try:
        ml_extractor = MLEnhancedPDFExtractor(model_path='enhanced_model.pkl', use_ml=True)
        print("✅ ML-Enhanced extractor loaded")
    except Exception as e:
        print(f"⚠️  ML-Enhanced extractor failed to load: {e}")
        ml_extractor = None
    
    improved_extractor = ImprovedPDFExtractor()
    print("✅ Improved extractor loaded")
    
    all_results = []
    all_outputs = {}
    
    # Test PDFs with ground truth
    print("\n📊 TESTING PDFs WITH GROUND TRUTH")
    print("=" * 80)
    
    all_pdfs = test_pdfs + additional_pdfs
    
    for pdf_name, pdf_path in all_pdfs:
        if not os.path.exists(pdf_path):
            print(f"⚠️  PDF not found: {pdf_path}")
            continue
        
        print(f"\n📄 TESTING: {pdf_name}")
        print("-" * 60)
        
        try:
            # Get ground truth (if available)
            ground_truth = load_ground_truth(pdf_name)
            gt_count = len(ground_truth.get('outline', []))
            has_ground_truth = gt_count > 0 or pdf_name in ['E0CCG5S239']  # E0CCG5S239 has valid GT with 0 items
            
            if has_ground_truth:
                print(f"📋 Ground Truth: {gt_count} items")
            else:
                print(f"📋 No ground truth available")
            
            results = {'pdf_name': pdf_name, 'gt_count': gt_count, 'has_ground_truth': has_ground_truth}
            pdf_outputs = {'pdf_name': pdf_name, 'pdf_path': pdf_path}
            
            # Save ground truth if available
            if has_ground_truth:
                gt_file = os.path.join(output_dir, f"{pdf_name}_ground_truth.json")
                with open(gt_file, 'w') as f:
                    json.dump(ground_truth, f, indent=2, ensure_ascii=False)
                pdf_outputs['ground_truth'] = ground_truth
            
            # Test Improved approach (rule-based)
            print("\n🔧 Improved Extractor (Advanced Rule-based):")
            start_time = time.time()
            improved_predicted = improved_extractor.extract_outline_json(pdf_path)
            improved_time = time.time() - start_time
            
            # Save improved output
            improved_file = os.path.join(output_dir, f"{pdf_name}_improved.json")
            with open(improved_file, 'w') as f:
                json.dump(improved_predicted, f, indent=2, ensure_ascii=False)
            pdf_outputs['improved'] = improved_predicted
            
            improved_count = len(improved_predicted.get('outline', []))
            print(f"   ⏱️  Time: {improved_time:.2f}s")
            print(f"   🎯 Found: {improved_count} headings")
            print(f"   💾 Saved to: {improved_file}")
            
            if has_ground_truth:
                improved_precision, improved_recall, improved_f1 = calculate_metrics(improved_predicted, ground_truth)
                print(f"   📊 P: {improved_precision:.3f}, R: {improved_recall:.3f}, F1: {improved_f1:.3f}")
                results['improved'] = {
                    'f1': improved_f1, 'precision': improved_precision, 'recall': improved_recall,
                    'count': improved_count, 'time': improved_time
                }
            
            # Test ML approach
            if ml_extractor:
                print("\n🤖 ML-Enhanced Approach:")
                start_time = time.time()
                ml_predicted = ml_extractor.extract_outline_json(pdf_path)
                ml_time = time.time() - start_time
                
                # Save ML output
                ml_file = os.path.join(output_dir, f"{pdf_name}_ml_enhanced.json")
                with open(ml_file, 'w') as f:
                    json.dump(ml_predicted, f, indent=2, ensure_ascii=False)
                pdf_outputs['ml_enhanced'] = ml_predicted
                
                ml_count = len(ml_predicted.get('outline', []))
                print(f"   ⏱️  Time: {ml_time:.2f}s")
                print(f"   🎯 Found: {ml_count} headings")
                print(f"   💾 Saved to: {ml_file}")
                
                if has_ground_truth:
                    ml_precision, ml_recall, ml_f1 = calculate_metrics(ml_predicted, ground_truth)
                    print(f"   📊 P: {ml_precision:.3f}, R: {ml_recall:.3f}, F1: {ml_f1:.3f}")
                    results['ml'] = {
                        'f1': ml_f1, 'precision': ml_precision, 'recall': ml_recall,
                        'count': ml_count, 'time': ml_time
                    }
            
            # Show best performer for PDFs with ground truth
            if has_ground_truth:
                f1_scores = [('Improved', results.get('improved', {}).get('f1', 0))]
                if ml_extractor and 'ml' in results:
                    f1_scores.append(('ML-Enhanced', results['ml']['f1']))
                
                best_approach, best_f1 = max(f1_scores, key=lambda x: x[1])
                print(f"\n🏆 Best performer: {best_approach} (F1: {best_f1:.3f})")
                all_results.append(results)
            
            # Store all outputs for this PDF
            all_outputs[pdf_name] = pdf_outputs
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    # Test additional PDFs
    print(f"\n📁 TESTING ADDITIONAL PDFs (No Ground Truth)")
    print("=" * 60)
    
    for pdf_name, pdf_path in additional_pdfs:
        if not os.path.exists(pdf_path):
            continue
            
        print(f"\n📄 {pdf_name}:")
        
        # Test all approaches
        approaches = [
            ('Improved', improved_extractor)
        ]
        if ml_extractor:
            approaches.append(('ML-Enhanced', ml_extractor))
        
        for approach_name, extractor in approaches:
            try:
                start_time = time.time()
                result = extractor.extract_outline_json(pdf_path)
                elapsed = time.time() - start_time
                
                title = result.get('title', 'N/A')
                outline_count = len(result.get('outline', []))
                
                print(f"   {approach_name}: ⏱️  {elapsed:.2f}s | 📋 '{title[:30]}...' | 🎯 {outline_count} headings")
                
            except Exception as e:
                print(f"   {approach_name}: ❌ Error - {e}")
    
    # Save comprehensive comparison file
    comparison_file = os.path.join(output_dir, "comprehensive_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved comprehensive comparison to: {comparison_file}")
    
    # Create a summary comparison file
    summary_data = {
        'test_info': {
            'timestamp': timestamp,
            'total_pdfs_tested': len(all_outputs),
            'pdfs_with_ground_truth': len([p for p in all_outputs.values() if p.get('ground_truth')])
        },
        'results': all_results if all_results else []
    }
    
    summary_file = os.path.join(output_dir, "test_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved test summary to: {summary_file}")
    
    # Create an index file for easy navigation
    index_data = {
        'output_directory': output_dir,
        'created_at': timestamp,
        'files_generated': []
    }
    
    for pdf_name in all_outputs.keys():
        pdf_files = {
            'pdf_name': pdf_name,
            'files': []
        }
        
        # Check which files were created for this PDF
        for approach in ['ground_truth', 'improved', 'ml_enhanced']:
            filename = f"{pdf_name}_{approach}.json"
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                pdf_files['files'].append({
                    'approach': approach,
                    'filename': filename,
                    'path': filepath
                })
        
        index_data['files_generated'].append(pdf_files)
    
    index_file = os.path.join(output_dir, "index.json")
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved file index to: {index_file}")
    
    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print("📊 SUMMARY REPORT")
        print(f"{'='*70}")
        
        print(f"{'PDF':<35} {'Improved':<10} {'ML-Enh':<10} {'GT':<5} {'Winner':<12}")
        print("-" * 70)
        
        improved_scores = []
        ml_scores = []
        
        for result in all_results:
            pdf_name = result['pdf_name']
            gt_count = result['gt_count']
            
            improved_f1 = result.get('improved', {}).get('f1', 0.0)
            ml_f1 = result.get('ml', {}).get('f1', 0.0)
            
            # Determine winner
            scores = [('Improved', improved_f1)]
            if ml_f1 > 0:
                scores.append(('ML-Enh', ml_f1))
            
            winner, best_score = max(scores, key=lambda x: x[1])
            
            ml_str = f"{ml_f1:.3f}" if ml_f1 > 0 else "N/A"
            
            print(f"{pdf_name:<35} {improved_f1:<10.3f} {ml_str:<10} {gt_count:<5} {winner:<12}")
            
            improved_scores.append(improved_f1)
            if ml_f1 > 0:
                ml_scores.append(ml_f1)
        
        # Calculate averages
        avg_improved = sum(improved_scores) / len(improved_scores) if improved_scores else 0
        avg_ml = sum(ml_scores) / len(ml_scores) if ml_scores else 0
        
        print("-" * 70)
        ml_avg_str = f"{avg_ml:.3f}" if avg_ml > 0 else "N/A"
        print(f"{'AVERAGE':<35} {avg_improved:<10.3f} {ml_avg_str:<10}")
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   🔧 Improved F1:    {avg_improved:.3f}")
        if avg_ml > 0:
            print(f"   🤖 ML-Enhanced F1: {avg_ml:.3f}")
        
        # Determine overall winner
        final_scores = [('Improved', avg_improved)]
        if avg_ml > 0:
            final_scores.append(('ML-Enhanced', avg_ml))
        
        overall_winner, overall_best = max(final_scores, key=lambda x: x[1])
        print(f"\n🏆 OVERALL WINNER: {overall_winner} (F1: {overall_best:.3f})")
        
        # Show improvements
        if avg_ml > 0 and avg_improved > 0:
            ml_vs_improved = avg_ml - avg_improved
            print(f"   📊 ML vs Improved: {ml_vs_improved:+.3f} ({(ml_vs_improved/avg_improved*100):+.1f}%)")
    
    print(f"\n✅ Performance test completed!")
    print(f"📁 All outputs saved in directory: {output_dir}")
    print(f"📋 Check {index_file} for a complete list of generated files")
    
    return output_dir

def test_both_approaches():
    """Legacy function - redirect to new comprehensive test"""
    test_all_approaches()

if __name__ == "__main__":
    test_all_approaches()

#!/usr/bin/env python3
"""
ML-Enhanced PDF Outline Extractor for Challenge 1A
Uses Random Forest classifier with intelligent feature engineering
Includes rule-based fallback for challenge compliance
"""

import fitz  # PyMuPDF
import json
import re
import statistics
import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import math

# Try to import ML components, fallback to rule-based if not available
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML libraries not available, using rule-based approach")

class MLEnhancedPDFExtractor:
    def __init__(self, model_path=None, use_ml=True):
        """Initialize with optional ML model"""
        self.doc = None
        self.all_text_blocks = []
        self.font_sizes = []
        self.doc_characteristics = {}
        self.model = None
        self.use_ml = use_ml and ML_AVAILABLE
        
        # Feature names based on actual training data
        self.feature_names = [
            'relative_font_size', 'is_bold', 'is_title_case', 'ends_with_punct',
            'starts_with_number', 'has_colon', 'capital_ratio', 'relative_y',
            'length_norm', 'line_density', 'prev_spacing', 'next_spacing'
        ]
        
        if self.use_ml and model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"✅ ML model loaded from: {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load ML model: {e}, using rule-based approach")
                self.use_ml = False
    
    def extract_outline_json(self, pdf_path: str) -> dict:
        """Main interface - extract outline and return JSON format"""
        headings = self.extract_outline(pdf_path)
        
        # Convert to Challenge 1A format
        json_outline = {"outline": []}
        
        for heading in headings:
            json_outline["outline"].append({
                "text": heading['text'],
                "page": heading['page'] - 1  # Convert to 0-based for Challenge 1A format
            })
        
        return json_outline
    
    def extract_outline(self, pdf_path: str) -> List[Dict]:
        """Extract outline using ML-enhanced approach with rule-based fallback"""
        self.doc = fitz.open(pdf_path)
        
        # Extract and analyze all text elements
        self._extract_all_text_blocks()
        self._analyze_document_characteristics()
        
        if not self.all_text_blocks:
            self.doc.close()
            return []
        
        # Use ML model if available, otherwise use rule-based approach
        if self.use_ml and self.model is not None:
            final_headings = self._ml_based_detection()
        else:
            final_headings = self._rule_based_detection()
        
        self.doc.close()
        return final_headings
    
    def _extract_all_text_blocks(self):
        """Extract and analyze all text blocks with advanced fragmentation handling"""
        self.all_text_blocks = []
        self.font_sizes = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    self._process_text_block_advanced(block, page_num)
    
    def _process_text_block_advanced(self, block, page_num):
        """Advanced text block processing with intelligent reassembly"""
        page = self.doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        
        for line in block["lines"]:
            line_text_parts = []
            line_sizes = []
            line_flags = []
            line_fonts = []
            line_bbox = None
            
            for span in line["spans"]:
                text = span.get("text", "").strip()
                if text:
                    line_text_parts.append(text)
                    line_sizes.append(span.get("size", 12))
                    line_flags.append(span.get("flags", 0))
                    line_fonts.append(span.get("font", ""))
                    if line_bbox is None:
                        line_bbox = list(span.get("bbox", [0, 0, 0, 0]))
                    else:
                        # Extend bbox to include this span
                        span_bbox = span.get("bbox", [0, 0, 0, 0])
                        line_bbox[2] = max(line_bbox[2], span_bbox[2])
            
            if line_text_parts:
                # Intelligent text reassembly
                full_text = self._reassemble_text_intelligently(line_text_parts)
                
                # Use dominant characteristics for the line
                dominant_size = max(set(line_sizes), key=line_sizes.count) if line_sizes else 12
                dominant_flags = max(set(line_flags), key=line_flags.count) if line_flags else 0
                dominant_font = max(set(line_fonts), key=line_fonts.count) if line_fonts else ""
                
                # Store processed block with additional ML features
                text_block = {
                    'text': full_text,
                    'size': dominant_size,
                    'flags': dominant_flags,
                    'font': dominant_font,
                    'page': page_num,
                    'bbox': line_bbox or [0, 0, 0, 0],
                    'line_count': len(line_text_parts),
                    'page_height': page_height,
                    'page_width': page_width
                }
                
                self.all_text_blocks.append(text_block)
                self.font_sizes.append(dominant_size)
    
    def _reassemble_text_intelligently(self, text_parts: List[str]) -> str:
        """Intelligently reassemble fragmented text"""
        if len(text_parts) == 1:
            return text_parts[0]
        
        result = text_parts[0]
        
        for i in range(1, len(text_parts)):
            current = text_parts[i]
            prev = text_parts[i-1]
            
            # Determine appropriate separator
            if (prev.endswith(('-', '—')) or current.startswith(('-', '—'))):
                if prev.endswith('-'):
                    result = result[:-1] + current
                else:
                    result += current
            elif (len(prev) == 1 and prev.isupper() and 
                  len(current) == 1 and current.isupper()):
                result += current
            elif (len(prev) == 1 and prev.isupper() and current[0].islower()):
                result += current
            elif (prev[-1].islower() and current[0].islower() and
                  len(prev) < 4 and len(current) < 4):
                result += current
            elif (prev[-1].isalnum() and current[0].isalnum() and
                  not (prev[-1].isupper() and current[0].isupper() and 
                       len(prev) > 2 and len(current) > 2)):
                result += " " + current
            else:
                result += " " + current
        
        # Post-processing cleanup
        result = ' '.join(result.split())
        result = re.sub(r'\b([A-Z])\s+([a-z]+)\b', r'\1\2', result)
        result = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', result)
        
        return result.strip()
    
    def _analyze_document_characteristics(self):
        """Analyze document characteristics for adaptive processing"""
        if not self.font_sizes:
            self.doc_characteristics = {'is_complex': True, 'has_toc': False}
            return
        
        font_counter = Counter(self.font_sizes)
        unique_sizes = len(font_counter)
        most_common_size = font_counter.most_common(1)[0][0]
        
        font_std = statistics.stdev(self.font_sizes) if len(self.font_sizes) > 1 else 0
        font_range = max(self.font_sizes) - min(self.font_sizes)
        
        page_count = len(self.doc)
        avg_blocks_per_page = len(self.all_text_blocks) / page_count if page_count > 0 else 0
        
        has_numbered_items = any(re.match(r'^\d+\.?\s', block['text']) for block in self.all_text_blocks)
        has_toc_indicators = any(re.search(r'(table.{0,10}contents|contents)', block['text'], re.I) 
                                for block in self.all_text_blocks)
        
        form_indicators = sum(1 for block in self.all_text_blocks 
                             if re.search(r'(name|date|signature|___|\.{3,})', block['text'], re.I))
        is_likely_form = form_indicators > len(self.all_text_blocks) * 0.2
        
        self.doc_characteristics = {
            'page_count': page_count,
            'unique_font_sizes': unique_sizes,
            'most_common_size': most_common_size,
            'font_std': font_std,
            'font_range': font_range,
            'avg_blocks_per_page': avg_blocks_per_page,
            'has_numbered_items': has_numbered_items,
            'has_toc': has_toc_indicators,
            'is_likely_form': is_likely_form,
            'is_complex': unique_sizes > 5 or font_std > 2,
            'font_distribution': font_counter
        }
    
    def _extract_ml_features(self, block: Dict, index: int) -> Dict:
        """Extract ML features for a text block"""
        text = block['text']
        size = block['size']
        flags = block['flags']
        bbox = block['bbox']
        page_height = block.get('page_height', 800)
        page_width = block.get('page_width', 600)
        
        # Basic features
        most_common_size = self.doc_characteristics['most_common_size']
        relative_font_size = size / most_common_size if most_common_size > 0 else 1.0
        is_bold = bool(flags & 16)
        is_title_case = text.istitle()
        ends_with_punct = text.rstrip().endswith(('.', '!', '?', ':', ';'))
        starts_with_number = bool(re.match(r'^\d+', text.strip()))
        has_colon = ':' in text
        
        # Text analysis features
        capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        length_norm = min(len(text) / 100.0, 1.0)  # Normalize to 0-1
        
        # Position features
        relative_y = bbox[1] / page_height if page_height > 0 else 0
        
        # Spacing features (simplified for single block)
        prev_spacing = 0.1  # Default
        next_spacing = 0.1  # Default
        if index > 0:
            prev_block = self.all_text_blocks[index-1]
            if prev_block['page'] == block['page']:
                prev_spacing = abs(bbox[1] - prev_block['bbox'][3]) / page_height
        
        if index < len(self.all_text_blocks) - 1:
            next_block = self.all_text_blocks[index+1]
            if next_block['page'] == block['page']:
                next_spacing = abs(next_block['bbox'][1] - bbox[3]) / page_height
        
        # Advanced features
        font_variation = self.doc_characteristics['font_std'] / most_common_size if most_common_size > 0 else 0
        
        # Position score (higher for top of page, left alignment)
        position_score = (1 - relative_y) * 0.5  # Top of page bonus
        if bbox[0] < page_width * 0.2:  # Left aligned
            position_score += 0.3
        
        # Structure score
        structure_score = 0
        if re.match(r'^\d+\.?\s+', text):
            structure_score += 0.5
        if text.endswith(':'):
            structure_score += 0.3
        if text.isupper() and len(text) > 3 and len(text) < 50:
            structure_score += 0.2
        
        # Semantic score
        semantic_score = 0
        semantic_patterns = [
            r'\b(introduction|overview|summary|conclusion|background)\b',
            r'\b(chapter|section|part|appendix)\s+\w+',
            r'\b(table\s+of\s+contents|references|bibliography)\b',
            r'\b(objectives?|methodology|results?|discussion)\b'
        ]
        for pattern in semantic_patterns:
            if re.search(pattern, text.lower()):
                semantic_score += 0.2
                break
        
        # Line density (simplified)
        line_density = block.get('line_count', 1) / 10.0  # Normalize
        
        return {
            'relative_font_size': relative_font_size,
            'is_bold': int(is_bold),
            'is_title_case': int(is_title_case),
            'ends_with_punct': int(ends_with_punct),
            'starts_with_number': int(starts_with_number),
            'has_colon': int(has_colon),
            'capital_ratio': capital_ratio,
            'relative_y': relative_y,
            'length_norm': length_norm,
            'line_density': line_density,
            'prev_spacing': min(prev_spacing, 1.0),
            'next_spacing': min(next_spacing, 1.0),
            'font_variation': font_variation,
            'position_score': min(position_score, 1.0),
            'structure_score': min(structure_score, 1.0),
            'semantic_score': min(semantic_score, 1.0)
        }
    
    def _ml_based_detection(self) -> List[Dict]:
        """Use ML model for heading detection"""
        candidates = []
        
        # Extract features for all blocks
        feature_data = []
        for i, block in enumerate(self.all_text_blocks):
            text = block['text'].strip()
            
            # Basic filtering
            if len(text) < 3 or len(text) > 300:
                continue
            if self._is_form_field(text) or self._is_non_heading_content(text):
                continue
            
            features = self._extract_ml_features(block, i)
            feature_data.append(features)
            candidates.append((block, features))
        
        if not candidates:
            return []
        
        # Prepare data for ML model
        feature_df = pd.DataFrame([features for _, features in candidates])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in feature_df.columns:
                feature_df[feature] = 0.0
        
            # Make predictions with adjusted threshold
            try:
                predictions_proba = self.model.predict_proba(feature_df[self.feature_names])
                
                # Get positive class probabilities
                if predictions_proba.shape[1] > 1:
                    positive_probs = predictions_proba[:, 1]
                else:
                    positive_probs = predictions_proba[:, 0]
                
                # Combine ML predictions with rule-based confidence
                final_candidates = []
                for i, (block, features) in enumerate(candidates):
                    ml_confidence = positive_probs[i]
                    rule_confidence = self._calculate_rule_based_score(block)
                    
                    # More conservative ML threshold due to training data bias
                    if ml_confidence < 0.7:  # High threshold for ML
                        ml_confidence *= 0.5  # Reduce ML confidence for uncertain predictions
                    
                    # Ensemble: emphasize rule-based for better precision
                    final_confidence = 0.4 * ml_confidence + 0.6 * rule_confidence
                    
                    # Higher threshold to reduce false positives
                    if final_confidence > 0.6:
                        final_candidates.append((block, final_confidence))
                
                # Sort by confidence and apply more aggressive filtering
                final_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # More conservative dynamic threshold
                if final_candidates:
                    confidences = [conf for _, conf in final_candidates]
                    threshold = max(0.7, statistics.median(confidences) * 0.9)
                    final_candidates = [(block, conf) for block, conf in final_candidates if conf >= threshold]
                
                # Limit results more aggressively
                max_headings = min(30, len(self.all_text_blocks) // 3)  # At most 1/3 of all blocks
                final_candidates = final_candidates[:max_headings]            # Format results
            # Format results
                final_headings = []
                for block, confidence in final_candidates:
                    text = self._clean_text(block['text'])
                    final_headings.append({
                        'text': text,
                        'page': block['page'] + 1,
                        'confidence': confidence
                    })
                
                return final_headings
                
            except Exception as e:
                print(f"⚠️  ML prediction failed: {e}, falling back to rule-based")
                return self._rule_based_detection()
    
    def _calculate_rule_based_score(self, block: Dict) -> float:
        """Calculate rule-based confidence score"""
        text = block['text']
        size = block['size']
        flags = block['flags']
        
        score = 0.0
        most_common = self.doc_characteristics['most_common_size']
        
        # Font size
        if size > most_common:
            score += min((size - most_common) / 10.0, 0.4)
        
        # Bold
        if flags & 16:
            score += 0.3
        
        # Structure patterns
        if re.match(r'^\d+\.?\s+[A-Z]', text):
            score += 0.3
        elif text.endswith(':') and len(text) < 100:
            score += 0.2
        elif text.isupper() and len(text) > 3 and len(text) < 50:
            score += 0.2
        
        return min(score, 1.0)
    
    def _rule_based_detection(self) -> List[Dict]:
        """Fallback rule-based detection (similar to advanced extractor)"""
        # Use ensemble approach from advanced extractor
        strategies = [
            ('font_size', self._detect_by_font_size, 0.35),
            ('formatting', self._detect_by_formatting, 0.25),
            ('structure', self._detect_by_structure_patterns, 0.20),
            ('position', self._detect_by_position, 0.20)
        ]
        
        strategy_votes = defaultdict(lambda: {'block': None, 'votes': 0, 'total_confidence': 0})
        
        for strategy_name, strategy_func, weight in strategies:
            candidates = strategy_func()
            
            for block, confidence in candidates:
                key = (block['text'].strip(), block['page'])
                if strategy_votes[key]['block'] is None:
                    strategy_votes[key]['block'] = block
                
                weighted_confidence = confidence * weight
                strategy_votes[key]['votes'] += 1
                strategy_votes[key]['total_confidence'] += weighted_confidence
        
        # Convert to list with ensemble scores
        ensemble_candidates = []
        for key, data in strategy_votes.items():
            block = data['block']
            consensus_bonus = (data['votes'] - 1) * 0.1
            final_score = data['total_confidence'] + consensus_bonus
            
            ensemble_candidates.append((block, final_score))
        
        return self._intelligent_filtering_and_ranking(ensemble_candidates)
    
    # Include detection methods from advanced extractor
    def _detect_by_font_size(self) -> List[Tuple[Dict, float]]:
        """Detect headings by font size with adaptive thresholds"""
        if not self.font_sizes:
            return []
        
        candidates = []
        font_dist = self.doc_characteristics['font_distribution']
        most_common = self.doc_characteristics['most_common_size']
        
        if self.doc_characteristics['font_range'] > 5:
            font_std = self.doc_characteristics['font_std']
            size_threshold = most_common + (0.5 * font_std)
        else:
            size_threshold = most_common + 1
        
        for block in self.all_text_blocks:
            size = block['size']
            if size > size_threshold:
                size_diff = size - most_common
                confidence = min(1.0, size_diff / 10.0)
                candidates.append((block, confidence))
        
        return candidates
    
    def _detect_by_formatting(self) -> List[Tuple[Dict, float]]:
        """Detect headings by formatting"""
        candidates = []
        
        for block in self.all_text_blocks:
            confidence = 0.0
            
            if block['flags'] & 16:
                confidence += 0.6
            
            font = block.get('font', '').lower()
            if any(keyword in font for keyword in ['bold', 'black', 'heavy']):
                confidence += 0.3
            
            text = block['text']
            if (text.isupper() and len(text) > 3 and len(text) < 100 and
                not self._is_form_field(text)):
                confidence += 0.4
            
            if confidence > 0.3:
                candidates.append((block, confidence))
        
        return candidates
    
    def _detect_by_structure_patterns(self) -> List[Tuple[Dict, float]]:
        """Detect headings by structural patterns"""
        candidates = []
        
        for block in self.all_text_blocks:
            text = block['text'].strip()
            confidence = 0.0
            
            if re.match(r'^\d+\.?\s+[A-Z]', text):
                confidence += 0.7
            elif re.match(r'^[A-Z]+\.\s+[A-Z]', text):
                confidence += 0.6
            elif re.match(r'^\w+\s*\d+', text):
                confidence += 0.5
            
            if re.match(r'^[IVX]+\.?\s', text):
                confidence += 0.6
            
            if text.endswith('?') and len(text.split()) > 2:
                confidence += 0.4
            
            if text.endswith(':') and len(text) < 100:
                confidence += 0.3
            
            if confidence > 0.2:
                candidates.append((block, confidence))
        
        return candidates
    
    def _detect_by_position(self) -> List[Tuple[Dict, float]]:
        """Detect headings by position"""
        candidates = []
        
        page_blocks = defaultdict(list)
        for block in self.all_text_blocks:
            page_blocks[block['page']].append(block)
        
        for page_num, blocks in page_blocks.items():
            if not blocks:
                continue
            
            blocks.sort(key=lambda b: b['bbox'][1])
            
            for i, block in enumerate(blocks):
                confidence = 0.0
                
                if i < 3:
                    confidence += 0.3 * (3 - i) / 3
                
                left_margin = block['bbox'][0]
                if left_margin < 100:
                    confidence += 0.2
                
                if i > 0 and i < len(blocks) - 1:
                    prev_bottom = blocks[i-1]['bbox'][3]
                    current_top = block['bbox'][1]
                    next_top = blocks[i+1]['bbox'][1]
                    current_bottom = block['bbox'][3]
                    
                    space_above = current_top - prev_bottom
                    space_below = next_top - current_bottom
                    
                    if space_above > 10 and space_below > 10:
                        confidence += 0.3
                
                if confidence > 0.1:
                    candidates.append((block, confidence))
        
        return candidates
    
    def _intelligent_filtering_and_ranking(self, candidates: List[Tuple[Dict, float]]) -> List[Dict]:
        """Apply intelligent filtering and ranking"""
        if not candidates:
            return []
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        filtered_candidates = []
        
        for block, score in candidates:
            text = block['text'].strip()
            
            if len(text) < 3 or len(text) > 300:
                continue
            
            if self._is_form_field(text) or self._is_non_heading_content(text):
                continue
            
            if re.match(r'^\d+[\d\s\-/]*$', text):
                continue
            
            if self._passes_document_specific_filter(text, score):
                filtered_candidates.append((block, score))
        
        # Dynamic threshold
        if filtered_candidates:
            scores = [score for _, score in filtered_candidates]
            if len(scores) > 1:
                score_threshold = statistics.median(scores) * 0.7
            else:
                score_threshold = 0.3
        else:
            score_threshold = 0.3
        
        final_headings = []
        for block, score in filtered_candidates:
            if score >= score_threshold:
                final_headings.append({
                    'text': self._clean_text(block['text']),
                    'page': block['page'] + 1,
                    'confidence': score
                })
        
        return final_headings
    
    def _is_form_field(self, text: str) -> bool:
        """Check if text appears to be a form field"""
        text_lower = text.lower()
        form_patterns = [
            r'^(name|date|signature|address|phone|email)[:.]?\s*$',
            r'__{3,}', r'\.{3,}', r'^\w+\s*:\s*$', r'^(yes|no)\s*\[\s*\]',
        ]
        return any(re.search(pattern, text_lower) for pattern in form_patterns)
    
    def _is_non_heading_content(self, text: str) -> bool:
        """Check if text is clearly not a heading"""
        if len(text.split()) > 15:
            return True
        if text and text[0].islower() and not text.startswith(('e.g.', 'i.e.')):
            return True
        if '. ' in text and len(text.split('.')) > 2:
            return True
        if re.search(r'(https?://|www\.|@\w+\.\w+)', text):
            return True
        return False
    
    def _passes_document_specific_filter(self, text: str, score: float) -> bool:
        """Apply document-specific filtering logic"""
        if self.doc_characteristics.get('is_likely_form', False):
            return score > 0.6 and not self._is_form_field(text)
        
        if self.doc_characteristics.get('has_toc', False):
            if re.match(r'^\d+\.?\s+', text):
                return score > 0.3
        
        if self.doc_characteristics.get('is_complex', False):
            return score > 0.4
        
        return score > 0.3
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text with advanced handling"""
        text = ' '.join(text.split())
        
        # Fix fragmentation
        text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', text)
        text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)
        text = re.sub(r'\b([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\b([A-Z])\s+([a-z])', r'\1\2', text)
        text = re.sub(r'\b([A-Z])\s+([A-Z][A-Z]+)\b', r'\1\2', text)
        
        # Fix punctuation
        text = re.sub(r'\s+([!?.,;:])', r'\1', text)
        text = re.sub(r'[.]{2,}$', '', text)
        text = re.sub(r'\s*:\s*$', '', text)
        
        return ' '.join(text.split()).strip()

    def train_model(self, training_data_path: str, model_save_path: str = None):
        """Train ML model from training data"""
        if not ML_AVAILABLE:
            print("❌ ML libraries not available for training")
            return False
        
        try:
            # Load training data
            if training_data_path.endswith('.xlsx'):
                df = pd.read_excel(training_data_path)
            else:
                df = pd.read_csv(training_data_path)
            
            print(f"📊 Loaded training data: {df.shape}")
            print(f"📋 Available features: {list(df.columns)}")
            
            # Create binary target from level (assuming level > 0 means heading)
            if 'level' in df.columns:
                df['is_heading'] = (df['level'] > 0).astype(int)
                target_col = 'is_heading'
            elif 'is_outline' in df.columns:
                target_col = 'is_outline'
            elif 'is_heading' in df.columns:
                target_col = 'is_heading'
            else:
                print("❌ No suitable target column found (level, is_outline, is_heading)")
                return False
            
            print(f"🎯 Target distribution: {df[target_col].value_counts().to_dict()}")
            
            # Check which features are available
            available_features = [f for f in self.feature_names if f in df.columns]
            missing_features = [f for f in self.feature_names if f not in df.columns]
            
            if missing_features:
                print(f"⚠️  Missing features: {missing_features}")
                print(f"✅ Using available features: {available_features}")
                self.feature_names = available_features
            
            if not available_features:
                print("❌ No matching features found")
                return False
            
            # Prepare features and labels
            X = df[available_features]
            y = df[target_col]
            
            # Handle missing values
            X = X.fillna(0)
            
            print(f"📈 Training with {len(available_features)} features on {len(X)} samples")
            
            # Split data (no stratification due to small dataset)
            if len(X) < 10:
                print("⚠️  Very small dataset, using simple train/validation split")
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest with optimized parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'  # Handle imbalanced data
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            print("📊 Model Performance:")
            print(classification_report(y_test, y_pred))
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("🎯 Top Feature Importances:")
            print(feature_importance.head(10))
            
            # Save model
            if model_save_path:
                joblib.dump(self.model, model_save_path)
                print(f"💾 Model saved to: {model_save_path}")
            
            self.use_ml = True
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='ML-Enhanced PDF Outline Extractor')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--model', '-m', help='Path to trained ML model')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--train', help='Path to training data for model training')
    parser.add_argument('--save-model', help='Path to save trained model')
    parser.add_argument('--no-ml', action='store_true', help='Use only rule-based approach')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = MLEnhancedPDFExtractor(
        model_path=args.model,
        use_ml=not args.no_ml
    )
    
    # Train model if requested
    if args.train:
        if extractor.train_model(args.train, args.save_model):
            print("✅ Model training completed")
        else:
            print("❌ Model training failed")
            return 1
    
    # Extract outline
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found")
        return 1
    
    try:
        result = extractor.extract_outline_json(args.pdf_path)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Outline saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
        
        return 0
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""
Improved PDF outline extractor based on detailed analysis insights.

Key improvements:
1. Better text fragment reassembly 
2. More accurate heading level classification
3. Improved filtering for false positives
4. Document-type adaptive thresholds
"""

import fitz
import json
import sys
import re
from collections import defaultdict, Counter
import statistics

class ImprovedPDFExtractor:
    def __init__(self):
        self.doc = None
        self.font_sizes = []
        self.all_text_blocks = []
        self.doc_characteristics = {}
        
    def extract_outline_json(self, pdf_path):
        """Extract outline in the required JSON format."""
        try:
            self.doc = fitz.open(pdf_path)
            self._analyze_document_characteristics()
            self._extract_all_text_blocks()
            
            # Extract title and outline
            title = self._extract_title()
            outline_items = self._extract_headings()
            
            result = {
                "title": title,
                "outline": outline_items
            }
            
            self.doc.close()
            return result
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            if self.doc:
                self.doc.close()
            return {"title": "", "outline": []}
    
    def _analyze_document_characteristics(self):
        """Analyze document to determine its characteristics for adaptive processing."""
        page_count = len(self.doc)
        total_blocks = 0
        font_size_counter = Counter()
        
        # Sample first few pages to understand document structure
        sample_pages = min(3, page_count)
        for page_num in range(sample_pages):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            total_blocks += len(blocks)
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size_counter[round(span["size"], 1)] += 1
        
        # Determine document characteristics
        self.doc_characteristics = {
            'page_count': page_count,
            'avg_blocks_per_page': total_blocks / sample_pages,
            'common_font_sizes': font_size_counter.most_common(5),
            'is_complex_document': total_blocks / sample_pages > 25,  # Increased threshold
            'has_toc': self._likely_has_toc(),
            'doc_type': 'unknown' # To be determined
        }
        
        # Determine document type
        self.doc_characteristics['doc_type'] = self._determine_document_type()
        
        print(f"📊 Document characteristics: {self.doc_characteristics}")
    
    def _determine_document_type(self):
        """Determine document type for more tailored processing."""
        text_sample = ""
        for page_num in range(min(3, len(self.doc))): # Sample more pages
            text_sample += self.doc[page_num].get_text().lower()

        # Form detection (more robust)
        form_keywords = ['date', 'name', 'signature', 's.no', 'serial', 'amount', 'total', 'form', 'application', 'invoice', 'bill to', 'ship to']
        if sum(1 for keyword in form_keywords if keyword in text_sample) >= 2 or text_sample.count('___') > 3:
            return 'form'

        # Academic paper/report detection
        academic_keywords = ['abstract', 'introduction', 'references', 'methodology', 'conclusion', 'figure', 'table', 'acknowledgments']
        if sum(1 for keyword in academic_keywords if keyword in text_sample) >= 3:
            return 'academic'
            
        # Presentation detection (fewer blocks, larger fonts)
        if self.doc_characteristics['avg_blocks_per_page'] < 12 and self.doc_characteristics['common_font_sizes'] and self.doc_characteristics['common_font_sizes'][0][0] > 18:
            return 'presentation'

        return 'general'
    
    def _likely_has_toc(self):
        """Check if document likely has a table of contents."""
        # Look for TOC indicators in first few pages
        for page_num in range(min(5, len(self.doc))):
            page = self.doc[page_num]
            text = page.get_text().lower()
            if any(phrase in text for phrase in ['table of contents', 'contents', 'toc']):
                return True
        return False
    
    def _extract_all_text_blocks(self):
        """Extract and analyze all text blocks from the document."""
        self.all_text_blocks = []
        self.font_sizes = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    self._process_text_block(block, page_num)
    
    def _process_text_block(self, block, page_num):
        """Process a text block to extract and reassemble fragmented text."""
        # Group spans by line and reassemble fragments
        for line in block["lines"]:
            line_spans = []
            for span in line["spans"]:
                text = span.get("text", "").strip()
                if text:
                    line_spans.append({
                        'text': text,
                        'size': span.get("size", 12),
                        'flags': span.get("flags", 0),
                        'font': span.get("font", ""),
                        'bbox': span.get("bbox", [0, 0, 0, 0])
                    })
            
            if line_spans:
                # Reassemble fragmented text
                reassembled = self._reassemble_line_fragments(line_spans)
                if reassembled:
                    self.all_text_blocks.append({
                        **reassembled,
                        'page_num': page_num,
                        'y_position': line['bbox'][1] if 'bbox' in line else 0
                    })
                    self.font_sizes.append(reassembled['size'])
    
    def _reassemble_line_fragments(self, spans):
        """Reassemble text fragments that should be together with improved logic."""
        if not spans:
            return None
            
        # Sort spans by x-position to get correct reading order
        spans.sort(key=lambda s: s['bbox'][0])
        
        # Reassemble text with intelligent spacing
        full_text = ""
        avg_size = statistics.mean([s['size'] for s in spans])
        combined_flags = 0
        
        for i, span in enumerate(spans):
            text = span['text'].strip()
            
            if i == 0:
                full_text = text
            else:
                prev_bbox = spans[i-1]['bbox']
                curr_bbox = span['bbox']
                
                # Calculate horizontal gap between spans
                gap = curr_bbox[0] - prev_bbox[2]  # Left edge of current - right edge of previous
                avg_char_width = (prev_bbox[2] - prev_bbox[0]) / max(1, len(spans[i-1]['text']))
                
                # Determine if we need a space based on gap and text content
                needs_space = self._should_add_space_between_fragments(
                    full_text, text, gap, avg_char_width
                )
                
                if needs_space and not full_text.endswith(' ') and not text.startswith(' '):
                    full_text += " "
                
                full_text += text
            
            combined_flags |= span['flags']  # Combine formatting flags
        
        # Advanced text cleanup
        full_text = self._clean_reassembled_text(full_text)
        
        if len(full_text) < 2:  # Skip very short text
            return None
            
        return {
            'text': full_text,
            'size': avg_size,
            'flags': combined_flags,
            'font': spans[0]['font']
        }
    
    def _should_add_space_between_fragments(self, prev_text, curr_text, gap, avg_char_width):
        """Determine if space should be added between text fragments."""
        # Don't add space if either text already has space at the boundary
        if prev_text.endswith(' ') or curr_text.startswith(' '):
            return False
        
        # Don't add space for hyphenated words
        if prev_text.endswith('-') or curr_text.startswith('-'):
            return False
        
        # Add space if there's a significant gap (more than half a character width)
        if gap > avg_char_width * 0.5:
            return True
        
        # Add space between separate words (both end/start with letters)
        if (prev_text and curr_text and 
            prev_text[-1].isalpha() and curr_text[0].isalpha()):
            return True
        
        # Add space before punctuation that should be separated
        if curr_text[0] in '!?.' and not prev_text.endswith(' '):
            return True
        
        return False
    
    def _clean_reassembled_text(self, text):
        """Clean up reassembled text with advanced normalization."""
        if not text:
            return text
        
        # Fix common OCR/fragmentation issues
        # Remove excessive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix broken words (common in OCR) - improved patterns
        # Pattern: single letter followed by space and then more letters (case sensitive)
        text = re.sub(r'\b([A-Za-z])\s+([a-z]{2,})\b', r'\1\2', text)
        
        # Fix split uppercase words (like "Y ou" -> "You")
        text = re.sub(r'\b([A-Z])\s+([a-z]+)\b', r'\1\2', text)
        
        # Fix split words where parts are separated (like "T HERE" -> "THERE")
        text = re.sub(r'\b([A-Z])\s+([A-Z]{2,})\b', r'\1\2', text)
        
        # Fix common word fragments
        common_fixes = {
            r'\bT HERE\b': 'THERE',
            r'\bY ou\b': 'You', 
            r'\bY our\b': 'Your',
            r'\bF rom\b': 'From',
            r'\bA nd\b': 'And',
            r'\bT he\b': 'The',
            r'\bW ith\b': 'With',
            r'\bF or\b': 'For',
            r'\bT o\b': 'To',
        }
        
        for pattern, replacement in common_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix numbers split from words
        text = re.sub(r'(\d+)\s+([a-zA-Z]+)', r'\1\2', text)
        text = re.sub(r'([a-zA-Z]+)\s+(\d+)', r'\1\2', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([!?.,:;])', r'\1', text)
        text = re.sub(r'([!?.,:;])\s+', r'\1 ', text)
        
        # Fix quotes and parentheses
        text = re.sub(r'\s+([\"\'\)])', r'\1', text)
        text = re.sub(r'([\"\'\(])\s+', r'\1', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_title(self):
        """Extract document title."""
        if not self.all_text_blocks:
            return ""
        
        # Look for title in first page, largest font size
        first_page_blocks = [b for b in self.all_text_blocks if b['page_num'] == 0]
        if not first_page_blocks:
            return ""
        
        # Find blocks with largest font size on first page
        max_size = max(b['size'] for b in first_page_blocks)
        title_candidates = [b for b in first_page_blocks if b['size'] >= max_size - 1]
        
        # Pick the most title-like candidate
        for block in title_candidates:
            text = block['text'].strip()
            # Skip very short or very long texts, or those that look like headers
            if 5 <= len(text) <= 150 and not re.match(r'^(page \d+|chapter \d+)', text.lower()):
                return text
        
        return ""
    
    def _extract_headings(self):
        """Extract headings with improved logic."""
        if not self.all_text_blocks:
            return []
        
        # Calculate adaptive thresholds based on document characteristics
        thresholds = self._calculate_adaptive_thresholds()
        
        # Identify potential headings
        heading_candidates = []
        for block in self.all_text_blocks:
            if self._is_heading_improved(block, thresholds):
                heading_candidates.append(block)
        
        # Filter and classify headings
        filtered_headings = self._filter_and_classify_headings(heading_candidates, thresholds)
        
        # Convert to required format
        outline_items = []
        for heading in filtered_headings:
            outline_items.append({
                "text": heading['text'],
                "level": heading['level'],
                "page": heading['page_num']
            })
        
        return outline_items
    
    def _calculate_adaptive_thresholds(self):
        """Calculate fully adaptive thresholds using document-specific intelligence."""
        if not self.font_sizes:
            return {'min_heading_size': 12, 'large_size': 14, 'very_large_size': 16}
        
        font_stats = {
            'mean': statistics.mean(self.font_sizes),
            'median': statistics.median(self.font_sizes),
            'max': max(self.font_sizes),
            'min': min(self.font_sizes),
            'std': statistics.stdev(self.font_sizes) if len(self.font_sizes) > 1 else 0
        }
        
        # Calculate font size distribution characteristics
        font_range = font_stats['max'] - font_stats['min']
        sorted_sizes = sorted(set(self.font_sizes))
        
        # Determine variation type based on distribution analysis
        unique_sizes = len(sorted_sizes)
        has_good_variation = (
            font_stats['std'] > 0.5 and  # Meaningful standard deviation
            font_range > 1.5 and         # Decent range of sizes
            unique_sizes >= 3             # At least 3 different sizes
        )
        
        # Calculate thresholds based on font size distribution
        if has_good_variation:
            # Use statistical approach for well-distributed fonts
            min_heading_size = font_stats['median'] + font_stats['std'] * 0.5
            large_size = font_stats['median'] + font_stats['std'] * 1.2
            very_large_size = font_stats['median'] + font_stats['std'] * 2.0
        else:
            # Use quartile approach for limited variation
            n = len(sorted_sizes)
            q1_idx = max(0, n // 4)
            q3_idx = max(0, (3 * n) // 4)
            
            min_heading_size = sorted_sizes[q1_idx] if q1_idx < len(sorted_sizes) else font_stats['median']
            large_size = sorted_sizes[q3_idx] if q3_idx < len(sorted_sizes) else font_stats['max']
            very_large_size = font_stats['max']
        
        # Ensure logical progression with minimum gaps
        min_gap = max(0.5, font_stats['std'] * 0.3)
        
        if large_size <= min_heading_size:
            large_size = min_heading_size + min_gap
        if very_large_size <= large_size:
            very_large_size = large_size + min_gap
            
        return {
            'min_heading_size': min_heading_size,
            'large_size': large_size,
            'very_large_size': very_large_size,
            'font_stats': font_stats,
            'has_good_variation': has_good_variation,
            'unique_sizes': unique_sizes,
            'font_range': font_range
        }
    
    def _is_heading_improved(self, block, thresholds):
        """Advanced heading detection using intelligent scoring with document awareness."""
        text = block['text'].strip()
        size = block['size']
        flags = block['flags']
        
        # Dynamic length constraints based on document analysis
        if len(text) < 2 or len(text) > 200:
            return False
        
        if self._is_obviously_not_heading_strict(text):
            return False
        
        if self._is_fragmentary(text):
            return False
        
        # Calculate comprehensive score
        score = 0
        confidence_factors = []
        
        # 1. Font size analysis (adaptive weight)
        font_stats = thresholds['font_stats']
        size_score = 0
        
        if size >= thresholds['very_large_size']:
            size_score = 4
            confidence_factors.append('very_large_font')
        elif size >= thresholds['large_size']:
            size_score = 3
            confidence_factors.append('large_font')
        elif size >= thresholds['min_heading_size']:
            size_score = 2
            confidence_factors.append('medium_font')
        elif size >= font_stats['median']:
            size_score = 1
            confidence_factors.append('above_median_font')
        
        # Weight font size based on document characteristics
        font_weight = 1.5 if thresholds['has_good_variation'] else 1.0
        score += size_score * font_weight
        
        # 2. Typography and formatting
        is_bold = bool(flags & 16)
        if is_bold:
            score += 2.5
            confidence_factors.append('bold')
        
        # 3. Advanced structural pattern recognition
        structure_score = self._calculate_advanced_structure_score(text)
        score += structure_score
        if structure_score > 2:
            confidence_factors.append('strong_structure')
        
        # 4. Position and context analysis
        position_score = self._calculate_position_score(block, thresholds)
        score += position_score
        if position_score > 0:
            confidence_factors.append('good_position')
        
        # 5. Document type specific adjustments
        doc_type_score = self._calculate_document_type_score(text, block)
        score += doc_type_score
        
        # Dynamic threshold calculation
        base_threshold = self._calculate_dynamic_threshold(thresholds, confidence_factors)
        
        return score >= base_threshold
    
    def _calculate_advanced_structure_score(self, text):
        """Advanced structural pattern analysis for heading detection."""
        score = 0
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Numbered section patterns (strong indicators)
        numbering_patterns = [
            (r'^\d+\.\s+[A-Z]', 5),           # "1. Title"
            (r'^\d+\.\d+\s+[A-Z]', 5),        # "1.1 Subtitle"  
            (r'^\d+\.\d+\.\d+\s+[A-Z]', 4),   # "1.1.1 Sub-subtitle"
            (r'^[A-Z]\.\s+[A-Z]', 4),         # "A. Title"
            (r'^\([a-z]\)\s+[A-Z]', 3),       # "(a) Item"
            (r'^\d+\)\s+[A-Z]', 3),           # "1) Item"
            (r'^[IVXLC]+\.\s+[A-Z]', 4),        # "I. Roman numerals"
        ]
        
        for pattern, points in numbering_patterns:
            if re.match(pattern, text):
                return points + 1  # Extra bonus for numbered sections
        
        # Case and formatting patterns
        if text.isupper() and word_count >= 2:
            score += min(4, word_count // 2 + 1.5)
        elif text.istitle() and word_count >= 2:
            score += min(3, word_count // 3 + 1.5)
        elif len(text) > 1 and text[0].isupper() and word_count < 5: # Only for short phrases
            score += 0.5
        
        # Structural keywords (context-aware)
        keyword_patterns = [
            # High-value structural indicators
            (r'\b(chapter|section|part|appendix|introduction|conclusion|summary|overview|abstract)\b', 5),
            (r'\b(background|methodology|results|discussion|references|acknowledgments|bibliography)\b', 4),
            (r'\b(preface|foreword|epilogue|index|glossary|table\s+of\s+contents)\b', 4),
            # Document-specific patterns
            (r'\b(phase\s+[ivxlc\d]+|step\s+\d+|stage\s+\d+)\b', 3),
            (r'\b(appendix\s+[a-z]|figure\s+\d+|table\s+\d+)\b', 2),
        ]
        
        for pattern, points in keyword_patterns:
            if re.search(pattern, text_lower):
                score += points
                break  # Only count the highest match
        
        # Question headings
        if text.endswith('?') and word_count >= 3:
            score += 2.5
        
        # Colon endings (selective)
        if text.endswith(':') and word_count >= 2:
            if not any(text_lower.startswith(prefix) for prefix in ['note:', 'tip:', 'warning:', 'example:']):
                score += 2.5
        
        return min(score, 7)  # Cap at 7 points
    
    def _calculate_position_score(self, block, thresholds):
        """Calculate position-based confidence score."""
        score = 0
        y_pos = block.get('y_position', float('inf'))
        page_num = block.get('page_num', 0)
        
        # Top of page bonus
        if y_pos < 100:
            score += 1
        
        # First page bonus for certain patterns
        if page_num == 0 and y_pos < 200:
            score += 0.5
        
        return score
    
    def _calculate_document_type_score(self, text, block):
        """Document type specific scoring adjustments."""
        score = 0
        text_lower = text.lower()
        
        # Form document penalties
        if self.doc_characteristics['doc_type'] == 'form':
            form_indicators = ['date', 'name', 'signature', 's.no', 'serial', 'amount', 'total']
            if any(indicator in text_lower for indicator in form_indicators) and len(text) < 30:
                score -= 5 # Stronger penalty
        
        # Academic/Technical document bonuses
        if self.doc_characteristics['doc_type'] in ['academic', 'general']:
            tech_indicators = ['system', 'process', 'method', 'approach', 'framework', 'model', 'algorithm', 'design', 'architecture']
            if any(indicator in text_lower for indicator in tech_indicators):
                score += 1.5 # Stronger bonus
        
        return score
    
    def _calculate_dynamic_threshold(self, thresholds, confidence_factors):
        """Calculate dynamic threshold based on document characteristics and confidence factors."""
        base_threshold = 4.5  # Increased starting point for better precision
        
        # Adjust based on document complexity and type
        if self.doc_characteristics['is_complex_document']:
            base_threshold += 1.5
        if self.doc_characteristics['doc_type'] == 'form':
            base_threshold += 2.5 # Be even more strict with forms
        if self.doc_characteristics['doc_type'] == 'academic':
            base_threshold -= 1.0 # More lenient with academic papers
        
        # Adjust based on font variation
        if not thresholds['has_good_variation']:
            base_threshold -= 1.5 # If only bold/not bold is an indicator, be more lenient
        
        # Adjust based on confidence factors
        strong_indicators = sum(1 for factor in ['very_large_font', 'bold', 'strong_structure'] 
                              if factor in confidence_factors)
        if strong_indicators >= 2:
            base_threshold -= 2.5 # Strong reduction for very confident headings
        elif strong_indicators == 1:
            base_threshold -= 1.5

        # Document type adjustments
        if thresholds['unique_sizes'] < 4:
            base_threshold -= 1.0
        
        return max(base_threshold, 3.5)  # Increased minimum threshold
    
    def _is_obviously_not_heading_strict(self, text):
        """Enhanced filtering with document-aware logic to prevent false positives."""
        text_lower = text.lower()
        
        # Regex patterns for obvious non-headings
        non_heading_patterns = [
            r'^\d{1,4}$',                                    # Just numbers
            r'^(page|figure|table)\s+\d+',                   # Page numbers and labels
            r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}',          # Dates
            r'^https?:\/\/',                                 # URLs
            r'^www\.',                                       # Web addresses
            r'\.com|\.org|\.net|\.edu',                      # Domain names
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', # Emails
            r'^version\s+\d+',                               # Version numbers
            r'^\d+\.\d+$',                                   # Decimal numbers
            r'^[\(\)\[\]]+$',                                # Just brackets
        ]
        
        for pattern in non_heading_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Form field indicators (major source of false positives)
        form_fields = [
            'date', 'name', 'signature', 'amount', 'total', 'subtotal',
            's.no', 'sr.no', 'serial no', 'sl.no', 'relationship',
            'pay', 'salary', 'grade', 'designation', 'department'
        ]
        
        # Single word form fields
        if text_lower.strip() in form_fields:
            return True
        
        # Form field patterns with colons or parentheses
        if len(text) < 20 and any(field in text_lower for field in form_fields):
            return True
        
        # Navigation and UI elements
        ui_elements = [
            'click here', 'read more', 'continue reading', 'next page',
            'previous page', 'home page', 'back to top', 'print', 'save',
            'download', 'submit', 'cancel', 'ok', 'yes', 'no'
        ]
        
        if any(element in text_lower for element in ui_elements):
            return True
        
        # Single common words that are rarely headings
        single_word_excludes = {
            'goals', 'applied', 'science', 'math', 'technology', 'computer',
            'family', 'consumer', 'regular', 'distinction', 'total', 'amount',
            'date', 'time', 'location', 'contact', 'email', 'phone', 'note',
            'figure', 'table', 'appendix', 'chapter', 'section', 'page', 'form'
        }
        
        clean_text = text_lower.strip().rstrip(':').rstrip('.')
        if clean_text in single_word_excludes:
            return True
        
        # Sentence-like text detection (improved)
        words = text_lower.split()
        if len(words) > 4: # Stricter check for shorter phrases
            # Common sentence words
            sentence_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
                'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should',
                'this', 'that', 'these', 'those', 'can', 'may', 'might'
            }
            
            sentence_word_count = sum(1 for word in words if word in sentence_words)
            sentence_ratio = sentence_word_count / len(words)
            
            # Dynamic threshold based on text length
            threshold = max(0.3, 0.5 - (len(words) / 30))
            if sentence_ratio > threshold:
                return True
        
        # Descriptive text patterns
        descriptive_starters = [
            'the ', 'this ', 'that ', 'these ', 'those ', 'a ', 'an ',
            'some ', 'many ', 'most ', 'all ', 'each ', 'every ',
            'while ', 'when ', 'where ', 'whether ', 'if ', 'unless '
        ]
        
        if len(text) > 25 and any(text_lower.startswith(starter) for starter in descriptive_starters):
            return True
        
        # Sentence structure indicators
        if text.count('.') > 1 or (': ' in text and len(text) > 40) or text.endswith(','):
            return True
        
        # Starts with lowercase (likely continuation)
        if len(text) > 1 and text[0].islower() and not text.istitle():
            return True
        
        # Common verb patterns in sentences
        sentence_patterns = [
            r'\b(can|will|should|would|could|may|might|must)\s+be\b',
            r'\b(is|are|was|were)\s+(a|an|the)\b',
            r'\byou\s+(can|will|should|are|were)\b',
            r'\bthis\s+(is|will|can|should)\b',
            r'\bto\s+(provide|ensure|maintain|develop|create)\b'
        ]
        
        for pattern in sentence_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _calculate_structure_score_strict(self, text):
        """Legacy method for backward compatibility - redirects to advanced method."""
        return self._calculate_advanced_structure_score(text)
    
    def _filter_and_classify_headings(self, candidates, thresholds):
        """Filter false positives and classify heading levels."""
        if not candidates:
            return []
        
        # Sort by page, then by position
        candidates.sort(key=lambda x: (x['page_num'], x.get('y_position', 0)))
        
        # Remove duplicates and very similar headings
        filtered = self._remove_duplicates(candidates)
        
        # Classify heading levels
        classified = self._classify_heading_levels(filtered, thresholds)
        
        return classified
    
    def _remove_duplicates(self, candidates):
        """Remove duplicate and very similar headings."""
        if not candidates:
            return []
        
        filtered = []
        seen_texts = set()
        
        for candidate in candidates:
            text = candidate['text'].strip()
            text_normalized = re.sub(r'\s+', ' ', text.lower())
            
            # Skip if we've seen this exact text
            if text_normalized in seen_texts:
                continue
            
            # Skip if very similar to an existing heading
            is_similar = False
            for seen in seen_texts:
                # Simple similarity check
                if abs(len(text_normalized) - len(seen)) < 3:
                    # Calculate simple similarity
                    common_words = set(text_normalized.split()) & set(seen.split())
                    total_words = set(text_normalized.split()) | set(seen.split())
                    if len(common_words) / len(total_words) > 0.8:
                        is_similar = True
                        break
            
            if not is_similar:
                filtered.append(candidate)
                seen_texts.add(text_normalized)
        
        return filtered
    
    def _classify_heading_levels(self, headings, thresholds):
        """Classify headings into H1, H2, H3 levels with improved accuracy."""
        if not headings:
            return []
        
        # Get font sizes and sort (largest first)
        sizes = [h['size'] for h in headings]
        unique_sizes = sorted(set(sizes), reverse=True)
        
        # Create more accurate size-to-level mapping
        size_to_level = {}
        
        # If we have multiple font sizes, use them for hierarchy
        if len(unique_sizes) >= 3:
            # Three or more sizes: map to H1, H2, H3
            size_to_level[unique_sizes[0]] = 'H1'
            size_to_level[unique_sizes[1]] = 'H2'
            # All remaining sizes become H3
            for size in unique_sizes[2:]:
                size_to_level[size] = 'H3'
        elif len(unique_sizes) == 2:
            # Two sizes: larger is H1, smaller is H2
            size_to_level[unique_sizes[0]] = 'H1'
            size_to_level[unique_sizes[1]] = 'H2'
        else:
            # Only one size: all become H2 (middle level)
            size_to_level[unique_sizes[0]] = 'H2'
        
        # Apply levels with pattern-based overrides
        classified = []
        for heading in headings:
            text = heading['text']
            base_level = size_to_level.get(heading['size'], 'H3')
            
            # Strong pattern-based overrides
            if re.match(r'^\d+\.\s+[A-Z]', text):  # "1. Something" -> H1
                level = 'H1'
            elif re.match(r'^\d+\.\d+\s+[A-Z]', text):  # "1.1 Something" -> H2
                level = 'H2'
            elif re.match(r'^\d+\.\d+\.\d+\s+[A-Z]', text):  # "1.1.1 Something" -> H3
                level = 'H3'
            elif re.match(r'^(chapter|section|part)\s+\d+', text.lower()):
                level = 'H1'
            elif re.match(r'^appendix\s+[a-z]', text.lower()):
                level = 'H1'
            else:
                # For non-numbered headings, trust the font size mapping more
                level = base_level
                
                # But apply some logic for all-caps vs title case
                if text.isupper() and len(text) > 10:
                    # All caps often indicates higher level
                    if base_level == 'H3':
                        level = 'H2'
                    elif base_level == 'H2':
                        level = 'H1'
            
            classified.append({
                **heading,
                'level': level
            })
        
        return classified

    def _is_fragmentary(self, text):
        """Check if text appears to be a fragment rather than a complete heading."""
        # Very short incomplete words
        if len(text) < 4 and not text.isupper():
            return True
        
        # Fragments that end abruptly (single letters, incomplete words)
        if re.match(r'^[A-Za-z]{1,2}$', text):
            return True
        
        # Text that looks cut off
        if text.endswith((' ', '-')) or text.startswith((' ', '-')):
            return True
        
        # Mathematical symbols or single special characters
        if re.match(r'^[\+\-\*\/\=\.\,\;\:]+$', text):
            return True
        
        # Single numbers (unless part of a numbering scheme)
        if text.isdigit() and len(text) < 3:
            return True
        
        return False

def main():
    """Command line interface."""
    if len(sys.argv) != 2:
        print("Usage: python improved_extractor.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    extractor = ImprovedPDFExtractor()
    result = extractor.extract_outline_json(pdf_path)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

"""
Product matching module: finds target products in extracted OCR text.
Supports exact matching, fuzzy matching, and Arabic variants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    from difflib import SequenceMatcher
except ImportError:
    SequenceMatcher = None

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


@dataclass
class MatchResult:
    """Result of a product match."""
    product_name: str
    confidence: float
    matched_text: str | None = None
    match_type: str = "unknown"  # exact, fuzzy, arabic


def _normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text by removing diacritics and standardizing characters.
    Maps common variations to canonical forms.
    """
    if not text:
        return ""
    
    # Common Arabic character normalizations
    normalizations = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alef variants → Alef
        'ى': 'ي',  # Alef maksura → Ya
        'ة': 'ه',  # Tah marbuta → Ha
        'و': 'و',  # Waw consistency
    }
    
    result = text
    for source, target in normalizations.items():
        result = result.replace(source, target)
    
    # Remove common diacritics
    diacritics = ['\u064B', '\u064C', '\u064D', '\u064E', '\u064F', 
                  '\u0650', '\u0651', '\u0652', '\u0653', '\u0654']
    for diacritic in diacritics:
        result = result.replace(diacritic, '')
    
    return result


def _to_ascii_approximate(text: str) -> str:
    """Convert text to ASCII-like representation for fuzzy matching."""
    if unidecode:
        return unidecode(text)
    # Fallback: just lowercase
    return text.lower()


def _fuzzy_similarity(str1: str, str2: str) -> float:
    """
    Calculate fuzzy similarity between two strings (0.0 to 1.0).
    Returns ratio of matching characters.
    """
    if not SequenceMatcher:
        # Fallback: exact match only
        return 1.0 if str1.lower() == str2.lower() else 0.0
    
    str1_lower = str1.lower()
    str2_lower = str2.lower()
    
    if str1_lower == str2_lower:
        return 1.0
    
    matcher = SequenceMatcher(None, str1_lower, str2_lower)
    return matcher.ratio()


def _extract_product_names_from_text(text: str) -> list[str]:
    """Extract potential product names from OCR text (lines, cells, etc)."""
    if isinstance(text, list):
        products = []
        for item in text:
            if isinstance(item, str):
                products.extend(_extract_product_names_from_text(item))
            elif isinstance(item, dict):
                for v in item.values():
                    products.extend(_extract_product_names_from_text(v))
        return products
    
    if not isinstance(text, str):
        return []
    
    # Split by common delimiters and filter empty strings
    lines = re.split(r'[\n\r|,;]', text)
    products = []
    
    for line in lines:
        line = line.strip()
        # Filter: product name should be 2-100 chars, avoid pure numbers
        if 2 < len(line) < 100 and not line.replace(' ', '').isdigit():
            products.append(line)
    
    return products


def _collect_all_text_values(data: Any, max_depth: int = 10) -> list[str]:
    """Recursively collect all text values from nested OCR structures."""
    if max_depth <= 0:
        return []
    
    if isinstance(data, str):
        return [data] if data.strip() else []
    
    if isinstance(data, list):
        result = []
        for item in data:
            result.extend(_collect_all_text_values(item, max_depth - 1))
        return result
    
    if isinstance(data, dict):
        result = []
        for v in data.values():
            result.extend(_collect_all_text_values(v, max_depth - 1))
        return result
    
    return []


def match_products(
    target_items: list[dict[str, str]] | None,
    extracted_data: dict[str, Any] | None,
    threshold_fuzzy: float = 0.75,
    threshold_arabic: float = 0.80,
) -> list[MatchResult]:
    """
    Match target products against extracted OCR data.
    
    Args:
        target_items: List of {"product_name": str, "strength": str|None, ...}
        extracted_data: OCR extraction result (tables, text, etc.)
        threshold_fuzzy: Confidence threshold for fuzzy matching (0.0-1.0)
        threshold_arabic: Confidence threshold for Arabic matching (0.0-1.0)
    
    Returns:
        List of MatchResult objects with confidence scores.
    """
    if not target_items or not extracted_data:
        return []
    
    # Collect all text from extracted data
    all_text = _collect_all_text_values(extracted_data)
    extracted_products = _extract_product_names_from_text('\n'.join(all_text))
    
    results = []
    seen = set()
    
    for target in target_items:
        target_name = target.get("product_name", "").strip()
        target_strength = target.get("strength", "").strip()
        
        if not target_name:
            continue
        
        # Build search query
        search_query = target_name
        if target_strength:
            search_query = f"{target_name} {target_strength}"
        
        best_match: MatchResult | None = None
        best_confidence = 0.0
        
        for extracted in extracted_products:
            if not extracted or extracted.lower() in seen:
                continue
            
            # 1. Exact match (case-insensitive)
            if extracted.lower() == target_name.lower():
                confidence = 1.0
                match_result = MatchResult(
                    product_name=target_name,
                    confidence=confidence,
                    matched_text=extracted,
                    match_type="exact",
                )
                results.append(match_result)
                seen.add(extracted.lower())
                best_match = None  # Found exact, stop searching
                break
            
            # 2. Try Arabic normalization
            arabic_extracted = _normalize_arabic_text(extracted)
            arabic_target = _normalize_arabic_text(target_name)
            
            if arabic_target and arabic_extracted.lower() == arabic_target.lower():
                confidence = threshold_arabic
                if confidence > best_confidence:
                    best_match = MatchResult(
                        product_name=target_name,
                        confidence=confidence,
                        matched_text=extracted,
                        match_type="arabic",
                    )
                    best_confidence = confidence
                continue
            
            # 3. Fuzzy matching (handles typos, partial matches)
            similarity = _fuzzy_similarity(extracted, target_name)
            
            if similarity >= threshold_fuzzy:
                if similarity > best_confidence:
                    best_match = MatchResult(
                        product_name=target_name,
                        confidence=similarity,
                        matched_text=extracted,
                        match_type="fuzzy",
                    )
                    best_confidence = similarity
        
        # Add best fuzzy/arabic match if found
        if best_match and best_match not in results:
            results.append(best_match)
            seen.add(best_match.matched_text.lower())
    
    return results


def format_matches_for_callback(
    matches: list[MatchResult],
    extracted_data: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    """
    Format match results for callback to backend.
    
    Returns list of items with: drug_name, company, price, confidence, review_required
    """
    items = []
    
    for match in matches:
        # Try to extract price from extracted_data if available
        price = 0.0
        company = ""
        
        # TODO: Enhance to extract actual prices from tables if present
        # This would require parsing table structure from extracted_data
        
        item = {
            "drug_name": match.product_name,
            "company": company,
            "price": price,
            "confidence": match.confidence,
            "review_required": match.confidence < 0.9,  # Flag low-confidence matches for review
            "match_type": match.match_type,
            "matched_text": match.matched_text,
        }
        items.append(item)
    
    return items

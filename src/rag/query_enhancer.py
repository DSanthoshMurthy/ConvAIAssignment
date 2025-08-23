#!/usr/bin/env python3
"""
Query Enhancement System
Spell correction and query preprocessing for better financial queries
"""

import logging
from typing import List, Dict, Any, Tuple
import re
from difflib import SequenceMatcher, get_close_matches

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialQueryEnhancer:
    def __init__(self):
        """Initialize the Financial Query Enhancer."""
        
        # Financial terms dictionary with common variations and misspellings
        # Note: Removed substring matches to prevent false corrections (e.g., 'revenu' in 'revenue')
        self.financial_terms_corrections = {
            # Revenue variations (only clear misspellings)
            'revunue': 'revenue',
            'reveune': 'revenue',
            'revanue': 'revenue',
            'revinue': 'revenue',
            # Removed 'revenu': 'revenue' - substring of correct word
            'sales': 'revenue',
            'income': 'revenue',
            'turnover': 'revenue',
            
            # Profit variations (only clear misspellings)
            'proffit': 'profit',
            'prfit': 'profit',
            'profi': 'profit',
            'earnings': 'profit',
            'net income': 'profit',
            'bottom line': 'profit',
            
            # Expense variations (only clear misspellings)
            'expence': 'expense',
            'expens': 'expense',
            'expanse': 'expense',
            # Keep 'expenses': 'expense' for singular standardization
            'expenses': 'expense',
            'costs': 'expense',
            'expenditure': 'expense',
            'outgoings': 'expense',
            
            # Asset variations (only clear misspellings)
            'assests': 'assets',
            'aseets': 'assets', 
            # Removed 'asset': 'assets' to avoid singular/plural conflicts
            'holdings': 'assets',
            
            # Liability variations (only clear misspellings)
            'liabilties': 'liabilities',
            'liabilites': 'liabilities',
            # Keep 'liability': 'liabilities' as it's a standardization, not a substring issue
            'liability': 'liabilities',
            'debts': 'liabilities',
            'obligations': 'liabilities',
            
            # Cash flow variations
            'cashflow': 'cash flow',
            'cash-flow': 'cash flow',
            'cash flows': 'cash flow',
            
            # Quarter variations
            'q1': 'Q1',
            'q2': 'Q2', 
            'q3': 'Q3',
            'q4': 'Q4',
            'quarter 1': 'Q1',
            'quarter 2': 'Q2',
            'quarter 3': 'Q3',
            'quarter 4': 'Q4',
            'first quarter': 'Q1',
            'second quarter': 'Q2',
            'third quarter': 'Q3',
            'fourth quarter': 'Q4',
        }
        
        # Standard financial terms for validation
        self.standard_financial_terms = [
            'revenue', 'profit', 'loss', 'earnings', 'expense', 'cost',
            'assets', 'liabilities', 'equity', 'debt', 'cash flow',
            'margin', 'ratio', 'growth', 'decline', 'increase', 'decrease',
            'quarter', 'annual', 'monthly', 'financial year', 'fiscal',
            'balance sheet', 'income statement', 'P&L', 'EBITDA'
        ]
        
        # Date period mappings
        self.date_period_mappings = {
            '2023-04-01 to 2024-03-31': 'FY 2023-24',
            '2022-04-01 to 2023-03-31': 'FY 2022-23', 
            '2021-04-01 to 2022-03-31': 'FY 2021-22',
            
            # Quarterly mappings
            '2023-04-01 to 2023-06-30': 'Q1 2023-24',
            '2023-07-01 to 2023-09-30': 'Q2 2023-24',
            '2023-10-01 to 2023-12-31': 'Q3 2023-24',
            '2024-01-01 to 2024-03-31': 'Q4 2023-24',
        }
        
        # Statistics
        self.enhancement_stats = {
            'queries_processed': 0,
            'corrections_made': 0,
            'date_conversions': 0,
            'term_enhancements': 0
        }
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """Enhance query with spell correction and preprocessing.
        
        Args:
            query: Original user query
            
        Returns:
            Dictionary with enhanced query and metadata
        """
        self.enhancement_stats['queries_processed'] += 1
        
        enhancement_metadata = {
            'original_query': query,
            'corrections_applied': [],
            'enhancements_made': [],
            'confidence_boost': 0
        }
        
        enhanced_query = query.lower().strip()
        
        # Step 1: Spell correction for financial terms
        corrected_query, corrections = self.apply_spell_corrections(enhanced_query)
        if corrections:
            enhancement_metadata['corrections_applied'] = corrections
            self.enhancement_stats['corrections_made'] += len(corrections)
        
        # Step 2: Date period conversion
        final_query, date_conversions = self.convert_date_periods(corrected_query)
        if date_conversions:
            enhancement_metadata['enhancements_made'].extend(date_conversions)
            self.enhancement_stats['date_conversions'] += len(date_conversions)
        
        # Step 3: Term standardization
        final_query, term_enhancements = self.standardize_financial_terms(final_query)
        if term_enhancements:
            enhancement_metadata['enhancements_made'].extend(term_enhancements)
            self.enhancement_stats['term_enhancements'] += len(term_enhancements)
        
        # Calculate confidence boost
        total_improvements = len(corrections) + len(date_conversions) + len(term_enhancements)
        enhancement_metadata['confidence_boost'] = min(total_improvements * 0.1, 0.3)  # Max 30% boost
        
        enhancement_metadata['enhanced_query'] = final_query
        enhancement_metadata['improvement_count'] = total_improvements
        
        if total_improvements > 0:
            logger.info(f"✅ Query enhanced: '{query}' → '{final_query}' ({total_improvements} improvements)")
        
        return enhancement_metadata
    
    def apply_spell_corrections(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """Apply spell corrections to financial terms using word boundaries.
        
        Args:
            query: Query to correct
            
        Returns:
            Tuple of (corrected_query, list_of_corrections)
        """
        corrected_query = query
        corrections = []
        
        # Direct mapping corrections with word boundaries
        for incorrect, correct in self.financial_terms_corrections.items():
            # Use word boundaries to avoid substring matches
            pattern = r'\b' + re.escape(incorrect) + r'\b'
            if re.search(pattern, corrected_query, re.IGNORECASE):
                # Only correct if it's not already the correct word
                if incorrect.lower() != correct.lower():
                    corrected_query = re.sub(pattern, correct, corrected_query, flags=re.IGNORECASE)
                    corrections.append({
                        'original': incorrect,
                        'corrected': correct,
                        'type': 'direct_mapping'
                    })
        
        # Fuzzy matching for remaining terms (more conservative)
        words = corrected_query.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Skip if word is already in standard terms (case insensitive)
            if any(word_lower == term.lower() for term in self.standard_financial_terms):
                continue
                
            # Skip if word is too short or looks like a proper word
            if len(word) < 4 or word.isdigit():
                continue
                
            # Find close matches with higher cutoff to avoid false positives
            close_matches = get_close_matches(
                word_lower, [term.lower() for term in self.standard_financial_terms], 
                n=1, cutoff=0.85  # Increased from 0.7 to 0.85 for fewer false positives
            )
            
            if close_matches:
                best_match_lower = close_matches[0]
                # Find the original case version
                best_match = next(term for term in self.standard_financial_terms if term.lower() == best_match_lower)
                
                # Only apply if the similarity is significant and words are clearly different
                similarity = SequenceMatcher(None, word_lower, best_match_lower).ratio()
                if similarity < 0.95:  # Only correct if words are significantly different
                    words[i] = best_match
                    corrections.append({
                        'original': word,
                        'corrected': best_match,
                        'type': 'fuzzy_matching',
                        'similarity': round(similarity, 2)
                    })
        
        corrected_query = ' '.join(words)
        return corrected_query, corrections
    
    def convert_date_periods(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """Convert date ranges to financial periods.
        
        Args:
            query: Query with potential date ranges
            
        Returns:
            Tuple of (converted_query, list_of_conversions)
        """
        converted_query = query
        conversions = []
        
        for date_range, period in self.date_period_mappings.items():
            if date_range.lower() in converted_query:
                converted_query = converted_query.replace(date_range.lower(), period)
                conversions.append({
                    'original': date_range,
                    'converted': period,
                    'type': 'date_period_mapping'
                })
        
        return converted_query, conversions
    
    def standardize_financial_terms(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """Standardize financial terminology.
        
        Args:
            query: Query to standardize
            
        Returns:
            Tuple of (standardized_query, list_of_enhancements)
        """
        standardized_query = query
        enhancements = []
        
        # Common term standardizations
        standardizations = {
            'operations': 'operations',
            'from operations': 'from operations',
            'net income': 'profit',
            'net profit': 'profit', 
            'gross profit': 'profit',
            'operating income': 'revenue',
            'total income': 'revenue',
            'total revenue': 'revenue'
        }
        
        for term, standard in standardizations.items():
            if term in standardized_query and term != standard:
                standardized_query = standardized_query.replace(term, standard)
                enhancements.append({
                    'original': term,
                    'standardized': standard,
                    'type': 'term_standardization'
                })
        
        return standardized_query, enhancements
    
    def get_enhancement_suggestions(self, query: str) -> List[str]:
        """Get query enhancement suggestions for users.
        
        Args:
            query: Original query
            
        Returns:
            List of suggested improvements
        """
        suggestions = []
        lower_query = query.lower()
        
        # Check for common issues
        if any(term in lower_query for term in ['revunue', 'reveune', 'revanue']):
            suggestions.append("Try spelling 'revenue' correctly for better results")
        
        if 'profit' in lower_query and 'tax' not in lower_query:
            suggestions.append("Specify 'profit before tax' or 'profit after tax' for precision")
        
        if any(date in lower_query for date in ['2023-04-01', '2024-03-31']):
            suggestions.append("Try using 'FY 2023-24' or 'financial year 2023-24' instead of date ranges")
        
        if len(query.split()) < 5:
            suggestions.append("Add more context like time period (Q1 2023, Dec 2023) for better results")
        
        return suggestions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhancement statistics."""
        return self.enhancement_stats.copy()

def main():
    """Test the query enhancement system."""
    print("="*80)
    print("FINANCIAL QUERY ENHANCEMENT SYSTEM")
    print("Spell Correction and Query Preprocessing")
    print("="*80)
    
    # Initialize enhancer
    enhancer = FinancialQueryEnhancer()
    
    # Test queries with common issues
    test_queries = [
        "What is the revunue from 2023-04-01 to 2024-03-31",  # Typo + date range
        "What was the proffit in Q3 2023?",  # Typo
        "Tell me about expences for last quarter",  # Typo
        "What are the assests of the company?",  # Typo
        "Show me cashflow data",  # Compound word
        "What was the net income in quarter 1?",  # Term standardization
    ]
    
    print(f"\n🧪 Testing Query Enhancement")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 60)
        
        # Enhance query
        result = enhancer.enhance_query(query)
        
        print(f"🔧 Enhanced Query: {result['enhanced_query']}")
        print(f"📊 Improvements: {result['improvement_count']}")
        print(f"🎯 Confidence Boost: +{result['confidence_boost']:.1%}")
        
        if result['corrections_applied']:
            print(f"✏️  Corrections:")
            for correction in result['corrections_applied']:
                print(f"   • '{correction['original']}' → '{correction['corrected']}' ({correction['type']})")
        
        if result['enhancements_made']:
            print(f"⚡ Enhancements:")
            for enhancement in result['enhancements_made']:
                if enhancement['type'] == 'date_period_mapping':
                    print(f"   • '{enhancement['original']}' → '{enhancement['converted']}' (date conversion)")
                else:
                    print(f"   • '{enhancement['original']}' → '{enhancement['standardized']}' (standardization)")
        
        # Get suggestions
        suggestions = enhancer.get_enhancement_suggestions(query)
        if suggestions:
            print(f"💡 Suggestions:")
            for suggestion in suggestions:
                print(f"   • {suggestion}")
    
    # Show statistics
    stats = enhancer.get_stats()
    print(f"\n📈 Enhancement Statistics:")
    print(f"   Queries processed: {stats['queries_processed']}")
    print(f"   Corrections made: {stats['corrections_made']}")
    print(f"   Date conversions: {stats['date_conversions']}")
    print(f"   Term enhancements: {stats['term_enhancements']}")
    
    print(f"\n🎉 Query enhancement system test completed!")

if __name__ == "__main__":
    main()

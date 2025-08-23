#!/usr/bin/env python3
"""
Financial Period Mapping System
Correct mapping between financial years, quarters, and calendar periods
"""

import logging
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialPeriodMapper:
    def __init__(self):
        """Initialize the Financial Period Mapper."""
        
        # Indian Financial Year mappings (April to March)
        self.financial_year_mappings = {
            # FY 2023-24 (April 1, 2023 to March 31, 2024)
            'FY2023-24': {
                'Q1': 'Apr-Jun 2023',  # Q1: April-June 2023
                'Q2': 'Jul-Sep 2023',  # Q2: July-September 2023  
                'Q3': 'Oct-Dec 2023',  # Q3: October-December 2023
                'Q4': 'Jan-Mar 2024'   # Q4: January-March 2024
            },
            'FY 2023-24': {
                'Q1': 'Apr-Jun 2023',
                'Q2': 'Jul-Sep 2023', 
                'Q3': 'Oct-Dec 2023',
                'Q4': 'Jan-Mar 2024'
            },
            
            # FY 2022-23 (April 1, 2022 to March 31, 2023)
            'FY2022-23': {
                'Q1': 'Apr-Jun 2022',
                'Q2': 'Jul-Sep 2022',
                'Q3': 'Oct-Dec 2022', 
                'Q4': 'Jan-Mar 2023'
            },
            'FY 2022-23': {
                'Q1': 'Apr-Jun 2022',
                'Q2': 'Jul-Sep 2022',
                'Q3': 'Oct-Dec 2022',
                'Q4': 'Jan-Mar 2023'
            }
        }
        
        # Calendar period to quarter mapping
        self.calendar_to_quarter = {
            # 2023 periods
            'Jun 2023': ('FY2023-24', 'Q1'),  # June is end of Q1 FY2023-24
            'Sep 2023': ('FY2023-24', 'Q2'),  # September is end of Q2 FY2023-24
            'Dec 2023': ('FY2023-24', 'Q3'),  # December is end of Q3 FY2023-24
            'Mar 2024': ('FY2023-24', 'Q4'),  # March is end of Q4 FY2023-24
            
            # 2022 periods
            'Jun 2022': ('FY2022-23', 'Q1'),  # June is end of Q1 FY2022-23
            'Sep 2022': ('FY2022-23', 'Q2'),  # September is end of Q2 FY2022-23
            'Dec 2022': ('FY2022-23', 'Q3'),  # December is end of Q3 FY2022-23
            'Mar 2023': ('FY2022-23', 'Q4'),  # March is end of Q4 FY2022-23
        }
        
        # Reverse mapping: quarter to calendar periods
        self.quarter_to_calendar = {}
        for period, (fy, quarter) in self.calendar_to_quarter.items():
            if fy not in self.quarter_to_calendar:
                self.quarter_to_calendar[fy] = {}
            self.quarter_to_calendar[fy][quarter] = period
        
        # Currency formatting patterns
        self.currency_patterns = [
            (r'₹\s*(\d+)\s*crores?', r'₹\1 crore'),
            (r'₹\s*(\d+\.?\d*)\s*billions?', r'₹\1 billion'),  
            (r'₹\s*-\s*(\d+\.?\d*)', r'₹-\1'),  # Fix negative spacing
            (r'₹\s+(\d)', r'₹\1'),              # Remove extra spaces after ₹
            (r'(\d+)\s*\.\s*(\d+)', r'\1.\2'),  # Fix decimal spacing
        ]
    
    def map_query_period(self, query: str) -> Tuple[str, Optional[Dict[str, str]]]:
        """Map query period references to correct financial periods.
        
        Args:
            query: Query string with period references
            
        Returns:
            Tuple of (mapped_query, mapping_info)
        """
        query_lower = query.lower()
        mapping_info = None
        mapped_query = query
        
        # Look for Q2 FY2023-24 pattern
        fy_quarter_pattern = r'q(\d+)\s+fy(\d{4})-?(\d{2,4})'
        match = re.search(fy_quarter_pattern, query_lower)
        
        if match:
            quarter_num = match.group(1)
            year1 = match.group(2)
            year2 = match.group(3)
            
            # Normalize year2
            if len(year2) == 2:
                year2 = '20' + year2
            
            quarter = f'Q{quarter_num}'
            fy = f'FY{year1}-{year2[-2:]}'
            
            # Find the corresponding calendar period
            if fy in self.quarter_to_calendar and quarter in self.quarter_to_calendar[fy]:
                calendar_period = self.quarter_to_calendar[fy][quarter]
                
                mapping_info = {
                    'original_period': f'Q{quarter_num} FY{year1}-{year2[-2:]}',
                    'financial_year': fy,
                    'quarter': quarter,
                    'calendar_period': calendar_period,
                    'mapping_type': 'quarter_to_calendar'
                }
                
                # Replace in query
                original = match.group(0)
                mapped_query = query.replace(original, calendar_period, 1)
                
                logger.info(f"✅ Period mapped: '{original}' → '{calendar_period}'")
        
        return mapped_query, mapping_info
    
    def fix_currency_formatting(self, text: str) -> str:
        """Fix currency formatting issues in text.
        
        Args:
            text: Text with potential currency formatting issues
            
        Returns:
            Text with corrected currency formatting
        """
        corrected_text = text
        
        for pattern, replacement in self.currency_patterns:
            corrected_text = re.sub(pattern, replacement, corrected_text)
        
        # Additional fixes for common issues
        corrected_text = re.sub(r'₹\s*-\s*', '₹-', corrected_text)  # Fix negative currency
        corrected_text = re.sub(r'(\d+)\s*\.\s*(\d+)\s*crores?', r'₹\1.\2 crore', corrected_text)
        corrected_text = re.sub(r'₹\s*(\d+)\s*\.\s*(\d+)', r'₹\1.\2', corrected_text)
        
        return corrected_text
    
    def get_correct_period_data(self, query: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        """Filter retrieved chunks to show correct period data.
        
        Args:
            query: Original query
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Filtered chunks with correct period data
        """
        mapped_query, mapping_info = self.map_query_period(query)
        
        if not mapping_info:
            return retrieved_chunks
        
        target_calendar_period = mapping_info['calendar_period']
        corrected_chunks = []
        
        for chunk in retrieved_chunks:
            chunk_text = chunk.get('text', '')
            chunk_quarter = chunk.get('metadata', {}).get('quarter', '')
            
            # Check if chunk matches the target period
            if target_calendar_period in chunk_quarter or target_calendar_period in chunk_text:
                # Fix currency formatting in the chunk
                chunk_copy = chunk.copy()
                chunk_copy['text'] = self.fix_currency_formatting(chunk_text)
                corrected_chunks.append(chunk_copy)
            # Also include if no specific period mentioned (general data)
            elif not any(period in chunk_text for period in ['Jun', 'Sep', 'Dec', 'Mar', 'Q1', 'Q2', 'Q3', 'Q4']):
                chunk_copy = chunk.copy()
                chunk_copy['text'] = self.fix_currency_formatting(chunk_text)
                corrected_chunks.append(chunk_copy)
        
        # If no period-specific data found, return original with currency fixes
        if not corrected_chunks:
            for chunk in retrieved_chunks:
                chunk_copy = chunk.copy()
                chunk_copy['text'] = self.fix_currency_formatting(chunk.get('text', ''))
                corrected_chunks.append(chunk_copy)
        
        return corrected_chunks[:len(retrieved_chunks)]  # Maintain original count
    
    def validate_financial_response(self, response: str, query: str) -> Tuple[str, Dict[str, str]]:
        """Validate and correct financial response.
        
        Args:
            response: Generated response
            query: Original query
            
        Returns:
            Tuple of (corrected_response, validation_info)
        """
        validation_info = {
            'currency_fixes': [],
            'period_corrections': [],
            'validation_status': 'passed'
        }
        
        corrected_response = response
        
        # Fix currency formatting
        original_currency = corrected_response
        corrected_response = self.fix_currency_formatting(corrected_response)
        
        if original_currency != corrected_response:
            validation_info['currency_fixes'].append({
                'original': original_currency,
                'corrected': corrected_response
            })
        
        # Check for period mismatches
        mapped_query, mapping_info = self.map_query_period(query)
        
        if mapping_info:
            expected_period = mapping_info['calendar_period']
            if expected_period not in corrected_response:
                validation_info['period_corrections'].append({
                    'expected_period': expected_period,
                    'query_period': mapping_info['original_period']
                })
                validation_info['validation_status'] = 'period_mismatch'
        
        return corrected_response, validation_info

def main():
    """Test the financial period mapping system."""
    print("="*80)
    print("FINANCIAL PERIOD MAPPING SYSTEM")
    print("Quarter Mapping and Currency Formatting")
    print("="*80)
    
    # Initialize mapper
    mapper = FinancialPeriodMapper()
    
    # Test period mappings
    test_queries = [
        "What was the revenue from operations in Q2 FY2023-24?",
        "Show me profit data for Q1 FY 2023-24",
        "Tell me about Q3 FY2022-23 performance",
        "What is the revenue in last financial year"
    ]
    
    print(f"\n🧪 Testing Period Mappings")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 60)
        
        mapped_query, mapping_info = mapper.map_query_period(query)
        
        print(f"🔧 Mapped Query: {mapped_query}")
        
        if mapping_info:
            print(f"📊 Mapping Details:")
            print(f"   Original Period: {mapping_info['original_period']}")
            print(f"   Financial Year: {mapping_info['financial_year']}")
            print(f"   Quarter: {mapping_info['quarter']}")
            print(f"   Calendar Period: {mapping_info['calendar_period']}")
        else:
            print(f"📊 No period mapping needed")
    
    # Test currency formatting
    print(f"\n🧪 Testing Currency Formatting")
    
    test_currencies = [
        "₹ - 42. 89 crores",
        "₹  33 crores",
        "₹18 . 40 billion",
        "₹ 14.29 billion",
        "₹-97. 01 crores"
    ]
    
    for currency in test_currencies:
        fixed = mapper.fix_currency_formatting(currency)
        print(f"'{currency}' → '{fixed}'")
    
    # Show correct mappings
    print(f"\n📅 Financial Year Quarter Mappings:")
    for fy, quarters in mapper.financial_year_mappings.items():
        if 'FY2023-24' in fy:
            print(f"\n{fy}:")
            for quarter, period in quarters.items():
                print(f"   {quarter}: {period}")
    
    print(f"\n🎉 Financial period mapping test completed!")

if __name__ == "__main__":
    main()

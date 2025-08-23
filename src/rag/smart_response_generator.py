#!/usr/bin/env python3
"""
Smart Financial Response Generation System
Enhanced Phase 5: Template-based response generation with financial context integration.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartFinancialResponseGenerator:
    def __init__(self):
        """Initialize the Smart Financial Response Generator."""
        
        # Response templates for different types of financial queries
        self.response_templates = {
            'revenue': {
                'single': "The revenue from operations was {amount} in {period}.",
                'comparison': "The revenue from operations was {amount} in {period}, showing a {change} compared to {comparison_period}.",
                'trend': "Revenue from operations has shown {trend} with {amount} in {period}.",
                'multiple': "Revenue figures across periods: {revenue_list}."
            },
            'profit': {
                'single': "The {profit_type} was {amount} in {period}.",
                'comparison': "The {profit_type} was {amount} in {period}, representing a {change} from {comparison_period}.",
                'loss': "The company reported a loss before tax of {amount} in {period}.",
                'trend': "{profit_type} performance shows {trend} with {amount} in {period}."
            },
            'expenses': {
                'single': "The {expense_type} was {amount} in {period}.",
                'breakdown': "Major expense categories in {period}: {expense_breakdown}.",
                'comparison': "{expense_type} was {amount} in {period}, {change} from the previous period.",
                'total': "Total expenses amounted to {amount} in {period}."
            },
            'assets_liabilities': {
                'single': "Total {item_type} stood at {amount} as of {period}.",
                'components': "{item_type} breakdown includes: {component_list}.",
                'ratio': "The {ratio_name} ratio is {ratio_value}, indicating {interpretation}."
            },
            'general': {
                'financial_summary': "Based on the financial data for {period}, {summary}.",
                'multi_metric': "Key financial metrics for {period}: {metrics_list}.",
                'no_data': "I don't have specific information about {query_topic} in the available financial data.",
                'clarification': "Could you please specify which time period or financial metric you're interested in?"
            }
        }
        
        # Financial patterns and extractors
        self.amount_pattern = r'₹([\d,]+\.?\d*)\s*(crore|billion|million|thousand|lakh)?'
        self.period_pattern = r'(Q[1-4]\s+\d{4}|[A-Z][a-z]+\s+\d{4}|\d{4})'
        
        # Financial keywords for query classification
        self.financial_keywords = {
            'revenue': ['revenue', 'sales', 'income from operations', 'operating income', 'turnover'],
            'profit': ['profit', 'earnings', 'net income', 'profit before tax', 'profit after tax', 'EBITDA'],
            'expenses': ['expenses', 'costs', 'expenditure', 'employee benefit', 'depreciation', 'interest'],
            'assets': ['assets', 'current assets', 'non-current assets', 'total assets'],
            'liabilities': ['liabilities', 'current liabilities', 'debt', 'borrowings'],
            'cash_flow': ['cash flow', 'operating cash flow', 'free cash flow'],
            'ratios': ['ratio', 'margin', 'ROE', 'ROA', 'debt-equity']
        }
        
        # Response quality tracking
        self.generation_stats = {
            'total_responses': 0,
            'template_usage': {},
            'avg_confidence': 0.0,
            'response_types': {}
        }
    
    def classify_query(self, query: str) -> Tuple[str, List[str]]:
        """Classify the financial query to determine response strategy.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (primary_category, matched_keywords)
        """
        query_lower = query.lower()
        category_scores = {}
        matched_keywords = []
        
        for category, keywords in self.financial_keywords.items():
            score = 0
            category_matches = []
            
            for keyword in keywords:
                if keyword in query_lower:
                    score += len(keyword.split())  # Multi-word keywords get higher scores
                    category_matches.append(keyword)
            
            if score > 0:
                category_scores[category] = score
                matched_keywords.extend(category_matches)
        
        # Determine primary category
        if category_scores:
            primary_category = max(category_scores, key=category_scores.get)
        else:
            primary_category = 'general'
        
        return primary_category, matched_keywords
    
    def extract_financial_data(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract structured financial data from retrieved chunks.
        
        Args:
            chunks: Retrieved document chunks
            
        Returns:
            Structured financial data
        """
        extracted_data = {
            'amounts': [],
            'periods': [],
            'companies': [],
            'metrics': [],
            'qa_pairs': []
        }
        
        for chunk in chunks:
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Extract amounts
            amounts = re.findall(self.amount_pattern, text, re.IGNORECASE)
            for amount_match in amounts:
                amount_str = amount_match[0]
                unit = amount_match[1] if amount_match[1] else ''
                extracted_data['amounts'].append({
                    'value': amount_str,
                    'unit': unit,
                    'context': text,
                    'period': metadata.get('quarter', 'Unknown'),
                    'section': metadata.get('section', 'Unknown')
                })
            
            # Extract periods
            periods = re.findall(self.period_pattern, text)
            extracted_data['periods'].extend(periods)
            
            # Check if it's a Q&A pair
            if 'question:' in text.lower() and 'answer:' in text.lower():
                qa_match = re.search(r'question:\s*(.+?)\s*answer:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
                if qa_match:
                    extracted_data['qa_pairs'].append({
                        'question': qa_match.group(1).strip(),
                        'answer': qa_match.group(2).strip(),
                        'period': metadata.get('quarter', 'Unknown'),
                        'score': chunk.get('score', 0)
                    })
        
        return extracted_data
    
    def generate_contextual_response(self, 
                                   query: str, 
                                   chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a contextual response based on query classification and extracted data.
        
        Args:
            query: User query
            chunks: Retrieved document chunks
            
        Returns:
            Generated response with metadata
        """
        start_time = datetime.now()
        
        # Classify query
        primary_category, matched_keywords = self.classify_query(query)
        
        # Extract financial data
        extracted_data = self.extract_financial_data(chunks)
        
        # Check for direct Q&A matches first
        direct_answer = self.find_direct_qa_match(query, extracted_data['qa_pairs'])
        if direct_answer:
            response = self.create_qa_response(direct_answer, query)
            confidence = 0.95  # High confidence for direct matches
        else:
            # Generate template-based response
            response, confidence = self.generate_template_response(
                query, primary_category, extracted_data, chunks
            )
            
            # Cloud confidence enhancement: Boost confidence for good BM25-only results
            if confidence < 0.6:  # Only boost if initially low
                enhanced_confidence = self.enhance_cloud_confidence(
                    query, response, chunks, confidence, primary_category
                )
                if enhanced_confidence > confidence:
                    logger.info(f"💡 Cloud confidence enhanced: {confidence:.2f} → {enhanced_confidence:.2f}")
                    confidence = enhanced_confidence
        
        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Update statistics
        self.update_generation_stats(primary_category, confidence)
        
        response_data = {
            'answer': response,
            'confidence': confidence,
            'query_classification': {
                'primary_category': primary_category,
                'matched_keywords': matched_keywords
            },
            'data_sources': {
                'chunks_used': len(chunks),
                'qa_pairs_found': len(extracted_data['qa_pairs']),
                'financial_amounts': len(extracted_data['amounts']),
                'periods_mentioned': len(set(extracted_data['periods']))
            },
            'generation_metadata': {
                'generation_time': generation_time,
                'response_method': 'direct_qa' if direct_answer else 'template_based',
                'template_category': primary_category
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Response generated in {generation_time:.3f}s (confidence: {confidence:.2f}, method: {response_data['generation_metadata']['response_method']})")
        
        return response_data
    
    def find_direct_qa_match(self, query: str, qa_pairs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find direct Q&A matches for the query.
        
        Args:
            query: User query
            qa_pairs: List of extracted Q&A pairs
            
        Returns:
            Best matching Q&A pair or None
        """
        if not qa_pairs:
            return None
        
        query_words = set(query.lower().split())
        best_match = None
        best_score = 0
        
        for qa_pair in qa_pairs:
            question = qa_pair['question'].lower()
            question_words = set(question.split())
            
            # Calculate word overlap score
            overlap = len(query_words.intersection(question_words))
            total_words = len(query_words.union(question_words))
            similarity_score = overlap / total_words if total_words > 0 else 0
            
            # Boost score for exact financial terms
            for word in ['revenue', 'profit', 'expense', 'assets', 'liabilities']:
                if word in query.lower() and word in question:
                    similarity_score += 0.2
            
            # Consider retrieval score
            combined_score = similarity_score * 0.7 + (qa_pair['score'] / 10.0) * 0.3
            
            if combined_score > best_score and combined_score > 0.3:  # Threshold for acceptance
                best_score = combined_score
                best_match = qa_pair
        
        return best_match
    
    def create_qa_response(self, qa_pair: Dict[str, Any], original_query: str) -> str:
        """Create response from a direct Q&A match.
        
        Args:
            qa_pair: Matching Q&A pair
            original_query: Original user query
            
        Returns:
            Formatted response
        """
        answer = qa_pair['answer']
        period = qa_pair['period']
        
        # Clean up the answer
        answer = answer.strip()
        if not answer.endswith('.'):
            answer += '.'
        
        # Add context if helpful
        if period != 'Unknown' and period not in answer:
            answer += f" (Data from {period})"
        
        return answer
    
    def generate_template_response(self, 
                                 query: str, 
                                 category: str, 
                                 extracted_data: Dict[str, Any], 
                                 chunks: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Generate template-based response.
        
        Args:
            query: User query
            category: Query category
            extracted_data: Extracted financial data
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (response, confidence)
        """
        if category not in self.response_templates:
            category = 'general'
        
        templates = self.response_templates[category]
        
        # Select appropriate template based on available data
        if category == 'revenue':
            return self.generate_revenue_response(query, extracted_data, templates)
        elif category == 'profit':
            return self.generate_profit_response(query, extracted_data, templates)
        elif category == 'expenses':
            return self.generate_expenses_response(query, extracted_data, templates)
        elif category in ['assets', 'liabilities']:
            return self.generate_assets_liabilities_response(query, category, extracted_data, templates)
        else:
            return self.generate_general_response(query, extracted_data, chunks, templates)
    
    def generate_revenue_response(self, 
                                query: str, 
                                extracted_data: Dict[str, Any], 
                                templates: Dict[str, str]) -> Tuple[str, float]:
        """Generate revenue-specific response."""
        revenue_amounts = [amt for amt in extracted_data['amounts'] 
                          if 'revenue' in amt['context'].lower() or 'operations' in amt['context'].lower()]
        
        if revenue_amounts:
            # Use the first relevant amount
            amount_data = revenue_amounts[0]
            amount_str = f"₹{amount_data['value']}"
            if amount_data['unit']:
                amount_str += f" {amount_data['unit']}"
            
            period = amount_data['period']
            
            # Check for comparison context
            context_text = amount_data['context'].lower()
            if 'increase' in context_text or 'decrease' in context_text or '%' in context_text:
                # Extract change information
                change_match = re.search(r'(\d+%\s*(increase|decrease|growth|decline))', context_text)
                if change_match:
                    change = change_match.group(1)
                    response = templates['comparison'].format(
                        amount=amount_str, 
                        period=period, 
                        change=change,
                        comparison_period="the previous period"
                    )
                    return response, 0.85
            
            # Single amount response
            response = templates['single'].format(amount=amount_str, period=period)
            return response, 0.80
        
        return "I don't have specific revenue information in the available data.", 0.30
    
    def generate_profit_response(self, 
                               query: str, 
                               extracted_data: Dict[str, Any], 
                               templates: Dict[str, str]) -> Tuple[str, float]:
        """Generate profit-specific response."""
        profit_amounts = [amt for amt in extracted_data['amounts'] 
                         if any(term in amt['context'].lower() for term in ['profit', 'loss', 'earnings'])]
        
        if profit_amounts:
            amount_data = profit_amounts[0]
            amount_str = f"₹{amount_data['value']}"
            if amount_data['unit']:
                amount_str += f" {amount_data['unit']}"
            
            period = amount_data['period']
            context_text = amount_data['context'].lower()
            
            # Determine profit type
            if 'profit before tax' in context_text:
                profit_type = "profit before tax"
            elif 'loss before tax' in context_text:
                response = templates['loss'].format(amount=amount_str, period=period)
                return response, 0.85
            elif 'net profit' in context_text:
                profit_type = "net profit"
            else:
                profit_type = "profit"
            
            response = templates['single'].format(
                profit_type=profit_type,
                amount=amount_str, 
                period=period
            )
            return response, 0.80
        
        return "I don't have specific profit information in the available data.", 0.30
    
    def generate_expenses_response(self, 
                                 query: str, 
                                 extracted_data: Dict[str, Any], 
                                 templates: Dict[str, str]) -> Tuple[str, float]:
        """Generate expense-specific response."""
        expense_amounts = [amt for amt in extracted_data['amounts'] 
                          if any(term in amt['context'].lower() for term in ['expense', 'cost', 'benefit'])]
        
        if expense_amounts:
            amount_data = expense_amounts[0]
            amount_str = f"₹{amount_data['value']}"
            if amount_data['unit']:
                amount_str += f" {amount_data['unit']}"
            
            period = amount_data['period']
            context_text = amount_data['context'].lower()
            
            # Determine expense type
            if 'employee' in context_text:
                expense_type = "employee benefit expense"
            elif 'depreciation' in context_text:
                expense_type = "depreciation expense"
            elif 'interest' in context_text:
                expense_type = "interest expense"
            else:
                expense_type = "expense"
            
            response = templates['single'].format(
                expense_type=expense_type,
                amount=amount_str, 
                period=period
            )
            return response, 0.80
        
        return "I don't have specific expense information in the available data.", 0.30
    
    def generate_assets_liabilities_response(self, 
                                           query: str, 
                                           category: str,
                                           extracted_data: Dict[str, Any], 
                                           templates: Dict[str, str]) -> Tuple[str, float]:
        """Generate assets/liabilities response."""
        relevant_amounts = [amt for amt in extracted_data['amounts'] 
                           if category in amt['context'].lower()]
        
        if relevant_amounts:
            amount_data = relevant_amounts[0]
            amount_str = f"₹{amount_data['value']}"
            if amount_data['unit']:
                amount_str += f" {amount_data['unit']}"
            
            period = amount_data['period']
            item_type = f"total {category}"
            
            response = templates['single'].format(
                item_type=item_type,
                amount=amount_str, 
                period=period
            )
            return response, 0.75
        
        return f"I don't have specific {category} information in the available data.", 0.30
    
    def generate_general_response(self, 
                                query: str, 
                                extracted_data: Dict[str, Any], 
                                chunks: List[Dict[str, Any]],
                                templates: Dict[str, str]) -> Tuple[str, float]:
        """Generate general financial response."""
        if extracted_data['amounts']:
            # Provide a summary of available financial data
            periods = list(set(extracted_data['periods']))
            amounts = len(extracted_data['amounts'])
            
            if periods:
                period = periods[0] if len(periods) == 1 else f"{len(periods)} periods"
                response = templates['financial_summary'].format(
                    period=period,
                    summary=f"there are {amounts} financial figures available covering various metrics"
                )
                return response, 0.60
        
        return templates['no_data'].format(query_topic="this specific information"), 0.25
    
    def enhance_cloud_confidence(self, 
                               query: str, 
                               response: str, 
                               chunks: List[Dict[str, Any]], 
                               current_confidence: float,
                               query_category: str) -> float:
        """Enhance confidence for cloud deployments with degraded retrieval.
        
        This method boosts confidence when BM25-only retrieval still produces
        good results, compensating for missing ChromaDB/cross-encoder.
        
        Args:
            query: Original query
            response: Generated response  
            chunks: Retrieved chunks
            current_confidence: Current confidence score
            query_category: Query classification category
            
        Returns:
            Enhanced confidence score
        """
        try:
            boost_factors = []
            
            # Factor 1: Strong keyword match in response
            query_keywords = set(query.lower().split())
            response_keywords = set(response.lower().split())
            keyword_overlap = len(query_keywords.intersection(response_keywords))
            if keyword_overlap >= 2:
                boost_factors.append(0.2)  # +0.2 for good keyword coverage
            
            # Factor 2: Financial data presence (numbers, crores, etc.)
            financial_patterns = [
                r'₹[\d,.]+ crore',  # Currency amounts
                r'\d+\.?\d*%',      # Percentages  
                r'Q[1-4] \d{4}',    # Quarters
                r'FY \d{4}-?\d*'    # Financial years
            ]
            
            financial_data_count = 0
            for pattern in financial_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    financial_data_count += 1
            
            if financial_data_count >= 2:
                boost_factors.append(0.25)  # +0.25 for rich financial data
            elif financial_data_count >= 1:
                boost_factors.append(0.15)  # +0.15 for some financial data
            
            # Factor 3: Revenue-specific queries get higher confidence
            if query_category in ['revenue', 'financial_performance'] and 'revenue' in response.lower():
                boost_factors.append(0.2)  # +0.2 for relevant revenue responses
            
            # Factor 4: Multiple relevant chunks retrieved (good BM25 performance)
            if len(chunks) >= 3:
                # Check if chunks are relevant by looking for query keywords
                relevant_chunks = 0
                for chunk in chunks[:5]:  # Check top 5 chunks
                    chunk_text = chunk.get('text', '').lower()
                    if any(keyword in chunk_text for keyword in query_keywords):
                        relevant_chunks += 1
                
                if relevant_chunks >= 2:
                    boost_factors.append(0.15)  # +0.15 for good chunk relevance
            
            # Factor 5: Response length and structure indicates good generation
            if len(response) > 100 and ('₹' in response or 'crore' in response.lower()):
                boost_factors.append(0.1)  # +0.1 for substantial financial response
            
            # Calculate enhanced confidence
            boost = sum(boost_factors)
            enhanced_confidence = min(0.85, current_confidence + boost)  # Cap at 0.85 for BM25-only
            
            # Log the enhancement details
            if boost > 0:
                logger.info(f"🔧 Cloud confidence boost: +{boost:.2f} from {len(boost_factors)} factors")
            
            return enhanced_confidence
            
        except Exception as e:
            logger.warning(f"Cloud confidence enhancement failed: {str(e)}")
            return current_confidence
    
    def update_generation_stats(self, category: str, confidence: float):
        """Update generation statistics."""
        self.generation_stats['total_responses'] += 1
        
        # Update template usage
        if category not in self.generation_stats['template_usage']:
            self.generation_stats['template_usage'][category] = 0
        self.generation_stats['template_usage'][category] += 1
        
        # Update average confidence
        n = self.generation_stats['total_responses']
        self.generation_stats['avg_confidence'] = (
            (self.generation_stats['avg_confidence'] * (n-1) + confidence) / n
        )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics."""
        return self.generation_stats.copy()

def main():
    """Test the smart response generation system."""
    print("="*80)
    print("SMART FINANCIAL RESPONSE GENERATION SYSTEM - PHASE 5")
    print("Template-Based Response Generation for Financial QA")
    print("="*80)
    
    # Initialize response generator
    generator = SmartFinancialResponseGenerator()
    
    # Mock retrieved chunks with realistic financial data
    mock_chunks = [
        {
            'chunk_id': 'qa_1',
            'text': 'Question: What was the revenue from operations in Q3 2023? Answer: The revenue from operations was ₹15.03 billion in Q3 2023, representing a 12% increase from the previous quarter.',
            'score': 9.15,
            'metadata': {'quarter': 'Q3 2023', 'section': 'qa_memory'}
        },
        {
            'chunk_id': 'qa_2',
            'text': 'Question: What was the profit before tax in Q3 2023? Answer: The loss before tax was ₹2.1 billion in Q3 2023, compared to a profit of ₹1.5 billion in Q2 2023.',
            'score': 8.75,
            'metadata': {'quarter': 'Q3 2023', 'section': 'qa_memory'}
        },
        {
            'chunk_id': 'expense_1',
            'text': 'Employee benefit expenses were ₹3.8 billion in Q3 2023, reflecting the company\'s continued investment in human resources.',
            'score': 7.22,
            'metadata': {'quarter': 'Q3 2023', 'section': 'expenses'}
        }
    ]
    
    # Test queries
    test_queries = [
        "What was the revenue from operations in Q3 2023?",
        "How much profit did the company make?",
        "Tell me about employee benefit expenses",
        "What were the total assets?"
    ]
    
    print(f"\n🧪 Testing Smart Response Generation")
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 60)
        
        # Generate contextual response
        response = generator.generate_contextual_response(query, mock_chunks)
        
        print(f"🤖 Generated Answer:")
        print(f"   {response['answer']}")
        print(f"")
        print(f"📊 Response Quality:")
        print(f"   Confidence: {response['confidence']:.2f}")
        print(f"   Method: {response['generation_metadata']['response_method']}")
        print(f"   Category: {response['query_classification']['primary_category']}")
        print(f"   Keywords: {response['query_classification']['matched_keywords']}")
        print(f"   Generation Time: {response['generation_metadata']['generation_time']:.3f}s")
        
        print(f"📋 Data Analysis:")
        print(f"   Chunks used: {response['data_sources']['chunks_used']}")
        print(f"   Q&A pairs found: {response['data_sources']['qa_pairs_found']}")
        print(f"   Financial amounts: {response['data_sources']['financial_amounts']}")
    
    # Show generation statistics
    stats = generator.get_generation_stats()
    print(f"\n📈 Generation Statistics:")
    print(f"   Total responses: {stats['total_responses']}")
    print(f"   Average confidence: {stats['avg_confidence']:.3f}")
    print(f"   Template usage: {stats['template_usage']}")
    
    print(f"\n🎉 Smart response generation test completed successfully!")

if __name__ == "__main__":
    main()

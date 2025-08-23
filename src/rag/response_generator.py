#!/usr/bin/env python3
"""
Financial Response Generation System
Phase 5: Convert retrieved financial information into natural language responses.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import json
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialResponseGenerator:
    def __init__(self,
                 model_name: str = "distilgpt2",
                 max_context_length: int = 800,
                 max_response_length: int = 200,
                 temperature: float = 0.7,
                 top_p: float = 0.9):
        """Initialize the Financial Response Generator.
        
        Args:
            model_name: HuggingFace model name for text generation
            max_context_length: Maximum context length for prompt
            max_response_length: Maximum tokens in generated response
            temperature: Sampling temperature for generation
            top_p: Top-p sampling parameter
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.max_response_length = max_response_length
        self.temperature = temperature
        self.top_p = top_p
        
        # Models will be loaded later
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Financial formatting patterns
        self.currency_patterns = {
            'inr': r'₹([\d,]+\.?\d*)\s*(crore|billion|million|thousand|lakh)?',
            'usd': r'\$([\d,]+\.?\d*)\s*(billion|million|thousand)?',
            'eur': r'€([\d,]+\.?\d*)\s*(billion|million|thousand)?'
        }
        
        # Response quality metrics
        self.generation_stats = {
            'total_responses': 0,
            'avg_response_length': 0.0,
            'avg_generation_time': 0.0,
            'confidence_scores': []
        }
    
    def load_model(self) -> bool:
        """Load the generative model and tokenizer."""
        try:
            logger.info(f"Loading generative model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Test generation
            test_input = self.tokenizer("Test", return_tensors="pt").to(self.device)
            with torch.no_grad():
                test_output = self.model.generate(
                    test_input['input_ids'],
                    max_length=test_input['input_ids'].shape[1] + 5,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            logger.info(f"Model: {self.model_name}, Context: {self.max_context_length}, Response: {self.max_response_length}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def format_context_from_retrieval(self, 
                                    query: str, 
                                    retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context for generation.
        
        Args:
            query: Original user query
            retrieved_chunks: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add query context
        context_parts.append(f"Question: {query}\n")
        context_parts.append("Based on the following financial information:\n")
        
        # Add retrieved information
        for i, chunk in enumerate(retrieved_chunks[:5]):  # Limit to top 5 chunks
            text = chunk.get('text', '').strip()
            quarter = chunk.get('metadata', {}).get('quarter', 'Unknown')
            section = chunk.get('metadata', {}).get('section', 'Unknown')
            score = chunk.get('score', 0)
            
            # Format chunk information
            context_parts.append(f"\n[Context {i+1}] {quarter} - {section} (Relevance: {score:.2f}):")
            context_parts.append(f"{text}")
        
        context_parts.append("\n\nAnswer:")
        
        full_context = " ".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > self.max_context_length:
            # Try to keep complete sentences
            truncated = full_context[:self.max_context_length]
            last_period = truncated.rfind('.')
            if last_period > self.max_context_length * 0.7:  # If we can keep 70% and end with period
                full_context = truncated[:last_period + 1] + "\n\nAnswer:"
            else:
                full_context = truncated + "...\n\nAnswer:"
        
        return full_context
    
    def generate_response(self, 
                         context: str, 
                         query: str,
                         retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a natural language response based on context.
        
        Args:
            context: Formatted context for generation
            query: Original user query
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = datetime.now()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length,
                padding=True
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=input_length + self.max_response_length,
                    min_length=input_length + 20,  # Ensure minimum response length
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
            
            # Decode response
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (after "Answer:")
            answer_start = full_output.rfind("Answer:")
            if answer_start != -1:
                generated_response = full_output[answer_start + 7:].strip()
            else:
                generated_response = full_output[input_length:].strip()
            
            # Clean up response
            cleaned_response = self.post_process_response(generated_response, query, retrieved_chunks)
            
            # Calculate confidence
            confidence = self.calculate_response_confidence(
                cleaned_response, query, retrieved_chunks
            )
            
            # Generation metadata
            generation_time = (datetime.now() - start_time).total_seconds()
            
            response_data = {
                'answer': cleaned_response,
                'confidence': confidence,
                'generation_time': generation_time,
                'input_tokens': input_length,
                'output_tokens': len(self.tokenizer.tokenize(cleaned_response)),
                'model_used': self.model_name,
                'temperature': self.temperature,
                'retrieved_chunks_count': len(retrieved_chunks),
                'timestamp': datetime.now().isoformat()
            }
            
            # Update statistics
            self.update_generation_stats(generation_time, len(cleaned_response), confidence)
            
            logger.info(f"✅ Response generated in {generation_time:.3f}s (confidence: {confidence:.2f})")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'answer': "I apologize, but I encountered an error generating the response. Please try rephrasing your question.",
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def post_process_response(self, 
                            response: str, 
                            query: str, 
                            retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Clean and enhance the generated response.
        
        Args:
            response: Raw generated response
            query: Original query
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Cleaned and enhanced response
        """
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and sentences[-1].strip() and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Fix common generation issues
        response = re.sub(r'\s+', ' ', response)  # Multiple spaces
        response = re.sub(r'\n+', ' ', response)  # Multiple newlines
        response = response.strip()
        
        # Ensure proper capitalization
        if response and not response[0].isupper():
            response = response[0].upper() + response[1:]
        
        # Add period if missing
        if response and not response.endswith('.'):
            response += '.'
        
        # Format financial figures consistently
        response = self.format_financial_figures(response)
        
        # Add source context if very short response
        if len(response) < 50 and retrieved_chunks:
            quarter = retrieved_chunks[0].get('metadata', {}).get('quarter', '')
            if quarter:
                response += f" (Data from {quarter})"
        
        return response
    
    def format_financial_figures(self, text: str) -> str:
        """Format financial figures in the response consistently.
        
        Args:
            text: Text containing financial figures
            
        Returns:
            Text with consistently formatted figures
        """
        # Format Indian Rupee amounts
        for currency, pattern in self.currency_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                original = match.group(0)
                amount = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else ''
                
                # Standardize formatting
                if currency == 'inr':
                    formatted = f"₹{amount}"
                    if unit:
                        formatted += f" {unit.lower()}"
                elif currency == 'usd':
                    formatted = f"${amount}"
                    if unit:
                        formatted += f" {unit.lower()}"
                elif currency == 'eur':
                    formatted = f"€{amount}"
                    if unit:
                        formatted += f" {unit.lower()}"
                
                text = text.replace(original, formatted)
        
        return text
    
    def calculate_response_confidence(self, 
                                   response: str, 
                                   query: str, 
                                   retrieved_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the generated response.
        
        Args:
            response: Generated response
            query: Original query
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence_factors = []
        
        # Factor 1: Response length (not too short, not too long)
        length_score = min(1.0, len(response) / 100.0)  # Optimal around 100 chars
        if len(response) > 200:
            length_score = max(0.5, 1.0 - (len(response) - 200) / 300.0)
        confidence_factors.append(length_score * 0.2)
        
        # Factor 2: Presence of specific financial information
        financial_terms = ['revenue', 'profit', 'loss', 'expense', 'assets', 'liabilities', 
                          'crore', 'billion', 'million', 'quarter', '₹', '%']
        financial_score = sum(1 for term in financial_terms if term.lower() in response.lower()) / len(financial_terms)
        confidence_factors.append(financial_score * 0.3)
        
        # Factor 3: Numerical precision (presence of specific numbers)
        number_pattern = r'[\d,]+\.?\d*'
        number_matches = len(re.findall(number_pattern, response))
        number_score = min(1.0, number_matches / 3.0)  # Optimal around 3 numbers
        confidence_factors.append(number_score * 0.2)
        
        # Factor 4: Context utilization (overlap with retrieved chunks)
        if retrieved_chunks:
            context_words = set()
            for chunk in retrieved_chunks[:3]:  # Top 3 chunks
                context_words.update(chunk.get('text', '').lower().split())
            
            response_words = set(response.lower().split())
            overlap = len(context_words.intersection(response_words))
            context_score = min(1.0, overlap / 20.0)  # Normalize by expected overlap
            confidence_factors.append(context_score * 0.2)
        else:
            confidence_factors.append(0.0)
        
        # Factor 5: Retrieval quality (average score of top chunks)
        if retrieved_chunks:
            top_scores = [chunk.get('score', 0) for chunk in retrieved_chunks[:3]]
            avg_retrieval_score = np.mean(top_scores) if top_scores else 0
            # Normalize retrieval score (assuming max score around 10 from cross-encoder)
            retrieval_score = min(1.0, avg_retrieval_score / 10.0)
            confidence_factors.append(retrieval_score * 0.1)
        else:
            confidence_factors.append(0.0)
        
        # Overall confidence
        total_confidence = sum(confidence_factors)
        return min(1.0, max(0.0, total_confidence))
    
    def update_generation_stats(self, generation_time: float, response_length: int, confidence: float):
        """Update generation statistics."""
        self.generation_stats['total_responses'] += 1
        n = self.generation_stats['total_responses']
        
        # Update averages
        self.generation_stats['avg_response_length'] = (
            (self.generation_stats['avg_response_length'] * (n-1) + response_length) / n
        )
        self.generation_stats['avg_generation_time'] = (
            (self.generation_stats['avg_generation_time'] * (n-1) + generation_time) / n
        )
        
        # Track confidence scores
        self.generation_stats['confidence_scores'].append(confidence)
    
    def create_structured_response(self, 
                                 query: str, 
                                 retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a complete structured response with context and generation.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Complete response with answer, context, and metadata
        """
        # Format context for generation
        context = self.format_context_from_retrieval(query, retrieved_chunks)
        
        # Generate response
        response_data = self.generate_response(context, query, retrieved_chunks)
        
        # Create structured output
        structured_response = {
            'query': query,
            'answer': response_data['answer'],
            'confidence': response_data.get('confidence', 0.0),
            'context_used': [
                {
                    'text': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                    'quarter': chunk.get('metadata', {}).get('quarter', 'Unknown'),
                    'section': chunk.get('metadata', {}).get('section', 'Unknown'),
                    'score': chunk.get('score', 0),
                    'chunk_id': chunk.get('chunk_id', 'Unknown')
                }
                for chunk in retrieved_chunks[:5]
            ],
            'generation_metadata': {
                'model': response_data.get('model_used', self.model_name),
                'generation_time': response_data.get('generation_time', 0),
                'input_tokens': response_data.get('input_tokens', 0),
                'output_tokens': response_data.get('output_tokens', 0),
                'temperature': response_data.get('temperature', self.temperature)
            },
            'timestamp': response_data.get('timestamp', datetime.now().isoformat())
        }
        
        return structured_response
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics."""
        stats = self.generation_stats.copy()
        
        if stats['confidence_scores']:
            stats['confidence_stats'] = {
                'mean': np.mean(stats['confidence_scores']),
                'std': np.std(stats['confidence_scores']),
                'min': np.min(stats['confidence_scores']),
                'max': np.max(stats['confidence_scores'])
            }
        
        stats['model_info'] = {
            'model_name': self.model_name,
            'device': str(self.device),
            'max_context_length': self.max_context_length,
            'max_response_length': self.max_response_length
        }
        
        return stats

def main():
    """Test the response generation system."""
    print("="*80)
    print("FINANCIAL RESPONSE GENERATION SYSTEM - PHASE 5")
    print("DistilGPT-2 Response Generation for Financial QA")
    print("="*80)
    
    # Initialize response generator
    generator = FinancialResponseGenerator()
    
    # Load model
    if not generator.load_model():
        print("❌ Failed to load model")
        return
    
    # Mock retrieved chunks for testing
    mock_chunks = [
        {
            'chunk_id': 'test_1',
            'text': 'Revenue from operations was ₹15.03 billion in Q3 2023, representing a 12% increase from the previous quarter.',
            'score': 9.15,
            'metadata': {'quarter': 'Q3 2023', 'section': 'revenue'}
        },
        {
            'chunk_id': 'test_2',
            'text': 'The company achieved a profit before tax of ₹2.1 billion, showing strong financial performance.',
            'score': 8.75,
            'metadata': {'quarter': 'Q3 2023', 'section': 'profit_loss'}
        },
        {
            'chunk_id': 'test_3',
            'text': 'Employee benefit expenses were ₹3.8 billion, reflecting the company\'s investment in human resources.',
            'score': 7.22,
            'metadata': {'quarter': 'Q3 2023', 'section': 'expenses'}
        }
    ]
    
    # Test queries
    test_queries = [
        "What was the revenue from operations in Q3 2023?",
        "How did the company perform financially in terms of profit?",
        "Tell me about employee-related expenses"
    ]
    
    print(f"\n🧪 Testing Response Generation")
    print(f"Model: {generator.model_name}")
    print(f"Device: {generator.device}")
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 60)
        
        # Generate structured response
        response = generator.create_structured_response(query, mock_chunks)
        
        print(f"🤖 Generated Answer:")
        print(f"   {response['answer']}")
        print(f"")
        print(f"📊 Response Quality:")
        print(f"   Confidence: {response['confidence']:.2f}")
        print(f"   Generation Time: {response['generation_metadata']['generation_time']:.3f}s")
        print(f"   Input Tokens: {response['generation_metadata']['input_tokens']}")
        print(f"   Output Tokens: {response['generation_metadata']['output_tokens']}")
        
        print(f"📋 Context Used:")
        for j, context in enumerate(response['context_used'], 1):
            print(f"   {j}. {context['quarter']} | Score: {context['score']:.2f}")
            print(f"      {context['text'][:80]}...")
    
    # Show generation statistics
    stats = generator.get_generation_stats()
    print(f"\n📈 Generation Statistics:")
    print(f"   Total responses: {stats['total_responses']}")
    print(f"   Avg response length: {stats['avg_response_length']:.1f} chars")
    print(f"   Avg generation time: {stats['avg_generation_time']:.3f}s")
    if 'confidence_stats' in stats:
        print(f"   Avg confidence: {stats['confidence_stats']['mean']:.3f}")
        print(f"   Confidence range: [{stats['confidence_stats']['min']:.3f}, {stats['confidence_stats']['max']:.3f}]")
    
    print(f"\n🎉 Response generation test completed successfully!")

if __name__ == "__main__":
    main()

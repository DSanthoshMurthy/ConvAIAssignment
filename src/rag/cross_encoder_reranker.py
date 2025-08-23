#!/usr/bin/env python3
"""
Cross-Encoder Re-Ranking System for Financial RAG
Advanced Technique #3: Re-Ranking with Cross-Encoders (Group Number 98 mod 5 = 3)
"""

import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
import numpy as np
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialCrossEncoderReranker:
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int = 512):
        """Initialize the Cross-Encoder Re-Ranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name
            max_length: Maximum sequence length for cross-encoder
        """
        self.model_name = model_name
        self.max_length = max_length
        self.cross_encoder = None
        self.tokenizer = None
        
        # Performance tracking
        self.reranking_stats = {
            'total_queries': 0,
            'total_candidates_processed': 0,
            'avg_reranking_time': 0.0,
            'score_improvements': []
        }
    
    def load_cross_encoder(self) -> bool:
        """Load the cross-encoder model."""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            
            # Load cross-encoder
            self.cross_encoder = CrossEncoder(self.model_name, max_length=self.max_length)
            
            # Load tokenizer for length checking
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Test the model
            test_pairs = [["test query", "test document"]]
            test_scores = self.cross_encoder.predict(test_pairs)
            
            logger.info(f"✅ Cross-encoder loaded successfully")
            logger.info(f"Model: {self.model_name}, Max Length: {self.max_length}")
            logger.info(f"Test score: {test_scores[0]:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading cross-encoder: {str(e)}")
            return False
    
    def prepare_query_document_pairs(self, query: str, candidates: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Prepare query-document pairs for cross-encoder scoring.
        
        Args:
            query: The search query
            candidates: List of candidate documents from initial retrieval
            
        Returns:
            List of (query, document, metadata) tuples
        """
        pairs = []
        
        for candidate in candidates:
            document = candidate['text']
            
            # Truncate document if too long (keep some room for query)
            if self.tokenizer:
                # Estimate tokens (rough approximation)
                query_tokens = len(self.tokenizer.tokenize(query))
                max_doc_tokens = self.max_length - query_tokens - 10  # Buffer for special tokens
                
                if max_doc_tokens > 0:
                    doc_tokens = self.tokenizer.tokenize(document)
                    if len(doc_tokens) > max_doc_tokens:
                        # Truncate document but try to keep meaningful content
                        truncated_tokens = doc_tokens[:max_doc_tokens]
                        document = self.tokenizer.convert_tokens_to_string(truncated_tokens)
            
            pairs.append((query, document, candidate))
        
        logger.info(f"Prepared {len(pairs)} query-document pairs for re-ranking")
        return pairs
    
    def compute_cross_encoder_scores(self, query_doc_pairs: List[Tuple[str, str, Dict[str, Any]]]) -> List[float]:
        """Compute cross-encoder relevance scores for query-document pairs.
        
        Args:
            query_doc_pairs: List of (query, document, metadata) tuples
            
        Returns:
            List of relevance scores
        """
        try:
            # Extract just query-document pairs for the model
            model_input = [(query, doc) for query, doc, _ in query_doc_pairs]
            
            # Batch process for efficiency
            start_time = time.time()
            scores = self.cross_encoder.predict(model_input)
            processing_time = time.time() - start_time
            
            logger.info(f"Cross-encoder scoring completed in {processing_time:.3f}s")
            logger.info(f"Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
            
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            
        except Exception as e:
            logger.error(f"Error computing cross-encoder scores: {str(e)}")
            # Fallback to original scores if cross-encoder fails
            return [pair[2]['score'] for pair in query_doc_pairs]
    
    def rerank_candidates(self, 
                         query: str, 
                         candidates: List[Dict[str, Any]], 
                         top_k: int = 10) -> List[Dict[str, Any]]:
        """Re-rank candidates using cross-encoder scores.
        
        Args:
            query: The search query
            candidates: Initial retrieval candidates
            top_k: Number of top results to return after re-ranking
            
        Returns:
            Re-ranked list of candidates with updated scores
        """
        try:
            start_time = time.time()
            
            if not self.cross_encoder:
                logger.warning("Cross-encoder not loaded, returning original ranking")
                return candidates[:top_k]
            
            if not candidates:
                logger.warning("No candidates provided for re-ranking")
                return []
            
            # Prepare query-document pairs
            query_doc_pairs = self.prepare_query_document_pairs(query, candidates)
            
            # Compute cross-encoder scores
            ce_scores = self.compute_cross_encoder_scores(query_doc_pairs)
            
            # Create re-ranked results
            reranked_results = []
            
            for i, (query_text, doc_text, original_candidate) in enumerate(query_doc_pairs):
                ce_score = ce_scores[i]
                
                # Create enhanced result with both original and cross-encoder scores
                enhanced_result = original_candidate.copy()
                enhanced_result.update({
                    'original_score': original_candidate['score'],
                    'cross_encoder_score': float(ce_score),
                    'score': float(ce_score),  # Use CE score as primary score
                    'reranking_method': 'cross_encoder',
                    'original_rank': i + 1,
                    'score_improvement': float(ce_score) - original_candidate['score']
                })
                
                reranked_results.append(enhanced_result)
            
            # Sort by cross-encoder score (descending)
            reranked_results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            
            # Update final ranks
            for i, result in enumerate(reranked_results):
                result['final_rank'] = i + 1
            
            # Update statistics
            processing_time = time.time() - start_time
            self.update_stats(query, candidates, reranked_results, processing_time)
            
            # Return top_k results
            final_results = reranked_results[:top_k]
            
            logger.info(f"✅ Re-ranking complete: {len(final_results)} results in {processing_time:.3f}s")
            
            # Log score improvements
            improvements = [r['score_improvement'] for r in final_results]
            avg_improvement = np.mean(improvements) if improvements else 0
            logger.info(f"Average score improvement: {avg_improvement:+.4f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in re-ranking: {str(e)}")
            # Fallback to original ranking
            return candidates[:top_k]
    
    def update_stats(self, query: str, original_candidates: List[Dict[str, Any]], 
                    reranked_results: List[Dict[str, Any]], processing_time: float):
        """Update performance statistics."""
        self.reranking_stats['total_queries'] += 1
        self.reranking_stats['total_candidates_processed'] += len(original_candidates)
        
        # Update average processing time
        total_time = (self.reranking_stats['avg_reranking_time'] * 
                     (self.reranking_stats['total_queries'] - 1) + processing_time)
        self.reranking_stats['avg_reranking_time'] = total_time / self.reranking_stats['total_queries']
        
        # Track score improvements
        for result in reranked_results:
            if 'score_improvement' in result:
                self.reranking_stats['score_improvements'].append(result['score_improvement'])
    
    def analyze_reranking_impact(self, original_results: List[Dict[str, Any]], 
                               reranked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the impact of re-ranking on result quality."""
        analysis = {
            'total_candidates': len(original_results),
            'position_changes': 0,
            'score_changes': [],
            'rank_changes': {},
            'top_result_changed': False
        }
        
        if not original_results or not reranked_results:
            return analysis
        
        # Check if top result changed
        orig_top_id = original_results[0].get('chunk_id', '')
        reranked_top_id = reranked_results[0].get('chunk_id', '')
        analysis['top_result_changed'] = orig_top_id != reranked_top_id
        
        # Create mapping from original positions
        orig_positions = {result.get('chunk_id', f'idx_{i}'): i for i, result in enumerate(original_results)}
        
        # Analyze position changes
        for new_pos, result in enumerate(reranked_results):
            chunk_id = result.get('chunk_id', f'idx_{new_pos}')
            orig_pos = orig_positions.get(chunk_id, -1)
            
            if orig_pos != -1:
                rank_change = orig_pos - new_pos
                if rank_change != 0:
                    analysis['position_changes'] += 1
                    analysis['rank_changes'][chunk_id] = rank_change
                
                # Track score changes
                if 'score_improvement' in result:
                    analysis['score_changes'].append(result['score_improvement'])
        
        # Summary statistics
        if analysis['score_changes']:
            analysis['avg_score_improvement'] = np.mean(analysis['score_changes'])
            analysis['max_score_improvement'] = np.max(analysis['score_changes'])
            analysis['min_score_improvement'] = np.min(analysis['score_changes'])
        
        return analysis
    
    def get_reranking_explanation(self, query: str, top_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Provide explanation for re-ranking decisions."""
        explanation = {
            'query': query,
            'reranking_model': self.model_name,
            'top_results_analysis': [],
            'methodology': 'Cross-encoder re-ranking with query-document pair scoring'
        }
        
        for i, result in enumerate(top_results[:3]):  # Explain top 3
            result_explanation = {
                'rank': i + 1,
                'chunk_id': result.get('chunk_id', 'Unknown'),
                'cross_encoder_score': result.get('cross_encoder_score', 0),
                'original_score': result.get('original_score', 0),
                'score_improvement': result.get('score_improvement', 0),
                'original_rank': result.get('original_rank', 0),
                'section': result.get('metadata', {}).get('section', 'Unknown'),
                'quarter': result.get('metadata', {}).get('quarter', 'Unknown')
            }
            explanation['top_results_analysis'].append(result_explanation)
        
        return explanation
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.reranking_stats.copy()
        
        if stats['score_improvements']:
            stats['score_improvement_stats'] = {
                'mean': np.mean(stats['score_improvements']),
                'std': np.std(stats['score_improvements']),
                'min': np.min(stats['score_improvements']),
                'max': np.max(stats['score_improvements']),
                'positive_improvements': sum(1 for x in stats['score_improvements'] if x > 0),
                'negative_improvements': sum(1 for x in stats['score_improvements'] if x < 0)
            }
        
        return stats

def main():
    """Test the cross-encoder re-ranking system."""
    print("="*80)
    print("CROSS-ENCODER RE-RANKING SYSTEM - GROUP NUMBER 98")
    print("Advanced RAG Technique #3: Re-Ranking with Cross-Encoders")
    print("="*80)
    
    # Initialize re-ranker
    reranker = FinancialCrossEncoderReranker()
    
    # Load cross-encoder model
    if not reranker.load_cross_encoder():
        print("❌ Failed to load cross-encoder model")
        return
    
    # Create some mock candidates for testing
    mock_candidates = [
        {
            'chunk_id': 'test_1',
            'text': 'Revenue from operations was ₹15.03 billion in Q3 2023',
            'score': 0.7,
            'method': 'hybrid',
            'metadata': {'section': 'revenue', 'quarter': 'Q3 2023'}
        },
        {
            'chunk_id': 'test_2', 
            'text': 'The company reported total income of ₹2.5 billion',
            'score': 0.6,
            'method': 'dense',
            'metadata': {'section': 'income', 'quarter': 'Q3 2023'}
        },
        {
            'chunk_id': 'test_3',
            'text': 'Employee benefit expense was ₹1.02 billion in the quarter',
            'score': 0.5,
            'method': 'sparse',
            'metadata': {'section': 'expenses', 'quarter': 'Q3 2023'}
        }
    ]
    
    # Test query
    test_query = "What was the revenue from operations in Q3 2023?"
    
    print(f"\n🔍 Test Query: {test_query}")
    print("\n📋 Original Candidates:")
    for i, candidate in enumerate(mock_candidates, 1):
        print(f"  {i}. Score: {candidate['score']:.3f} | {candidate['text'][:60]}...")
    
    # Perform re-ranking
    reranked_results = reranker.rerank_candidates(test_query, mock_candidates, top_k=3)
    
    print("\n🔄 Re-ranked Results:")
    for i, result in enumerate(reranked_results, 1):
        print(f"  {i}. CE Score: {result['cross_encoder_score']:.3f} | "
              f"Orig: {result['original_score']:.3f} | "
              f"Improvement: {result['score_improvement']:+.3f}")
        print(f"     {result['text'][:60]}...")
    
    # Show analysis
    analysis = reranker.analyze_reranking_impact(mock_candidates, reranked_results)
    print(f"\n📊 Re-ranking Impact Analysis:")
    print(f"  Position changes: {analysis['position_changes']}")
    print(f"  Top result changed: {analysis['top_result_changed']}")
    if 'avg_score_improvement' in analysis:
        print(f"  Avg score improvement: {analysis['avg_score_improvement']:+.3f}")
    
    # Show explanation
    explanation = reranker.get_reranking_explanation(test_query, reranked_results)
    print(f"\n💡 Re-ranking Explanation:")
    print(f"  Model: {explanation['reranking_model']}")
    print(f"  Methodology: {explanation['methodology']}")
    
    # Performance stats
    stats = reranker.get_performance_stats()
    print(f"\n⚡ Performance Statistics:")
    print(f"  Total queries processed: {stats['total_queries']}")
    print(f"  Avg re-ranking time: {stats['avg_reranking_time']:.3f}s")
    
    print(f"\n🎉 Cross-encoder re-ranking test completed successfully!")

if __name__ == "__main__":
    main()

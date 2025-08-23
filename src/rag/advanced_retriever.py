#!/usr/bin/env python3
"""
Advanced Financial RAG Retriever with Cross-Encoder Re-Ranking
Integrates hybrid retrieval with Advanced Technique #3 for Group Number 98.
"""

import logging
from typing import List, Dict, Any, Optional
# Fix imports for direct execution
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from hybrid_retriever import FinancialHybridRetriever
from cross_encoder_reranker import FinancialCrossEncoderReranker
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFinancialRetriever:
    def __init__(self, 
                 indexes_dir: str = "data/indexes",
                 chroma_db_dir: str = "data/indexes/chroma_db",
                 enable_reranking: bool = True,
                 reranking_candidates: int = 20,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the Advanced Financial Retriever.
        
        Args:
            indexes_dir: Directory containing all indexes
            chroma_db_dir: ChromaDB persistence directory  
            enable_reranking: Whether to enable cross-encoder re-ranking
            reranking_candidates: Number of candidates to retrieve before re-ranking
            cross_encoder_model: Cross-encoder model for re-ranking
        """
        self.enable_reranking = enable_reranking
        self.reranking_candidates = reranking_candidates
        
        # Initialize hybrid retriever
        self.hybrid_retriever = FinancialHybridRetriever(
            indexes_dir=indexes_dir,
            chroma_db_dir=chroma_db_dir
        )
        
        # Initialize cross-encoder re-ranker if enabled
        self.reranker = None
        if self.enable_reranking:
            self.reranker = FinancialCrossEncoderReranker(
                model_name=cross_encoder_model
            )
        
        self.loaded = False
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'hybrid_retrieval_time': 0.0,
            'reranking_time': 0.0,
            'total_time': 0.0,
            'reranking_improvements': []
        }
    
    def load_models_and_indexes(self) -> bool:
        """Load all models and indexes."""
        try:
            logger.info("🚀 Loading Advanced Financial Retriever...")
            
            # Load hybrid retriever
            if not self.hybrid_retriever.load_models_and_indexes():
                logger.error("Failed to load hybrid retriever")
                return False
            
            # Load cross-encoder re-ranker if enabled
            if self.enable_reranking:
                if not self.reranker.load_cross_encoder():
                    logger.warning("Failed to load cross-encoder, disabling re-ranking")
                    self.enable_reranking = False
                else:
                    logger.info("✅ Cross-encoder re-ranking enabled")
            
            self.loaded = True
            logger.info("🎉 Advanced Financial Retriever loaded successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading advanced retriever: {str(e)}")
            return False
    
    def advanced_search(self, 
                       query: str, 
                       top_k: int = 10,
                       fusion_method: str = 'weighted',
                       filters: Optional[Dict[str, Any]] = None,
                       explanation: bool = False) -> Dict[str, Any]:
        """Perform advanced search with optional cross-encoder re-ranking.
        
        Args:
            query: Search query string
            top_k: Number of final results to return
            fusion_method: 'weighted' or 'rrf' for hybrid fusion
            filters: Optional metadata filters
            explanation: Whether to include detailed explanation
            
        Returns:
            Dictionary with results and metadata
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models_and_indexes() first.")
        
        start_time = time.time()
        search_metadata = {
            'query': query,
            'top_k': top_k,
            'fusion_method': fusion_method,
            'filters': filters,
            'reranking_enabled': self.enable_reranking,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Stage 1: Hybrid Retrieval (get more candidates if re-ranking)
            retrieval_candidates = self.reranking_candidates if self.enable_reranking else top_k
            
            logger.info(f"🔍 Stage 1: Hybrid retrieval for '{query}' (candidates: {retrieval_candidates})")
            hybrid_start = time.time()
            
            hybrid_results = self.hybrid_retriever.hybrid_search(
                query=query,
                top_k=retrieval_candidates,
                fusion_method=fusion_method,
                filters=filters
            )
            
            hybrid_time = time.time() - hybrid_start
            search_metadata['hybrid_retrieval_time'] = hybrid_time
            search_metadata['hybrid_candidates'] = len(hybrid_results)
            
            # Stage 2: Cross-Encoder Re-Ranking (if enabled)
            final_results = hybrid_results
            reranking_time = 0.0
            reranking_analysis = None
            
            if self.enable_reranking and self.reranker and hybrid_results:
                logger.info(f"🔄 Stage 2: Cross-encoder re-ranking ({len(hybrid_results)} candidates)")
                reranking_start = time.time()
                
                # Store original results for comparison
                original_results = hybrid_results.copy()
                
                # Perform re-ranking
                reranked_results = self.reranker.rerank_candidates(
                    query=query,
                    candidates=hybrid_results,
                    top_k=top_k
                )
                
                reranking_time = time.time() - reranking_start
                final_results = reranked_results
                
                # Analyze re-ranking impact
                reranking_analysis = self.reranker.analyze_reranking_impact(
                    original_results, reranked_results
                )
                
                search_metadata['reranking_time'] = reranking_time
                search_metadata['reranking_analysis'] = reranking_analysis
                
                if explanation:
                    search_metadata['reranking_explanation'] = self.reranker.get_reranking_explanation(
                        query, final_results
                    )
            
            # Final processing
            total_time = time.time() - start_time
            search_metadata['total_time'] = total_time
            search_metadata['final_results_count'] = len(final_results)
            
            # Update performance stats
            self.update_performance_stats(hybrid_time, reranking_time, total_time, reranking_analysis)
            
            # Generate retrieval statistics
            retrieval_stats = self.hybrid_retriever.get_retrieval_stats(final_results)
            search_metadata['retrieval_stats'] = retrieval_stats
            
            logger.info(f"✅ Advanced search complete: {len(final_results)} results in {total_time:.3f}s")
            
            # Build response
            response = {
                'results': final_results,
                'metadata': search_metadata,
                'performance': {
                    'total_time': total_time,
                    'hybrid_time': hybrid_time, 
                    'reranking_time': reranking_time,
                    'results_count': len(final_results)
                }
            }
            
            if explanation:
                response['explanation'] = self.generate_search_explanation(
                    query, hybrid_results, final_results, search_metadata
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in advanced search: {str(e)}")
            return {
                'results': [],
                'metadata': search_metadata,
                'error': str(e)
            }
    
    def update_performance_stats(self, hybrid_time: float, reranking_time: float, 
                               total_time: float, reranking_analysis: Optional[Dict]):
        """Update performance statistics."""
        self.retrieval_stats['total_queries'] += 1
        
        # Update average times
        n = self.retrieval_stats['total_queries']
        self.retrieval_stats['hybrid_retrieval_time'] = (
            (self.retrieval_stats['hybrid_retrieval_time'] * (n-1) + hybrid_time) / n
        )
        self.retrieval_stats['reranking_time'] = (
            (self.retrieval_stats['reranking_time'] * (n-1) + reranking_time) / n
        )
        self.retrieval_stats['total_time'] = (
            (self.retrieval_stats['total_time'] * (n-1) + total_time) / n
        )
        
        # Track re-ranking improvements
        if reranking_analysis and 'avg_score_improvement' in reranking_analysis:
            self.retrieval_stats['reranking_improvements'].append(
                reranking_analysis['avg_score_improvement']
            )
    
    def generate_search_explanation(self, query: str, hybrid_results: List[Dict], 
                                  final_results: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Generate detailed explanation of the search process."""
        explanation = {
            'search_process': {
                'stage_1': 'Hybrid retrieval combining dense semantic search and sparse keyword matching',
                'stage_2': 'Cross-encoder re-ranking for precise relevance scoring' if self.enable_reranking else 'No re-ranking applied'
            },
            'query_analysis': {
                'original_query': query,
                'processed_tokens': 'Tokenized and filtered for financial terms',
                'expansion': 'Financial synonyms added for better recall'
            },
            'retrieval_pipeline': {
                'fusion_method': metadata.get('fusion_method', 'weighted'),
                'candidates_retrieved': len(hybrid_results),
                'final_results': len(final_results),
                'reranking_enabled': self.enable_reranking
            },
            'performance_breakdown': {
                'hybrid_retrieval': f"{metadata.get('hybrid_retrieval_time', 0):.3f}s",
                'cross_encoder_reranking': f"{metadata.get('reranking_time', 0):.3f}s",
                'total_time': f"{metadata.get('total_time', 0):.3f}s"
            }
        }
        
        if self.enable_reranking and 'reranking_analysis' in metadata:
            analysis = metadata['reranking_analysis']
            explanation['reranking_impact'] = {
                'position_changes': analysis.get('position_changes', 0),
                'top_result_changed': analysis.get('top_result_changed', False),
                'avg_score_improvement': analysis.get('avg_score_improvement', 0)
            }
        
        return explanation
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance statistics."""
        stats = self.retrieval_stats.copy()
        
        # Add cross-encoder stats if available
        if self.reranker:
            ce_stats = self.reranker.get_performance_stats()
            stats['cross_encoder_stats'] = ce_stats
        
        # Add hybrid retriever stats
        stats['hybrid_retriever_config'] = {
            'dense_weight': self.hybrid_retriever.dense_weight,
            'sparse_weight': self.hybrid_retriever.sparse_weight,
            'embedding_model': self.hybrid_retriever.embedding_model_name
        }
        
        return stats
    
    def benchmark_search_methods(self, test_queries: List[str], top_k: int = 5) -> Dict[str, Any]:
        """Benchmark different search methods with test queries."""
        benchmark_results = {
            'test_queries': test_queries,
            'methods_compared': ['hybrid_only', 'hybrid_with_reranking'],
            'results': []
        }
        
        for query in test_queries:
            query_results = {'query': query}
            
            # Test 1: Hybrid only
            hybrid_start = time.time()
            hybrid_only = self.hybrid_retriever.hybrid_search(query, top_k=top_k)
            hybrid_time = time.time() - hybrid_start
            
            query_results['hybrid_only'] = {
                'results_count': len(hybrid_only),
                'time': hybrid_time,
                'top_score': hybrid_only[0]['score'] if hybrid_only else 0
            }
            
            # Test 2: Hybrid with re-ranking (if enabled)
            if self.enable_reranking:
                advanced_start = time.time()
                advanced_result = self.advanced_search(query, top_k=top_k)
                advanced_time = time.time() - advanced_start
                
                advanced_results = advanced_result.get('results', [])
                query_results['hybrid_with_reranking'] = {
                    'results_count': len(advanced_results),
                    'time': advanced_time,
                    'top_score': advanced_results[0]['score'] if advanced_results else 0,
                    'reranking_improvement': advanced_result.get('metadata', {}).get('reranking_analysis', {}).get('avg_score_improvement', 0)
                }
            
            benchmark_results['results'].append(query_results)
        
        return benchmark_results

def main():
    """Test the advanced retrieval system."""
    print("="*80)
    print("ADVANCED FINANCIAL RETRIEVER - GROUP NUMBER 98")  
    print("Hybrid Retrieval + Cross-Encoder Re-Ranking (Technique #3)")
    print("="*80)
    
    # Initialize advanced retriever
    retriever = AdvancedFinancialRetriever(enable_reranking=True)
    
    # Load models
    if not retriever.load_models_and_indexes():
        print("❌ Failed to load models and indexes")
        return
    
    # Test queries
    test_queries = [
        "What was the revenue from operations in Q3 2023?",
        "Show me profit before tax information",
        "Employee benefit expenses across quarters"
    ]
    
    print(f"\n🧪 Testing Advanced Retrieval System")
    print(f"Re-ranking enabled: {retriever.enable_reranking}")
    print(f"Cross-encoder model: {retriever.reranker.model_name if retriever.reranker else 'N/A'}")
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 60)
        
        # Perform advanced search with explanation
        result = retriever.advanced_search(
            query=query,
            top_k=3,
            fusion_method='weighted',
            explanation=True
        )
        
        # Display results
        results = result.get('results', [])
        metadata = result.get('metadata', {})
        performance = result.get('performance', {})
        
        print(f"📊 Results ({len(results)} found):")
        for j, res in enumerate(results, 1):
            ce_score = res.get('cross_encoder_score', res.get('score', 0))
            orig_score = res.get('original_score', 'N/A')
            improvement = res.get('score_improvement', 0)
            
            print(f"  {j}. CE Score: {ce_score:.3f} | Orig: {orig_score} | Δ: {improvement:+.3f}")
            print(f"     Section: {res.get('metadata', {}).get('section', 'N/A')} | "
                  f"Quarter: {res.get('metadata', {}).get('quarter', 'N/A')}")
            print(f"     {res.get('text', '')[:100]}...")
        
        print(f"\n⚡ Performance:")
        print(f"  Total Time: {performance.get('total_time', 0):.3f}s")
        print(f"  Hybrid: {performance.get('hybrid_time', 0):.3f}s")
        print(f"  Re-ranking: {performance.get('reranking_time', 0):.3f}s")
        
        # Show re-ranking impact
        if 'reranking_analysis' in metadata:
            analysis = metadata['reranking_analysis']
            print(f"  Position Changes: {analysis.get('position_changes', 0)}")
            print(f"  Top Result Changed: {analysis.get('top_result_changed', False)}")
    
    # System performance summary
    system_stats = retriever.get_system_performance()
    print(f"\n📈 System Performance Summary:")
    print(f"  Total queries: {system_stats['total_queries']}")
    print(f"  Avg hybrid time: {system_stats['hybrid_retrieval_time']:.3f}s")
    print(f"  Avg reranking time: {system_stats['reranking_time']:.3f}s")
    print(f"  Avg total time: {system_stats['total_time']:.3f}s")
    
    print(f"\n🎉 Advanced retrieval system test completed!")

if __name__ == "__main__":
    main()

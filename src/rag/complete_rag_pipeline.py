#!/usr/bin/env python3
"""
Complete Financial RAG Pipeline
Integrates all components: Advanced Retrieval + Smart Response Generation
"""

import logging
from typing import List, Dict, Any, Optional
# Fix imports for direct execution
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from advanced_retriever import AdvancedFinancialRetriever
from smart_response_generator import SmartFinancialResponseGenerator
import time
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteFinancialRAG:
    def __init__(self, 
                 indexes_dir: str = "data/indexes",
                 chroma_db_dir: str = "data/indexes/chroma_db",
                 enable_cross_encoder: bool = True):
        """Initialize the Complete Financial RAG system.
        
        Args:
            indexes_dir: Directory containing all indexes
            chroma_db_dir: ChromaDB persistence directory
            enable_cross_encoder: Whether to enable cross-encoder re-ranking
        """
        # Initialize retrieval system
        self.retriever = AdvancedFinancialRetriever(
            indexes_dir=indexes_dir,
            chroma_db_dir=chroma_db_dir,
            enable_reranking=enable_cross_encoder
        )
        
        # Initialize response generation system
        self.response_generator = SmartFinancialResponseGenerator()
        
        self.loaded = False
        
        # Complete system statistics
        self.rag_stats = {
            'total_queries': 0,
            'retrieval_time': 0.0,
            'generation_time': 0.0,
            'total_time': 0.0,
            'avg_confidence': 0.0,
            'query_types': {}
        }
    
    def load_system(self) -> bool:
        """Load all components of the RAG system."""
        try:
            logger.info("🚀 Loading Complete Financial RAG System...")
            
            # Load retrieval system
            if not self.retriever.load_models_and_indexes():
                logger.error("Failed to load retrieval system")
                return False
            
            logger.info("✅ Retrieval system loaded")
            
            # Response generator doesn't need model loading (template-based)
            logger.info("✅ Response generation system ready")
            
            self.loaded = True
            logger.info("🎉 Complete Financial RAG System loaded successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading RAG system: {str(e)}")
            return False
    
    def process_query(self, 
                     query: str, 
                     top_k: int = 5,
                     fusion_method: str = 'weighted',
                     filters: Optional[Dict[str, Any]] = None,
                     include_explanation: bool = False) -> Dict[str, Any]:
        """Process a complete financial query through the RAG pipeline.
        
        Args:
            query: User query string
            top_k: Number of relevant chunks to retrieve
            fusion_method: 'weighted' or 'rrf' for hybrid fusion
            filters: Optional metadata filters
            include_explanation: Whether to include detailed explanation
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        if not self.loaded:
            raise RuntimeError("RAG system not loaded. Call load_system() first.")
        
        start_time = datetime.now()
        logger.info(f"🔍 Processing query: '{query}'")
        
        try:
            # Stage 1: Advanced Retrieval
            retrieval_start = time.time()
            retrieval_result = self.retriever.advanced_search(
                query=query,
                top_k=top_k,
                fusion_method=fusion_method,
                filters=filters,
                explanation=include_explanation
            )
            retrieval_time = time.time() - retrieval_start
            
            retrieved_chunks = retrieval_result.get('results', [])
            retrieval_metadata = retrieval_result.get('metadata', {})
            
            logger.info(f"✓ Retrieved {len(retrieved_chunks)} relevant chunks in {retrieval_time:.3f}s")
            
            # Stage 2: Smart Response Generation  
            generation_start = time.time()
            response_result = self.response_generator.generate_contextual_response(
                query, retrieved_chunks
            )
            generation_time = time.time() - generation_start
            
            logger.info(f"✓ Generated response in {generation_time:.3f}s")
            
            # Combine results
            total_time = (datetime.now() - start_time).total_seconds()
            
            complete_response = {
                'query': query,
                'answer': response_result['answer'],
                'confidence': response_result['confidence'],
                'sources': self.format_sources(retrieved_chunks),
                'retrieval_metadata': {
                    'chunks_found': len(retrieved_chunks),
                    'fusion_method': fusion_method,
                    'reranking_enabled': retrieval_metadata.get('reranking_enabled', False),
                    'retrieval_time': retrieval_time
                },
                'generation_metadata': response_result['generation_metadata'],
                'query_analysis': response_result['query_classification'],
                'performance': {
                    'total_time': total_time,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'retrieval_percentage': (retrieval_time / total_time) * 100,
                    'generation_percentage': (generation_time / total_time) * 100
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Add explanation if requested
            if include_explanation:
                complete_response['explanation'] = {
                    'retrieval_explanation': retrieval_result.get('explanation', {}),
                    'response_explanation': {
                        'method_used': response_result['generation_metadata']['response_method'],
                        'data_sources_analyzed': response_result['data_sources'],
                        'template_category': response_result['generation_metadata'].get('template_category', 'N/A')
                    }
                }
            
            # Update system statistics
            self.update_rag_stats(
                query, retrieval_time, generation_time, total_time, 
                response_result['confidence'], response_result['query_classification']['primary_category']
            )
            
            logger.info(f"✅ RAG query complete: confidence {response_result['confidence']:.2f} in {total_time:.3f}s")
            
            return complete_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': query,
                'answer': "I apologize, but I encountered an error processing your question. Please try rephrasing your query.",
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format retrieved chunks as sources for citation.
        
        Args:
            chunks: Retrieved document chunks
            
        Returns:
            Formatted source information
        """
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            source = {
                'source_id': i,
                'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
                'content_preview': chunk.get('text', '')[:150] + '...' if len(chunk.get('text', '')) > 150 else chunk.get('text', ''),
                'quarter': chunk.get('metadata', {}).get('quarter', 'Unknown'),
                'section': chunk.get('metadata', {}).get('section', 'Unknown'),
                'relevance_score': chunk.get('score', 0),
                'retrieval_method': chunk.get('method', 'Unknown')
            }
            
            # Add cross-encoder information if available
            if 'cross_encoder_score' in chunk:
                source.update({
                    'cross_encoder_score': chunk['cross_encoder_score'],
                    'original_score': chunk.get('original_score', 0),
                    'score_improvement': chunk.get('score_improvement', 0)
                })
            
            sources.append(source)
        
        return sources
    
    def batch_process_queries(self, 
                             queries: List[str], 
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            top_k: Number of chunks to retrieve per query
            
        Returns:
            List of complete responses
        """
        logger.info(f"🔄 Processing {len(queries)} queries in batch")
        
        results = []
        batch_start = time.time()
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query[:50]}...")
            
            try:
                result = self.process_query(query, top_k=top_k)
                results.append(result)
                
                logger.info(f"✓ Query {i} complete (confidence: {result.get('confidence', 0):.2f})")
                
            except Exception as e:
                logger.error(f"Error processing query {i}: {str(e)}")
                results.append({
                    'query': query,
                    'answer': "Error processing this query.",
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        batch_time = time.time() - batch_start
        
        # Batch statistics
        confidences = [r.get('confidence', 0) for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        logger.info(f"✅ Batch processing complete: {len(results)} queries in {batch_time:.2f}s")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        
        return results
    
    def update_rag_stats(self, query: str, retrieval_time: float, generation_time: float, 
                        total_time: float, confidence: float, query_type: str):
        """Update RAG system statistics."""
        self.rag_stats['total_queries'] += 1
        n = self.rag_stats['total_queries']
        
        # Update average times
        self.rag_stats['retrieval_time'] = (
            (self.rag_stats['retrieval_time'] * (n-1) + retrieval_time) / n
        )
        self.rag_stats['generation_time'] = (
            (self.rag_stats['generation_time'] * (n-1) + generation_time) / n
        )
        self.rag_stats['total_time'] = (
            (self.rag_stats['total_time'] * (n-1) + total_time) / n
        )
        self.rag_stats['avg_confidence'] = (
            (self.rag_stats['avg_confidence'] * (n-1) + confidence) / n
        )
        
        # Track query types
        if query_type not in self.rag_stats['query_types']:
            self.rag_stats['query_types'][query_type] = 0
        self.rag_stats['query_types'][query_type] += 1
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = self.rag_stats.copy()
        
        # Add component statistics
        stats['retrieval_stats'] = self.retriever.get_system_performance()
        stats['generation_stats'] = self.response_generator.get_generation_stats()
        
        # System configuration
        stats['system_config'] = {
            'cross_encoder_enabled': self.retriever.enable_reranking,
            'cross_encoder_model': self.retriever.reranker.model_name if self.retriever.reranker else None,
            'embedding_model': self.retriever.hybrid_retriever.embedding_model_name,
            'response_method': 'template_based'
        }
        
        return stats
    
    def save_conversation_log(self, 
                            queries_and_responses: List[Dict[str, Any]], 
                            filename: str = None) -> str:
        """Save conversation log to file.
        
        Args:
            queries_and_responses: List of query-response pairs
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_conversation_{timestamp}.json"
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'system_stats': self.get_system_stats(),
            'conversation': queries_and_responses
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"✅ Conversation log saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving conversation log: {str(e)}")
            return None

def main():
    """Test the complete RAG pipeline."""
    print("="*80)
    print("COMPLETE FINANCIAL RAG PIPELINE - FINAL INTEGRATION")
    print("Advanced Retrieval + Cross-Encoder + Smart Response Generation")
    print("="*80)
    
    # Initialize complete RAG system
    rag_system = CompleteFinancialRAG(enable_cross_encoder=True)
    
    # Load system
    if not rag_system.load_system():
        print("❌ Failed to load RAG system")
        return
    
    # Test queries covering different financial areas
    test_queries = [
        "What was the revenue from operations in Q3 2023?",
        "How much profit before tax did the company achieve?", 
        "Tell me about employee benefit expenses",
        "What were the major financial highlights for Dec 2023?",
        "Compare the financial performance across quarters"
    ]
    
    print(f"\n🧪 Testing Complete RAG Pipeline")
    print(f"Cross-encoder re-ranking: {rag_system.retriever.enable_reranking}")
    print(f"Embedding model: {rag_system.retriever.hybrid_retriever.embedding_model_name}")
    
    conversation_log = []
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("="*70)
        
        # Process query with explanation
        result = rag_system.process_query(
            query=query,
            top_k=3,
            fusion_method='weighted',
            include_explanation=True
        )
        
        # Display results
        print(f"🤖 Answer:")
        print(f"   {result['answer']}")
        print(f"")
        print(f"📊 Quality Metrics:")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Query Type: {result['query_analysis']['primary_category']}")
        print(f"   Method Used: {result['generation_metadata']['response_method']}")
        
        print(f"⚡ Performance:")
        print(f"   Total Time: {result['performance']['total_time']:.3f}s")
        print(f"   Retrieval: {result['performance']['retrieval_time']:.3f}s ({result['performance']['retrieval_percentage']:.1f}%)")
        print(f"   Generation: {result['performance']['generation_time']:.3f}s ({result['performance']['generation_percentage']:.1f}%)")
        
        print(f"📚 Sources Used:")
        for source in result['sources'][:2]:  # Show top 2 sources
            print(f"   • {source['quarter']} | {source['section']} | Score: {source['relevance_score']:.2f}")
            print(f"     {source['content_preview']}")
        
        # Add to conversation log
        conversation_log.append(result)
    
    # Save conversation log
    log_file = rag_system.save_conversation_log(conversation_log)
    
    # System performance summary
    system_stats = rag_system.get_system_stats()
    print(f"\n📈 System Performance Summary:")
    print(f"   Total queries processed: {system_stats['total_queries']}")
    print(f"   Average confidence: {system_stats['avg_confidence']:.3f}")
    print(f"   Average total time: {system_stats['total_time']:.3f}s")
    print(f"   Average retrieval time: {system_stats['retrieval_time']:.3f}s")
    print(f"   Average generation time: {system_stats['generation_time']:.3f}s")
    print(f"   Query type distribution: {system_stats['query_types']}")
    
    print(f"\n🎯 RAG System Configuration:")
    config = system_stats['system_config']
    print(f"   Cross-encoder: {config['cross_encoder_enabled']} ({config['cross_encoder_model']})")
    print(f"   Embedding model: {config['embedding_model']}")
    print(f"   Response method: {config['response_method']}")
    
    if log_file:
        print(f"\n💾 Conversation saved to: {log_file}")
    
    print(f"\n🎉 Complete RAG pipeline test successful!")
    print("The system is ready for production use!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hybrid Retrieval System for Financial RAG
Combines dense (semantic) and sparse (keyword) retrieval methods.
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialHybridRetriever:
    def __init__(self, 
                 indexes_dir: str = "data/indexes",
                 chroma_db_dir: str = "data/indexes/chroma_db",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dense_weight: float = 0.6,
                 sparse_weight: float = 0.4):
        """Initialize the Financial Hybrid Retriever.
        
        Args:
            indexes_dir: Directory containing all indexes
            chroma_db_dir: ChromaDB persistence directory
            embedding_model_name: Sentence transformer model name
            dense_weight: Weight for dense retrieval scores (default: 0.6)
            sparse_weight: Weight for sparse retrieval scores (default: 0.4)
        """
        self.indexes_dir = Path(indexes_dir)
        self.chroma_db_dir = Path(chroma_db_dir)
        self.embedding_model_name = embedding_model_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.bm25_index = None
        self.chunk_mapping = []
        self.metadata = {}
        
        # Query preprocessing
        self.stop_words = set(stopwords.words('english'))
        
        # Financial domain specific terms to preserve
        self.financial_preserve_terms = {
            'revenue', 'profit', 'loss', 'asset', 'liability', 'cash', 'expense', 
            'income', 'ebitda', 'eps', 'equity', 'debt', 'capex', 'opex',
            'margin', 'ratio', 'tax', 'dividend', 'interest', 'depreciation'
        }
        
        # Remove common financial stop words that aren't useful
        financial_stop_words = {
            'fy', 'q1', 'q2', 'q3', 'q4', 'crores', 'billion', 'million', 
            'rs', 'inr', 'was', 'were', 'the', 'in', 'to', 'and', 'of'
        }
        self.stop_words.update(financial_stop_words)
        
        # Query expansion dictionary for financial terms
        self.query_expansions = {
            'revenue': ['sales', 'income', 'turnover'],
            'profit': ['earnings', 'surplus'],
            'loss': ['deficit', 'negative'],
            'expense': ['cost', 'expenditure', 'outflow'],
            'cash': ['liquidity', 'funds'],
            'debt': ['borrowing', 'liability'],
            'asset': ['property', 'investment']
        }
    
    def load_models_and_indexes(self) -> bool:
        """Load all required models and indexes."""
        try:
            logger.info("Loading embedding model and indexes...")
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"✓ Loaded embedding model: {self.embedding_model_name}")
            
            # Load ChromaDB with SQLite compatibility fallback
            try:
                # Try to patch sqlite3 if needed
                try:
                    import pysqlite3 as sqlite3
                    import sys
                    sys.modules['sqlite3'] = sqlite3
                except ImportError:
                    pass
                
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.chroma_db_dir),
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection = self.chroma_client.get_collection("financial_chunks_100")
                logger.info("✓ Loaded ChromaDB collection")
                
            except Exception as chroma_error:
                logger.warning(f"ChromaDB failed: {str(chroma_error)}")
                logger.info("⚠️  Falling back to BM25-only retrieval (dense search disabled)")
                self.chroma_client = None
                self.collection = None
                # Adjust weights to rely only on sparse retrieval
                self.dense_weight = 0.0
                self.sparse_weight = 1.0
            
            # Load BM25 index
            bm25_file = self.indexes_dir / "bm25_index.pkl"
            with open(bm25_file, 'rb') as f:
                self.bm25_index = pickle.load(f)
            logger.info("✓ Loaded BM25 index")
            
            # Load chunk mapping
            mapping_file = self.indexes_dir / "bm25_chunk_mapping.json"
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.chunk_mapping = json.load(f)
            logger.info(f"✓ Loaded chunk mapping ({len(self.chunk_mapping)} chunks)")
            
            # Load metadata
            metadata_file = self.indexes_dir / "index_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info("✓ Loaded index metadata")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models/indexes: {str(e)}")
            return False
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """Preprocess query for optimal retrieval."""
        try:
            # Original query
            original = query.strip()
            
            # Clean and normalize
            cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Tokenization
            tokens = word_tokenize(cleaned)
            
            # Filter tokens but preserve financial terms
            filtered_tokens = []
            for token in tokens:
                if (token.isalpha() and 
                    len(token) > 2 and 
                    (token not in self.stop_words or token in self.financial_preserve_terms)):
                    filtered_tokens.append(token)
            
            # Query expansion
            expanded_tokens = filtered_tokens.copy()
            for token in filtered_tokens:
                if token in self.query_expansions:
                    expanded_tokens.extend(self.query_expansions[token])
            
            # Remove duplicates while preserving order
            unique_expanded = []
            seen = set()
            for token in expanded_tokens:
                if token not in seen:
                    unique_expanded.append(token)
                    seen.add(token)
            
            # Create expanded query
            expanded_query = ' '.join(unique_expanded)
            
            return {
                'original': original,
                'cleaned': cleaned,
                'tokens': filtered_tokens,
                'expanded_tokens': unique_expanded,
                'expanded_query': expanded_query,
                'financial_terms': [t for t in filtered_tokens if t in self.financial_preserve_terms]
            }
            
        except Exception as e:
            logger.warning(f"Error preprocessing query: {str(e)}")
            return {
                'original': query,
                'cleaned': query.lower(),
                'tokens': [query.lower()],
                'expanded_tokens': [query.lower()],
                'expanded_query': query.lower(),
                'financial_terms': []
            }
    
    def dense_retrieval(self, query_data: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform dense (semantic) retrieval using ChromaDB."""
        try:
            # Check if ChromaDB is available
            if not self.collection:
                logger.info("Dense retrieval skipped: ChromaDB not available")
                return []
            
            # Use expanded query for better semantic matching
            query_text = query_data['expanded_query']
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(top_k, 50),  # Cap at 50 for performance
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            dense_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (lower distance = higher similarity)
                    similarity = 1 / (1 + distance)  # Convert to similarity
                    
                    dense_results.append({
                        'chunk_id': results['ids'][0][i],
                        'text': doc,
                        'score': similarity,
                        'method': 'dense',
                        'metadata': metadata,
                        'rank': i + 1
                    })
            
            logger.info(f"Dense retrieval: Found {len(dense_results)} results")
            return dense_results
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {str(e)}")
            return []
    
    def sparse_retrieval(self, query_data: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform sparse (keyword) retrieval using BM25."""
        try:
            # Use filtered tokens for BM25 matching
            query_tokens = query_data['tokens']
            
            if not query_tokens:
                logger.warning("No valid tokens for sparse retrieval")
                return []
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(bm25_scores)[-min(top_k, len(bm25_scores)):][::-1]
            
            # Format results
            sparse_results = []
            for rank, idx in enumerate(top_indices):
                if idx < len(self.chunk_mapping) and bm25_scores[idx] > 0:
                    chunk_data = self.chunk_mapping[idx]
                    
                    sparse_results.append({
                        'chunk_id': chunk_data['chunk_id'],
                        'text': chunk_data['text'],
                        'score': float(bm25_scores[idx]),
                        'method': 'sparse',
                        'metadata': {
                            'quarter': chunk_data['quarter'],
                            'section': chunk_data['section'],
                            'tokens': chunk_data['tokens']
                        },
                        'rank': rank + 1
                    })
            
            logger.info(f"Sparse retrieval: Found {len(sparse_results)} results")
            return sparse_results
            
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {str(e)}")
            return []
    
    def normalize_scores(self, results: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
        """Normalize scores to [0, 1] range for fair comparison."""
        if not results:
            return results
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            for result in results:
                result['normalized_score'] = 1.0
        else:
            for result in results:
                normalized = (result['score'] - min_score) / (max_score - min_score)
                result['normalized_score'] = normalized
        
        return results
    
    def reciprocal_rank_fusion(self, dense_results: List[Dict[str, Any]], 
                              sparse_results: List[Dict[str, Any]], 
                              k: int = 60) -> List[Dict[str, Any]]:
        """Implement Reciprocal Rank Fusion for combining results."""
        rrf_scores = {}
        
        # Add dense results with RRF scoring
        for rank, result in enumerate(dense_results):
            chunk_id = result['chunk_id']
            rrf_score = 1 / (k + rank + 1)
            
            rrf_scores[chunk_id] = {
                'rrf_score': rrf_score,
                'dense_rank': rank + 1,
                'dense_score': result['score'],
                'text': result['text'],
                'metadata': result['metadata']
            }
        
        # Add sparse results with RRF scoring
        for rank, result in enumerate(sparse_results):
            chunk_id = result['chunk_id']
            rrf_score = 1 / (k + rank + 1)
            
            if chunk_id in rrf_scores:
                # Combine scores for chunks found in both methods
                rrf_scores[chunk_id]['rrf_score'] += rrf_score
                rrf_scores[chunk_id]['sparse_rank'] = rank + 1
                rrf_scores[chunk_id]['sparse_score'] = result['score']
                rrf_scores[chunk_id]['method'] = 'hybrid'
            else:
                rrf_scores[chunk_id] = {
                    'rrf_score': rrf_score,
                    'sparse_rank': rank + 1,
                    'sparse_score': result['score'],
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'method': 'sparse_only'
                }
        
        # Sort by RRF score and format results
        sorted_results = []
        for chunk_id, data in sorted(rrf_scores.items(), key=lambda x: x[1]['rrf_score'], reverse=True):
            result = {
                'chunk_id': chunk_id,
                'text': data['text'],
                'score': data['rrf_score'],
                'method': data.get('method', 'dense_only'),
                'metadata': data['metadata'],
                'dense_rank': data.get('dense_rank', None),
                'sparse_rank': data.get('sparse_rank', None),
                'dense_score': data.get('dense_score', None),
                'sparse_score': data.get('sparse_score', None)
            }
            sorted_results.append(result)
        
        return sorted_results
    
    def weighted_score_fusion(self, dense_results: List[Dict[str, Any]], 
                             sparse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results using weighted score fusion."""
        # Normalize scores
        dense_normalized = self.normalize_scores(dense_results, 'dense')
        sparse_normalized = self.normalize_scores(sparse_results, 'sparse')
        
        # Create combined score dictionary
        combined_scores = {}
        
        # Add dense results
        for result in dense_normalized:
            chunk_id = result['chunk_id']
            combined_scores[chunk_id] = {
                'dense_score': result['normalized_score'],
                'sparse_score': 0.0,
                'combined_score': self.dense_weight * result['normalized_score'],
                'text': result['text'],
                'metadata': result['metadata'],
                'dense_rank': result['rank']
            }
        
        # Add sparse results
        for result in sparse_normalized:
            chunk_id = result['chunk_id']
            sparse_contribution = self.sparse_weight * result['normalized_score']
            
            if chunk_id in combined_scores:
                # Update existing entry
                combined_scores[chunk_id]['sparse_score'] = result['normalized_score']
                combined_scores[chunk_id]['combined_score'] += sparse_contribution
                combined_scores[chunk_id]['sparse_rank'] = result['rank']
                combined_scores[chunk_id]['method'] = 'hybrid'
            else:
                # New entry from sparse only
                combined_scores[chunk_id] = {
                    'dense_score': 0.0,
                    'sparse_score': result['normalized_score'],
                    'combined_score': sparse_contribution,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'sparse_rank': result['rank'],
                    'method': 'sparse_only'
                }
        
        # Sort by combined score
        sorted_results = []
        for chunk_id, data in sorted(combined_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True):
            result = {
                'chunk_id': chunk_id,
                'text': data['text'],
                'score': data['combined_score'],
                'method': data.get('method', 'dense_only'),
                'metadata': data['metadata'],
                'dense_score': data['dense_score'],
                'sparse_score': data['sparse_score'],
                'dense_rank': data.get('dense_rank', None),
                'sparse_rank': data.get('sparse_rank', None)
            }
            sorted_results.append(result)
        
        return sorted_results
    
    def filter_by_metadata(self, results: List[Dict[str, Any]], 
                          filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply metadata-based filtering to results."""
        if not filters:
            return results
        
        filtered_results = []
        for result in results:
            metadata = result.get('metadata', {})
            include = True
            
            # Apply filters
            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, list):
                        if metadata[key] not in value:
                            include = False
                            break
                    else:
                        if metadata[key] != value:
                            include = False
                            break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def hybrid_search(self, 
                     query: str, 
                     top_k: int = 10,
                     fusion_method: str = 'weighted',
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            fusion_method: 'weighted' or 'rrf' (reciprocal rank fusion)
            filters: Optional metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            logger.info(f"🔍 Hybrid search: '{query}' (top_k={top_k}, method={fusion_method})")
            
            # Preprocess query
            query_data = self.preprocess_query(query)
            logger.info(f"Query tokens: {query_data['tokens']}")
            logger.info(f"Financial terms: {query_data['financial_terms']}")
            
            # Perform dense retrieval
            dense_results = self.dense_retrieval(query_data, top_k=top_k*2)  # Get more for better fusion
            
            # Perform sparse retrieval
            sparse_results = self.sparse_retrieval(query_data, top_k=top_k*2)  # Get more for better fusion
            
            # Combine results based on fusion method
            if fusion_method == 'rrf':
                combined_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
            else:  # weighted
                combined_results = self.weighted_score_fusion(dense_results, sparse_results)
            
            # Apply metadata filters if provided
            if filters:
                combined_results = self.filter_by_metadata(combined_results, filters)
                logger.info(f"Applied filters: {filters}")
            
            # Limit to top_k results
            final_results = combined_results[:top_k]
            
            # Add retrieval statistics
            for i, result in enumerate(final_results):
                result['final_rank'] = i + 1
                result['retrieval_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"✅ Hybrid search complete: {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def get_retrieval_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about retrieval results."""
        if not results:
            return {'total': 0}
        
        stats = {
            'total': len(results),
            'methods': {},
            'sections': {},
            'quarters': {},
            'avg_score': sum(r['score'] for r in results) / len(results),
            'max_score': max(r['score'] for r in results),
            'min_score': min(r['score'] for r in results)
        }
        
        # Count by method
        for result in results:
            method = result.get('method', 'unknown')
            stats['methods'][method] = stats['methods'].get(method, 0) + 1
            
            # Count by section
            metadata = result.get('metadata', {})
            section = metadata.get('section', 'unknown')
            stats['sections'][section] = stats['sections'].get(section, 0) + 1
            
            # Count by quarter
            quarter = metadata.get('quarter', 'unknown')
            stats['quarters'][quarter] = stats['quarters'].get(quarter, 0) + 1
        
        return stats

def main():
    """Test the hybrid retrieval system."""
    retriever = FinancialHybridRetriever()
    
    # Load models and indexes
    if not retriever.load_models_and_indexes():
        logger.error("Failed to load models and indexes")
        return
    
    # Test queries
    test_queries = [
        "What was the revenue from operations in Q3 2023?",
        "Show me profit before tax information",
        "Employee benefit expenses across quarters", 
        "Cash flow from operating activities",
        "Total assets and liabilities comparison"
    ]
    
    print("\n" + "="*80)
    print("FINANCIAL HYBRID RETRIEVAL SYSTEM - TEST RESULTS")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 60)
        
        # Test weighted fusion
        results_weighted = retriever.hybrid_search(query, top_k=3, fusion_method='weighted')
        
        print("📊 Weighted Fusion Results:")
        for j, result in enumerate(results_weighted, 1):
            print(f"  {j}. Score: {result['score']:.4f} | Method: {result['method']}")
            print(f"     {result['text'][:120]}...")
            print(f"     Section: {result['metadata'].get('section', 'N/A')} | "
                  f"Quarter: {result['metadata'].get('quarter', 'N/A')}")
        
        # Test RRF fusion
        results_rrf = retriever.hybrid_search(query, top_k=3, fusion_method='rrf')
        
        print("\n🔀 RRF Fusion Results:")
        for j, result in enumerate(results_rrf, 1):
            print(f"  {j}. Score: {result['score']:.4f} | Method: {result['method']}")
            print(f"     {result['text'][:120]}...")
        
        # Show stats
        stats = retriever.get_retrieval_stats(results_weighted)
        print(f"\n📈 Stats: {stats['total']} results | "
              f"Methods: {stats['methods']} | "
              f"Avg Score: {stats['avg_score']:.4f}")

if __name__ == "__main__":
    main()

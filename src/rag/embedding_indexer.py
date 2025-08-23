#!/usr/bin/env python3
"""
Embedding & Indexing System for Financial RAG
Creates dense and sparse indexes from processed chunks.
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialEmbeddingIndexer:
    def __init__(self, 
                 chunks_file: str = "data/chunks/financial_chunks_100_tokens.json",
                 indexes_dir: str = "data/indexes",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chroma_db_dir: str = "data/indexes/chroma_db"):
        """Initialize the Financial Embedding Indexer.
        
        Args:
            chunks_file: Path to the processed chunks JSON file
            indexes_dir: Directory to save all indexes
            embedding_model_name: Name of the sentence transformer model
            chroma_db_dir: Directory for ChromaDB persistence
        """
        self.chunks_file = Path(chunks_file)
        self.indexes_dir = Path(indexes_dir)
        self.chroma_db_dir = Path(chroma_db_dir)
        self.embedding_model_name = embedding_model_name
        
        # Create directories
        self.indexes_dir.mkdir(exist_ok=True)
        self.chroma_db_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.chunks = []
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.bm25_index = None
        self.tokenized_chunks = []
        
        # Stop words for BM25
        self.stop_words = set(stopwords.words('english'))
        
        # Add financial-specific stop words to remove
        financial_stop_words = {
            'fy', 'q1', 'q2', 'q3', 'q4', 'crores', 'billion', 'million', 
            'rs', 'inr', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
            '2022', '2023', '2024', 'was', 'were', 'the', 'in', 'to', 'and', 'of'
        }
        self.stop_words.update(financial_stop_words)
    
    def load_chunks(self) -> bool:
        """Load processed chunks from JSON file."""
        try:
            logger.info(f"Loading chunks from {self.chunks_file}...")
            
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            logger.info(f"✓ Loaded {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading chunks: {str(e)}")
            return False
    
    def initialize_embedding_model(self) -> bool:
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}...")
            
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Test the model
            test_embedding = self.embedding_model.encode(["Test sentence"])
            logger.info(f"✓ Model loaded successfully. Embedding dimension: {test_embedding.shape[1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            return False
    
    def initialize_chromadb(self) -> bool:
        """Initialize ChromaDB client and collection."""
        try:
            logger.info("Initializing ChromaDB...")
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_db_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            collection_name = "financial_chunks_100"
            
            # Delete existing collection if it exists (for clean restart)
            try:
                self.chroma_client.delete_collection(collection_name)
                logger.info("Deleted existing collection for fresh start")
            except:
                pass  # Collection doesn't exist, which is fine
            
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Financial chunks with 100 tokens for RAG system"}
            )
            
            logger.info(f"✓ ChromaDB collection '{collection_name}' created")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            return False
    
    def preprocess_text_for_bm25(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing."""
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Filter tokens
            filtered_tokens = [
                token for token in tokens 
                if token.isalpha() and 
                len(token) > 2 and 
                token not in self.stop_words
            ]
            
            return filtered_tokens
            
        except Exception as e:
            logger.warning(f"Error preprocessing text: {str(e)}")
            return []
    
    def create_dense_embeddings(self) -> bool:
        """Create dense embeddings and store in ChromaDB."""
        try:
            logger.info("Creating dense embeddings...")
            
            if not self.chunks:
                logger.error("No chunks loaded!")
                return False
            
            # Prepare data for ChromaDB
            chunk_texts = []
            chunk_ids = []
            chunk_metadatas = []
            
            for chunk in self.chunks:
                chunk_texts.append(chunk['text'])
                chunk_ids.append(chunk['chunk_id'])
                
                # Prepare metadata (ChromaDB doesn't support nested dicts)
                metadata = {
                    'quarter': chunk['quarter'],
                    'fiscal_year': chunk['fiscal_year'],
                    'calendar_year': chunk['calendar_year'],
                    'section': chunk['section'],
                    'source_file': chunk['source_file'],
                    'tokens': chunk['tokens'],
                    'chunk_index': chunk['chunk_index']
                }
                
                # Add company info if available
                if 'metadata' in chunk and 'company' in chunk['metadata']:
                    metadata['company'] = chunk['metadata']['company']
                
                chunk_metadatas.append(metadata)
            
            # Generate embeddings in batches for memory efficiency
            batch_size = 100
            total_batches = len(chunk_texts) // batch_size + (1 if len(chunk_texts) % batch_size else 0)
            
            for i in range(0, len(chunk_texts), batch_size):
                end_idx = min(i + batch_size, len(chunk_texts))
                batch_texts = chunk_texts[i:end_idx]
                batch_ids = chunk_ids[i:end_idx]
                batch_metadatas = chunk_metadatas[i:end_idx]
                
                # Generate embeddings for batch
                batch_embeddings = self.embedding_model.encode(
                    batch_texts, 
                    show_progress_bar=True,
                    batch_size=32
                ).tolist()  # Convert to list for ChromaDB
                
                # Add to ChromaDB
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                batch_num = (i // batch_size) + 1
                logger.info(f"✓ Processed batch {batch_num}/{total_batches}")
            
            logger.info(f"✅ Successfully created dense embeddings for {len(chunk_texts)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating dense embeddings: {str(e)}")
            return False
    
    def create_sparse_index(self) -> bool:
        """Create BM25 sparse index."""
        try:
            logger.info("Creating BM25 sparse index...")
            
            if not self.chunks:
                logger.error("No chunks loaded!")
                return False
            
            # Preprocess all chunks for BM25
            self.tokenized_chunks = []
            chunk_mapping = []  # Maps BM25 index to chunk data
            
            for chunk in self.chunks:
                tokenized_text = self.preprocess_text_for_bm25(chunk['text'])
                
                # Only add non-empty tokenized texts (must have at least 1 token)
                if tokenized_text and len(tokenized_text) > 0:
                    self.tokenized_chunks.append(tokenized_text)
                    chunk_mapping.append({
                        'chunk_id': chunk['chunk_id'],
                        'text': chunk['text'],
                        'quarter': chunk['quarter'],
                        'section': chunk['section'],
                        'tokens': chunk['tokens']
                    })
            
            # Create BM25 index (check for empty chunks)
            if not self.tokenized_chunks:
                logger.error("No tokenized chunks available for BM25 index!")
                return False
            
            # Filter out any remaining empty chunks and corresponding mappings  
            valid_chunks = []
            valid_mapping = []
            
            for i, chunk in enumerate(self.tokenized_chunks):
                if chunk and len(chunk) > 0:
                    valid_chunks.append(chunk)
                    valid_mapping.append(chunk_mapping[i])
            
            if not valid_chunks:
                logger.error("No valid tokenized chunks after final filtering!")
                return False
            
            self.bm25_index = BM25Okapi(valid_chunks)
            self.tokenized_chunks = valid_chunks
            chunk_mapping = valid_mapping
            
            # Save BM25 index and mapping
            bm25_file = self.indexes_dir / "bm25_index.pkl"
            mapping_file = self.indexes_dir / "bm25_chunk_mapping.json"
            
            with open(bm25_file, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_mapping, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Created BM25 index for {len(self.tokenized_chunks)} chunks")
            logger.info(f"✓ Saved BM25 index to {bm25_file}")
            logger.info(f"✓ Saved chunk mapping to {mapping_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating sparse index: {str(e)}")
            return False
    
    def test_indexes(self) -> bool:
        """Test both dense and sparse indexes with sample queries."""
        try:
            logger.info("Testing indexes with sample queries...")
            
            test_queries = [
                "revenue from operations",
                "profit before tax", 
                "employee benefit expense",
                "cash flow from operating activities"
            ]
            
            for query in test_queries:
                logger.info(f"\n🔍 Testing query: '{query}'")
                
                # Test dense retrieval
                try:
                    dense_results = self.collection.query(
                        query_texts=[query],
                        n_results=3
                    )
                    
                    logger.info(f"Dense retrieval: Found {len(dense_results['documents'][0])} results")
                    if dense_results['documents'][0]:
                        logger.info(f"  Top result: {dense_results['documents'][0][0][:100]}...")
                
                except Exception as e:
                    logger.warning(f"Dense retrieval test failed: {str(e)}")
                
                # Test sparse retrieval
                try:
                    query_tokens = self.preprocess_text_for_bm25(query)
                    if query_tokens:
                        bm25_scores = self.bm25_index.get_scores(query_tokens)
                        top_indices = np.argsort(bm25_scores)[-3:][::-1]  # Top 3
                        
                        top_scores = [f"{bm25_scores[i]:.3f}" for i in top_indices]
                        logger.info(f"Sparse retrieval: Top 3 scores: {top_scores}")
                        
                        # Load chunk mapping for details
                        mapping_file = self.indexes_dir / "bm25_chunk_mapping.json"
                        with open(mapping_file, 'r') as f:
                            chunk_mapping = json.load(f)
                        
                        if top_indices[0] < len(chunk_mapping):
                            top_chunk = chunk_mapping[top_indices[0]]
                            logger.info(f"  Top result: {top_chunk['text'][:100]}...")
                
                except Exception as e:
                    logger.warning(f"Sparse retrieval test failed: {str(e)}")
            
            logger.info("✅ Index testing completed")
            return True
            
        except Exception as e:
            logger.error(f"Error testing indexes: {str(e)}")
            return False
    
    def save_index_metadata(self) -> bool:
        """Save metadata about the created indexes."""
        try:
            metadata = {
                'creation_timestamp': datetime.now().isoformat(),
                'embedding_model': self.embedding_model_name,
                'total_chunks': len(self.chunks),
                'dense_index': {
                    'collection_name': 'financial_chunks_100',
                    'embedding_dimension': 384,
                    'storage_type': 'ChromaDB'
                },
                'sparse_index': {
                    'algorithm': 'BM25Okapi',
                    'tokenized_chunks': len(self.tokenized_chunks),
                    'preprocessing': 'stopwords_removed_financial_terms_filtered'
                },
                'chunk_statistics': {
                    'sections': list(set(chunk['section'] for chunk in self.chunks)),
                    'quarters': list(set(chunk['quarter'] for chunk in self.chunks)),
                    'avg_tokens': sum(chunk['tokens'] for chunk in self.chunks) / len(self.chunks)
                }
            }
            
            metadata_file = self.indexes_dir / "index_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Saved index metadata to {metadata_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            return False
    
    def build_all_indexes(self) -> bool:
        """Build complete indexing system."""
        logger.info("🚀 Starting complete indexing process...")
        
        # Step 1: Load chunks
        if not self.load_chunks():
            logger.error("Failed to load chunks")
            return False
        
        # Step 2: Initialize embedding model  
        if not self.initialize_embedding_model():
            logger.error("Failed to initialize embedding model")
            return False
        
        # Step 3: Initialize ChromaDB
        if not self.initialize_chromadb():
            logger.error("Failed to initialize ChromaDB")
            return False
        
        # Step 4: Create dense embeddings
        if not self.create_dense_embeddings():
            logger.error("Failed to create dense embeddings")
            return False
        
        # Step 5: Create sparse index
        if not self.create_sparse_index():
            logger.error("Failed to create sparse index")
            return False
        
        # Step 6: Test indexes
        if not self.test_indexes():
            logger.warning("Index testing had issues, but continuing...")
        
        # Step 7: Save metadata
        if not self.save_index_metadata():
            logger.warning("Failed to save metadata, but indexes are created")
        
        logger.info("🎉 Indexing process completed successfully!")
        return True
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about created indexes."""
        try:
            metadata_file = self.indexes_dir / "index_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"error": "Index metadata not found"}
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main function to build the indexing system."""
    indexer = FinancialEmbeddingIndexer()
    
    success = indexer.build_all_indexes()
    
    if success:
        print("\n" + "="*60)
        print("FINANCIAL EMBEDDING & INDEXING RESULTS")
        print("="*60)
        
        # Display results
        info = indexer.get_index_info()
        if "error" not in info:
            print(f"✅ Total chunks indexed: {info.get('total_chunks', 'Unknown')}")
            print(f"📊 Embedding model: {info.get('embedding_model', 'Unknown')}")
            print(f"🗂️  Sections covered: {len(info.get('chunk_statistics', {}).get('sections', []))}")
            print(f"📅 Quarters covered: {len(info.get('chunk_statistics', {}).get('quarters', []))}")
            print(f"🔤 Average tokens per chunk: {info.get('chunk_statistics', {}).get('avg_tokens', 0):.1f}")
            
            dense_info = info.get('dense_index', {})
            print(f"\n📈 Dense Index (ChromaDB):")
            print(f"   Collection: {dense_info.get('collection_name', 'Unknown')}")
            print(f"   Dimensions: {dense_info.get('embedding_dimension', 'Unknown')}")
            
            sparse_info = info.get('sparse_index', {})
            print(f"\n📋 Sparse Index (BM25):")
            print(f"   Algorithm: {sparse_info.get('algorithm', 'Unknown')}")
            print(f"   Tokenized chunks: {sparse_info.get('tokenized_chunks', 'Unknown')}")
            
            print(f"\n🎯 Indexes are ready for hybrid retrieval!")
        else:
            print("❌ Error retrieving index information")
    else:
        print("❌ Indexing process failed!")

if __name__ == "__main__":
    main()

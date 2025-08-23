#!/usr/bin/env python3
"""
Financial RAG System - Streamlit Cloud Version
Optimized for cloud deployment with memory and resource constraints
"""

import streamlit as st
import pandas as pd
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Financial RAG System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main > div {padding-top: 2rem;}
.stAlert > div {padding: 1rem; margin: 0.5rem 0;}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.confidence-high {color: #28a745; font-weight: bold;}
.confidence-medium {color: #ffc107; font-weight: bold;}
.confidence-low {color: #dc3545; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

class CloudFinancialRAG:
    """Lightweight Financial RAG system optimized for Streamlit Cloud."""
    
    def __init__(self):
        self.qa_data = None
        self.spell_corrections = {
            'revunue': 'revenue', 'reveune': 'revenue', 'revanue': 'revenue',
            'proffit': 'profit', 'prfit': 'profit',
            'expences': 'expenses', 'expence': 'expenses',
            'assests': 'assets', 'aseets': 'assets',
        }
        
        # Performance stats
        self.stats = {
            'queries_processed': 0,
            'avg_response_time': 0.0,
            'corrections_made': 0
        }
    
    @st.cache_data
    def load_qa_data(_self):
        """Load Q&A data with caching for cloud performance."""
        try:
            # Try to load from different possible locations
            possible_paths = [
                'data/processed/xbrl_qa_pairs.json',
                'data/processed/qa_pairs_compressed.json',
                'xbrl_qa_pairs.json'
            ]
            
            qa_data = None
            for path in possible_paths:
                try:
                    with open(path, 'r') as f:
                        qa_data = json.load(f)
                    st.success(f"✅ Loaded {len(qa_data)} Q&A pairs from {path}")
                    break
                except FileNotFoundError:
                    continue
            
            if qa_data is None:
                # Fallback: Create sample data
                qa_data = _self.create_sample_qa_data()
                st.warning("⚠️ Using sample data - Q&A file not found")
            
            return qa_data
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return _self.create_sample_qa_data()
    
    def create_sample_qa_data(self):
        """Create sample Q&A data for demonstration."""
        return [
            {
                "question": "What was the revenue from operations in Dec 2023?",
                "answer": "The revenue from operations was ₹15.03 billion",
                "quarter": "Dec 2023"
            },
            {
                "question": "What was the revenue from operations in Sep 2023?", 
                "answer": "The revenue from operations was ₹18.96 billion",
                "quarter": "Sep 2023"
            },
            {
                "question": "What was the profit before tax in Dec 2023?",
                "answer": "The loss before tax was ₹4.52 billion",
                "quarter": "Dec 2023"
            },
            {
                "question": "What was the employee benefit expense in Dec 2023?",
                "answer": "The employee benefit expense was ₹98.60 crores",
                "quarter": "Dec 2023"
            }
        ]
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """Simple spell correction and query enhancement."""
        enhanced_query = query.lower().strip()
        corrections = []
        
        # Apply spell corrections
        for incorrect, correct in self.spell_corrections.items():
            if incorrect in enhanced_query:
                enhanced_query = enhanced_query.replace(incorrect, correct)
                corrections.append({'original': incorrect, 'corrected': correct})
                self.stats['corrections_made'] += 1
        
        # Simple financial year conversion
        if '2023-04-01 to 2024-03-31' in enhanced_query:
            enhanced_query = enhanced_query.replace('2023-04-01 to 2024-03-31', 'FY 2023-24')
            corrections.append({'original': '2023-04-01 to 2024-03-31', 'corrected': 'FY 2023-24'})
        
        return {
            'enhanced_query': enhanced_query,
            'corrections_applied': corrections,
            'improvement_count': len(corrections)
        }
    
    def simple_similarity_search(self, query: str, qa_data: List[Dict], top_k: int = 5) -> List[Dict]:
        """Simple similarity search without heavy NLP models."""
        query_words = set(query.lower().split())
        results = []
        
        for qa in qa_data:
            question = qa['question'].lower()
            question_words = set(question.split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(question_words))
            total_words = len(query_words.union(question_words))
            similarity = overlap / total_words if total_words > 0 else 0
            
            # Boost for exact financial term matches
            financial_terms = ['revenue', 'profit', 'loss', 'expense', 'asset', 'liability']
            for term in financial_terms:
                if term in query.lower() and term in question:
                    similarity += 0.3
            
            # Boost for period matches
            if any(period in query.lower() and period in question for period in ['dec', 'sep', 'jun', 'mar', '2023', '2022']):
                similarity += 0.2
            
            if similarity > 0.1:  # Minimum threshold
                results.append({
                    'qa': qa,
                    'similarity': similarity,
                    'source': f"Q&A Database - {qa['quarter']}"
                })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def generate_response(self, query: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Generate response from search results."""
        start_time = time.time()
        
        if not search_results:
            return {
                'answer': "I don't have specific information about that in the available financial data.",
                'confidence': 0.2,
                'method': 'fallback',
                'sources': []
            }
        
        # Use the best match
        best_result = search_results[0]
        qa_pair = best_result['qa']
        
        # Calculate confidence based on similarity
        confidence = min(0.95, best_result['similarity'] * 1.5)
        
        # Format response
        answer = qa_pair['answer']
        if qa_pair['quarter'] not in answer:
            answer += f" (Data from {qa_pair['quarter']})"
        
        processing_time = time.time() - start_time
        
        return {
            'answer': answer,
            'confidence': confidence,
            'method': 'direct_match' if confidence > 0.8 else 'similarity_match',
            'sources': [best_result],
            'processing_time': processing_time
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main query processing pipeline."""
        start_time = time.time()
        
        # Load data
        if self.qa_data is None:
            self.qa_data = self.load_qa_data()
        
        # Step 1: Query enhancement
        enhancement = self.enhance_query(query)
        enhanced_query = enhancement['enhanced_query']
        
        # Step 2: Search
        search_results = self.simple_similarity_search(enhanced_query, self.qa_data)
        
        # Step 3: Generate response
        response = self.generate_response(enhanced_query, search_results)
        
        # Update stats
        total_time = time.time() - start_time
        self.stats['queries_processed'] += 1
        self.stats['avg_response_time'] = (
            (self.stats['avg_response_time'] * (self.stats['queries_processed'] - 1) + total_time) / 
            self.stats['queries_processed']
        )
        
        return {
            **response,
            'query_enhancement': enhancement,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat()
        }

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    """Initialize and cache the RAG system."""
    return CloudFinancialRAG()

def format_confidence_badge(confidence):
    """Format confidence score with color coding."""
    if confidence >= 0.8:
        return f'<span class="confidence-high">High ({confidence:.2f})</span>'
    elif confidence >= 0.6:
        return f'<span class="confidence-medium">Medium ({confidence:.2f})</span>'
    else:
        return f'<span class="confidence-low">Low ({confidence:.2f})</span>'

def main():
    """Main application interface."""
    
    # Sidebar
    st.sidebar.title("💰 Financial RAG System")
    st.sidebar.markdown("**Cloud-Optimized Version**")
    
    # System status
    rag_system = get_rag_system()
    st.sidebar.success("✅ System Online")
    st.sidebar.info("🚀 Deployed on Streamlit Cloud")
    
    # Main interface
    st.header("💰 Financial Question Answering")
    st.markdown("Ask questions about financial performance and get AI-powered answers.")
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "💬 Your Financial Question:",
            placeholder="What was the revenue from operations in Sep 2023?",
            help="Try asking about revenue, profit, expenses, or other financial metrics."
        )
    
    with col2:
        st.markdown("&nbsp;")
        ask_button = st.button("🔍 Get Answer", type="primary")
    
    # Sample queries
    st.markdown("### 🎯 Sample Queries")
    sample_queries = [
        "What was the revenue from operations in Sep 2023?",
        "How much profit did the company make in Dec 2023?", 
        "Tell me about employee benefit expenses",
        "What was the revenue in Q2 FY2023-24?"
    ]
    
    cols = st.columns(2)
    for i, sample_query in enumerate(sample_queries):
        col = cols[i % 2]
        if col.button(f"📝 {sample_query}", key=f"sample_{i}"):
            query = sample_query
            ask_button = True
    
    # Process query
    if ask_button and query:
        with st.spinner("🔍 Processing your financial query..."):
            result = rag_system.process_query(query)
        
        # Show enhancement if improvements were made
        enhancement = result.get('query_enhancement', {})
        if enhancement.get('improvement_count', 0) > 0:
            st.info(f"🔧 **Query Enhanced**: Applied {enhancement['improvement_count']} improvements")
            if enhancement.get('corrections_applied'):
                corrections = [f"'{c['original']}' → '{c['corrected']}'" for c in enhancement['corrections_applied']]
                st.success(f"✏️ **Corrections**: {', '.join(corrections)}")
        
        # Display answer
        st.markdown("### 🤖 Answer")
        confidence_html = format_confidence_badge(result['confidence'])
        st.markdown(f"**Confidence:** {confidence_html}", unsafe_allow_html=True)
        st.markdown(f"**{result['answer']}**")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⚡ Response Time", f"{result['total_time']:.3f}s")
        with col2:
            st.metric("🎯 Confidence", f"{result['confidence']:.2f}")
        with col3:
            st.metric("🔧 Method", result['method'])
        with col4:
            st.metric("📊 Sources", len(result.get('sources', [])))
        
        # Show sources
        sources = result.get('sources', [])
        if sources:
            st.markdown("### 📚 Information Sources")
            for i, source in enumerate(sources, 1):
                with st.expander(f"📄 Source {i}: {source['source']} (Similarity: {source['similarity']:.3f})"):
                    qa = source['qa']
                    st.markdown(f"**Question**: {qa['question']}")
                    st.markdown(f"**Answer**: {qa['answer']}")
                    st.markdown(f"**Quarter**: {qa['quarter']}")
    
    # System statistics
    st.markdown("---")
    st.markdown("### 📊 System Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Queries Processed", rag_system.stats['queries_processed'])
    with col2:
        st.metric("⚡ Avg Response Time", f"{rag_system.stats['avg_response_time']:.3f}s")
    with col3:
        st.metric("✏️ Corrections Made", rag_system.stats['corrections_made'])
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**🚀 Deployed on Streamlit Community Cloud** | "
        "**💰 Financial RAG System** | "
        f"**🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
    )

if __name__ == "__main__":
    main()

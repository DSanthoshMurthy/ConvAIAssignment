#!/usr/bin/env python3
"""
Financial RAG System - Streamlit Web Interface
Cloud-Optimized Production Deployment with Full RAG Features
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import os

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cloud-compatible import handling with NLTK setup
RAG_SYSTEM_AVAILABLE = True
IMPORT_ERROR_DETAILS = None
DATA_DIAGNOSTIC = {"error": "Not run yet"}

def setup_sqlite_for_chromadb():
    """Setup SQLite compatibility for ChromaDB"""
    try:
        import sys
        import sqlite3
        
        # Check if we need to patch sqlite3
        if sqlite3.sqlite_version_info < (3, 35, 0):
            try:
                import pysqlite3
                sys.modules['sqlite3'] = pysqlite3
                st.success("✅ Successfully patched SQLite with pysqlite3")
                return True
            except ImportError as e:
                st.error(f"❌ Failed to import pysqlite3. Please ensure pysqlite3-binary is installed: {str(e)}")
                return False
        else:
            st.success(f"✅ System SQLite version {sqlite3.sqlite_version} is compatible")
            return True
    except Exception as e:
        st.error(f"❌ SQLite setup failed: {str(e)}")
        return False

def setup_nltk_data():
    """Download required NLTK data if not present"""
    try:
        import nltk
        import os
        
        # Create NLTK data directory in a writable location
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        # List of required NLTK resources
        required_resources = [
            'punkt',  # Main tokenizer
            'stopwords',  # Stopwords
            'punkt_tab'  # Tab-separated punkt data
        ]
        
        # Download and verify each resource
        for resource_name in required_resources:
            try:
                nltk.download(resource_name, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"✓ NLTK resource downloaded/verified: {resource_name}")
            except Exception as e:
                logger.warning(f"Failed to download {resource_name}: {str(e)}")
                # Continue anyway - we have fallback tokenization
        
        # Basic verification - try tokenization
        try:
            from nltk.tokenize import word_tokenize
            test_tokens = word_tokenize("Testing NLTK tokenization.")
            logger.info("✅ NLTK tokenization test successful")
            return True
        except Exception as e:
            logger.warning(f"NLTK tokenization test failed: {str(e)}")
            logger.info("⚠️ Will use fallback tokenization")
            return True  # Return True to continue with fallback
        
    except Exception as e:
        error_msg = f"NLTK setup failed: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return False

try:
    # Setup SQLite compatibility for ChromaDB
    setup_sqlite_for_chromadb()
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Run data loading diagnostic
    try:
        from src.rag.data_loader_diagnostic import DataLoadingDiagnostic
        diagnostic = DataLoadingDiagnostic()
        diagnostic_results = diagnostic.run_complete_diagnostic()
        DATA_DIAGNOSTIC = diagnostic_results
    except Exception as e:
        st.warning(f"Diagnostic failed: {str(e)}")
        DATA_DIAGNOSTIC = {"error": str(e)}
    
    # Import RAG and Fine-tuned components
    from src.rag.secured_rag_pipeline import SecuredFinancialRAG
    from src.rag.guardrails import FinancialRAGGuardrails
    from src.rag.query_enhancer import FinancialQueryEnhancer
    from src.fine_tuning.inference.model_wrapper import FineTunedFinancialQA
    st.success("✅ Full system components loaded successfully")
    
except ImportError as e:
    IMPORT_ERROR_DETAILS = str(e)
    st.error(f"⚠️ RAG system import failed: {IMPORT_ERROR_DETAILS}")
    st.error(f"🔍 Check System Diagnostic Information below for detailed analysis")
    RAG_SYSTEM_AVAILABLE = False
    # Cloud fallback - create lightweight alternatives
    SecuredFinancialRAG = None
    FinancialRAGGuardrails = None  
    FinancialQueryEnhancer = None
    
except Exception as e:
    IMPORT_ERROR_DETAILS = f"Unexpected error: {str(e)}"
    st.error(f"⚠️ RAG system initialization failed: {IMPORT_ERROR_DETAILS}")
    st.error(f"🔍 Check System Diagnostic Information below for detailed analysis")
    RAG_SYSTEM_AVAILABLE = False
    SecuredFinancialRAG = None
    FinancialRAGGuardrails = None
    FinancialQueryEnhancer = None

# Configure logging is already done at the top of the file

# Page configuration
st.set_page_config(
    page_title="Financial RAG System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Financial RAG System\n\nAdvanced AI-powered financial question answering with enterprise security."
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}

.stAlert > div {
    padding: 1rem;
    margin: 0.5rem 0;
}

.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.security-approved {
    color: #28a745;
    font-weight: bold;
}

.security-blocked {
    color: #dc3545;
    font-weight: bold;
}

.confidence-high {
    color: #28a745;
    font-weight: bold;
}

.confidence-medium {
    color: #ffc107;
    font-weight: bold;
}

.confidence-low {
    color: #dc3545;
    font-weight: bold;
}

.source-citation {
    background-color: #e9ecef;
    padding: 0.5rem;
    border-left: 4px solid #007bff;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_qa_systems():
    """Load both RAG and Fine-tuned systems with caching and cloud fallback."""
    if not RAG_SYSTEM_AVAILABLE:
        return load_cloud_fallback_system(), None
    
    with st.spinner("🚀 Loading QA Systems..."):
        try:
            # Load RAG system
            rag_system = SecuredFinancialRAG(enable_strict_guardrails=True)
            
            # Set cloud-friendly parameters
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            rag_success = rag_system.load_system()
            
            # Load Fine-tuned system
            try:
                # Load from local checkpoint
                local_path = "checkpoints/best_checkpoint.pt"
                if os.path.exists(local_path):
                    model_path = local_path
                    fine_tuned_system = FineTunedFinancialQA(model_path=model_path)
                    st.success("✅ Fine-tuned model loaded successfully from local checkpoint")
                    ft_success = True
                else:
                    raise FileNotFoundError("Model not found in local checkpoints")
            except Exception as e:
                st.warning(f"⚠️ Fine-tuned model loading failed: {str(e)}")
                fine_tuned_system = None
                ft_success = False
            
            if rag_success and ft_success:
                st.success("✅ Both RAG and Fine-tuned systems loaded successfully!")
                return rag_system, fine_tuned_system
            elif rag_success:
                st.warning("⚠️ Only RAG system available")
                return rag_system, None
            else:
                st.warning("⚠️ Full systems failed, using fallback")
                return load_cloud_fallback_system(), None
                
        except Exception as e:
            st.warning(f"⚠️ System loading error: {str(e)}, using fallback")
            return load_cloud_fallback_system(), None

@st.cache_resource
def load_cloud_fallback_system():
    """Load a cloud-compatible fallback system."""
    from typing import Dict, List, Any
    import re
    
    class CloudFallbackRAG:
        def __init__(self):
            self.qa_data = None
            self.spell_corrections = {
                'revunue': 'revenue', 'reveune': 'revenue', 'revanue': 'revenue',
                'proffit': 'profit', 'prfit': 'profit',
                'expences': 'expenses', 'expence': 'expenses',
                'assests': 'assets', 'aseets': 'assets',
            }
            self.stats = {'queries_processed': 0, 'avg_response_time': 0.0}
        
        def load_system(self):
            """Load Q&A data for fallback system."""
            try:
                possible_paths = [
                    'data/processed/xbrl_qa_pairs.json',
                    'xbrl_qa_pairs.json'
                ]
                
                for path in possible_paths:
                    try:
                        with open(path, 'r') as f:
                            self.qa_data = json.load(f)
                        st.info(f"✅ Loaded {len(self.qa_data)} Q&A pairs (Cloud Mode)")
                        return True
                    except FileNotFoundError:
                        continue
                
                # Create sample data if no files found
                self.qa_data = self.create_sample_data()
                st.warning("⚠️ Using sample data - Q&A file not found")
                return True
                
            except Exception as e:
                st.error(f"❌ Fallback system error: {str(e)}")
                return False
        
        def create_sample_data(self):
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
                }
            ]
        
        def secure_query_processing(self, query, user_id, top_k=5, fusion_method='weighted', include_explanation=True):
            """Process query with simple search."""
            start_time = time.time()
            
            # Simple spell correction
            corrected_query = query.lower()
            for incorrect, correct in self.spell_corrections.items():
                corrected_query = corrected_query.replace(incorrect, correct)
            
            # Simple similarity search
            best_match = None
            best_score = 0
            
            if self.qa_data:
                for qa in self.qa_data:
                    question = qa['question'].lower()
                    query_words = set(corrected_query.split())
                    question_words = set(question.split())
                    
                    overlap = len(query_words.intersection(question_words))
                    total = len(query_words.union(question_words))
                    score = overlap / total if total > 0 else 0
                    
                    # Boost for financial terms
                    for term in ['revenue', 'profit', 'loss']:
                        if term in corrected_query and term in question:
                            score += 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_match = qa
            
            response_time = time.time() - start_time
            self.stats['queries_processed'] += 1
            
            if best_match and best_score > 0.2:
                return {
                    'status': 'approved',
                    'answer': best_match['answer'],
                    'confidence': min(0.95, best_score * 1.5),
                    'sources': [{'content_preview': best_match['question'][:100], 
                               'quarter': best_match['quarter'], 'relevance_score': best_score}],
                    'processing_time': response_time,
                    'system_mode': 'Cloud Fallback',
                    'query_enhancement': {'enhanced_query': corrected_query, 'improvement_count': 0}
                }
            else:
                return {
                    'status': 'approved',
                    'answer': "I don't have specific information about that in the available financial data.",
                    'confidence': 0.1,
                    'sources': [],
                    'processing_time': response_time,
                    'system_mode': 'Cloud Fallback'
                }
    
    fallback_system = CloudFallbackRAG()
    fallback_system.load_system()
    st.info("🔄 Running in Cloud Fallback Mode - Core Q&A functionality available")
    return fallback_system

def format_confidence_badge(confidence):
    """Format confidence score with color coding."""
    if confidence >= 0.8:
        return f'<span class="confidence-high">High ({confidence:.2f})</span>'
    elif confidence >= 0.6:
        return f'<span class="confidence-medium">Medium ({confidence:.2f})</span>'
    else:
        return f'<span class="confidence-low">Low ({confidence:.2f})</span>'

def format_security_status(status):
    """Format security status with color coding."""
    if status == "approved":
        return f'<span class="security-approved">✅ APPROVED</span>'
    elif status == "blocked":
        return f'<span class="security-blocked">🚫 BLOCKED</span>'
    else:
        return f'<span class="security-blocked">❌ ERROR</span>'

def main_query_interface():
    """Main financial query interface."""
    st.header("💰 Financial Question Answering")
    st.markdown("Ask questions about financial performance, metrics, and business insights.")
    
    # Load QA systems
    rag_system, fine_tuned_system = load_qa_systems()
    if not rag_system:
        st.error("QA systems not available. Please check system status.")
        return
        
    # System selection
    qa_system = st.radio(
        "🤖 Select QA System:",
        ["RAG System", "Fine-Tuned Model", "Hybrid (Compare Both)"],
        help="Choose which system to use for answering questions"
    )
    
    # Show system mode
    system_mode = getattr(rag_system, '__class__', type(rag_system)).__name__
    if system_mode == 'CloudFallbackRAG':
        st.info("🔄 **Running in Cloud Fallback Mode** - Core Q&A functionality with your financial data")
    elif RAG_SYSTEM_AVAILABLE:
        st.success("🎯 **Running Full RAG System** - Advanced retrieval with cross-encoder re-ranking")
    else:
        st.warning("⚠️ **Limited Mode** - Basic functionality available")
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "💬 Your Financial Question:",
            placeholder="What was the revenue from operations in Q3 2023?",
            help="Ask questions about revenue, profit, expenses, assets, or financial performance."
        )
    
    with col2:
        st.markdown("&nbsp;")  # Spacing
        ask_button = st.button("🔍 Get Answer", type="primary")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_k = st.slider("📊 Results to Retrieve", 1, 10, 5, help="Number of relevant chunks to retrieve")
        
        with col2:
            fusion_method = st.selectbox("🔄 Fusion Method", ["weighted", "rrf"], help="Method for combining dense and sparse search results")
        
        with col3:
            include_explanation = st.checkbox("📝 Include Explanation", help="Show detailed explanation of the retrieval process")
    
    # Process query
    if ask_button and query:
        # User identification for security
        user_id = st.session_state.get('user_id', f"streamlit_user_{int(time.time())}")
        st.session_state.user_id = user_id
        
        # Initialize query enhancer
        query_enhancer = FinancialQueryEnhancer()
        
        # Process with progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Query enhancement
            status_text.text("✨ Enhancing query (spell check, optimization)...")
            progress_bar.progress(10)
            
            enhancement_result = query_enhancer.enhance_query(query)
            enhanced_query = enhancement_result['enhanced_query']
            
            # Show enhancement info if improvements were made
            if enhancement_result['improvement_count'] > 0:
                st.info(f"🔧 **Query Enhanced**: '{query}' → '{enhanced_query}' ({enhancement_result['improvement_count']} improvements)")
                
                if enhancement_result['corrections_applied']:
                    corrections_text = ", ".join([f"'{c['original']}' → '{c['corrected']}'" for c in enhancement_result['corrections_applied']])
                    st.success(f"✏️ **Corrections Applied**: {corrections_text}")
                
                if enhancement_result['enhancements_made']:
                    enhancements_text = ", ".join([f"'{e.get('original', e.get('original', 'term'))}' → '{e.get('converted', e.get('standardized', 'enhanced'))}'" for e in enhancement_result['enhancements_made']])
                    st.success(f"⚡ **Enhancements**: {enhancements_text}")
            
            status_text.text("🔒 Validating query security...")
            progress_bar.progress(25)
            
            status_text.text("🔍 Searching financial database...")
            progress_bar.progress(60)
            
            # Process query based on selected system
            if qa_system == "RAG System" or not fine_tuned_system:
                result = rag_system.secure_query_processing(
                    query=enhanced_query,
                    user_id=user_id,
                    top_k=top_k,
                    fusion_method=fusion_method,
                    include_explanation=include_explanation
                )
            elif qa_system == "Fine-Tuned Model":
                result = fine_tuned_system.answer_question(
                    question=enhanced_query,
                    user_id=user_id
                )
            else:  # Hybrid mode
                # Get results from both systems
                rag_result = rag_system.secure_query_processing(
                    query=enhanced_query,
                    user_id=user_id,
                    top_k=top_k,
                    fusion_method=fusion_method,
                    include_explanation=include_explanation
                )
                
                ft_result = fine_tuned_system.answer_question(
                    question=enhanced_query,
                    user_id=user_id
                )
                
                # Combine results
                result = {
                    'status': 'approved' if rag_result['status'] == 'approved' and ft_result['status'] == 'approved' else 'blocked',
                    'rag_result': rag_result,
                    'ft_result': ft_result,
                    'is_hybrid': True
                }
            
            # Add enhancement metadata to result
            result['query_enhancement'] = enhancement_result
            result['original_query'] = query
            
            status_text.text("🤖 Generating response...")
            progress_bar.progress(90)
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            # Display results
            display_query_results(result, query)
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            progress_bar.empty()
            status_text.empty()

def display_query_results(result, query):
    """Display query results with proper formatting."""
    status = result.get('status', 'unknown')
    
    # Handle hybrid results differently
    if result.get('is_hybrid', False):
        st.markdown("### 🤖 System Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RAG System")
            display_single_result(result['rag_result'], query, "RAG")
        
        with col2:
            st.markdown("#### Fine-Tuned Model")
            display_single_result(result['ft_result'], query, "Fine-Tuned")
        
        return
    
    # Display single system result
    display_single_result(result, query)

def display_single_result(result, query, system_name=None):
    """Display a single system's query results."""
    result_status = result.get('status', 'unknown')
    
    # Security status header
    st.markdown("### 🛡️ Security Status")
    st.markdown(f"**Status:** {format_security_status(result_status)}", unsafe_allow_html=True)
    
    if result_status == "blocked":
        st.warning(f"🚫 **Query Blocked:** {result.get('reason', 'Unknown reason')}")
        st.info("💡 **Tip:** Ensure your query is related to financial topics and doesn't contain sensitive information.")
        return
    
    if result_status == "error":
        st.error(f"❌ **Processing Error:** {result.get('reason', 'Unknown error')}")
        return
    
    # Approved query results
    if result_status == "approved":
        answer = result.get('answer', 'No answer generated')
        confidence = result.get('confidence', 0)
        
        # Answer section
        st.markdown("### 🤖 Answer")
        
        # Confidence indicator
        confidence_html = format_confidence_badge(confidence)
        st.markdown(f"**Confidence:** {confidence_html}", unsafe_allow_html=True)
        
        # Main answer
        st.markdown(f"**{answer}**")
        
        # Performance metrics
        processing_metadata = result.get('processing_metadata', {})
        total_time = processing_metadata.get('total_time', 0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⚡ Response Time", f"{total_time:.3f}s")
        with col2:
            # Show confidence with enhancement boost if applicable
            enhancement = result.get('query_enhancement', {})
            confidence_boost = enhancement.get('confidence_boost', 0)
            if confidence_boost > 0:
                st.metric("🎯 Confidence", f"{confidence:.2f}", delta=f"+{confidence_boost:.1%} boost")
            else:
                st.metric("🎯 Confidence", f"{confidence:.2f}")
        with col3:
            st.metric("📊 Sources Used", len(result.get('sources', [])))
        with col4:
            st.metric("🛡️ Security Checks", len(result.get('security_metadata', {}).get('validation_checks_passed', [])))
        
        # Sources section
        sources = result.get('sources', [])
        if sources:
            st.markdown("### 📚 Information Sources")
            
            for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                with st.expander(f"📄 Source {i}: {source.get('quarter', 'Unknown')} - {source.get('section', 'Unknown')}"):
                    st.markdown(f"**Relevance Score:** {source.get('relevance_score', 0):.3f}")
                    
                    if 'cross_encoder_score' in source:
                        st.markdown(f"**Cross-Encoder Score:** {source['cross_encoder_score']:.3f}")
                        st.markdown(f"**Score Improvement:** {source.get('score_improvement', 0):+.3f}")
                    
                    st.markdown("**Content Preview:**")
                    st.markdown(f'<div class="source-citation">{source.get("content_preview", "No preview available")}</div>', unsafe_allow_html=True)
        
        # Detailed explanation (if requested)
        explanation = result.get('explanation')
        if explanation:
            st.markdown("### 📝 Detailed Explanation")
            
            with st.expander("🔍 Retrieval Process"):
                st.json(explanation.get('retrieval_explanation', {}))
            
            with st.expander("🤖 Response Generation"):
                st.json(explanation.get('response_explanation', {}))
            
            with st.expander("🛡️ Security Analysis"):
                st.json(explanation.get('security_explanation', {}))
        
        # Query enhancement details (if available)
        enhancement = result.get('query_enhancement', {})
        if enhancement and enhancement.get('improvement_count', 0) > 0:
            st.markdown("### ✨ Query Enhancement Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Original Query**: {result.get('original_query', 'N/A')}")
                st.markdown(f"**Enhanced Query**: {enhancement.get('enhanced_query', 'N/A')}")
                st.markdown(f"**Improvements Made**: {enhancement.get('improvement_count', 0)}")
                st.markdown(f"**Confidence Boost**: +{enhancement.get('confidence_boost', 0):.1%}")
            
            with col2:
                if enhancement.get('corrections_applied'):
                    st.markdown("**Spell Corrections**:")
                    for correction in enhancement['corrections_applied']:
                        st.markdown(f"- '{correction['original']}' → '{correction['corrected']}'")
                
                if enhancement.get('enhancements_made'):
                    st.markdown("**Query Enhancements**:")
                    for enh in enhancement['enhancements_made']:
                        if 'converted' in enh:
                            st.markdown(f"- '{enh['original']}' → '{enh['converted']}'")
                        elif 'standardized' in enh:
                            st.markdown(f"- '{enh['original']}' → '{enh['standardized']}'")
            
            with st.expander("🔍 Enhancement Analysis"):
                st.json(enhancement)

def security_dashboard():
    """Security monitoring dashboard."""
    st.header("🛡️ Security Dashboard")
    st.markdown("Monitor system security, threats, and access patterns.")
    
    # Load QA systems for dashboard data
    rag_system, fine_tuned_system = load_qa_systems()
    if not rag_system:
        st.error("QA systems not available for dashboard.")
        return
    
    # Get security dashboard data
    dashboard_data = rag_system.get_security_dashboard()
    
    # System status indicators
    st.markdown("### 🚨 System Status")
    
    status = dashboard_data.get('system_status', {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rag_loaded = status.get('rag_pipeline_loaded', False)
        st.metric("RAG Pipeline", "✅ Online" if rag_loaded else "❌ Offline")
    
    with col2:
        guardrails_active = status.get('guardrails_active', False)
        st.metric("Security Guardrails", "🛡️ Active" if guardrails_active else "⚠️ Inactive")
    
    with col3:
        cross_encoder = status.get('cross_encoder_enabled', False)
        st.metric("Cross-Encoder", "🎯 Enabled" if cross_encoder else "❌ Disabled")
    
    with col4:
        emergency = status.get('emergency_mode', False)
        st.metric("Emergency Mode", "🚨 Active" if emergency else "✅ Normal")
    
    # Security metrics
    st.markdown("### 📊 Security Metrics")
    
    security_metrics = dashboard_data.get('security_metrics', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_requests = security_metrics.get('total_requests_processed', 0)
        approval_rate = security_metrics.get('approval_rate_percentage', 0)
        
        # Create approval rate gauge
        fig_approval = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=approval_rate,
            title={"text": "Approval Rate (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]},
                  'bar': {'color': "darkgreen"},
                  'steps': [{'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "green"}],
                  'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75,
                               'value': 90}}))
        fig_approval.update_layout(height=300)
        st.plotly_chart(fig_approval, use_container_width=True)
    
    with col2:
        blocked_requests = security_metrics.get('blocked_requests', 0)
        security_incidents = security_metrics.get('security_incidents', 0)
        
        # Security incidents chart
        incidents_data = pd.DataFrame({
            'Type': ['Approved', 'Blocked', 'Incidents'],
            'Count': [total_requests - blocked_requests, blocked_requests, security_incidents],
            'Color': ['green', 'red', 'orange']
        })
        
        fig_incidents = px.bar(incidents_data, x='Type', y='Count', color='Color',
                              title="Request Status Breakdown",
                              color_discrete_map={'green': 'green', 'red': 'red', 'orange': 'orange'})
        fig_incidents.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_incidents, use_container_width=True)
    
    with col3:
        avg_processing_time = security_metrics.get('average_processing_time', 0)
        
        st.metric("Total Requests", f"{total_requests:,}")
        st.metric("Blocked Requests", f"{blocked_requests:,}")
        st.metric("Security Incidents", f"{security_incidents:,}")
        st.metric("Avg Processing Time", f"{avg_processing_time:.3f}s")
    
    # Guardrails effectiveness
    st.markdown("### 🔒 Guardrails Effectiveness")
    
    effectiveness = dashboard_data.get('guardrails_effectiveness', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rate Limit Protection", effectiveness.get('rate_limit_protection', 0))
    with col2:
        st.metric("Content Filtering", effectiveness.get('content_filtering', 0))
    with col3:
        st.metric("Context Validation", effectiveness.get('context_validation', 0))
    with col4:
        st.metric("Output Filtering", effectiveness.get('output_filtering', 0))
    
    # Response quality metrics
    st.markdown("### 📈 Response Quality")
    
    quality = dashboard_data.get('response_quality', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        high_conf = quality.get('high_confidence_responses', 0)
        low_conf = quality.get('low_confidence_responses', 0)
        total_responses = high_conf + low_conf
        
        if total_responses > 0:
            quality_data = pd.DataFrame({
                'Confidence': ['High (≥0.8)', 'Low (<0.6)'],
                'Count': [high_conf, low_conf]
            })
            
            fig_quality = px.pie(quality_data, values='Count', names='Confidence',
                               title="Response Quality Distribution")
            fig_quality.update_layout(height=300)
            st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        avg_confidence = quality.get('average_rag_confidence', 0)
        avg_response_time = quality.get('response_time_average', 0)
        
        st.metric("Average Confidence", f"{avg_confidence:.3f}")
        st.metric("Average Response Time", f"{avg_response_time:.3f}s")
        
        # Performance over time (simulated data for demo)
        time_data = pd.DataFrame({
            'Time': pd.date_range('2024-01-01', periods=7, freq='D'),
            'Confidence': [0.85, 0.82, 0.88, 0.90, 0.87, 0.89, 0.91],
            'Response Time': [0.25, 0.23, 0.28, 0.22, 0.26, 0.24, 0.21]
        })
        
        fig_trends = px.line(time_data, x='Time', y=['Confidence', 'Response Time'],
                           title="Performance Trends")
        fig_trends.update_layout(height=300)
        st.plotly_chart(fig_trends, use_container_width=True)

def system_analytics():
    """System performance analytics."""
    st.header("📊 System Analytics")
    st.markdown("Analyze system performance, usage patterns, and optimization opportunities.")
    
    # Load QA systems
    rag_system, fine_tuned_system = load_qa_systems()
    if not rag_system:
        st.error("QA systems not available for analytics.")
        return
    
    # Get system statistics
    system_stats = rag_system.get_security_dashboard()
    
    # Performance overview
    st.markdown("### ⚡ Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    security_metrics = system_stats.get('security_metrics', {})
    response_quality = system_stats.get('response_quality', {})
    
    with col1:
        total_requests = security_metrics.get('total_requests_processed', 0)
        st.metric("Total Queries Processed", f"{total_requests:,}")
    
    with col2:
        avg_time = security_metrics.get('average_processing_time', 0)
        st.metric("Average Response Time", f"{avg_time:.3f}s")
    
    with col3:
        avg_confidence = response_quality.get('average_rag_confidence', 0)
        st.metric("Average Confidence", f"{avg_confidence:.3f}")
    
    with col4:
        approval_rate = security_metrics.get('approval_rate_percentage', 0)
        st.metric("Security Approval Rate", f"{approval_rate:.1f}%")
    
    # System configuration
    st.markdown("### ⚙️ System Configuration")
    
    config = system_stats.get('system_configuration', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Security Settings:**")
        st.write(f"• Max Query Length: {config.get('max_query_length', 0)} characters")
        st.write(f"• Rate Limit (per minute): {config.get('rate_limit_per_minute', 0)}")
        st.write(f"• Rate Limit (per hour): {config.get('rate_limit_per_hour', 0)}")
        st.write(f"• Strict Guardrails: {'✅' if config.get('strict_guardrails', False) else '❌'}")
    
    with col2:
        system_config = system_stats.get('system_status', {})
        st.markdown("**System Components:**")
        st.write(f"• RAG Pipeline: {'✅ Active' if system_config.get('rag_pipeline_loaded', False) else '❌ Inactive'}")
        st.write(f"• Cross-Encoder: {'✅ Enabled' if system_config.get('cross_encoder_enabled', False) else '❌ Disabled'}")
        st.write(f"• Security Guardrails: {'✅ Active' if system_config.get('guardrails_active', False) else '❌ Inactive'}")
        st.write(f"• Emergency Mode: {'🚨 Active' if system_config.get('emergency_mode', False) else '✅ Normal'}")

def admin_panel():
    """Administrative controls and settings."""
    st.header("⚙️ Admin Panel")
    st.markdown("System administration, user management, and emergency controls.")
    
    # Authentication check (simplified)
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.warning("🔒 Admin authentication required")
        
        admin_password = st.text_input("Enter Admin Password:", type="password")
        if st.button("🔐 Authenticate"):
            # Simple password check (in production, use proper authentication)
            if admin_password == "admin123":
                st.session_state.admin_authenticated = True
                st.success("✅ Authentication successful!")
                st.rerun()
            else:
                st.error("❌ Invalid password")
        return
    
    # Admin controls
    st.success("✅ Admin access granted")
    
    # Load QA systems
    rag_system, fine_tuned_system = load_qa_systems()
    if not rag_system:
        st.error("QA systems not available for administration.")
        return
    
    # Emergency controls
    st.markdown("### 🚨 Emergency Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚨 Emergency Lockdown", help="Block all queries immediately"):
            rag_system.emergency_lockdown()
            st.error("🚨 Emergency lockdown activated! All queries will be blocked.")
    
    with col2:
        user_id_reset = st.text_input("Reset User Access:")
        if st.button("🔄 Reset User Limits") and user_id_reset:
            rag_system.reset_user_access(user_id_reset)
            st.success(f"✅ Access reset for user: {user_id_reset}")
    
    # System management
    st.markdown("### 🛠️ System Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Security Configuration:**")
        
        new_query_limit = st.number_input("Max Query Length", 100, 1000, 500)
        new_rate_limit_min = st.number_input("Rate Limit (per minute)", 1, 100, 30)
        new_rate_limit_hour = st.number_input("Rate Limit (per hour)", 100, 2000, 500)
        
        if st.button("💾 Update Security Settings"):
            # Update guardrails settings
            rag_system.guardrails.max_query_length = new_query_limit
            rag_system.guardrails.rate_limit_per_minute = new_rate_limit_min
            rag_system.guardrails.rate_limit_per_hour = new_rate_limit_hour
            st.success("✅ Security settings updated!")
    
    with col2:
        st.markdown("**System Statistics:**")
        dashboard_data = rag_system.get_security_dashboard()
        
        # Display key metrics
        security_metrics = dashboard_data.get('security_metrics', {})
        st.write(f"Total Requests: {security_metrics.get('total_requests_processed', 0)}")
        st.write(f"Blocked Requests: {security_metrics.get('blocked_requests', 0)}")
        st.write(f"Security Incidents: {security_metrics.get('security_incidents', 0)}")
        st.write(f"Average Processing Time: {security_metrics.get('average_processing_time', 0):.3f}s")
    
    # Logs and monitoring
    st.markdown("### 📋 System Logs")
    
    if st.button("📥 Download System Logs"):
        # Generate log data
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'system_status': dashboard_data.get('system_status', {}),
            'security_metrics': security_metrics,
            'detailed_stats': dashboard_data.get('detailed_stats', {})
        }
        
        st.download_button(
            label="💾 Download Logs (JSON)",
            data=json.dumps(log_data, indent=2),
            file_name=f"rag_system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def documentation():
    """System documentation and help."""
    st.header("📚 Documentation")
    st.markdown("Learn how to use the Financial RAG System effectively.")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    **1. Ask Financial Questions**
    - Navigate to the main interface
    - Type your financial question in plain English
    - Click "Get Answer" to receive AI-powered responses
    
    **2. Question Examples**
    - "What was the revenue from operations in Q3 2023?"
    - "How much profit did the company make?"
    - "Tell me about employee benefit expenses"
    - "Compare financial performance across quarters"
    
    **3. Understanding Results**
    - **High Confidence (≥0.8)**: Reliable answer with strong evidence
    - **Medium Confidence (0.6-0.8)**: Good answer, verify if needed
    - **Low Confidence (<0.6)**: Uncertain answer, additional verification recommended
    """)
    
    # System features
    st.markdown("### 🎯 System Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 AI Capabilities**
        - Natural language understanding
        - Financial context awareness  
        - Cross-encoder re-ranking for precision
        - Smart response generation
        - Multi-source information synthesis
        
        **🛡️ Security Features**
        - Real-time threat detection
        - Input validation and sanitization
        - Rate limiting and access control
        - Content filtering
        - Emergency lockdown capabilities
        """)
    
    with col2:
        st.markdown("""
        **📊 Performance Features**
        - Sub-second response times
        - High-confidence answers
        - Source attribution and citations
        - Comprehensive error handling
        - Performance monitoring
        
        **⚙️ Enterprise Features**
        - Security dashboard and monitoring
        - Administrative controls
        - System analytics and reporting
        - Audit logging
        - Configurable security settings
        """)
    
    # Technical architecture
    st.markdown("### 🏗️ Technical Architecture")
    
    st.markdown("""
    **System Components:**
    
    1. **Text Processing**: 1,520 financial chunks with 100-token precision
    2. **Hybrid Retrieval**: Dense (ChromaDB) + Sparse (BM25) search
    3. **Advanced Re-Ranking**: Cross-encoder for precise relevance scoring
    4. **Smart Response Generation**: Template-based financial responses
    5. **Security Guardrails**: Multi-layer input/output validation
    6. **Web Interface**: Professional Streamlit-based deployment
    """)
    
    # System Status & Debugging
    st.markdown("### 🔧 System Status & Troubleshooting")
    
    with st.expander("🔍 System Diagnostic Information"):
        st.markdown("### 📊 Comprehensive System Analysis")
        
        # Display diagnostic results
        if DATA_DIAGNOSTIC and "error" not in DATA_DIAGNOSTIC:
            diag = DATA_DIAGNOSTIC
            
            st.markdown(f"""
            **🌍 Deployment Mode**: {diag.get('deployment_mode', 'Unknown').title()}
            
            **📁 Base Directory**: `{diag.get('base_directory', 'Unknown')}`
            
            **🐍 Python Version**: {sys.version.split()[0]}
            
            **🤖 RAG System Status**: {"✅ Available" if RAG_SYSTEM_AVAILABLE else "❌ Unavailable"}
            
            **❗ Import Error Details**: {IMPORT_ERROR_DETAILS if IMPORT_ERROR_DETAILS else "None"}
            """)
            
            # Critical Files Status
            st.markdown("### 📋 Critical Files Status")
            critical_files = diag.get('critical_files_check', {})
            
            for filename, info in critical_files.items():
                status = info.get('status', 'unknown')
                if status == 'available':
                    size = info.get('size_mb', 0)
                    loadable = info.get('loadable', info.get('valid_json', True))
                    loadable_icon = "✅" if loadable else "⚠️"
                    st.markdown(f"- **{filename}**: ✅ Available ({size}MB) {loadable_icon}")
                elif status == 'missing':
                    st.markdown(f"- **{filename}**: ❌ Missing")
                else:
                    st.markdown(f"- **{filename}**: ⚠️ Error - {info.get('error', 'Unknown')}")
            
            # Data Integrity
            integrity = diag.get('data_integrity_check', {})
            if integrity:
                st.markdown("### 🔗 Data Integrity")
                
                chunk_consistency = integrity.get('chunk_consistency', {})
                if 'error' not in chunk_consistency:
                    chunks_count = chunk_consistency.get('chunks_count', 0)
                    mapping_count = chunk_consistency.get('mapping_count', 0) 
                    consistent = chunk_consistency.get('consistent', False)
                    
                    consistency_icon = "✅" if consistent else "❌"
                    st.markdown(f"- **Chunk Consistency**: {consistency_icon} {chunks_count} chunks, {mapping_count} mappings")
                
                qa_data = integrity.get('qa_data', {})
                if 'error' not in qa_data:
                    qa_pairs = qa_data.get('total_pairs', 0)
                    has_revenue = qa_data.get('has_revenue_data', False)
                    revenue_icon = "✅" if has_revenue else "❌"
                    st.markdown(f"- **Q&A Data**: {revenue_icon} {qa_pairs} pairs (Revenue data: {has_revenue})")
            
        else:
            st.error(f"❌ Diagnostic Error: {DATA_DIAGNOSTIC.get('error', 'Unknown error')}")
            
        # Recovery suggestions for cloud issues
        if DATA_DIAGNOSTIC and DATA_DIAGNOSTIC.get('deployment_mode') == 'cloud':
            st.markdown("### 🛠️ Cloud Deployment Issues")
            missing_files = []
            if 'critical_files_check' in DATA_DIAGNOSTIC:
                for filename, info in DATA_DIAGNOSTIC['critical_files_check'].items():
                    if info.get('status') in ['missing', 'error']:
                        missing_files.append(filename)
            
            if missing_files:
                st.error(f"🚨 Missing critical files in cloud: {', '.join(missing_files)}")
                st.markdown("""
                **Suggested Solutions:**
                1. Ensure all data files are committed to git (check .gitignore)
                2. Verify cloud deployment includes data/ directory 
                3. Check file size limits for your cloud provider
                4. Consider using alternative data loading mechanisms
                """)
        
    
    with st.expander("❓ Common Issues"):
        st.markdown("""
        **Query Blocked by Security**
        - Ensure your question is finance-related
        - Avoid sensitive personal information
        - Check for malicious patterns or special characters
        
        **Low Confidence Responses**
        - Try rephrasing your question more specifically
        - Include time periods (e.g., "Q3 2023", "Dec 2022")
        - Use financial terminology (revenue, profit, expenses)
        
        **System Performance Issues**
        - Check system status in the Security Dashboard
        - Verify all components are loaded
        - Contact administrator for system maintenance
        """)
    
    # API reference
    st.markdown("### 📖 API Reference")
    
    with st.expander("🔧 System Configuration"):
        st.code("""
# Security Settings
max_query_length: 500 characters
rate_limit_per_minute: 30 queries
rate_limit_per_hour: 500 queries

# Performance Settings  
top_k_results: 1-10 chunks
fusion_methods: ["weighted", "rrf"]
confidence_threshold: 0.0-1.0

# Models Used
embedding_model: sentence-transformers/all-MiniLM-L6-v2
cross_encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
response_generation: template-based + direct Q&A matching
        """)

# Main application
def main():
    """Main application with sidebar navigation."""
    
    # Sidebar navigation
    st.sidebar.title("💰 Financial RAG System")
    st.sidebar.markdown("**Enterprise AI Financial Assistant**")
    
    # Navigation menu
    page = st.sidebar.selectbox("📍 Navigate to:", [
        "🏠 Main Interface",
        "🛡️ Security Dashboard", 
        "📊 System Analytics",
        "📚 Documentation"
    ])
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚨 System Status")
    
    rag_system, fine_tuned_system = load_qa_systems()
    if rag_system:
        st.sidebar.success("✅ RAG System Online")
        if fine_tuned_system:
            st.sidebar.success("✅ Fine-Tuned Model Ready")
        st.sidebar.success("🛡️ Security Active")
        st.sidebar.success("🎯 Cross-Encoder Ready")
    else:
        st.sidebar.error("❌ Systems Offline")
    
    # Current time
    st.sidebar.markdown(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Page routing
    if page == "🏠 Main Interface":
        main_query_interface()
    elif page == "🛡️ Security Dashboard":
        security_dashboard()
    elif page == "📊 System Analytics":
        system_analytics()

    elif page == "📚 Documentation":
        documentation()

if __name__ == "__main__":
    main()

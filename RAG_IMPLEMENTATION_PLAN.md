# RAG System Implementation Plan
**Retrieval-Augmented Generation for Financial QA System**

## 📋 Overview
Implement a comprehensive RAG system using the processed Jaiprakash Associates financial data (2022-2024) with hybrid retrieval and response generation capabilities.

---

## 🗂️ Current Data Assets
✅ **Financial Reports**: 8 quarterly reports (April 2022 - March 2024)  
✅ **Q&A Dataset**: 136 question-answer pairs across all quarters  
✅ **Processed Data**: CSV files with structured XBRL financial elements  
✅ **Combined Report**: 2-year consolidated financial analysis  

---

## 🎯 Implementation Steps

### **Phase 1: Data Processing & Chunking (2.1)**

#### **1.1 Text Preparation**
**Input Sources:**
- Raw financial text from combined 2-year report
- Individual quarterly CSV data
- Generated Q&A pairs as reference

**Processing Tasks:**
- [ ] Extract clean financial text from processed reports
- [ ] Remove headers, footers, page numbers
- [ ] Segment by financial statement sections (P&L, Balance Sheet, Cash Flow)
- [ ] Create document metadata structure

**Tools Required:**
```python
# Text processing
import nltk
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
```

#### **1.2 Text Chunking Strategy**
**Chunk Size:** 100 tokens only (simplified approach)

**Chunking Approach:**
1. **Sentence-boundary chunking** - Preserve financial statement coherence
2. **Overlapping windows** - 20% overlap between chunks
3. **Metadata preservation** - Quarter, section type, financial metrics

**Chunk Structure:**
```python
{
    "chunk_id": "Q2_2023_P&L_001",
    "text": "Revenue from operations was ₹15.03 billion...",
    "tokens": 98,
    "quarter": "Q2 2023", 
    "section": "Profit & Loss",
    "start_pos": 0,
    "end_pos": 500,
    "metadata": {
        "company": "Jaiprakash Associates",
        "period": "Jul-Sep 2023",
        "key_metrics": ["revenue", "operations"]
    }
}
```

---

### **Phase 2: Embedding & Indexing (2.2)**

#### **2.1 Embedding Model Selection**
**Primary Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Size:** 22.7M parameters
- **Embedding Dimension:** 384
- **Optimized for:** Sentence similarity

**Alternative:** `intfloat/e5-small-v2`
- **Size:** 33.4M parameters  
- **Embedding Dimension:** 384
- **Optimized for:** General text retrieval

#### **2.2 Dense Vector Store**
**ChromaDB Implementation:**
```python
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

# Create collection for 100-token chunks
collection = client.create_collection("financial_chunks_100")
```

**FAISS Alternative:**
```python
import faiss
import numpy as np

# Create FAISS index
dimension = 384  # all-MiniLM-L6-v2 embedding size
index = faiss.IndexFlatL2(dimension)
```

#### **2.3 Sparse Index (BM25)**
**Implementation:**
```python
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# BM25 for keyword retrieval
bm25 = BM25Okapi(tokenized_chunks)
```

---

### **Phase 3: Hybrid Retrieval Pipeline (2.3)**

#### **3.1 Query Preprocessing**
**Pipeline:**
```python
def preprocess_query(query: str) -> dict:
    # Clean and normalize
    query_clean = query.lower().strip()
    
    # Tokenization
    tokens = word_tokenize(query_clean)
    
    # Remove stopwords  
    stop_words = set(stopwords.words('english'))
    tokens_filtered = [w for w in tokens if w not in stop_words]
    
    # Financial term expansion
    query_expanded = expand_financial_terms(tokens_filtered)
    
    return {
        "original": query,
        "cleaned": query_clean,
        "tokens": tokens_filtered,
        "expanded": query_expanded
    }
```

#### **3.2 Dense Retrieval**
**Vector Similarity Search:**
```python
def dense_retrieval(query: str, top_k: int = 5) -> List[dict]:
    # Generate query embedding
    query_embedding = embedding_model.encode([query])
    
    # Search in the collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    return format_dense_results(results)
```

#### **3.3 Sparse Retrieval**
**BM25 Keyword Search:**
```python
def sparse_retrieval(query: str, top_k: int = 5) -> List[dict]:
    processed_query = preprocess_query(query)
    
    # BM25 scoring
    scores = bm25.get_scores(processed_query["tokens"])
    
    # Get top-k results
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    
    return format_sparse_results(top_indices, scores)
```

#### **3.4 Result Fusion**
**Weighted Score Combination:**
```python
def hybrid_retrieval(query: str, top_k: int = 10) -> List[dict]:
    # Get results from both methods
    dense_results = dense_retrieval(query, top_k)
    sparse_results = sparse_retrieval(query, top_k)
    
    # Weighted fusion (60% dense, 40% sparse)
    combined_scores = {}
    
    for result in dense_results:
        chunk_id = result["chunk_id"]
        combined_scores[chunk_id] = 0.6 * result["score"]
    
    for result in sparse_results:
        chunk_id = result["chunk_id"]
        if chunk_id in combined_scores:
            combined_scores[chunk_id] += 0.4 * result["score"]
        else:
            combined_scores[chunk_id] = 0.4 * result["score"]
    
    # Sort and return top-k
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

---

### **Phase 4: Advanced RAG Technique (2.4)**
**Selected Technique: #4 - Hybrid Search (Sparse + Dense Retrieval)**

#### **4.1 Enhanced Hybrid Architecture**
**Multi-Stage Retrieval:**
1. **Stage 1:** Broad retrieval (top-20 from each method)
2. **Stage 2:** Score fusion and re-ranking
3. **Stage 3:** Context-aware filtering

**Implementation:**
```python
class AdvancedHybridRetriever:
    def __init__(self):
        self.dense_weight = 0.6
        self.sparse_weight = 0.4
        self.rerank_threshold = 0.7
        
    def multi_stage_retrieval(self, query: str) -> List[dict]:
        # Stage 1: Broad retrieval
        broad_dense = dense_retrieval(query, top_k=20)
        broad_sparse = sparse_retrieval(query, top_k=20)
        
        # Stage 2: Advanced score fusion
        fused_results = self.reciprocal_rank_fusion(broad_dense, broad_sparse)
        
        # Stage 3: Context filtering
        filtered_results = self.context_aware_filter(query, fused_results)
        
        return filtered_results
    
    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        # RRF scoring for better fusion
        scores = {}
        for rank, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            
        for rank, result in enumerate(sparse_results):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### **4.2 Enhanced Context Selection**
**Smart Chunk Filtering:**
```python
def enhanced_context_selection(query: str, retrieved_chunks: List[dict]) -> List[dict]:
    # Analyze query complexity
    query_tokens = len(word_tokenize(query))
    financial_terms = count_financial_terms(query)
    
    # Filter and prioritize chunks based on query complexity
    if query_tokens > 10 or financial_terms > 2:
        # For complex queries, select more diverse chunks
        return select_diverse_chunks(retrieved_chunks, diversity_threshold=0.7)
    else:
        # For simple queries, prioritize highest scoring chunks
        return retrieved_chunks[:5]  # Top 5 most relevant
```

---

### **Phase 5: Response Generation (2.5)**

#### **5.1 Model Selection**
**Primary Option:** `distilgpt2`
- **Size:** 82M parameters
- **Context Window:** 1024 tokens
- **Speed:** Fast inference

**Alternative:** `microsoft/DialoGPT-small`
- **Size:** 117M parameters
- **Optimized for:** Conversational responses

#### **5.2 Response Generation Pipeline**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class FinancialResponseGenerator:
    def __init__(self):
        self.model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.max_input_tokens = 800  # Leave room for generation
        
    def generate_response(self, query: str, retrieved_chunks: List[str]) -> str:
        # Construct prompt
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""
Financial Context:
{context}

Question: {query}
Answer: """

        # Tokenize and truncate if needed
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        answer = response.split("Answer: ")[-1].strip()
        return answer
```

---

### **Phase 6: Guardrail Implementation (2.6)**

#### **6.1 Input-Side Guardrails**
**Query Validation:**
```python
class InputGuardrails:
    def __init__(self):
        self.financial_keywords = ["revenue", "profit", "loss", "assets", "liability", "cash", "expense", "income"]
        self.forbidden_topics = ["personal", "medical", "political", "inappropriate"]
        
    def validate_query(self, query: str) -> dict:
        query_lower = query.lower()
        
        # Check if query is financial-related
        has_financial_terms = any(keyword in query_lower for keyword in self.financial_keywords)
        
        # Check for forbidden content
        has_forbidden_content = any(topic in query_lower for topic in self.forbidden_topics)
        
        # Check query length
        is_reasonable_length = 5 <= len(query.split()) <= 50
        
        return {
            "valid": has_financial_terms and not has_forbidden_content and is_reasonable_length,
            "reason": self.get_validation_reason(has_financial_terms, has_forbidden_content, is_reasonable_length),
            "confidence": self.calculate_confidence(query)
        }
```

#### **6.2 Output-Side Guardrails**
**Response Validation:**
```python
class OutputGuardrails:
    def __init__(self):
        self.factual_indicators = ["according to", "based on", "the data shows"]
        self.uncertainty_phrases = ["I'm not sure", "it appears", "possibly"]
        
    def validate_response(self, response: str, retrieved_chunks: List[str]) -> dict:
        # Check for factual grounding
        has_factual_indicators = any(indicator in response.lower() for indicator in self.factual_indicators)
        
        # Check for hallucination indicators
        has_uncertainty = any(phrase in response.lower() for phrase in self.uncertainty_phrases)
        
        # Verify response aligns with retrieved context
        context_alignment = self.calculate_context_alignment(response, retrieved_chunks)
        
        return {
            "valid": context_alignment > 0.7 and (has_factual_indicators or has_uncertainty),
            "context_alignment": context_alignment,
            "factual_grounding": has_factual_indicators,
            "shows_uncertainty": has_uncertainty
        }
```

---

### **Phase 7: Interface Development (2.7)**

#### **7.1 Streamlit Application**
```python
import streamlit as st
from datetime import datetime

class FinancialQAInterface:
    def __init__(self):
        self.rag_system = FinancialRAGSystem()
        
    def main(self):
        st.title("🏦 Financial QA System - RAG Implementation")
        st.markdown("Ask questions about Jaiprakash Associates financial data (2022-2024)")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            retrieval_method = st.selectbox("Retrieval Method", ["Hybrid", "Dense Only", "Sparse Only"])
            top_k = st.slider("Top-K Results", 1, 10, 5)
            diversity_threshold = st.slider("Context Diversity", 0.5, 1.0, 0.7)
            
        # Main interface
        query = st.text_input("💬 Ask a financial question:", placeholder="What was the revenue in Q3 2023?")
        
        if st.button("🔍 Get Answer", type="primary"):
            if query:
                with st.spinner("Processing your question..."):
                    start_time = datetime.now()
                    
                    # Process query
                    result = self.rag_system.process_query(
                        query=query,
                        method=retrieval_method,
                        top_k=top_k,
                        diversity_threshold=diversity_threshold
                    )
                    
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds()
                
                # Display results
                st.success("✅ Answer Generated")
                st.markdown(f"**Answer:** {result['answer']}")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Response Time", f"{response_time:.2f}s")
                with col2:
                    st.metric("🎯 Confidence", f"{result['confidence']:.2f}")
                with col3:
                    st.metric("📊 Method", retrieval_method)
                
                # Retrieved context
                with st.expander("📋 Retrieved Context"):
                    for i, chunk in enumerate(result['retrieved_chunks']):
                        st.markdown(f"**Chunk {i+1}** (Score: {chunk['score']:.3f})")
                        st.text(chunk['text'])
```

#### **7.2 Gradio Alternative**
```python
import gradio as gr

def create_gradio_interface():
    def process_query(query, method, top_k, diversity):
        result = rag_system.process_query(query, method, top_k, diversity)
        return result['answer'], result['confidence'], result['response_time']
    
    interface = gr.Interface(
        fn=process_query,
        inputs=[
            gr.Textbox(label="Financial Question", placeholder="What was the total income in 2023?"),
            gr.Dropdown(["Hybrid", "Dense Only", "Sparse Only"], label="Retrieval Method", value="Hybrid"),
            gr.Slider(1, 10, value=5, label="Top-K Results"),
            gr.Slider(0.5, 1.0, value=0.7, label="Context Diversity")
        ],
        outputs=[
            gr.Textbox(label="Answer"),
            gr.Number(label="Confidence Score"),
            gr.Number(label="Response Time (s)")
        ],
        title="🏦 Financial QA System - RAG (100-token chunks)",
        description="Ask questions about Jaiprakash Associates financial data (2022-2024)"
    )
    
    return interface
```

---

## 🛠️ Implementation Timeline

### **Week 1: Data Processing & Setup** (Simplified)
- [ ] Install missing dependency (rank-bm25)
- [ ] Implement text chunking (100 tokens only)
- [ ] Set up embedding models
- [ ] Create chunk metadata structure

### **Week 2: Indexing & Retrieval**
- [ ] Build ChromaDB dense index (single collection)
- [ ] Implement BM25 sparse indexing
- [ ] Develop hybrid retrieval pipeline
- [ ] Test retrieval accuracy

### **Week 3: Advanced Features**
- [ ] Implement advanced RAG technique
- [ ] Set up response generation model
- [ ] Develop guardrail systems
- [ ] Performance optimization

### **Week 4: Interface & Testing**
- [ ] Build Streamlit interface
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] Documentation and cleanup

---

## 📦 Required Dependencies

```python
# Core ML/NLP
torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
datasets>=2.0.0

# Vector Storage
chromadb>=0.4.0
faiss-cpu>=1.7.4

# Search & Retrieval
rank-bm25>=0.2.2
nltk>=3.8

# UI Framework
streamlit>=1.20.0
gradio>=3.35.0

# Utilities
pandas>=1.5.0
numpy>=1.21.0
tqdm>=4.64.0
```

---

## 🎯 Success Metrics

### **Performance Targets:**
- **Response Time:** < 2 seconds average
- **Retrieval Accuracy:** > 80% relevant chunks in top-5
- **Answer Quality:** > 85% factually correct responses
- **Guardrail Effectiveness:** > 95% inappropriate query detection

### **Evaluation Framework:**
- **Test Set:** 50 financial questions (from our generated Q&A pairs)
- **Metrics:** Accuracy, BLEU score, response time, confidence calibration
- **Comparison:** Against ground truth answers from financial reports

---

This comprehensive plan provides a roadmap for implementing a production-ready RAG system specifically tailored for financial question-answering using your processed Jaiprakash Associates data.

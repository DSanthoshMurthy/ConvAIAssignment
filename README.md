# Financial Question-Answering System

A production-ready financial question-answering system that combines two powerful approaches:

1. RAG (Retrieval-Augmented Generation) for accurate information retrieval
2. Fine-tuned Language Models for domain-specific understanding

## System Architecture

The system operates through two parallel pipelines that work together to provide accurate financial answers:

### 1. RAG Pipeline

```
Query → Hybrid Retrieval → Re-Ranking → Response Generation
```

1. **Text Processing & Chunking**

   - Processes financial documents into 1,520 chunks (100 tokens each)
   - Extracts metadata (quarter, section, financial metrics)
   - Generates Q&A pairs for evaluation
2. **Advanced Retrieval System**

   - **Dense Search**: ChromaDB with sentence-transformers/all-MiniLM-L6-v2
   - **Sparse Search**: BM25 with financial domain optimization
   - **Cross-Encoder Re-Ranking**: Using cross-encoder/ms-marco-MiniLM-L-6-v2
   - **Response Generation**: Template-based + Direct Q&A matching

### 2. Fine-Tuning Pipeline

```
Financial Data → Expert Models → MoE Training → Specialized Model
```

Located in `src/fine_tuning/`, this pipeline creates domain-specialized models:

1. **Data Processing** (`fine_tuning/data/`)

   - Custom dataset preparation for financial domain
   - Advanced Q&A pair generation
   - Financial context enrichment
2. **Expert Models** (`fine_tuning/models/`)

   - Specialized financial domain experts
   - Mixture of Experts (MoE) architecture
   - Advanced model gating and routing
   - Domain-specific embeddings
3. **Training Pipeline** (`fine_tuning/training/`)

   - Financial domain-specific data loaders
   - Custom loss functions and metrics
   - Checkpoint management and validation
   - Performance monitoring
4. **Inference & Validation** (`fine_tuning/inference/`)

   - Model serving and optimization
   - Response quality validation
   - Financial accuracy checks

## Running the System

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Processing

```bash
# Process raw financial data
python data_processor.py

# Generate Q&A pairs
python -m src.preprocessing.xbrl_qa_generator
```

### 3. Train Fine-Tuned Model

```bash
# Download base model
python src/fine_tuning/model_downloader.py

# Train the model
python src/fine_tuning/train_model.py --config configs/training_config.json

# Convert checkpoints if needed
python src/fine_tuning/checkpoint_converter.py
```

### 4. Initialize RAG Pipeline

```bash
# Build indexes
python -m src.rag.embedding_indexer

# Start retrieval system
python -m src.rag.complete_rag_pipeline
```

### 5. Start the Application

```bash
# Launch the Streamlit interface
streamlit run streamlit_app.py
```

The system will be available at: http://localhost:8501

## Project Structure

```
.
├── configs/                  # Configuration files
├── data/                    # Data processing and storage
│   ├── raw/                # Raw financial documents
│   ├── processed/          # Processed data and Q&A pairs
│   ├── chunks/             # Text chunks for retrieval
│   └── indexes/            # Vector and sparse indexes
├── src/
│   ├── preprocessing/      # Data preprocessing modules
│   ├── rag/               # RAG implementation
│   │   ├── embedding_indexer.py
│   │   ├── hybrid_retriever.py
│   │   └── complete_rag_pipeline.py
│   └── fine_tuning/       # Model fine-tuning
│       ├── models/        # Expert models
│       ├── training/      # Training pipeline
│       ├── inference/     # Model serving
│       └── guardrails/    # Quality checks
└── streamlit_app.py       # Web interface
```

## System Requirements

- Python 3.13+
- 4GB+ RAM recommended
- 2GB storage for models and indexes
- Internet connection for initial model downloads

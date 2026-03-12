# Document Intelligence RAG API

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A lightweight **Retrieval-Augmented Generation (RAG) API** that enables intelligent document querying using semantic search and AI-powered answer generation. Upload documents, ask questions in natural language, and receive context-aware answers with source citations.

---

## ✨ Features

- 📄 **Document Upload** - Support for PDF and TXT files
- 🔪 **Smart Chunking** - Automatic text extraction and intelligent document segmentation
- 🧠 **Semantic Embeddings** - Convert text to dense vector representations using sentence-transformers
- 🔍 **Vector Search** - Fast similarity search powered by FAISS
- 💬 **AI Answers** - Context-aware responses using LLM with retrieved document chunks
- 📚 **Source Citations** - Track which document sections were used to generate answers
- 🚀 **RESTful API** - Easy integration with any application
- ⚡ **Fast & Efficient** - Optimized for performance with minimal resource usage

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.9 or higher
- **RAM**: 8GB minimum (16GB recommended for large documents)
- **Storage**: 2GB for models and vector index
- **OS**: Linux, macOS, or Windows

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/GarimaPrakash20/doc-intelligence-rag-api.git
cd doc-intelligence-rag-api
```

2. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

**Note**: On first run, the sentence-transformer model (~80MB) will be downloaded automatically to `~/.cache/torch/sentence_transformers/`.
This takes ~5 minutes but only happens once.

4. **Run the server**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

5. **View API documentation**

Visit `http://localhost:8000/docs` for interactive Swagger documentation

---

## 📋 Requirements

### System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.9+ |
| **RAM** | 8GB minimum, 16GB recommended |
| **Storage** | 2GB for models and vector store |
| **OS** | Linux, macOS, or Windows |
| **GPU** | Optional (CUDA-compatible for faster embeddings) |

### Python Dependencies

**Web Framework**
```
fastapi>=0.109.0        # Modern async web framework
uvicorn>=0.27.0         # ASGI server for FastAPI
python-multipart>=0.0.6 # File upload support
```

**Machine Learning & AI**
```
torch>=2.1.0                      # Deep learning framework
transformers>=4.37.0              # Hugging Face transformers
sentence-transformers>=2.2.0      # Sentence embeddings
accelerate>=0.26.0                # Model acceleration utilities
sentencepiece>=0.1.99             # Tokenization
```

**Vector Store**
```
faiss-cpu>=1.7.4        # CPU-only vector search (use faiss-gpu for GPU support)
```

**Document Processing**
```
pypdf>=3.17.0           # PDF text extraction
numpy>=1.24.0           # Numerical operations
regex>=2023.12.0        # Advanced text processing
```

### Optional Dependencies

- `faiss-gpu` - GPU-accelerated vector search (requires CUDA)
- `pytest>=7.4.0` - Testing framework
- `black>=23.12.0` - Code formatting
- `python-dotenv` - Environment variable management

---

## 🏗️ Architecture

### RAG Pipeline Flow

```
User uploads document
        │
        ▼
    Text Extraction (PyPDF/Plain text)
        │
        ▼
    Text Chunking (512 tokens/chunk)
        │
        ▼
    Embedding Generation (sentence-transformers)
        │
        ▼
    Store in Vector Database (FAISS)
        │
        ▼
    Ready for queries

User submits query
        │
        ▼
    Embed query text
        │
        ▼
    Semantic Search (top-k retrieval)
        │
        ▼
    Retrieve relevant chunks
        │
        ▼
    LLM Answer Generation (with context)
        │
        ▼
    Return answer + source citations
```

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                       │
├─────────────────────────────────────────────────────────┤
│  Routes           │  Services                            │
│  ├─ upload.py     │  ├─ embeddings.py (text → vectors) │
│  └─ query.py      │  ├─ chunker.py (text splitting)    │
│                   │  ├─ vector_store.py (FAISS ops)    │
│                   │  └─ llm.py (answer generation)     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
            ┌──────────────────────────────┐
            │   FAISS Vector Database      │
            │  (384-dimensional embeddings)│
            └──────────────────────────────┘
```

---

## 📂 Project Structure

```
doc-intelligence-rag-api/
├── app/
│   ├── main.py                  # FastAPI application entry point
│   ├── routes/                  # API endpoints
│   │   ├── __init__.py
│   │   ├── upload.py           # Document upload endpoint
│   │   └── query.py            # Query endpoint
│   └── services/               # Business logic
│       ├── __init__.py
│       ├── embeddings.py       # Embedding generation (sentence-transformers)
│       ├── chunker.py          # Text chunking strategies
│       ├── vector_store.py     # FAISS vector database operations
│       └── llm.py              # LLM-based answer generation
├── data/                       # Processed documents and metadata
├── uploads/                    # Temporary uploaded files
├── .gitignore
├── .python-version             # Python version specification
├── README.md
└── requirements.txt            # Python dependencies
```

---

## 🔌 API Endpoints

### 1. Upload Document

**Endpoint**: `POST /upload`

Upload a document (PDF or TXT) for processing and indexing.

**Request**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Response**:
```json
{
  "document_id": "doc_abc123",
  "filename": "document.pdf",
  "status": "processed",
  "chunks_count": 45,
  "processing_time": 2.34
}
```

---

### 2. Query Documents

**Endpoint**: `POST /query`

Ask questions about uploaded documents and receive AI-generated answers.

**Request**:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings of the research?",
    "top_k": 5
  }'
```

**Response**:
```json
{
  "query": "What are the main findings of the research?",
  "answer": "The main findings indicate that...",
  "sources": [
    {
      "document_id": "doc_abc123",
      "chunk_text": "Our research demonstrates...",
      "relevance_score": 0.89,
      "page": 3
    }
  ],
  "processing_time": 1.12
}
```

---

## 📖 How It Works

### Step 1: Document Processing

1. **Text Extraction**: Extracts plain text from uploaded PDFs or text files
2. **Chunking**: Splits text into semantic chunks (default: 512 tokens with 50-token overlap)
3. **Metadata Preservation**: Maintains page numbers, sections, and document structure

### Step 2: Embedding Generation

1. **Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` (default)
2. **Vector Dimensions**: 384-dimensional dense vectors
3. **Speed**: ~1000 chunks/second on CPU, faster with GPU

### Step 3: Vector Storage

1. **Database**: FAISS (Facebook AI Similarity Search)
2. **Index Type**: Flat L2 (exact search) - can be upgraded to IVF for larger datasets
3. **Capacity**: Supports millions of document chunks

### Step 4: Query Processing

1. **Query Embedding**: Converts user query to 384-dimensional vector
2. **Semantic Search**: Finds top-k most similar chunks (default k=5)
3. **Ranking**: Results sorted by cosine similarity

### Step 5: Answer Generation

1. **Context Assembly**: Combines retrieved chunks with original query
2. **LLM Prompt**: Constructs prompt with context and instructions
3. **Answer Generation**: LLM generates coherent answer from context
4. **Source Attribution**: Links answer back to source documents

---

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Vector Store
VECTOR_STORE_PATH=./data/vector_store
INDEX_TYPE=Flat

# Query Configuration
DEFAULT_TOP_K=5
MAX_CONTEXT_LENGTH=2000
```

### Changing the Embedding Model

To use a different embedding model, update your configuration:

```python
# In app/services/embeddings.py
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Higher quality, larger size
# OR
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Smaller, faster
```

---

## 🧪 Testing

### Test Document Upload

```bash
# Upload a PDF
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sample.pdf"

# Expected: 200 OK with document_id
```

### Test Query

```bash
# Query the uploaded document
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the main points"
  }'

# Expected: 200 OK with answer and sources
```

### Run Unit Tests (if implemented)

```bash
pytest tests/ -v
```

---

## 🚀 Deployment

### Docker Deployment (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t doc-rag-api .
docker run -p 8000:8000 -v $(pwd)/data:/app/data doc-rag-api
```

### Cloud Deployment

**AWS**:
- Deploy to EC2 with Docker
- Use S3 for document storage
- Use ECS/Fargate for container orchestration

**Google Cloud**:
- Deploy to Cloud Run
- Use Cloud Storage for documents
- Use Cloud SQL for metadata

**Azure**:
- Deploy to Azure Container Instances
- Use Blob Storage for documents

### Production Considerations

- Use environment variables for sensitive configuration
- Implement authentication/authorization (JWT, API keys)
- Add rate limiting
- Set up logging and monitoring
- Use a production ASGI server (gunicorn + uvicorn workers)
- Enable HTTPS with SSL certificates
- Consider using a managed vector database (Pinecone, Weaviate, Qdrant)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to function signatures
- Write docstrings for all public functions
- Add tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Technologies Used

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Sentence Transformers](https://www.sbert.net/) - State-of-the-art sentence embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [PyPDF](https://pypdf.readthedocs.io/) - PDF text extraction
- [Hugging Face Transformers](https://huggingface.co/transformers/) - NLP models

### Inspiration

This project was inspired by the growing need for intelligent document search and question-answering systems. RAG combines the benefits of retrieval-based and generation-based approaches for more accurate and contextual responses.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/GarimaPrakash20/doc-intelligence-rag-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GarimaPrakash20/doc-intelligence-rag-api/discussions)

---

**Built with ❤️ by [GarimaPrakash20](https://github.com/GarimaPrakash20)**

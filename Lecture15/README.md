# Autonomous MirAI Student Policy Advisor

A Python-based Retrieval-Augmented Generation (RAG) system for querying and advising on student policies using LangChain, Google Gemini, ChromaDB, FastAPI, and Streamlit.

## Project Structure

```text
├── backend.py            # FastAPI backend service
├── frontend.py           # Streamlit user interface
├── rag_pipeline.py       # LangChain + Gemini + ChromaDB RAG logic
├── evaluate.py           # Evaluation script for pipeline quality
├── requirements.txt      # Python dependencies
├── .env.example          # Template for environment variables
├── .gitignore            # Git ignore configuration
├── data/                 # Directory for source policy documents/PDFs
├── chroma_db/            # ChromaDB persistent vector database storage
└── tests/                # Test suite
```

## Setup Instructions

### 1. Prerequisites
- Python 3.9+ (Python 3.11+ recommended)
- Google AI Studio API Key

### 2. Virtual Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the `.env.example` template and configure your API key:
```bash
cp .env.example .env
```
Edit `.env` and set:
- `GOOGLE_API_KEY`: Your Google Gemini API key
- `EMBEDDING_MODEL`: `models/embedding-001` (or desired embedding model)
- `GENERATION_MODEL`: `gemini-1.5-flash` (or desired Gemini model)
- `CHROMA_PERSIST_DIRECTORY`: `chroma_db`
- `BACKEND_URL`: `http://localhost:8000`

### 4. Running Tests
```bash
pytest
```

## Next Steps
- Implement document ingestion and vector indexing in `rag_pipeline.py`.
- Build FastAPI endpoints in `backend.py`.
- Build the conversational interface in `frontend.py`.
- Implement RAG evaluation in `evaluate.py`.

# Autonomous MirAI Student Policy Advisor

An enterprise-grade, evidence-grounded Retrieval-Augmented Generation (RAG) system engineered to advise students on institutional policies. Built with **LangChain**, **Google Gemini**, **ChromaDB**, **FastAPI**, and **Streamlit**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Setup & Installation](#-setup--installation)
- [Environment Configuration](#-environment-configuration)
- [Handbook Ingestion](#-handbook-ingestion)
- [Running the Services](#-running-the-services)
- [API Reference & Examples](#-api-reference--examples)
- [RAG Guardrails & Grounding](#-rag-guardrails--grounding)
- [Retrieval Modes](#-retrieval-modes)
- [Evaluation & Comparison Benchmark](#-evaluation--comparison-benchmark)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Model Limitations & API Considerations](#-model-limitations--api-considerations)
- [Troubleshooting](#-troubleshooting)
- [Submission Checklist](#-submission-checklist)

---

## 🌟 Overview

The **Autonomous MirAI Student Policy Advisor** provides students with instant, accurate, and verified policy guidance derived exclusively from the official *Mirai School of Technology Student Policy Handbook 2026*.

### Core Capabilities
- **Strict Evidence Grounding**: The advisor only answers if substantiated by retrieved handbook text.
- **Hallucination Prevention**: Explicit penalty guardrails forbid inventing monetary fines, contacts, deadlines, or disciplinary steps.
- **Deterministic Refusal**: Standardized refusal response when context is ambiguous or insufficient.
- **Dual Retrieval Engine**: Configurable between standard similarity vector search and `MultiQueryRetriever` query expansion.
- **Auditable Citations**: Every generated response includes exact source filenames, page numbers, and expandable context chunks.
- **Automated LLM-as-a-Judge Benchmark**: Evaluates factual precision, completeness, and adherence to safety criteria with incremental CSV logging.

---

## 🏛 System Architecture

```mermaid
flowchart TD
    subgraph Ingestion Pipeline
        PDF[("Mirai Policy Handbook (PDF)")] -->|PyPDFLoader| Load[PDF Parsing & Page Tracking]
        Load -->|RecursiveCharacterTextSplitter| Split[Chunking (1000 chars, 200 overlap)]
        Split -->|GoogleGenerativeAIEmbeddings| Embed[Embedding Generation]
        Embed -->|Chroma.from_documents| Chroma[Persistent ChromaDB (mirai_policies)]
    end

    subgraph User Interaction
        User([Student / Admin]) <-->|HTTP / WebSocket| UI[Streamlit Frontend (:8501)]
        UI <-->|REST API| API[FastAPI Backend (:8000)]
    end

    subgraph RAG Generation & Guardrails
        API -->|Query| Route{Retrieval Mode}
        Route -->|Standard| VRet[Vector Similarity Search]
        Route -->|MultiQuery| MQRet[Gemini Query Expansion + Multi-Search]
        VRet --> Chroma
        MQRet --> Chroma
        Chroma -->|Relevant Chunks| Val[Evidence Validation & Deduplication]
        Val -->|Formatted Context + System Prompt| LCEL[LCEL Generation Chain]
        LCEL -->|Gemini 3.8 Flash (temp=0.0)| Response[Grounded Answer + Page Citations]
        Response --> API
    end

    subgraph LLM Judge Benchmark
        API -.-> Eval[evaluate.py]
        Eval -->|Run Q1-Q4| Benchmark[Comparison Engine]
        Benchmark --> CSV1[(rag_eval_scores.csv)]
        Benchmark --> CSV2[(rag_comparison_report.csv)]
    end
```

---

## 🛠 Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Core Framework** | Python 3.9+ / LangChain 0.3+ | Orchestration, document loading, and LCEL chaining |
| **LLM & Embeddings** | Google Gemini (`gemini-3.8-flash`) | Strict zero-temperature reasoning and vector embedding |
| **Vector Store** | ChromaDB (`chromadb`) | Local persistent vector indexing (`mirai_policies` collection) |
| **Backend API** | FastAPI / Uvicorn | High-performance REST endpoints with Pydantic validation |
| **Frontend UI** | Streamlit | Responsive student chat and administrative document ingestion |
| **Evaluation** | Custom LLM-as-a-Judge | Automated grading, fingerprint-based caching, and CSV reporting |
| **Test Suite** | Pytest | 66+ offline unit, regression, and integration tests |

---

## 📂 Repository Structure

```text
Lecture15/
├── backend.py                  # FastAPI REST backend service (/health, /status, /ingest, /chat)
├── frontend.py                 # Streamlit chat interface with citation inspector & PDF uploader
├── rag_pipeline.py             # Ingestion, ChromaDB storage, retrieval, and LCEL generation logic
├── evaluate.py                 # LLM judge benchmark and side-by-side comparison workflow
├── requirements.txt            # Locked project dependencies
├── .env.example                # Template for required environment variables
├── .gitignore                  # Git ignore rules for secrets, DB files, and environments
├── README.md                   # System documentation and submission guide
├── data/                       # Source handbook PDFs (e.g. Mirai_SoT_Policy_Handbook_2026.pdf)
├── chroma_db/                  # Persistent ChromaDB vector storage (auto-generated)
├── rag_eval_scores.csv         # Full historical evaluation records log
├── rag_comparison_report.csv   # Side-by-side Standard vs MultiQuery comparison report
└── tests/                      # Pytest automated test suite
    ├── test_backend.py         # REST API endpoint tests
    ├── test_evaluate.py        # Judge parsing, fingerprinting, comparison, and cache tests
    ├── test_frontend.py        # Streamlit HTTP client tests
    ├── test_placeholder.py     # Environment sanity tests
    └── test_rag_pipeline.py    # Document ingestion, retrieval, refusal, and cert tests
```

---

## ⚙️ Prerequisites

1. **Python 3.9 or higher** (Python 3.10+ recommended).
2. **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/)).
3. **Official Handbook PDF**: `Mirai_SoT_Policy_Handbook_2026.pdf` placed in `data/`.

---

## 🚀 Setup & Installation

### 1. Clone & Navigate to Project Directory
```bash
cd Lecture15
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

Copy `.env.example` to create your local `.env`:
```bash
cp .env.example .env
```

Edit `.env` with your actual settings:
```dotenv
# Google Gemini API Key
GOOGLE_API_KEY=your_actual_gemini_api_key_here

# Model Configurations
EMBEDDING_MODEL=text-embedding-004
GENERATION_MODEL=gemini-3.8-flash

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=chroma_db

# Backend URL for Frontend Communication
BACKEND_URL=http://127.0.0.1:8000
```

> **Security Note**: Never commit `.env` or hardcode API keys. `.env` is ignored by `.gitignore`.

---

## 📄 Handbook Ingestion

Place the official handbook in the `data/` directory:
```bash
mkdir -p data
cp /path/to/Mirai_SoT_Policy_Handbook_2026.pdf data/
```

To ingest the document into ChromaDB programmatically or via CLI:
```bash
python -c "import rag_pipeline; res = rag_pipeline.ingest_pdf(); print(res)"
```

Alternatively, upload the PDF directly through the **Admin Handbook Ingestion** panel in the Streamlit UI or via `POST /ingest`.

---

## 💻 Running the Services

### 1. Launch FastAPI Backend
```bash
uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at: `http://127.0.0.1:8000/docs`

### 2. Launch Streamlit Frontend
In a new terminal:
```bash
source .venv/bin/activate
streamlit run frontend.py --server.port 8501
```
The student advisor interface will be accessible at: `http://localhost:8501`

---

## 📡 API Reference & Examples

### Health Check
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
```json
{
  "status": "healthy",
  "service": "MirAI Student Policy Advisor API"
}
```

### Check Database Status
```bash
curl -X GET "http://127.0.0.1:8000/status"
```
```json
{
  "status": "ready",
  "collection_name": "mirai_policies",
  "chunk_count": 50,
  "is_ingested": true
}
```

### Ingest Policy Document
```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "file=@data/Mirai_SoT_Policy_Handbook_2026.pdf"
```

### Query Advisor (`/chat`)
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "I have 72% attendance. How many attendance marks will I get?",
    "mode": "standard",
    "top_k": 4
  }'
```
**Response**:
```json
{
  "question": "I have 72% attendance. How many attendance marks will I get?",
  "answer": "According to the handbook, an attendance of 72% falls within the 60% – 74.99% bracket, which awards 4 marks out of 10.",
  "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
  "pages": [3],
  "retrieved_passages": [
    {
      "content": "Attendance Marks Allocation:\n60% – 74.99%: 4 marks...",
      "source": "Mirai_SoT_Policy_Handbook_2026.pdf",
      "page": 3,
      "chunk_id": "chunk_0"
    }
  ],
  "retrieval_mode": "standard",
  "evidence_found": true
}
```

---

## 🛡 RAG Guardrails & Grounding

The advisor adheres to strict safety and factual grounding rules:

1. **Zero Hallucination Mandate**: The model is forbidden from guessing or synthesizing policies not directly present in retrieved context chunks.
2. **Explicit Refusal**: If evidence is absent, irrelevant, or below the similarity cutoff, the system returns:
   > *"I could not find sufficient information in the official Mirai School of Technology Student Policy Handbook to answer your question."*
3. **No Fabricated Fines**: Disciplinary matters (such as smoking on campus) mention disciplinary committee action without fabricating monetary amounts.
4. **Multi-Source Synthesis**: For complex questions (e.g. Ratnam campus medical leave), multiple policy clauses across pages are synthesized into a coherent procedure.

---

## 🔍 Retrieval Modes

| Feature | Standard Vector Retrieval | MultiQueryRetriever |
|---|---|---|
| **Mechanism** | Cosine similarity on dense query embedding | LLM-generated query variations + multi-pass retrieval |
| **Strengths** | Fast, deterministic, single embedding call | Captures synonyms, colloquial terms, and multi-part queries |
| **API Overhead** | 0 LLM generation calls | 1 LLM generation call for query expansion |
| **Configuration** | `mode="standard"` | `mode="multi_query"` |

---

## 📊 Evaluation & Comparison Benchmark

[`evaluate.py`](evaluate.py) provides a side-by-side evaluation harness for comparing retrieval modes against four mandatory certification questions:

1. **Q1 (Attendance)**: `72% attendance` $\rightarrow$ Exactly **4 marks**.
2. **Q2 (Medical Leave)**: `Ratnam campus medical leave` $\rightarrow$ **Yashaswini Ma'am within 7 days**.
3. **Q3 (New Society)**: `Cybersecurity society` $\rightarrow$ **40% batch support + Faculty Coordinator** (not direct Management).
4. **Q4 (Smoking Policy)**: `Smoking on campus fine` $\rightarrow$ **Disciplinary action; no monetary fine**.

### Execution Commands

#### 1. Run Reproducible Comparison Workflow (Recommended):
```bash
python evaluate.py --mode compare --output rag_eval_scores.csv --comparison-output rag_comparison_report.csv
```

#### 2. Run Single Mode:
```bash
python evaluate.py --mode standard
python evaluate.py --mode multi_query
```

#### 3. Invalidate / Bypass Cache:
```bash
python evaluate.py --mode compare --no-cache
```

### Benchmark Metrics & CSV Schema
- **`rag_eval_scores.csv`**: Contains `question`, `retrieval_mode`, `retrieved_context`, `generated_answer`, `score` (1–5), `reasoning`, `passed` (boolean), and `timestamp`.
- **`rag_comparison_report.csv`**: Contains side-by-side delta columns (`std_score`, `mq_score`, `score_difference`, chunk counts, page citations, answer excerpts, and handbook fingerprint).

---

## 🧪 Testing & Quality Assurance

The test suite runs completely offline with deterministic mocks and fixtures:

```bash
pytest -v
```

### Test Coverage Breakdown
- **[`tests/test_backend.py`](tests/test_backend.py)**: Tests `/health`, `/status`, `/ingest`, `/chat`, and error handling.
- **[`tests/test_frontend.py`](tests/test_frontend.py)**: Tests HTTP client interactions, UI state formatting, timeouts, and connection retries.
- **[`tests/test_rag_pipeline.py`](tests/test_rag_pipeline.py)**: Tests chunking overlap, duplicate prevention, vector indexing, refusal triggers, and Q1–Q4 certification logic.
- **[`tests/test_evaluate.py`](tests/test_evaluate.py)**: Tests judge JSON parsing, multiline CSV formatting, fingerprint caching, and comparison reporting.

---

## ⚠️ Model Limitations & API Considerations

- **Gemini Free-Tier Rate Limits**: Free-tier Google AI Studio keys have a daily ceiling of 20 requests/day for `gemini-3.8-flash`. When rate limits are reached, the API returns `429 ResourceExhausted`. In production, use a Pay-As-You-Go API key to ensure uninterrupted generation.
- **Model Availability**: Always ensure your target generation and embedding models are active in your Google AI Studio account. If `gemini-3.8-flash` is unavailable, configure an approved active model in `.env`.

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `400 Bad Request: No handbook ingested` | Vector database empty | Ingest the handbook PDF via `POST /ingest` or admin UI panel. |
| `500 ConfigurationError: GOOGLE_API_KEY missing` | `.env` file not configured | Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`. |
| `429 ResourceExhausted` | Gemini free-tier daily quota exceeded | Wait for daily quota reset or attach a billing account in Google AI Studio. |
| Streamlit cannot reach backend | FastAPI backend not running | Start backend via `uvicorn backend:app --port 8000`. |
| Retrieval returns irrelevant context | Obsolete index chunks | Re-run ingestion; `ingest_pdf()` automatically replaces outdated chunks. |

---

## ✅ Submission Checklist

- [x] All 66 automated tests pass cleanly (`pytest -v`).
- [x] Vector store persistence in ChromaDB verified (`mirai_policies`).
- [x] Strict LCEL generation chain with hallucination prevention active.
- [x] All 4 mandatory certification tests implemented and verified.
- [x] Fast, reproducible retrieval comparison workflow implemented.
- [x] `.env` excluded from version control via `.gitignore`.
- [x] `.env.example` provided with placeholder values.
- [x] Clean architecture separating backend, frontend, pipeline, and evaluation.


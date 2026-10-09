"""FastAPI backend service for Autonomous MirAI Student Policy Advisor.

Exposes endpoints for document ingestion and policy question-answering.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag_pipeline

# Load environment variables
load_dotenv()

app = FastAPI(
    title="MirAI Student Policy Advisor Backend",
    description="API for student policy question answering with LangChain, Gemini, and ChromaDB.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The student policy query.")
    mode: str = Field(default="standard", description="Retrieval mode: 'standard' or 'multi_query'.")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve.")
    score_threshold: Optional[float] = Field(default=None, description="Optional distance score cutoff.")


class PassageItem(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    pages: List[int]
    retrieved_passages: List[PassageItem]
    retrieval_mode: str
    evidence_found: bool


class IngestResponse(BaseModel):
    status: str
    file_name: str
    file_path: str
    page_count: int
    chunk_count: int
    collection_name: str
    persist_directory: str


class StatusResponse(BaseModel):
    status: str
    collection_name: str
    chunk_count: int
    is_ingested: bool


@app.get("/", tags=["Health"])
def root() -> Dict[str, str]:
    """Root status endpoint."""
    return {
        "service": "MirAI Student Policy Advisor API",
        "status": "online",
    }


@app.get("/health", tags=["Health"])
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "MirAI Student Policy Advisor API",
    }


@app.get("/status", response_model=StatusResponse, tags=["Health"])

def get_status() -> StatusResponse:
    """Check whether the policy handbook is currently indexed and ready."""
    try:
        store = rag_pipeline.get_vector_store()
        count = store._collection.count()
        return StatusResponse(
            status="ready" if count > 0 else "empty",
            collection_name=rag_pipeline.COLLECTION_NAME,
            chunk_count=count,
            is_ingested=count > 0,
        )
    except Exception:
        return StatusResponse(
            status="not_initialized",
            collection_name=rag_pipeline.COLLECTION_NAME,
            chunk_count=0,
            is_ingested=False,
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and ingest a new policy handbook PDF into ChromaDB."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported for ingestion.",
        )

    # Save uploaded file to project data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    saved_path = data_dir / file.filename

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = rag_pipeline.ingest_pdf(pdf_path=saved_path)
        return IngestResponse(**result)
    except rag_pipeline.ConfigurationError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(err),
        ) from err
    except (rag_pipeline.PDFNotFoundError, rag_pipeline.PDFProcessingError, rag_pipeline.EmptyDocumentError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err
    except rag_pipeline.EmbeddingError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(err),
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected ingestion failure: {err}",
        ) from err


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """Process a student question and return a policy-grounded answer."""
    try:
        result = rag_pipeline.answer_question(
            question=request.question,
            mode=request.mode,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
        return ChatResponse(**result)
    except rag_pipeline.ConfigurationError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(err),
        ) from err
    except rag_pipeline.EmptyCollectionError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No handbook has been ingested yet. Please upload and ingest a handbook PDF first.",
        ) from err
    except (rag_pipeline.RetrievalError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err
    except rag_pipeline.GenerationError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(err),
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal service error: {err}",
        ) from err


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

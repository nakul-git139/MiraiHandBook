"""RAG pipeline module for Autonomous MirAI Student Policy Advisor.

Implements document ingestion, persistent vector store management with ChromaDB,
standard & MultiQuery retrieval, and strict LCEL answer generation with Google Gemini.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import chromadb
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "mirai_policies"
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_GENERATION_MODEL = "gemini-3.8-flash"
DEFAULT_PERSIST_DIRECTORY = "chroma_db"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PDF_NAME = "Mirai_SoT_Policy_Handbook_2026.pdf"

STRICT_REFUSAL_MESSAGE = (
    "I could not find sufficient information in the official Mirai School of Technology "
    "Student Policy Handbook to answer your question."
)

SYSTEM_PROMPT = """You are the Autonomous MirAI Student Policy Advisor. Your sole role is to answer questions from students strictly and accurately using ONLY the provided official Mirai School of Technology Student Policy Handbook Context.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided Handbook Context.
2. If the context does not contain clear, factual, and sufficient evidence to answer the student's question, you MUST respond EXACTLY with:
"I could not find sufficient information in the official Mirai School of Technology Student Policy Handbook to answer your question."
3. NEVER invent, hallucinate, extrapolate, or assume policies, attendance percentages, penalties, fines, emails, office contacts, deadlines, or escalation workflows.
4. When answering questions that span multiple policies or sections, synthesize the information concisely and accurately.
5. Provide precise, factual, and polite answers without speculative advice."""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Handbook Context:\n{context}\n\nStudent Question: {question}\n\nHelpful Policy Answer:",
        ),
    ]
)

# Global in-memory cache to prevent repeated re-initialization
_CACHE: Dict[str, Any] = {
    "embeddings": {},
    "llms": {},
    "vector_stores": {},
}


class PolicyAdvisorError(Exception):
    """Base exception for MirAI Policy Advisor."""


class PDFNotFoundError(PolicyAdvisorError, FileNotFoundError):
    """Raised when the target PDF handbook file is not found."""


class PDFProcessingError(PolicyAdvisorError):
    """Raised when the PDF is malformed, corrupt, or cannot be parsed."""


class EmptyDocumentError(PolicyAdvisorError):
    """Raised when the PDF yields empty or whitespace-only content."""


class ConfigurationError(PolicyAdvisorError):
    """Raised when required configuration (e.g. API keys) is missing."""


class EmbeddingError(PolicyAdvisorError):
    """Raised when vector embedding generation fails."""


class RetrievalError(PolicyAdvisorError):
    """Raised when retrieval operations fail."""


class EmptyCollectionError(RetrievalError):
    """Raised when retrieval is attempted on an empty or missing collection."""


class GenerationError(PolicyAdvisorError):
    """Raised when answer generation fails."""


def get_embedding_function(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> GoogleGenerativeAIEmbeddings:
    """Initialize and return cached Google Generative AI Embeddings.

    Args:
        api_key: Optional API key. If not provided, reads GOOGLE_API_KEY from env.
        model: Optional embedding model name. Defaults to EMBEDDING_MODEL env or text-embedding-004.

    Returns:
        GoogleGenerativeAIEmbeddings instance.

    Raises:
        ConfigurationError: If GOOGLE_API_KEY is not set or empty.
    """
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        raise ConfigurationError(
            "GOOGLE_API_KEY is missing or empty. Please set it in your .env file or pass it explicitly."
        )

    resolved_model = model or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    cache_key = f"{resolved_model}_{resolved_api_key[:8]}"

    if cache_key not in _CACHE["embeddings"]:
        _CACHE["embeddings"][cache_key] = GoogleGenerativeAIEmbeddings(
            model=resolved_model,
            google_api_key=resolved_api_key,
        )

    return _CACHE["embeddings"][cache_key]


def get_llm(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatGoogleGenerativeAI:
    """Initialize and return cached Google Gemini Chat Model.

    Args:
        api_key: Optional API key. If not provided, reads GOOGLE_API_KEY from env.
        model: Optional generation model name. Defaults to GENERATION_MODEL env or gemini-3.8-flash.
        temperature: Sampling temperature (defaults to 0.0 for deterministic answers).

    Returns:
        ChatGoogleGenerativeAI instance.

    Raises:
        ConfigurationError: If GOOGLE_API_KEY is not set or empty.
    """
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        raise ConfigurationError(
            "GOOGLE_API_KEY is missing or empty. Please set it in your .env file or pass it explicitly."
        )

    resolved_model = model or os.getenv("GENERATION_MODEL") or DEFAULT_GENERATION_MODEL
    cache_key = f"{resolved_model}_{temperature}_{resolved_api_key[:8]}"

    if cache_key not in _CACHE["llms"]:
        _CACHE["llms"][cache_key] = ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=resolved_api_key,
            temperature=temperature,
            max_retries=2,
        )

    return _CACHE["llms"][cache_key]



def load_and_validate_pdf(pdf_path: Union[str, Path]) -> List[Document]:
    """Load and validate a PDF handbook document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of Document objects representing pages.

    Raises:
        PDFNotFoundError: If the PDF does not exist.
        PDFProcessingError: If the PDF is malformed or unreadable.
        EmptyDocumentError: If the document contains no extractable text.
    """
    path_obj = Path(pdf_path).resolve()
    if not path_obj.is_file():
        raise PDFNotFoundError(f"PDF document not found at: {path_obj}")

    if path_obj.stat().st_size == 0:
        raise PDFProcessingError(f"PDF file is empty (0 bytes): {path_obj}")

    try:
        loader = PyPDFLoader(str(path_obj))
        pages = loader.load()
    except Exception as exc:
        raise PDFProcessingError(f"Failed to parse PDF document at {path_obj}: {exc}") from exc

    if not pages:
        raise EmptyDocumentError(f"No pages loaded from PDF document: {path_obj}")

    # Validate that at least some text was extracted across all pages
    total_text = "".join(page.page_content.strip() for page in pages)
    if not total_text:
        raise EmptyDocumentError(
            f"Extracted content from PDF document is entirely empty or whitespace: {path_obj}"
        )

    # Standardize source metadata to file name while preserving original page indices
    for page in pages:
        page.metadata["source"] = path_obj.name
        page.metadata["file_path"] = str(path_obj)

    return pages


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into chunks with preserved metadata and unique IDs.

    Args:
        documents: List of loaded page Documents.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of chunked Document objects with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    # Ensure source and page metadata are present and clean on every chunk
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "document")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{source}_p{page}_c{idx}"

    return chunks


def get_vector_store(
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = COLLECTION_NAME,
    embedding_function: Optional[Embeddings] = None,
) -> Chroma:
    """Retrieve or create the Chroma vector store instance with caching.

    Args:
        persist_directory: Optional path for Chroma persistence.
        collection_name: Collection name, defaults to mirai_policies.
        embedding_function: Optional embeddings instance.

    Returns:
        Chroma vector store instance.
    """
    resolved_persist_dir = str(
        persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY") or DEFAULT_PERSIST_DIRECTORY
    )
    resolved_embedding = embedding_function or get_embedding_function()

    cache_key = f"{resolved_persist_dir}_{collection_name}"
    if cache_key not in _CACHE["vector_stores"]:
        client = chromadb.PersistentClient(path=resolved_persist_dir)
        _CACHE["vector_stores"][cache_key] = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=resolved_embedding,
        )

    return _CACHE["vector_stores"][cache_key]


def ingest_pdf(
    pdf_path: Optional[Union[str, Path]] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = COLLECTION_NAME,
    embedding_function: Optional[Embeddings] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """Ingest a PDF policy handbook into the persistent ChromaDB collection.

    Cleans up any existing collection records first to guarantee idempotency,
    preventing duplicate chunks or leftover obsolete content from previous versions.

    Args:
        pdf_path: Path to the PDF file. Defaults to data/Mirai_SoT_Policy_Handbook_2026.pdf.
        persist_directory: ChromaDB persist directory. Defaults to env or chroma_db.
        collection_name: Collection name. Defaults to mirai_policies.
        embedding_function: Optional custom Embeddings instance.
        chunk_size: Chunk size in characters. Defaults to 1000.
        chunk_overlap: Chunk overlap in characters. Defaults to 200.

    Returns:
        Dictionary containing ingestion metadata:
            - status: "success"
            - file_name: Name of ingested file
            - file_path: Absolute path of ingested file
            - page_count: Total loaded pages
            - chunk_count: Total generated chunks
            - collection_name: Target Chroma collection name
            - persist_directory: Path to Chroma database directory

    Raises:
        PDFNotFoundError: If PDF does not exist.
        PDFProcessingError: If PDF cannot be parsed.
        EmptyDocumentError: If PDF has no extractable text.
        ConfigurationError: If API key is missing.
        EmbeddingError: If embedding generation or storage fails.
    """
    if pdf_path is None:
        default_dir = Path(__file__).parent / "data"
        target_path = default_dir / DEFAULT_PDF_NAME
        if not target_path.exists():
            # Check for alternative name variants such as 'Mirai_SoT_Policy_Handbook_2026 (1).pdf'
            pdf_candidates = list(default_dir.glob("*Mirai*Policy*.pdf")) or list(default_dir.glob("*.pdf"))
            if pdf_candidates:
                target_path = pdf_candidates[0]
    else:
        target_path = Path(pdf_path)

    # 1. Load and validate PDF
    pages = load_and_validate_pdf(target_path)

    # 2. Chunk documents
    chunks = split_documents(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 3. Resolve embedding function
    embeddings = embedding_function or get_embedding_function()

    # 4. Resolve persistence directory
    resolved_persist_dir = str(
        persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY") or DEFAULT_PERSIST_DIRECTORY
    )

    # 5. Persistent Chroma Client & idempotency management
    try:
        client = chromadb.PersistentClient(path=resolved_persist_dir)
        # Clear existing records from collection to guarantee idempotency and remove obsolete chunks
        try:
            existing_coll = client.get_collection(name=collection_name)
            existing_ids = existing_coll.get()["ids"]
            if existing_ids:
                existing_coll.delete(ids=existing_ids)
        except Exception:
            pass

        # Create/retrieve Chroma vector store and add fresh chunks
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

        chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        vector_store.add_documents(documents=chunks, ids=chunk_ids)

        # Invalidate vector store cache so new state is reflected
        cache_key = f"{resolved_persist_dir}_{collection_name}"
        _CACHE["vector_stores"][cache_key] = vector_store
    except PolicyAdvisorError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Failed to generate embeddings and populate ChromaDB: {exc}") from exc

    return {
        "status": "success",
        "file_name": target_path.name,
        "file_path": str(target_path.resolve()),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "collection_name": collection_name,
        "persist_directory": resolved_persist_dir,
    }


def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """Deduplicate retrieved Document objects by chunk_id or content while preserving order.

    Args:
        documents: List of retrieved Documents.

    Returns:
        Deduplicated list of Documents.
    """
    seen_keys: Set[str] = set()
    deduped: List[Document] = []

    for doc in documents:
        key = doc.metadata.get("chunk_id") or doc.page_content.strip()
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(doc)

    return deduped


def validate_evidence(
    query: str,
    documents: List[Document],
    min_content_length: int = 15,
) -> bool:
    """Validate whether retrieved documents contain sufficient evidence for generation.

    Performs basic evidence validation prior to calling LLM generation:
    - Verifies document collection is non-empty.
    - Verifies retrieved documents contain substantive non-whitespace text.

    Args:
        query: Student user question.
        documents: Retrieved documents.
        min_content_length: Minimum total characters required across documents.

    Returns:
        True if evidence meets basic validity standards, False otherwise.
    """
    if not documents:
        return False

    total_chars = sum(len(doc.page_content.strip()) for doc in documents)
    if total_chars < min_content_length:
        return False

    return True


def retrieve_context(
    query: str,
    mode: str = "standard",
    top_k: int = DEFAULT_TOP_K,
    score_threshold: Optional[float] = None,
    vector_store: Optional[Chroma] = None,
    llm: Optional[BaseChatModel] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = COLLECTION_NAME,
    embedding_function: Optional[Embeddings] = None,
) -> List[Document]:
    """Retrieve relevant context chunks from the policy handbook collection.

    Supports both 'standard' similarity search and 'multi_query' expansion retrieval.

    Note on similarity thresholds:
    Embedding models produce different vector distributions and distance metrics
    (e.g., L2 vs Cosine distance). A universal threshold cannot be assumed across models;
    when score_threshold is supplied, results exceeding the maximum distance cutoff are filtered out.

    Args:
        query: Question string.
        mode: Retrieval mode ('standard' or 'multi_query').
        top_k: Number of chunks to retrieve.
        score_threshold: Optional maximum distance / minimum similarity threshold.
        vector_store: Optional Chroma vector store instance.
        llm: Optional LLM instance (used for MultiQuery expansion).
        persist_directory: Optional ChromaDB path.
        collection_name: Collection name.
        embedding_function: Optional embedding instance.

    Returns:
        List of deduplicated relevant Document chunks.

    Raises:
        ValueError: If mode is invalid or query is malformed.
        EmptyCollectionError: If target vector collection has zero items.
        RetrievalError: If retrieval or query expansion fails.
    """
    if not query or not query.strip():
        return []

    normalized_mode = mode.strip().lower()
    if normalized_mode not in ("standard", "multi_query"):
        raise ValueError(
            f"Unsupported retrieval mode '{mode}'. Supported modes are 'standard' and 'multi_query'."
        )

    store = vector_store or get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    # Check collection population
    try:
        if store._collection.count() == 0:
            raise EmptyCollectionError(
                f"ChromaDB collection '{collection_name}' is empty. Ingest the handbook first."
            )
    except EmptyCollectionError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Failed to access ChromaDB collection: {exc}") from exc

    try:
        if normalized_mode == "multi_query":
            generator_llm = llm or get_llm()
            base_retriever = store.as_retriever(search_kwargs={"k": top_k})
            multi_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=generator_llm,
            )
            raw_docs = multi_retriever.invoke(query)
        else:
            # Standard vector similarity retrieval
            if score_threshold is not None:
                docs_and_scores = store.similarity_search_with_score(query, k=top_k)
                # Chroma returns L2/cosine distance where lower is closer
                raw_docs = [doc for doc, score in docs_and_scores if score <= score_threshold]
            else:
                raw_docs = store.similarity_search(query, k=top_k)

        return deduplicate_documents(raw_docs)
    except PolicyAdvisorError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Retrieval failed for query '{query}': {exc}") from exc


def answer_question(
    question: str,
    mode: str = "standard",
    top_k: int = DEFAULT_TOP_K,
    score_threshold: Optional[float] = None,
    vector_store: Optional[Chroma] = None,
    llm: Optional[BaseChatModel] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = COLLECTION_NAME,
    embedding_function: Optional[Embeddings] = None,
) -> Dict[str, Any]:
    """Generate a verified, policy-grounded answer using LCEL and retrieved context.

    Executes:
    1. Context retrieval (standard or multi-query).
    2. Evidence validation before LLM invocation.
    3. Strict LCEL generation chain if evidence is valid; refusal otherwise.
    4. Structured return format with source attribution, page citations, and passages.

    Args:
        question: Student query.
        mode: 'standard' or 'multi_query'.
        top_k: Number of retrieved context chunks.
        score_threshold: Optional similarity cutoff.
        vector_store: Optional Chroma vector store.
        llm: Optional LLM instance.
        persist_directory: Optional Chroma persistence directory.
        collection_name: Collection name.
        embedding_function: Optional embeddings instance.

    Returns:
        Structured dictionary:
            - question: Original question
            - answer: Policy answer or strict refusal
            - sources: Unique list of source documents
            - pages: Sorted list of 1-indexed page numbers
            - retrieved_passages: List of passage dicts with text and metadata
            - retrieval_mode: Mode used ('standard' | 'multi_query')
            - evidence_found: Boolean flag indicating if evidence was validated

    Raises:
        ConfigurationError: If API key is missing.
        RetrievalError: If vector search or collection access fails.
        GenerationError: If LLM generation fails.
    """
    if not question or not question.strip():
        return {
            "question": question,
            "answer": STRICT_REFUSAL_MESSAGE,
            "sources": [],
            "pages": [],
            "retrieved_passages": [],
            "retrieval_mode": mode,
            "evidence_found": False,
        }

    # 1. Retrieve context chunks
    try:
        retrieved_docs = retrieve_context(
            query=question,
            mode=mode,
            top_k=top_k,
            score_threshold=score_threshold,
            vector_store=vector_store,
            llm=llm,
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
    except EmptyCollectionError:
        return {
            "question": question,
            "answer": STRICT_REFUSAL_MESSAGE,
            "sources": [],
            "pages": [],
            "retrieved_passages": [],
            "retrieval_mode": mode,
            "evidence_found": False,
        }

    # 2. Evidence validation before LLM invocation
    if not validate_evidence(question, retrieved_docs):
        return {
            "question": question,
            "answer": STRICT_REFUSAL_MESSAGE,
            "sources": [],
            "pages": [],
            "retrieved_passages": [],
            "retrieval_mode": mode,
            "evidence_found": False,
        }

    # Extract source files and 1-indexed page numbers without fabricating missing fields
    sources = sorted(list({str(doc.metadata["source"]) for doc in retrieved_docs if doc.metadata.get("source")}))
    pages = sorted(list({
        int(doc.metadata["page"]) + 1
        for doc in retrieved_docs
        if "page" in doc.metadata and doc.metadata["page"] is not None and str(doc.metadata["page"]).isdigit()
    }))

    passages = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": (int(doc.metadata["page"]) + 1) if ("page" in doc.metadata and doc.metadata["page"] is not None and str(doc.metadata["page"]).isdigit()) else None,
            "chunk_id": doc.metadata.get("chunk_id", ""),
        }
        for doc in retrieved_docs
    ]

    # 3. Format context string
    formatted_context = "\n\n---\n\n".join(
        f"[Section: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    # 4. LCEL Answer Generation Chain
    try:
        generator_llm = llm or get_llm()
        rag_chain = RAG_PROMPT_TEMPLATE | generator_llm | StrOutputParser()
        raw_answer = rag_chain.invoke(
            {
                "context": formatted_context,
                "question": question,
            }
        )
        answer_text = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer)
    except PolicyAdvisorError:
        raise
    except Exception as exc:
        raise GenerationError(f"Failed to generate answer for query '{question}': {exc}") from exc

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "pages": pages,
        "retrieved_passages": passages,
        "retrieval_mode": mode,
        "evidence_found": True,
    }

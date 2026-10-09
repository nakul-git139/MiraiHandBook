"""Streamlit frontend interface for Autonomous MirAI Student Policy Advisor.

Connects to the FastAPI backend service via HTTP requests. Does not contain RAG logic
or direct Gemini API keys.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def get_backend_url() -> str:
    """Retrieve configured Backend API URL."""
    url = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).strip()
    return url.rstrip("/")


def check_backend_status(backend_url: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Check backend health and handbook ingestion status.

    Returns:
        (is_online, status_dict, error_message)
    """
    url = backend_url or get_backend_url()
    try:
        resp = requests.get(f"{url}/status", timeout=5.0)
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, None, f"Backend returned status {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        return False, None, f"Cannot connect to backend server at {url}. Ensure FastAPI is running."
    except requests.exceptions.Timeout:
        return False, None, f"Backend connection timed out at {url}."
    except Exception as exc:
        return False, None, f"Unexpected connection error: {exc}"


def upload_and_ingest_pdf(
    file_bytes: bytes,
    filename: str,
    backend_url: Optional[str] = None,
    timeout: float = 60.0,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Send uploaded PDF to backend POST /ingest.

    Returns:
        (success, result_dict, error_message)
    """
    url = backend_url or get_backend_url()
    try:
        files = {"file": (filename, file_bytes, "application/pdf")}
        resp = requests.post(f"{url}/ingest", files=files, timeout=timeout)
        if resp.status_code == 200:
            return True, resp.json(), None
        
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return False, None, f"Ingestion error ({resp.status_code}): {detail}"
    except requests.exceptions.ConnectionError:
        return False, None, f"Cannot connect to backend at {url}."
    except requests.exceptions.Timeout:
        return False, None, "Ingestion request timed out."
    except Exception as exc:
        return False, None, f"Ingestion failed: {exc}"


def send_chat_query(
    question: str,
    mode: str = "standard",
    top_k: int = 4,
    score_threshold: Optional[float] = None,
    backend_url: Optional[str] = None,
    timeout: float = 45.0,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Send user question to backend POST /chat.

    Returns:
        (success, chat_response_dict, error_message)
    """
    url = backend_url or get_backend_url()
    payload = {
        "question": question,
        "mode": mode,
        "top_k": top_k,
        "score_threshold": score_threshold,
    }

    try:
        resp = requests.post(f"{url}/chat", json=payload, timeout=timeout)
        if resp.status_code == 200:
            return True, resp.json(), None

        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return False, None, f"Error ({resp.status_code}): {detail}"
    except requests.exceptions.ConnectionError:
        return False, None, f"Cannot connect to backend service at {url}. Please ensure the server is active."
    except requests.exceptions.Timeout:
        return False, None, "The request timed out while generating an answer. Please try again."
    except Exception as exc:
        return False, None, f"Query failed: {exc}"


def main() -> None:
    """Streamlit Application Entrypoint."""
    st.set_page_config(
        page_title="MirAI Student Policy Advisor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    backend_url = get_backend_url()

    # App Header
    st.title("🎓 MirAI Student Policy Advisor")
    st.caption("Official policy assistance for Mirai School of Technology students.")

    # Sidebar: System Status & Administration
    with st.sidebar:
        st.header("⚙️ System Status")
        is_online, status_data, status_err = check_backend_status(backend_url)

        if is_online and status_data:
            st.success("🟢 Backend Service Online")
            is_ingested = status_data.get("is_ingested", False)
            chunk_count = status_data.get("chunk_count", 0)

            if is_ingested:
                st.info(f"📚 **Handbook Indexed**: {chunk_count} policy chunks")
            else:
                st.warning("⚠️ **No Handbook Indexed**. Please upload a handbook below.")
        else:
            st.error("🔴 Backend Offline")
            if status_err:
                st.caption(status_err)

        st.divider()

        # Administrative PDF Ingestion Section
        st.header("📄 Admin Document Ingestion")
        uploaded_file = st.file_uploader(
            "Upload Official Policy Handbook (PDF)",
            type=["pdf"],
            help="Upload the official Mirai SoT Policy Handbook to index into ChromaDB.",
        )

        if uploaded_file is not None:
            if st.button("🚀 Ingest Document", use_container_width=True):
                with st.spinner("Processing & indexing handbook..."):
                    success, res, err = upload_and_ingest_pdf(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        backend_url=backend_url,
                    )
                    if success and res:
                        st.success("✅ Ingestion Successful!")
                        st.json(res)
                        st.rerun()
                    else:
                        st.error(f"❌ Ingestion Failed: {err}")

        st.divider()

        # Retrieval Settings
        st.header("🔍 Query Settings")
        retrieval_mode = st.radio(
            "Retrieval Strategy",
            options=["standard", "multi_query"],
            format_func=lambda x: "⚡ Standard Vector Search" if x == "standard" else "🧠 Multi-Query Expansion",
            help="Standard uses direct similarity search. Multi-Query generates query variants for complex topics.",
        )
        top_k = st.slider("Context Passages (top_k)", min_value=1, max_value=8, value=4)

        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display notice if no handbook is indexed
    if is_online and status_data and not status_data.get("is_ingested", False):
        st.warning(
            "⚠️ **No policy handbook has been ingested into the system yet.**\n\n"
            "Please use the **Admin Document Ingestion** section in the left sidebar to upload "
            "`Mirai_SoT_Policy_Handbook_2026.pdf`."
        )

    # Render previous conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("pages"):
                pages_str = ", ".join(str(p) for p in msg["pages"])
                st.caption(f"📑 **Handbook Citations**: Page(s) {pages_str} ({msg.get('source', 'Handbook')})")

            # Render expandable retrieved passages
            if msg.get("passages"):
                with st.expander(f"🔍 View {len(msg['passages'])} Retrieved Evidence Passages"):
                    for idx, p in enumerate(msg["passages"]):
                        page_label = f"Page {p.get('page')}" if p.get('page') else "Unknown Page"
                        st.markdown(f"**Passage {idx + 1} ({page_label} - `{p.get('chunk_id', 'chunk')}`)**")
                        st.text(p.get("content", "").strip())

    # User Input
    if prompt := st.chat_input("Ask a question about Mirai School of Technology policies..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching official handbook & generating answer..."):
                success, response_data, error_msg = send_chat_query(
                    question=prompt,
                    mode=retrieval_mode,
                    top_k=top_k,
                    backend_url=backend_url,
                )

                if success and response_data:
                    answer = response_data.get("answer", "")
                    sources = response_data.get("sources", [])
                    pages = response_data.get("pages", [])
                    passages = response_data.get("retrieved_passages", [])
                    primary_source = sources[0] if sources else "Handbook"

                    st.markdown(answer)

                    if pages:
                        pages_str = ", ".join(str(p) for p in pages)
                        st.caption(f"📑 **Handbook Citations**: Page(s) {pages_str} ({primary_source})")

                    if passages:
                        with st.expander(f"🔍 View {len(passages)} Retrieved Evidence Passages"):
                            for idx, p in enumerate(passages):
                                page_label = f"Page {p.get('page')}" if p.get('page') else "Unknown Page"
                                st.markdown(f"**Passage {idx + 1} ({page_label} - `{p.get('chunk_id', 'chunk')}`)**")
                                st.text(p.get("content", "").strip())

                    # Save to session history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "pages": pages,
                            "passages": passages,
                            "source": primary_source,
                        }
                    )
                else:
                    err_text = f"⚠️ {error_msg}"
                    st.error(err_text)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": err_text,
                        }
                    )


if __name__ == "__main__":
    main()

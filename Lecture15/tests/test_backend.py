"""Unit tests for FastAPI backend service endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend import app

client = TestClient(app)


def test_root_endpoint():
    """Verify GET / returns online status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_status_endpoint():
    """Verify GET /status checks Chroma collection status."""
    mock_vs = MagicMock()
    mock_vs._collection.count.return_value = 50

    with patch("rag_pipeline.get_vector_store", return_value=mock_vs):
        response = client.get("/status")

    assert response.status_code == 200
    data = response.json()
    assert data["is_ingested"] is True
    assert data["chunk_count"] == 50


def test_chat_endpoint_success():
    """Verify POST /chat returns grounded response."""
    mock_result = {
        "question": "What is the attendance threshold?",
        "answer": "Attendance requirement is 75% minimum.",
        "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
        "pages": [3, 5],
        "retrieved_passages": [
            {
                "content": "75% attendance mandatory.",
                "source": "Mirai_SoT_Policy_Handbook_2026.pdf",
                "page": 3,
                "chunk_id": "c1",
            }
        ],
        "retrieval_mode": "standard",
        "evidence_found": True,
    }

    with patch("rag_pipeline.answer_question", return_value=mock_result):
        response = client.post(
            "/chat",
            json={"question": "What is the attendance threshold?", "mode": "standard"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "75%" in data["answer"]
    assert data["pages"] == [3, 5]
    assert data["evidence_found"] is True


def test_chat_endpoint_empty_collection():
    """Verify POST /chat returns 400 when handbook is unindexed."""
    with patch(
        "rag_pipeline.answer_question",
        side_effect=__import__("rag_pipeline").EmptyCollectionError("Collection empty"),
    ):
        response = client.post(
            "/chat",
            json={"question": "What is the attendance threshold?"},
        )

    assert response.status_code == 400
    assert "No handbook has been ingested" in response.json()["detail"]


def test_ingest_endpoint_non_pdf():
    """Verify POST /ingest rejects non-PDF file formats."""
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]

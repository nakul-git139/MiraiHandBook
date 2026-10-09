"""Unit tests for Streamlit frontend API communication and error handling."""

from unittest.mock import MagicMock, patch

import pytest
import requests

import frontend


def test_get_backend_url_default(monkeypatch):
    """Verify backend URL defaults to http://127.0.0.1:8000 when unset."""
    monkeypatch.delenv("BACKEND_URL", raising=False)
    url = frontend.get_backend_url()
    assert url == "http://127.0.0.1:8000"


def test_get_backend_url_custom(monkeypatch):
    """Verify backend URL reads custom configured environment variable."""
    monkeypatch.setenv("BACKEND_URL", "http://api.mirai.edu:9000/")
    url = frontend.get_backend_url()
    assert url == "http://api.mirai.edu:9000"


def test_check_backend_status_success():
    """Verify successful backend health check."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ready",
        "collection_name": "mirai_policies",
        "chunk_count": 50,
        "is_ingested": True,
    }

    with patch("requests.get", return_value=mock_resp):
        online, data, err = frontend.check_backend_status("http://127.0.0.1:8000")

    assert online is True
    assert data["is_ingested"] is True
    assert err is None


def test_check_backend_status_connection_error():
    """Verify graceful handling when backend is unreachable."""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection Refused")):
        online, data, err = frontend.check_backend_status("http://127.0.0.1:8000")

    assert online is False
    assert data is None
    assert "Cannot connect" in err


def test_check_backend_status_timeout():
    """Verify timeout handling in status check."""
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Request Timeout")):
        online, data, err = frontend.check_backend_status("http://127.0.0.1:8000")

    assert online is False
    assert data is None
    assert "timed out" in err


def test_check_backend_status_non_200():
    """Verify handling of non-200 HTTP responses in status check."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"

    with patch("requests.get", return_value=mock_resp):
        online, data, err = frontend.check_backend_status("http://127.0.0.1:8000")

    assert online is False
    assert "503" in err


def test_upload_and_ingest_pdf_success():
    """Verify successful PDF upload and ingestion request."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "file_name": "Mirai_SoT_Policy_Handbook_2026.pdf",
        "file_path": "/data/Mirai_SoT_Policy_Handbook_2026.pdf",
        "page_count": 16,
        "chunk_count": 50,
        "collection_name": "mirai_policies",
        "persist_directory": "chroma_db",
    }

    with patch("requests.post", return_value=mock_resp):
        success, res, err = frontend.upload_and_ingest_pdf(
            file_bytes=b"%PDF-1.4 dummy",
            filename="Mirai_SoT_Policy_Handbook_2026.pdf",
            backend_url="http://127.0.0.1:8000",
        )

    assert success is True
    assert res["chunk_count"] == 50
    assert err is None


def test_upload_and_ingest_pdf_error_response():
    """Verify handling of backend ingestion error responses."""
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"detail": "Corrupted PDF file."}

    with patch("requests.post", return_value=mock_resp):
        success, res, err = frontend.upload_and_ingest_pdf(
            file_bytes=b"invalid",
            filename="test.pdf",
            backend_url="http://127.0.0.1:8000",
        )

    assert success is False
    assert res is None
    assert "Corrupted PDF file." in err


def test_send_chat_query_success():
    """Verify successful student question chat query."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "question": "What is the attendance rule?",
        "answer": "Attendance requirement is 75% minimum.",
        "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
        "pages": [3, 5],
        "retrieved_passages": [
            {
                "content": "75% minimum attendance required.",
                "source": "Mirai_SoT_Policy_Handbook_2026.pdf",
                "page": 3,
                "chunk_id": "c1",
            }
        ],
        "retrieval_mode": "standard",
        "evidence_found": True,
    }

    with patch("requests.post", return_value=mock_resp):
        success, res, err = frontend.send_chat_query(
            question="What is the attendance rule?",
            mode="standard",
            top_k=4,
            backend_url="http://127.0.0.1:8000",
        )

    assert success is True
    assert "75%" in res["answer"]
    assert res["pages"] == [3, 5]
    assert len(res["retrieved_passages"]) == 1
    assert err is None


def test_send_chat_query_connection_error():
    """Verify chat query connection failure handling."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection Refused")):
        success, res, err = frontend.send_chat_query(
            question="What is the attendance rule?",
            backend_url="http://127.0.0.1:8000",
        )

    assert success is False
    assert res is None
    assert "Cannot connect" in err


def test_send_chat_query_timeout():
    """Verify chat query timeout handling."""
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Timeout")):
        success, res, err = frontend.send_chat_query(
            question="What is the attendance rule?",
            backend_url="http://127.0.0.1:8000",
        )

    assert success is False
    assert res is None
    assert "timed out" in err


def test_send_chat_query_http_error():
    """Verify non-200 HTTP error detail extraction."""
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.json.return_value = {"detail": "Gemini generation upstream failure."}

    with patch("requests.post", return_value=mock_resp):
        success, res, err = frontend.send_chat_query(
            question="What is the attendance rule?",
            backend_url="http://127.0.0.1:8000",
        )

    assert success is False
    assert res is None
    assert "502" in err
    assert "Gemini generation upstream failure" in err

"""Unit tests for RAG pipeline evaluation module and LLM judge."""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import evaluate


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset evaluation cache before and after each test."""
    evaluate.clear_eval_cache()
    yield
    evaluate.clear_eval_cache()


def test_parse_judge_response_valid_json():
    """Verify parsing valid JSON judge output."""
    raw = json.dumps({"score": 5, "reasoning": "Fully factual and grounded.", "passed": True})
    parsed = evaluate.parse_judge_response(raw)
    assert parsed["score"] == 5
    assert parsed["passed"] is True
    assert "factual" in parsed["reasoning"]


def test_parse_judge_response_markdown_code_blocks():
    """Verify parsing JSON wrapped in markdown code blocks."""
    raw = '```json\n{"score": 4, "reasoning": "Accurate with minor verbosity.", "passed": true}\n```'
    parsed = evaluate.parse_judge_response(raw)
    assert parsed["score"] == 4
    assert parsed["passed"] is True


def test_parse_judge_response_fallback():
    """Verify fallback handling when judge output is non-JSON."""
    raw = 'The score is "score": 2 because the model hallucinated.'
    parsed = evaluate.parse_judge_response(raw)
    assert parsed["score"] == 2
    assert parsed["passed"] is False


def test_append_eval_result_to_csv_multiline_escaping():
    """Verify CSV writer creates valid headers and escapes multiline strings."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_eval.csv"
        record = {
            "question": "What is the policy?\n(Second line)",
            "retrieval_mode": "standard",
            "retrieved_context": "Line 1: Policy.\nLine 2: Clause.\nLine 3: Details.",
            "generated_answer": "The policy states:\n- 75% attendance\n- 4 marks",
            "score": 5,
            "reasoning": "Excellent grounding.",
            "passed": True,
            "timestamp": "2026-10-09T00:00:00Z",
        }

        evaluate.append_eval_result_to_csv(csv_path, record)

        assert csv_path.is_file()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 2  # Header + 1 row
            assert reader[0] == [
                "question",
                "retrieval_mode",
                "retrieved_context",
                "generated_answer",
                "score",
                "reasoning",
                "passed",
                "timestamp",
            ]
            assert "Second line" in reader[1][0]
            assert reader[1][4] == "5"
            assert reader[1][6] == "True"


def test_evaluate_single_question_passing():
    """Verify single question evaluation pipeline with passing mock judge."""
    mock_rag_res = {
        "question": "I have 72% attendance. How many attendance marks will I get?",
        "answer": "Based on the handbook, 72% attendance awards 4 marks.",
        "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
        "pages": [3],
        "retrieved_passages": [
            {"page": 3, "content": "60% – 74.99%: 4 marks"}
        ],
        "evidence_found": True,
    }

    judge_response = json.dumps({
        "score": 5,
        "reasoning": "Exact match with expected criteria (4 marks).",
        "passed": True,
    })
    mock_judge = FakeListChatModel(responses=[judge_response])

    with patch("rag_pipeline.answer_question", return_value=mock_rag_res):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "eval.csv"
            res = evaluate.evaluate_single_question(
                question_spec=evaluate.CERTIFICATION_QUESTIONS[0],
                retrieval_mode="standard",
                judge_llm=mock_judge,
                csv_path=csv_path,
            )

            assert res["score"] == 5
            assert res["passed"] is True
            assert csv_path.is_file()


def test_evaluate_single_question_smoking_fine_penalized():
    """Verify that fabricating a fine for smoking results in a failing score."""
    mock_rag_res = {
        "question": "How much is the fine for smoking a cigarette on campus?",
        "answer": "The fine for smoking on campus is ₹500.",
        "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
        "pages": [9],
        "retrieved_passages": [
            {"page": 9, "content": "Tobacco is prohibited on campus. Actions by Disciplinary Committee."}
        ],
        "evidence_found": True,
    }

    judge_response = json.dumps({
        "score": 1,
        "reasoning": "Answer fabricated a monetary fine (₹500) not present in the handbook.",
        "passed": False,
    })
    mock_judge = FakeListChatModel(responses=[judge_response])

    with patch("rag_pipeline.answer_question", return_value=mock_rag_res):
        res = evaluate.evaluate_single_question(
            question_spec=evaluate.CERTIFICATION_QUESTIONS[3],
            retrieval_mode="standard",
            judge_llm=mock_judge,
        )

        assert res["score"] == 1
        assert res["passed"] is False


def test_run_evaluation_full_suite():
    """Verify complete evaluation run across all 4 certification questions."""
    mock_rag_res = {
        "question": "sample",
        "answer": "sample answer",
        "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
        "pages": [1],
        "retrieved_passages": [{"page": 1, "content": "sample content"}],
        "evidence_found": True,
    }

    judge_responses = [
        json.dumps({"score": 5, "reasoning": "Q1 correct", "passed": True}),
        json.dumps({"score": 5, "reasoning": "Q2 correct", "passed": True}),
        json.dumps({"score": 4, "reasoning": "Q3 correct", "passed": True}),
        json.dumps({"score": 5, "reasoning": "Q4 correct", "passed": True}),
    ]
    mock_judge = FakeListChatModel(responses=judge_responses)

    with patch("rag_pipeline.answer_question", return_value=mock_rag_res):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "full_eval.csv"
            results = evaluate.run_evaluation(
                retrieval_mode="standard",
                csv_path=csv_path,
                judge_llm=mock_judge,
            )

            assert len(results) == 4
            assert all(r["passed"] for r in results)
            assert csv_path.is_file()

            # Verify CSV content
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                assert len(reader) == 5  # Header + 4 questions


def test_get_handbook_fingerprint():
    """Verify handbook fingerprint generation and sensitivity to changes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Empty dir
        fp_empty = evaluate.get_handbook_fingerprint(tmp_path)
        assert fp_empty == "empty_handbook"

        # Add a dummy pdf
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")
        fp1 = evaluate.get_handbook_fingerprint(tmp_path)
        assert len(fp1) == 16
        assert fp1 != fp_empty

        # Modify file -> fingerprint changes
        dummy_pdf.write_bytes(b"%PDF-1.4 updated content that is longer")
        fp2 = evaluate.get_handbook_fingerprint(tmp_path)
        assert fp2 != fp1


def test_run_comparison_workflow():
    """Verify run_comparison executes both standard and multi-query pipelines and writes comparison CSV."""
    mock_rag_res_std = {
        "question": "sample",
        "answer": "Standard answer",
        "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
        "pages": [1],
        "retrieved_passages": [{"page": 1, "content": "standard passage"}],
        "evidence_found": True,
    }
    mock_rag_res_mq = {
        "question": "sample",
        "answer": "MultiQuery answer",
        "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
        "pages": [1, 2],
        "retrieved_passages": [
            {"page": 1, "content": "standard passage"},
            {"page": 2, "content": "multiquery expanded passage"},
        ],
        "evidence_found": True,
    }

    # 4 judge calls for standard + 4 judge calls for multi_query
    judge_responses = [
        json.dumps({"score": 4, "reasoning": "Std Q1", "passed": True}),
        json.dumps({"score": 5, "reasoning": "Std Q2", "passed": True}),
        json.dumps({"score": 4, "reasoning": "Std Q3", "passed": True}),
        json.dumps({"score": 5, "reasoning": "Std Q4", "passed": True}),
        json.dumps({"score": 5, "reasoning": "MQ Q1", "passed": True}),
        json.dumps({"score": 5, "reasoning": "MQ Q2", "passed": True}),
        json.dumps({"score": 5, "reasoning": "MQ Q3", "passed": True}),
        json.dumps({"score": 5, "reasoning": "MQ Q4", "passed": True}),
    ]
    mock_judge = FakeListChatModel(responses=judge_responses)

    def mock_answer_fn(question, mode="standard", top_k=4):
        if mode == "multi_query":
            return dict(mock_rag_res_mq, question=question)
        return dict(mock_rag_res_std, question=question)

    with patch("rag_pipeline.answer_question", side_effect=mock_answer_fn):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "scores.csv"
            comp_csv_path = Path(tmp_dir) / "comparison.csv"

            summary = evaluate.run_comparison(
                csv_path=csv_path,
                comparison_csv=comp_csv_path,
                judge_llm=mock_judge,
                use_cache=False,
            )

            assert summary["total_questions"] == 4
            assert summary["standard_passed"] == 4
            assert summary["multiquery_passed"] == 4
            assert summary["standard_avg_score"] == 4.5
            assert summary["multiquery_avg_score"] == 5.0
            assert comp_csv_path.is_file()

            # Verify comparison CSV content
            with open(comp_csv_path, newline="", encoding="utf-8") as f:
                comp_rows = list(csv.DictReader(f))
                assert len(comp_rows) == 4
                assert "std_score" in comp_rows[0]
                assert "mq_score" in comp_rows[0]
                assert "score_difference" in comp_rows[0]
                assert "std_retrieval_status" in comp_rows[0]
                assert "mq_retrieval_status" in comp_rows[0]
                assert comp_rows[0]["std_score"] == "4"
                assert comp_rows[0]["mq_score"] == "5"
                assert comp_rows[0]["score_difference"] == "1"


def test_retrieval_failure_reported_separately():
    """Verify that retrieval failures are recorded with FAILED status and not masked."""
    def mock_failing_answer(question, mode="standard", top_k=4):
        raise RuntimeError("Vector database unreachable")

    with patch("rag_pipeline.answer_question", side_effect=mock_failing_answer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "failure_eval.csv"
            results = evaluate.run_evaluation(
                questions=[evaluate.CERTIFICATION_QUESTIONS[0]],
                retrieval_mode="standard",
                csv_path=csv_path,
                use_cache=False,
            )

            assert len(results) == 1
            assert "FAILED" in results[0]["retrieval_status"]
            assert results[0]["score"] == 1
            assert results[0]["passed"] is False
            assert "Vector database unreachable" in results[0]["reasoning"]


def test_eval_cache_hit_and_invalidation():
    """Verify that caching avoids redundant LLM judge calls and invalidates on key change."""
    mock_rag_res = {
        "question": "sample",
        "answer": "Answer 1",
        "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
        "pages": [1],
        "retrieved_passages": [{"page": 1, "content": "passage"}],
        "evidence_found": True,
    }
    mock_judge = MagicMock()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = json.dumps({"score": 5, "reasoning": "Judge ok", "passed": True})

    with patch("evaluate.JUDGE_PROMPT_TEMPLATE") as mock_prompt:
        mock_prompt.__or__.return_value.__or__.return_value = mock_chain
        with patch("rag_pipeline.answer_question", return_value=mock_rag_res):
            q_spec = evaluate.CERTIFICATION_QUESTIONS[0]

            # 1. First call: populates cache
            res1 = evaluate.evaluate_single_question(q_spec, retrieval_mode="standard", use_cache=True)
            assert res1["score"] == 5
            assert mock_chain.invoke.call_count == 1

            # 2. Second call with identical params: hits cache (no new LLM judge call)
            res2 = evaluate.evaluate_single_question(q_spec, retrieval_mode="standard", use_cache=True)
            assert res2["score"] == 5
            assert mock_chain.invoke.call_count == 1

            # 3. Third call with different mode: cache miss, calls judge again
            res3 = evaluate.evaluate_single_question(q_spec, retrieval_mode="multi_query", use_cache=True)
            assert res3["score"] == 5
            assert mock_chain.invoke.call_count == 2



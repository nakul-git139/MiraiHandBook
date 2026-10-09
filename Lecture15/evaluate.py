"""Evaluation module for Autonomous MirAI Student Policy Advisor.

Evaluates the RAG pipeline against the four assignment certification questions
using an LLM judge for factual correctness, grounding, and hallucination prevention,
logs results incrementally to rag_eval_scores.csv, and provides a reproducible
comparison workflow between Standard Vector Retrieval and MultiQueryRetriever.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import rag_pipeline

# Load environment configuration
load_dotenv()

DEFAULT_CSV_PATH = "rag_eval_scores.csv"
DEFAULT_COMPARISON_CSV_PATH = "rag_comparison_report.csv"

# Global in-memory evaluation cache keyed by (fingerprint, model, mode, question_hash)
_EVAL_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_eval_cache() -> None:
    """Clear all entries in the evaluation cache."""
    global _EVAL_CACHE
    _EVAL_CACHE.clear()


CERTIFICATION_QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "Q1",
        "question": "I have 72% attendance. How many attendance marks will I get?",
        "expected_criteria": (
            "The student should receive 4 marks out of 10 based on the 60% – 74.99% bracket. "
            "The answer must not state any other mark count."
        ),
    },
    {
        "id": "Q2",
        "question": (
            "I study at the Ratnam campus. I got sick and need medical leave. "
            "Who do I email and how many days do I have to submit my documents?"
        ),
        "expected_criteria": (
            "The student must submit medical leave documents to the Ratnam Campus Manager "
            "(Yashaswini Ma'am) within exactly 7 days of the illness or treatment."
        ),
    },
    {
        "id": "Q3",
        "question": "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
        "expected_criteria": (
            "Starting a new society requires at least 40% batch support. The proposal must be submitted "
            "to the Faculty Coordinator first; students must not approach Management directly."
        ),
    },
    {
        "id": "Q4",
        "question": "How much is the fine for smoking a cigarette on campus?",
        "expected_criteria": (
            "Smoking/tobacco is prohibited on campus and subject to Disciplinary Committee action. "
            "The answer must NOT invent a monetary fine."
        ),
    },
]

JUDGE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an objective AI evaluator assessing the performance of a RAG-based Student Policy Advisor.

You must evaluate the Generated Answer against the Question, Retrieved Context, and Expected Criteria.

Evaluation Criteria:
1. Factual Correctness: Are facts accurately aligned with the Retrieved Context?
2. Completeness: Does the answer address all parts of the Expected Criteria?
3. Grounding & Hallucination: Does the answer invent policies, contacts, deadlines, or monetary fines?
   - CRITICAL RULE: If a monetary fine is invented for smoking on campus, score MUST be <= 2 and passed MUST be false.
4. Appropriate Refusal: If evidence is absent, did it refuse appropriately?

Score Guidelines:
5 - Outstanding: Completely truthful to context, fully satisfies expected criteria, zero hallucinations.
4 - Good: Factual, satisfies criteria with minor phrasing differences, zero hallucinations.
3 - Borderline: Partial answer or minor omission.
2 - Poor: Contains factual errors, unsupported claims, or invented penalties/fines.
1 - Fail: Complete hallucination, contradicted by context, or fabricates non-existent rules.

You must respond ONLY with a valid JSON object matching this schema:
{{
  "score": <integer from 1 to 5>,
  "reasoning": "<concise 1-2 sentence justification>",
  "passed": <true if score >= 4 else false>
}}""",
        ),
        (
            "human",
            """Student Question: {question}

Expected Evaluation Criteria: {expected_criteria}

Retrieved Handbook Context:
{retrieved_context}

Generated Answer to Evaluate:
{generated_answer}

JSON Evaluation:""",
        ),
    ]
)


def get_handbook_fingerprint(data_dir: Optional[Union[str, Path]] = None) -> str:
    """Calculate a hash fingerprint for source handbook files.

    Guarantees cache invalidation whenever the handbook is modified or replaced.

    Args:
        data_dir: Optional path to data folder.

    Returns:
        Hex hash string of file names, modification times, and sizes.
    """
    target_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
    if not target_dir.exists():
        return "no_data_dir"

    pdf_files = sorted(list(target_dir.glob("*.pdf")))
    if not pdf_files:
        return "empty_handbook"

    hasher = hashlib.sha256()
    for f in pdf_files:
        hasher.update(f.name.encode("utf-8"))
        try:
            stat = f.stat()
            hasher.update(str(stat.st_mtime).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
        except OSError:
            pass
    return hasher.hexdigest()[:16]


def parse_judge_response(raw_response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse JSON evaluation response from LLM judge.

    Args:
        raw_response: LLM string output or dictionary.

    Returns:
        Structured dict with score, reasoning, and passed.
    """
    if isinstance(raw_response, dict):
        return {
            "score": int(raw_response.get("score", 1)),
            "reasoning": str(raw_response.get("reasoning", "No reasoning provided.")),
            "passed": bool(raw_response.get("passed", False)),
        }

    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        score = int(data.get("score", 1))
        score = max(1, min(5, score))
        return {
            "score": score,
            "reasoning": str(data.get("reasoning", "Evaluation completed.")),
            "passed": bool(data.get("passed", score >= 4)),
        }
    except Exception:
        score_match = re.search(r'"score"\s*:\s*(\d)', cleaned)
        score = int(score_match.group(1)) if score_match else 1
        return {
            "score": score,
            "reasoning": cleaned[:200],
            "passed": score >= 4,
        }


def append_eval_result_to_csv(
    csv_path: Union[str, Path],
    record: Dict[str, Any],
) -> None:
    """Incrementally append an evaluation result row to the CSV file.

    Guarantees proper quoting of multiline strings and preserves existing records.

    Args:
        csv_path: Path to CSV output file.
        record: Dictionary containing evaluation columns.
    """
    target_path = Path(csv_path)
    file_exists = target_path.is_file() and target_path.stat().st_size > 0

    headers = [
        "question",
        "retrieval_mode",
        "retrieved_context",
        "generated_answer",
        "score",
        "reasoning",
        "passed",
        "timestamp",
    ]

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=headers,
            quoting=csv.QUOTE_ALL,
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "question": record.get("question", ""),
                "retrieval_mode": record.get("retrieval_mode", "standard"),
                "retrieved_context": record.get("retrieved_context", ""),
                "generated_answer": record.get("generated_answer", ""),
                "score": record.get("score", 1),
                "reasoning": record.get("reasoning", ""),
                "passed": record.get("passed", False),
                "timestamp": record.get("timestamp", datetime.now(timezone.utc).isoformat()),
            }
        )
        f.flush()


def evaluate_single_question(
    question_spec: Dict[str, str],
    retrieval_mode: str = "standard",
    judge_llm: Optional[BaseChatModel] = None,
    csv_path: Optional[Union[str, Path]] = None,
    top_k: int = 4,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Execute RAG pipeline for a question and evaluate with LLM judge.

    Args:
        question_spec: Dict with 'question' and 'expected_criteria'.
        retrieval_mode: 'standard' or 'multi_query'.
        judge_llm: Optional custom judge LLM instance.
        csv_path: Optional CSV output path for incremental logging.
        top_k: Context chunk count.
        use_cache: Whether to use version-aware caching.

    Returns:
        Dictionary containing evaluation results and retrieval metadata.
    """
    question = question_spec["question"]
    expected_criteria = question_spec["expected_criteria"]
    timestamp = datetime.now(timezone.utc).isoformat()

    # Version-aware cache key computation
    handbook_fp = get_handbook_fingerprint()
    q_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
    gen_model = os.getenv("GENERATION_MODEL", rag_pipeline.DEFAULT_GENERATION_MODEL)
    cache_key = f"{handbook_fp}_{gen_model}_{retrieval_mode}_{top_k}_{q_hash}"

    if use_cache and cache_key in _EVAL_CACHE:
        cached = _EVAL_CACHE[cache_key].copy()
        cached["timestamp"] = timestamp
        if csv_path:
            append_eval_result_to_csv(csv_path, cached)
        return cached

    # 1. Run question through RAG pipeline
    rag_result = rag_pipeline.answer_question(
        question=question,
        mode=retrieval_mode,
        top_k=top_k,
    )

    generated_answer = rag_result.get("answer", "")
    passages = rag_result.get("retrieved_passages", [])
    pages = rag_result.get("pages", [])
    evidence_found = rag_result.get("evidence_found", False)

    retrieved_context = "\n\n---\n\n".join(
        f"[Page {p.get('page', '?')}]\n{p.get('content', '')}" for p in passages
    )

    # 2. Invoke LLM Judge
    eval_llm = judge_llm or rag_pipeline.get_llm()
    judge_chain = JUDGE_PROMPT_TEMPLATE | eval_llm | StrOutputParser()

    judge_output = judge_chain.invoke(
        {
            "question": question,
            "expected_criteria": expected_criteria,
            "retrieved_context": retrieved_context,
            "generated_answer": generated_answer,
        }
    )

    parsed_eval = parse_judge_response(judge_output)

    result_record = {
        "question_id": question_spec.get("id", ""),
        "question": question,
        "retrieval_mode": retrieval_mode,
        "retrieval_status": "OK" if passages else "EMPTY",
        "retrieved_chunks_count": len(passages),
        "retrieved_pages": pages,
        "retrieved_context": retrieved_context,
        "generated_answer": generated_answer,
        "evidence_found": evidence_found,
        "score": parsed_eval["score"],
        "reasoning": parsed_eval["reasoning"],
        "passed": parsed_eval["passed"],
        "timestamp": timestamp,
    }

    if use_cache:
        _EVAL_CACHE[cache_key] = result_record

    # 3. Incrementally persist to CSV
    if csv_path:
        append_eval_result_to_csv(csv_path, result_record)

    return result_record


def run_evaluation(
    questions: Optional[List[Dict[str, str]]] = None,
    retrieval_mode: str = "standard",
    csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
    judge_llm: Optional[BaseChatModel] = None,
    top_k: int = 4,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Run full benchmark evaluation across certification questions for a given mode.

    Args:
        questions: Optional list of question specifications.
        retrieval_mode: 'standard' or 'multi_query'.
        csv_path: CSV output filepath.
        judge_llm: Optional judge LLM.
        top_k: Retrieval context chunk count.
        use_cache: Whether to use version-aware caching.

    Returns:
        List of evaluation records.
    """
    target_questions = questions or CERTIFICATION_QUESTIONS
    results: List[Dict[str, Any]] = []

    print(f"Starting RAG Benchmark Evaluation [{retrieval_mode.upper()} mode]")
    print(f"Total Benchmark Questions: {len(target_questions)}")
    print(f"Logging Results to: {csv_path}\n" + "=" * 75)

    for idx, q_spec in enumerate(target_questions, 1):
        print(f"\nEvaluating [{idx}/{len(target_questions)}]: {q_spec['question']}")
        try:
            res = evaluate_single_question(
                question_spec=q_spec,
                retrieval_mode=retrieval_mode,
                judge_llm=judge_llm,
                csv_path=csv_path,
                top_k=top_k,
                use_cache=use_cache,
            )
            results.append(res)
            status_icon = "✅ PASSED" if res["passed"] else "❌ FAILED"
            print(f"Outcome: {status_icon} (Score: {res['score']}/5)")
            print(f"Pages Cited: {res['retrieved_pages']}")
            print(f"Reasoning: {res['reasoning']}")
        except Exception as exc:
            print(f"Evaluation encountered an error on question {idx}: {exc}")
            err_record = {
                "question_id": q_spec.get("id", f"Q{idx}"),
                "question": q_spec["question"],
                "retrieval_mode": retrieval_mode,
                "retrieval_status": f"FAILED: {exc}",
                "retrieved_chunks_count": 0,
                "retrieved_pages": [],
                "retrieved_context": "",
                "generated_answer": f"ERROR: {exc}",
                "evidence_found": False,
                "score": 1,
                "reasoning": f"Execution error: {exc}",
                "passed": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            results.append(err_record)
            if csv_path:
                append_eval_result_to_csv(csv_path, err_record)

    total_passed = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    print("\n" + "=" * 75)
    print(f"Benchmark Summary [{retrieval_mode}]: {total_passed}/{len(results)} Passed | Average Score: {avg_score:.2f}/5")
    print(f"All records written to: {csv_path}")

    return results


def run_comparison(
    questions: Optional[List[Dict[str, str]]] = None,
    csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
    comparison_csv: Union[str, Path] = DEFAULT_COMPARISON_CSV_PATH,
    judge_llm: Optional[BaseChatModel] = None,
    top_k: int = 4,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Execute reproducible side-by-side comparison between Standard and MultiQuery retrieval.

    Evaluates the exact same questions with identical prompt, handbook version, and judge,
    records results in both rag_eval_scores.csv and rag_comparison_report.csv, and prints
    a formatted comparative terminal summary.

    Args:
        questions: Questions to test. Defaults to CERTIFICATION_QUESTIONS.
        csv_path: Path to comprehensive scores CSV.
        comparison_csv: Path to side-by-side comparison CSV.
        judge_llm: Optional judge LLM.
        top_k: Context chunk count.
        use_cache: Whether to use version-aware caching.

    Returns:
        Dictionary containing comparative summaries and records.
    """
    target_questions = questions or CERTIFICATION_QUESTIONS
    handbook_fp = get_handbook_fingerprint()

    print("=" * 80)
    print("🔬 RUNNING REPRODUCIBLE RAG RETRIEVAL COMPARISON")
    print(f"Handbook Fingerprint: {handbook_fp}")
    print(f"Model Configuration: {os.getenv('GENERATION_MODEL', rag_pipeline.DEFAULT_GENERATION_MODEL)}")
    print("=" * 80)

    # 1. Run Standard Retrieval
    std_results = run_evaluation(
        questions=target_questions,
        retrieval_mode="standard",
        csv_path=csv_path,
        judge_llm=judge_llm,
        top_k=top_k,
        use_cache=use_cache,
    )

    # 2. Run Multi-Query Retrieval
    mq_results = run_evaluation(
        questions=target_questions,
        retrieval_mode="multi_query",
        csv_path=csv_path,
        judge_llm=judge_llm,
        top_k=top_k,
        use_cache=use_cache,
    )

    # 3. Build Side-by-Side Comparison Records
    comparison_rows: List[Dict[str, Any]] = []
    comp_path = Path(comparison_csv)
    comp_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "question_id",
        "question",
        "std_retrieval_status",
        "std_chunks_count",
        "std_pages",
        "std_score",
        "std_passed",
        "mq_retrieval_status",
        "mq_chunks_count",
        "mq_pages",
        "mq_score",
        "mq_passed",
        "score_difference",
        "std_answer_excerpt",
        "mq_answer_excerpt",
        "std_reasoning",
        "mq_reasoning",
        "handbook_fingerprint",
        "timestamp",
    ]

    for std, mq in zip(std_results, mq_results):
        score_diff = mq["score"] - std["score"]
        row = {
            "question_id": std.get("question_id", ""),
            "question": std.get("question", ""),
            "std_retrieval_status": std.get("retrieval_status", "OK"),
            "std_chunks_count": std.get("retrieved_chunks_count", 0),
            "std_pages": str(std.get("retrieved_pages", [])),
            "std_score": std.get("score", 1),
            "std_passed": std.get("passed", False),
            "mq_retrieval_status": mq.get("retrieval_status", "OK"),
            "mq_chunks_count": mq.get("retrieved_chunks_count", 0),
            "mq_pages": str(mq.get("retrieved_pages", [])),
            "mq_score": mq.get("score", 1),
            "mq_passed": mq.get("passed", False),
            "score_difference": score_diff,
            "std_answer_excerpt": std.get("generated_answer", "")[:120].replace("\n", " "),
            "mq_answer_excerpt": mq.get("generated_answer", "")[:120].replace("\n", " "),
            "std_reasoning": std.get("reasoning", ""),
            "mq_reasoning": mq.get("reasoning", ""),
            "handbook_fingerprint": handbook_fp,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        comparison_rows.append(row)

    # Write comparison CSV
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(comparison_rows)

    # 4. Print Formatted Terminal Summary
    std_avg = sum(r["score"] for r in std_results) / len(std_results) if std_results else 0.0
    mq_avg = sum(r["score"] for r in mq_results) / len(mq_results) if mq_results else 0.0
    std_pass_count = sum(1 for r in std_results if r["passed"])
    mq_pass_count = sum(1 for r in mq_results if r["passed"])

    print("\n" + "=" * 80)
    print("📊 RETRIEVAL STRATEGY COMPARISON REPORT")
    print("=" * 80)
    print(f"{'QID':<5} | {'Question Summary':<30} | {'Standard (Score/Pass)':<20} | {'Multi-Query (Score/Pass)':<22}")
    print("-" * 80)

    for r in comparison_rows:
        q_short = r["question"][:28] + ".." if len(r["question"]) > 30 else r["question"]
        std_p = "PASS" if r["std_passed"] else "FAIL"
        mq_p = "PASS" if r["mq_passed"] else "FAIL"
        std_res = f"{r['std_score']}/5 ({std_p})"
        mq_res = f"{r['mq_score']}/5 ({mq_p})"
        print(f"{r['question_id']:<5} | {q_short:<30} | {std_res:<20} | {mq_res:<22}")

    print("-" * 80)
    print(f"Standard Vector Search : {std_pass_count}/{len(std_results)} Passed | Average Score: {std_avg:.2f}/5")
    print(f"Multi-Query Expansion  : {mq_pass_count}/{len(mq_results)} Passed | Average Score: {mq_avg:.2f}/5")
    print("=" * 80)
    print(f"Detailed comparison saved to: {comparison_csv}\n")

    return {
        "standard_avg_score": std_avg,
        "multiquery_avg_score": mq_avg,
        "standard_passed": std_pass_count,
        "multiquery_passed": mq_pass_count,
        "total_questions": len(target_questions),
        "comparison_rows": comparison_rows,
        "comparison_csv": str(comp_path),
    }


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate & Compare MirAI RAG Pipeline.")
    parser.add_argument(
        "--mode",
        choices=["standard", "multi_query", "compare"],
        default="compare",
        help="Evaluation strategy ('standard', 'multi_query', or 'compare').",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_CSV_PATH,
        help="Target CSV file for evaluation scores (default: rag_eval_scores.csv).",
    )
    parser.add_argument(
        "--comparison-output",
        default=DEFAULT_COMPARISON_CSV_PATH,
        help="Target CSV file for side-by-side comparison (default: rag_comparison_report.csv).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable evaluation cache.",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        run_comparison(
            csv_path=args.output,
            comparison_csv=args.comparison_output,
            use_cache=not args.no_cache,
        )
    else:
        run_evaluation(
            retrieval_mode=args.mode,
            csv_path=args.output,
            use_cache=not args.no_cache,
        )


if __name__ == "__main__":
    main()

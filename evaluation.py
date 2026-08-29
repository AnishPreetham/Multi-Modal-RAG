"""
evaluation.py -- reproducible evaluation against a fixed, versioned dataset.

Run:  .venv\\Scripts\\python evaluation.py

Every run writes three artefacts into evaluation/reports/:
    report_<timestamp>.json   full per-query detail plus aggregates
    report_<timestamp>.csv    one row per query
    report_<timestamp>.md     human-readable summary

Each report stamps the dataset version, all four model names, the full
configuration, and the git commit, so two runs can be compared meaningfully.

Determinism, stated honestly:
    Retrieval, reranking, relationships and citations are deterministic --
    identical inputs give identical outputs on every run.
    Generation is NOT. Qwen is sampled and Ollama offers no cross-version
    seed guarantee. Temperature is pinned to 0.0 to minimise variance, so
    answer-level metrics are APPROXIMATELY reproducible, not exactly so.

Groundedness and correctness are scored by keyword and expected-source
overlap, never by an LLM judge: a 3B model grading its own output would be
neither reliable nor honest.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from schemas import QueryModality, QueryRequest

DATASET_VERSION = "1.0"
DATASET_PATH = settings.evaluation_dir / "dataset" / "queries.json"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall(retrieved: list[str], expected: list[str]) -> tuple[float, float]:
    """Set precision/recall of retrieved source filenames against expected."""
    if not expected:
        return (1.0 if not retrieved else 0.0), 1.0
    hits = len(set(retrieved) & set(expected))
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(expected)
    return precision, recall


def reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    """1/rank of the first relevant source; 0 if none appear."""
    for position, filename in enumerate(ranked, start=1):
        if filename in expected:
            return 1.0 / position
    return 0.0


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of required facts present verbatim in the answer."""
    if not keywords:
        return 1.0
    lowered = answer.lower()
    return sum(1 for k in keywords if k.lower() in lowered) / len(keywords)


def citation_accuracy(citation_files: list[str], expected: list[str]) -> float:
    """Fraction of citations that point at an expected source."""
    if not citation_files:
        return 0.0
    return sum(1 for f in citation_files if f in expected) / len(citation_files)


def hallucinated_numbers(answer: str, evidence_text: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the supplied evidence.

    A blunt but honest check: it catches the common 99.4 -> 99.44 style
    corruption without pretending to be a semantic entailment model.
    """
    import re

    answer_numbers = set(re.findall(r"\d+\.\d+|\d{2,}", answer))
    evidence_numbers = set(re.findall(r"\d+\.\d+|\d{2,}", evidence_text))
    return sorted(answer_numbers - evidence_numbers)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def default_dataset() -> dict:
    """The fixed six-query evaluation set. Written to disk on first run so
    that later runs use the committed file rather than regenerating it."""
    return {
        "dataset_version": DATASET_VERSION,
        "description": "Evidence AI fixed evaluation set over the Project Alpha "
                       "synthetic demo corpus.",
        "queries": [
            {
                "id": "Q1-pdf",
                "query": "How many records did the ingestion platform process in 2024?",
                "query_modality": "text",
                "expected_sources": ["project_report.pdf"],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": ["1.2 million"],
            },
            {
                "id": "Q2-docx",
                "query": "What is the top open risk carried into 2025?",
                "query_modality": "text",
                "expected_sources": ["project_notes.docx", "project_report.pdf"],
                "expected_modality": "text",
                "expect_abstention": False,
                "answer_keywords": ["latency"],
            },
            {
                "id": "Q3-image",
                "query": "What does the project dashboard show about team size?",
                "query_modality": "text",
                "expected_sources": ["project_dashboard.png"],
                "expected_modality": "image",
                "expect_abstention": False,
                "answer_keywords": ["6"],
            },
            {
                "id": "Q4-audio",
                "query": "What did the quarterly review recording say about accuracy?",
                "query_modality": "text",
                "expected_sources": ["project_meeting.wav"],
                "expected_modality": "audio",
                "expect_abstention": False,
                "answer_keywords": ["99.4"],
            },
            {
                "id": "Q5-crossmodal",
                "query": "What evidence do we have about Project Alpha's 2024 "
                         "development?",
                "query_modality": "text",
                "expected_sources": [
                    "project_report.pdf", "project_dashboard.png",
                    "project_meeting.wav", "project_notes.docx",
                ],
                "expected_modality": "mixed",
                "expect_abstention": False,
                "answer_keywords": ["1.2 million", "99.4"],
            },
            {
                "id": "Q6-unanswerable",
                "query": "What was Project Alpha's budget in 2040?",
                "query_modality": "text",
                "expected_sources": [],
                "expected_modality": "none",
                "expect_abstention": True,
                "answer_keywords": [],
            },
        ],
    }


def load_dataset() -> dict:
    if DATASET_PATH.exists():
        return json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = default_dataset()
    DATASET_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    _write_expectations(dataset)
    return dataset


def _write_expectations(dataset: dict) -> None:
    """Mirror expected answers and sources into their own directories, as
    the specification requires them to be separately inspectable."""
    answers_dir = settings.evaluation_dir / "expected_answers"
    sources_dir = settings.evaluation_dir / "expected_sources"
    answers_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    for query in dataset["queries"]:
        (answers_dir / f"{query['id']}.json").write_text(
            json.dumps({
                "id": query["id"],
                "answer_keywords": query["answer_keywords"],
                "expect_abstention": query["expect_abstention"],
            }, indent=2), encoding="utf-8",
        )
        (sources_dir / f"{query['id']}.json").write_text(
            json.dumps({
                "id": query["id"],
                "expected_sources": query["expected_sources"],
                "expected_modality": query["expected_modality"],
            }, indent=2), encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent, timeout=10,
        ).stdout.strip() or "unversioned"
    except Exception:
        return "unversioned"


def run_evaluation() -> dict:
    from rag_pipeline import get_pipeline

    dataset = load_dataset()
    pipeline = get_pipeline()
    per_query: list[dict] = []

    print(f"Evidence AI evaluation -- dataset v{dataset['dataset_version']}")
    print("=" * 78)

    for spec in dataset["queries"]:
        started = time.perf_counter()
        response = pipeline.query(QueryRequest(
            query=spec["query"],
            top_k=settings.top_k,
            rerank_top_k=settings.rerank_top_k,
            query_modality=QueryModality(spec["query_modality"]),
        ))
        latency = time.perf_counter() - started

        retrieved_files, seen = [], set()
        for result in response.retrieved_items:
            source = pipeline.store.get_source(result.item.source_id)
            name = source.filename if source else "unknown"
            if name not in seen:
                seen.add(name)
                retrieved_files.append(name)

        citation_files = [c.filename for c in response.citations]
        expected = spec["expected_sources"]

        precision, recall = precision_recall(retrieved_files, expected)
        mrr = reciprocal_rank(retrieved_files, expected)
        correctness = keyword_coverage(response.answer, spec["answer_keywords"])
        cite_accuracy = (
            1.0 if spec["expect_abstention"] and not citation_files
            else citation_accuracy(citation_files, expected)
        )
        evidence_text = " ".join(r.item.content for r in response.retrieved_items)
        invented = (
            [] if response.abstained
            else hallucinated_numbers(response.answer, evidence_text)
        )
        abstention_ok = response.abstained == spec["expect_abstention"]

        record = {
            "id": spec["id"],
            "query": spec["query"],
            "expected_sources": expected,
            "retrieved_sources": retrieved_files,
            "citation_sources": citation_files,
            "modalities": sorted({r.item.modality.value for r in response.retrieved_items}),
            "abstained": response.abstained,
            "expect_abstention": spec["expect_abstention"],
            "abstention_correct": abstention_ok,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mrr": round(mrr, 4),
            "citation_accuracy": round(cite_accuracy, 4),
            "answer_correctness": round(correctness, 4),
            "groundedness": round(1.0 if not invented else 0.0, 4),
            "hallucinated_numbers": invented,
            "confidence": response.confidence,
            "latency_s": round(latency, 2),
            "answer": response.answer,
        }
        per_query.append(record)

        status = "PASS" if abstention_ok and correctness >= 0.5 else "CHECK"
        print(f"[{status}] {spec['id']:18s} P={precision:.2f} R={recall:.2f} "
              f"MRR={mrr:.2f} corr={correctness:.2f} "
              f"cite={cite_accuracy:.2f} {latency:.1f}s")

    def mean(field: str) -> float:
        values = [r[field] for r in per_query if isinstance(r[field], (int, float))]
        return round(sum(values) / len(values), 4) if values else 0.0

    aggregate = {
        "retrieval_precision": mean("precision"),
        "retrieval_recall": mean("recall"),
        "mrr": mean("mrr"),
        "citation_accuracy": mean("citation_accuracy"),
        "answer_correctness": mean("answer_correctness"),
        "groundedness": mean("groundedness"),
        "hallucination_rate": round(
            sum(1 for r in per_query if r["hallucinated_numbers"]) / len(per_query), 4
        ),
        "abstention_accuracy": round(
            sum(1 for r in per_query if r["abstention_correct"]) / len(per_query), 4
        ),
        "mean_latency_s": mean("latency_s"),
    }

    return {
        "dataset_version": dataset["dataset_version"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "models": {
            "llm": settings.ollama_model,
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "whisper": f"{settings.whisper_model_size} "
                       f"({settings.whisper_device}/{settings.whisper_compute_type})",
        },
        "configuration": {
            "top_k": settings.top_k,
            "rerank_top_k": settings.rerank_top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "related_to_threshold": settings.related_to_threshold,
            "citation_threshold": settings.citation_threshold,
            "abstention_threshold": settings.abstention_threshold,
            "max_context_items": settings.max_context_items,
            "temperature": settings.ollama_temperature,
            "num_ctx": settings.ollama_num_ctx,
        },
        "index": pipeline.store.stats(),
        "aggregate": aggregate,
        "per_query": per_query,
        "determinism_note": (
            "Retrieval, reranking, relationships and citations are deterministic. "
            "Generation is sampled at temperature 0.0 to minimise variance but is "
            "not bit-reproducible; answer-level metrics are approximately, not "
            "exactly, reproducible."
        ),
    }


def write_reports(report: dict) -> dict[str, Path]:
    reports_dir = settings.evaluation_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"report_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = reports_dir / f"report_{stamp}.csv"
    fields = [
        "id", "abstained", "expect_abstention", "abstention_correct",
        "precision", "recall", "mrr", "citation_accuracy",
        "answer_correctness", "groundedness", "latency_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["per_query"])

    md_path = reports_dir / f"report_{stamp}.md"
    aggregate = report["aggregate"]
    lines = [
        "# Evidence AI Evaluation Report", "",
        f"- Dataset version: **{report['dataset_version']}**",
        f"- Timestamp (UTC): {report['timestamp']}",
        f"- Git commit: `{report['git_commit']}`", "",
        "## Models", "",
        *[f"- {k}: `{v}`" for k, v in report["models"].items()], "",
        "## Configuration", "",
        *[f"- {k}: `{v}`" for k, v in report["configuration"].items()], "",
        "## Aggregate metrics", "",
        "| Metric | Value |", "| --- | --- |",
        *[f"| {k.replace('_', ' ').title()} | {v} |" for k, v in aggregate.items()], "",
        "## Per-query results", "",
        "| ID | Abstain (exp) | P | R | MRR | Cite | Correct | Ground | s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["per_query"]:
        lines.append(
            f"| {record['id']} | {record['abstained']} "
            f"({record['expect_abstention']}) | {record['precision']} | "
            f"{record['recall']} | {record['mrr']} | {record['citation_accuracy']} | "
            f"{record['answer_correctness']} | {record['groundedness']} | "
            f"{record['latency_s']} |"
        )
    lines += ["", "## Determinism", "", report["determinism_note"], ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def main() -> int:
    report = run_evaluation()
    paths = write_reports(report)

    print("\n" + "=" * 78)
    print("AGGREGATE")
    for key, value in report["aggregate"].items():
        print(f"  {key:24s} {value}")
    print("\nReports written:")
    for label, path in paths.items():
        print(f"  {label:9s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Automated benchmark runner for retrieval, agent decisions, and output quality."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from .graph import AssessmentEngine
from .schemas import AssessmentRequest


def evaluate(cases_path: Path, output_path: Path, limit: int | None = None) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if limit:
        cases = cases[:limit]
    engine = AssessmentEngine()
    rows: list[dict] = []
    for case in cases:
        started = time.perf_counter()
        error: str | None = None
        try:
            response = engine.assess(AssessmentRequest.model_validate(case["request"]))
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            categories = {item.category for item in response.evidence}
            groups = set(categories)
            if categories & {"sentiment", "independent"}:
                groups.add("public_signal")
            expected = set(case.get("expected_groups", []))
            group_recall = len(groups & expected) / len(expected) if expected else 1.0
            cited_dimensions = sum(1 for score in response.scores if score.citations)
            citation_coverage = cited_dimensions / len(response.scores)
            returned_ids = {item.evidence_id for item in response.evidence}
            citations = [citation for score in response.scores for citation in score.citations]
            citation_integrity = all(citation in returned_ids for citation in citations)
            grounded_findings_valid = all(
                evidence_id in returned_ids
                for finding in response.main_findings
                for evidence_id in finding.evidence_ids
            )
            material_dimension = next(
                score for score in response.scores if score.dimension == "material_sustainability"
            )
            expected_material_assessable = case.get("expected_material_assessable")
            material_grounding_correct = (
                material_dimension.status == ("assessed" if expected_material_assessable else "not_assessable")
                if expected_material_assessable is not None
                else True
            )
            brand_correct = response.product_summary.brand == case.get("expected_brand")
            rows.append(
                {
                    "id": case["id"],
                    "success": True,
                    "latency_ms": elapsed_ms,
                    "brand_correct": brand_correct,
                    "expected_category_recall": round(group_recall, 4),
                    "citation_coverage": round(citation_coverage, 4),
                    "citation_integrity": citation_integrity,
                    "grounded_findings_valid": grounded_findings_valid,
                    "material_grounding_correct": material_grounding_correct,
                    "material_status": material_dimension.status,
                    "source_diversity": len({item.source_file for item in response.evidence}),
                    "evidence_count": len(response.evidence),
                    "tool_calls": response.metadata.tool_calls,
                    "retrieval_rounds": response.metadata.retrieval_rounds,
                    "contradictions": len(response.contradictions),
                    "confidence": response.confidence,
                }
            )
        except Exception as exc:
            error = str(exc)
            rows.append({"id": case["id"], "success": False, "error": error})

    successful = [row for row in rows if row["success"]]
    report = {
        "cases": len(rows),
        "successful_cases": len(successful),
        "kpis": {
            "assessment_success_rate": round(len(successful) / len(rows), 4) if rows else 0.0,
            "brand_parsing_accuracy": round(mean(row["brand_correct"] for row in successful), 4) if successful else 0.0,
            "expected_category_recall": round(mean(row["expected_category_recall"] for row in successful), 4) if successful else 0.0,
            "citation_coverage": round(mean(row["citation_coverage"] for row in successful), 4) if successful else 0.0,
            "citation_integrity": round(mean(row["citation_integrity"] for row in successful), 4) if successful else 0.0,
            "grounded_finding_validity": round(mean(row["grounded_findings_valid"] for row in successful), 4) if successful else 0.0,
            "material_grounding_accuracy": round(mean(row["material_grounding_correct"] for row in successful), 4) if successful else 0.0,
            "average_source_diversity": round(mean(row["source_diversity"] for row in successful), 2) if successful else 0.0,
            "average_tool_calls": round(mean(row["tool_calls"] for row in successful), 2) if successful else 0.0,
            "average_latency_ms": round(mean(row["latency_ms"] for row in successful), 2) if successful else 0.0,
            "average_confidence": round(mean(row["confidence"] for row in successful), 4) if successful else 0.0,
        },
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the sustainability assessment agent")
    parser.add_argument("--cases", type=Path, default=Path("data/evaluation_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("output/evaluation_report.json"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.cases, args.output, args.limit), indent=2))


if __name__ == "__main__":
    main()

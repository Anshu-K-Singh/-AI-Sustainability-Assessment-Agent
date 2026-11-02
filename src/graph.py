"""LangGraph orchestration for evidence planning and grounded assessment."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import LocalLLM
from .parser import parse_product
from .policy import DIMENSION_TERMS, DEPTH_POLICIES, best_claim_sentence, detect_conflicts, evidence_quality, score_assessment, sufficiency, text_polarity, validate_dimension_citations
from .retrieval import HybridRetriever
from .schemas import (
    AssessmentMetadata,
    AssessmentRequest,
    AssessmentResponse,
    Contradiction,
    EvidenceItem,
    GroundedFinding,
    ProductSummary,
    TraceEvent,
)
from .tools import RetrievalAction, RetrievalTools


class AssessmentState(TypedDict, total=False):
    request: dict[str, Any]
    product: dict[str, Any]
    policy: dict[str, Any]
    plan: list[dict[str, str]]
    source_plan: list[dict[str, str]]
    required_sources: list[str]
    evidence: list[dict[str, Any]]
    retrieval_round: int
    tool_calls: int
    missing_dimensions: list[str]
    coverage: float
    evidence_quality: dict[str, dict[str, float | int]]
    llm_roles_used: list[str]
    contradictions: list[dict[str, Any]]
    unconfirmed_conflicts: int
    scores: list[dict[str, Any]]
    overall_score: int
    confidence: float
    uncertainties: list[str]
    trace: list[dict[str, Any]]
    response: dict[str, Any]


def _trace(state: AssessmentState, step: str, action: str, **details: Any) -> list[dict[str, Any]]:
    events = list(state.get("trace", []))
    events.append(TraceEvent(step=step, action=action, details=details).model_dump())
    return events


class AssessmentEngine:
    def __init__(self, retriever: HybridRetriever | None = None, llm: LocalLLM | None = None) -> None:
        self.retriever = retriever or HybridRetriever()
        self.tools = RetrievalTools(self.retriever)
        self.llm = llm or LocalLLM()
        self.graph = self._build_graph()

    def _parse_input(self, state: AssessmentState) -> dict[str, Any]:
        request = AssessmentRequest.model_validate(state["request"])
        product = parse_product(request, self.llm)
        policy = DEPTH_POLICIES[request.depth.value]
        return {
            "product": product.model_dump(),
            "policy": {
                "depth": request.depth.value,
                "top_k": policy.top_k,
                "max_rounds": policy.max_rounds,
                "sources": list(policy.sources),
                "check_conflicts": policy.check_conflicts,
            },
            "evidence": [],
            "retrieval_round": 0,
            "tool_calls": 0,
            "llm_roles_used": ["query_extraction"] if product.extraction_method == "rules_and_llm_fallback" else [],
            "unconfirmed_conflicts": 0,
            "uncertainties": [f"Missing input attribute: {name}" for name in product.missing_attributes],
            "trace": _trace(
                state,
                "parse_input",
                "Parsed product attributes",
                brand=product.brand,
                category=product.category,
                materials=product.materials,
                missing=product.missing_attributes,
                extraction_method=product.extraction_method,
            ),
        }

    def _plan_sources(self, state: AssessmentState) -> dict[str, Any]:
        product = ProductSummary.model_validate(state["product"])
        source_tools = list(state["policy"]["sources"])
        actions: list[RetrievalAction] = []
        brand = product.brand or "unknown brand"
        materials = ", ".join(product.materials) or product.category or "apparel material"
        if "brand_evidence" in source_tools and product.brand:
            actions.append(
                RetrievalAction(
                    "brand_evidence",
                    f"{brand} sustainability commitments progress climate water circularity supply chain goals metrics",
                    "Brand-level claims and reported metrics are required.",
                )
            )
        if "material_evidence" in source_tools and product.materials:
            actions.append(
                RetrievalAction(
                    "material_evidence",
                    f"{materials} environmental impact lifecycle water energy preferred recycled material apparel",
                    "Material impacts are required for the product assessment.",
                )
            )
        if "certification_evidence" in source_tools:
            actions.append(
                RetrievalAction(
                    "certification_evidence",
                    f"{materials} GOTS certification criteria traceability labeling textile standard",
                    "Deep mode checks certification and traceability criteria.",
                )
            )
        if "public_signal" in source_tools and product.brand:
            actions.append(
                RetrievalAction(
                    "public_signal",
                    f"{brand} sustainability criticism greenwashing ethical durability consumer sentiment",
                    "Public and independent signals are required for contradiction checks.",
                )
            )
        return {
            "plan": [action.__dict__ for action in actions],
            # Keep the original plan separately. A retry has a narrower plan,
            # but coverage must continue to be measured against this full one.
            "source_plan": [action.__dict__ for action in actions],
            "required_sources": [action.tool for action in actions],
            "trace": _trace(
                state,
                "plan_sources",
                "Selected retrieval tools",
                tools=[action.tool for action in actions],
                reasons={action.tool: action.reason for action in actions},
            ),
        }

    def _retrieve(self, state: AssessmentState) -> dict[str, Any]:
        product = ProductSummary.model_validate(state["product"])
        round_number = state.get("retrieval_round", 0) + 1
        collected = {item["evidence_id"]: item for item in state.get("evidence", [])}
        failures: list[str] = []
        calls = state.get("tool_calls", 0)
        counts: dict[str, int] = {}
        for action_data in state.get("plan", []):
            action = RetrievalAction(**action_data)
            calls += 1
            try:
                results = self.tools.execute(action, product, int(state["policy"]["top_k"]))
                counts[action.tool] = len(results)
                for result in results:
                    collected[result.evidence_id] = result.to_dict()
            except Exception as exc:  # a single source failure must degrade gracefully
                failures.append(f"{action.tool}: {exc}")
                counts[action.tool] = 0
        uncertainties = list(state.get("uncertainties", []))
        uncertainties.extend(f"Retrieval failure: {failure}" for failure in failures)
        return {
            "evidence": list(collected.values()),
            "retrieval_round": round_number,
            "tool_calls": calls,
            "uncertainties": list(dict.fromkeys(uncertainties)),
            "trace": _trace(
                state,
                "retrieve_evidence",
                f"Completed retrieval round {round_number}",
                result_counts=counts,
                failures=failures,
                total_unique_evidence=len(collected),
            ),
        }

    def _check_sufficiency(self, state: AssessmentState) -> dict[str, Any]:
        required = state.get("required_sources", [])
        sufficient, missing, coverage = sufficiency(required, state.get("evidence", []))
        quality = evidence_quality(required, state.get("evidence", []))
        return {
            "missing_dimensions": missing,
            "coverage": coverage,
            "evidence_quality": quality,
            "trace": _trace(
                state,
                "check_sufficiency",
                "Evaluated evidence coverage",
                sufficient=sufficient,
                missing=missing,
                coverage=coverage,
                quality=quality,
                current_round=state.get("retrieval_round", 0),
                max_rounds=state["policy"]["max_rounds"],
            ),
        }

    @staticmethod
    def _route_after_sufficiency(state: AssessmentState) -> str:
        if state.get("missing_dimensions") and state.get("retrieval_round", 0) < state["policy"]["max_rounds"]:
            return "refine"
        return "conflict"

    def _refine_plan(self, state: AssessmentState) -> dict[str, Any]:
        missing = set(state.get("missing_dimensions", []))
        refined: list[dict[str, str]] = []
        for action in state.get("source_plan", []):
            if action["tool"] in missing:
                updated = dict(action)
                updated["query"] += " verified evidence recent report certification independent assessment"
                updated["reason"] = f"Refined because {action['tool']} coverage was insufficient."
                refined.append(updated)
        return {
            "plan": refined,
            "trace": _trace(
                state,
                "refine_plan",
                "Refined missing evidence queries within the retrieval budget",
                missing=sorted(missing),
                tools=[action["tool"] for action in refined],
            ),
        }

    def _detect_conflicts(self, state: AssessmentState) -> dict[str, Any]:
        product = ProductSummary.model_validate(state["product"])
        conflicts = detect_conflicts(product, state.get("evidence", [])) if state["policy"]["check_conflicts"] else []
        uncertainties = list(state.get("uncertainties", []))
        if state.get("missing_dimensions"):
            uncertainties.append(
                "Evidence remained incomplete after the configured retrieval budget: "
                + ", ".join(state["missing_dimensions"])
            )
        if not any(item.get("category") == "certifications" for item in state.get("evidence", [])):
            uncertainties.append("No product-level certification was verified.")
        return {
            "contradictions": [conflict.model_dump() for conflict in conflicts],
            "uncertainties": list(dict.fromkeys(uncertainties)),
            "trace": _trace(
                state,
                "detect_conflicts",
                "Compared official claims with independent and public signals",
                contradictions=len(conflicts),
            ),
        }

    def _llm_conflict_analysis(self, state: AssessmentState) -> dict[str, Any]:
        """Let the local model review bounded, deterministic conflict candidates.

        The candidate pairs, source roles, dates, and IDs are selected by policy
        first. A model response can refine or dismiss a candidate, never add a
        new source or change score/confidence logic.
        """
        candidates = [Contradiction.model_validate(item) for item in state.get("contradictions", [])]
        if not candidates:
            return {
                "trace": _trace(
                    state,
                    "llm_conflict_analysis",
                    "Skipped local LLM conflict review",
                    enabled=self.llm.enabled,
                    candidates=len(candidates),
                )
            }
        if not self.llm.enabled:
            # A heuristic match is useful for deciding what to investigate,
            # but it is not enough to call two sources contradictory.
            candidate_count = len(candidates)
            return {
                "contradictions": [],
                "unconfirmed_conflicts": candidate_count,
                "uncertainties": list(dict.fromkeys(
                    list(state.get("uncertainties", []))
                    + [f"{candidate_count} potential conflict candidate(s) could not be semantically confirmed because local LLM review is disabled."]
                )),
                "trace": _trace(
                    state,
                    "llm_conflict_analysis",
                    "Did not promote heuristic conflict candidates without local LLM review",
                    enabled=False,
                    candidates=candidate_count,
                    accepted=0,
                    dismissed=0,
                    unconfirmed_candidates=candidate_count,
                ),
            }
        by_id = {item["evidence_id"]: item for item in state.get("evidence", [])}
        reviewed: list[Contradiction] = []
        accepted = 0
        dismissed = 0
        unavailable = 0
        for candidate in candidates:
            if accepted + dismissed + unavailable >= self.llm.settings.max_conflict_pairs:
                unavailable += 1
                continue
            official = by_id.get(candidate.supporting_evidence[0]) if candidate.supporting_evidence else None
            external = by_id.get(candidate.opposing_evidence[0]) if candidate.opposing_evidence else None
            if not official or not external or official.get("category") != "brand" or external.get("category") not in {"independent", "sentiment"}:
                unavailable += 1
                continue
            result = self.llm.review_conflict(official, external, list(DIMENSION_TERMS))
            if result is None:
                # Do not present an unvalidated keyword match as a confirmed
                # contradiction. It becomes an explicit uncertainty instead.
                unavailable += 1
            elif result.decision == "no_conflict":
                dismissed += 1
            elif result.decision == "potential_conflict":
                reviewed.append(
                    Contradiction(
                        dimension=result.topic,
                        description=result.explanation,
                        supporting_evidence=candidate.supporting_evidence,
                        opposing_evidence=candidate.opposing_evidence,
                        severity=candidate.severity,
                    )
                )
                accepted += 1
            else:
                unavailable += 1
        roles = list(state.get("llm_roles_used", []))
        if accepted or dismissed:
            roles.append("conflict_analysis")
        return {
            "contradictions": [item.model_dump() for item in reviewed],
            "unconfirmed_conflicts": unavailable,
            "llm_roles_used": list(dict.fromkeys(roles)),
            "uncertainties": list(dict.fromkeys(
                list(state.get("uncertainties", []))
                + ([f"{unavailable} potential conflict candidate(s) could not be semantically confirmed."] if unavailable else [])
            )),
            "trace": _trace(
                state,
                "llm_conflict_analysis",
                "Reviewed deterministic conflict candidates with local Ollama",
                model=self.llm.settings.model,
                candidates=len(candidates),
                accepted=accepted,
                dismissed=dismissed,
                unconfirmed_candidates=unavailable,
            ),
        }

    def _score(self, state: AssessmentState) -> dict[str, Any]:
        product = ProductSummary.model_validate(state["product"])
        contradictions = [Contradiction.model_validate(item) for item in state.get("contradictions", [])]
        scores, overall, confidence = score_assessment(
            product,
            state.get("evidence", []),
            contradictions,
            state.get("coverage", 0.0),
            state.get("unconfirmed_conflicts", 0),
        )
        return {
            "scores": [score.model_dump() for score in scores],
            "overall_score": overall,
            "confidence": confidence,
            "trace": _trace(
                state,
                "score_assessment",
                "Applied transparent dimension weights and confidence policy",
                overall_score=overall,
                confidence=confidence,
                dimensions={score.dimension: score.score for score in scores},
            ),
        }

    def _synthesize(self, state: AssessmentState) -> dict[str, Any]:
        product = ProductSummary.model_validate(state["product"])
        evidence = sorted(
            state.get("evidence", []), key=lambda item: item.get("retrieval_score", 0.0), reverse=True
        )
        evidence_items = [
            EvidenceItem(
                evidence_id=item["evidence_id"],
                snippet=" ".join(item["text"].split())[:700],
                source_file=item["source_file"],
                source_url=item["source_url"],
                source_title=item.get("source_title"),
                publisher=item.get("publisher"),
                published_date=item.get("published_date"),
                captured_at=item.get("captured_at"),
                source_type=item["source_type"],
                category=item["category"],
                entity=item["entity"],
                section=item.get("section", ""),
                page=item.get("page"),
                credibility_tier=item["credibility_tier"],
                retrieval_score=item["retrieval_score"],
                recency_score=item.get("recency_score", 0.6),
                retrieval_methods=item.get("retrieval_methods", []),
            )
            for item in evidence
        ]
        scores = state["scores"]
        cited = [citation for score in scores for citation in score.get("citations", [])]
        dimension_citations_valid, invalid_dimension_citations = validate_dimension_citations(scores, evidence)
        citation_note = ", ".join(list(dict.fromkeys(cited))[:6]) or "no supporting evidence IDs"
        score = state["overall_score"]
        findings: list[GroundedFinding] = []
        finding_groups = {
            "material": {"materials", "certifications"},
            "brand": {"brand"},
            "public_signal": {"independent", "sentiment"},
        }
        for category, categories in finding_groups.items():
            item = next((candidate for candidate in evidence if candidate.get("category") in categories), None)
            if not item:
                continue
            claim = best_claim_sentence(item["text"])
            findings.append(
                GroundedFinding(
                    finding=claim,
                    evidence_ids=[item["evidence_id"]],
                    category=category,
                    polarity=("positive" if text_polarity(item["text"]) > 0 else "negative" if text_polarity(item["text"]) < 0 else "mixed"),
                )
            )
        label = "stronger evidence-backed profile" if score >= 70 else "mixed evidence-backed profile" if score >= 45 else "weak or incomplete evidence profile"
        assessment = (
            f"{product.brand or 'The product'} {product.category or 'apparel product'} received {score}/100, indicating a {label}. "
            f"This is an evidence assessment, not a lifecycle certification or a claim that the product is sustainable. "
            f"Supporting evidence IDs: {citation_note}."
        )
        next_actions: list[str] = []
        if "brand" in product.missing_attributes:
            next_actions.append("Provide the brand to enable company-level evidence retrieval.")
        if "materials" in product.missing_attributes:
            next_actions.append("Provide the material composition to improve material scoring.")
        if not any(item.category == "certifications" for item in evidence_items):
            next_actions.append("Verify product-level certification in an official certification directory.")
        if state.get("missing_dimensions"):
            next_actions.append("Add or refresh corpus sources for: " + ", ".join(state["missing_dimensions"]))
        draft = self.llm.synthesize(
            product=product.model_dump(),
            evidence=evidence,
            scores=scores,
            overall_score=score,
            confidence=state["confidence"],
            contradictions=state.get("contradictions", []),
            uncertainties=state.get("uncertainties", []),
            allowed_categories=["material", "brand", "public_signal", "certification", "independent", "sentiment", "mixed"],
        )
        roles = list(state.get("llm_roles_used", []))
        if draft:
            # If a small local model supplies cited findings but omits the
            # optional narrative, keep the deterministic narrative instead.
            assessment = draft.overall_assessment.strip() or assessment
            findings = [
                GroundedFinding(
                    finding=finding.finding,
                    evidence_ids=finding.evidence_ids,
                    category=finding.category,
                    polarity=finding.polarity,
                )
                for finding in draft.main_findings
            ]
            # Policy-required actions remain present even when the model adds a
            # clearer phrasing or an additional evidence-based action.
            next_actions = list(dict.fromkeys(next_actions + draft.recommended_next_actions))[:6]
            roles.append("grounded_synthesis")
        response = AssessmentResponse(
            product_summary=product,
            scores=scores,
            overall_score=score,
            overall_assessment=assessment,
            main_findings=findings,
            evidence=evidence_items,
            confidence=state["confidence"],
            contradictions=state.get("contradictions", []),
            uncertainties=state.get("uncertainties", []),
            recommended_next_actions=next_actions,
            trace=state["trace"]
            + [
                TraceEvent(
                    step="synthesize",
                    action="Generated grounded structured response",
                    details={
                        "citations_validated": bool(cited)
                        and all(citation in {item.evidence_id for item in evidence_items} for citation in cited),
                        "dimension_citations_valid": dimension_citations_valid,
                        "invalid_dimension_citations": invalid_dimension_citations,
                        "returned_evidence": len(evidence_items),
                        "grounded_findings": len(findings),
                        "llm_grounded_synthesis_used": bool(draft),
                    },
                )
            ],
            metadata=AssessmentMetadata(
                depth=state["policy"]["depth"],
                retrieval_rounds=state["retrieval_round"],
                tool_calls=state["tool_calls"],
                evidence_count=len(state.get("evidence", [])),
                llm_model=self.llm.settings.model if self.llm.enabled else None,
                llm_attempted=self.llm.enabled,
                llm_roles_used=list(dict.fromkeys(roles)),
            ),
        )
        return {"response": response.model_dump(mode="json")}

    def _build_graph(self):
        builder = StateGraph(AssessmentState)
        builder.add_node("parse_input", self._parse_input)
        builder.add_node("plan_sources", self._plan_sources)
        builder.add_node("retrieve_evidence", self._retrieve)
        builder.add_node("check_sufficiency", self._check_sufficiency)
        builder.add_node("refine_plan", self._refine_plan)
        builder.add_node("detect_conflicts", self._detect_conflicts)
        builder.add_node("llm_conflict_analysis", self._llm_conflict_analysis)
        builder.add_node("score_assessment", self._score)
        builder.add_node("synthesize", self._synthesize)
        builder.add_edge(START, "parse_input")
        builder.add_edge("parse_input", "plan_sources")
        builder.add_edge("plan_sources", "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "check_sufficiency")
        builder.add_conditional_edges(
            "check_sufficiency",
            self._route_after_sufficiency,
            {"refine": "refine_plan", "conflict": "detect_conflicts"},
        )
        builder.add_edge("refine_plan", "retrieve_evidence")
        builder.add_edge("detect_conflicts", "llm_conflict_analysis")
        builder.add_edge("llm_conflict_analysis", "score_assessment")
        builder.add_edge("score_assessment", "synthesize")
        builder.add_edge("synthesize", END)
        return builder.compile()

    def assess(self, request: AssessmentRequest) -> AssessmentResponse:
        final_state = self.graph.invoke({"request": request.model_dump(mode="json"), "trace": []})
        return AssessmentResponse.model_validate(final_state["response"])

    def graph_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

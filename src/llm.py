"""Constrained local LLM helpers using Ollama's OpenAI-compatible endpoint.

The model never retrieves evidence, calculates scores, or sets confidence. It
only interprets an ambiguous query, reviews a pre-selected conflict pair, and
phrases an evidence-bounded final summary. Every structured output is validated
before the graph accepts it; callers always have deterministic fallbacks.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float
    max_conflict_pairs: int

    @classmethod
    def from_env(cls) -> "LLMSettings":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return cls(
            enabled=_truthy(os.getenv("LLM_ENABLED", "false")),
            base_url=base_url,
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45")),
            max_conflict_pairs=max(0, int(os.getenv("LLM_MAX_CONFLICT_PAIRS", "3"))),
        )


class ProductHints(BaseModel):
    brand: str | None = None
    category: str | None = None
    materials: list[str] = Field(default_factory=list)
    sustainability_keywords: list[str] = Field(default_factory=list)


class ConflictReview(BaseModel):
    decision: str
    topic: str
    explanation: str = Field(default="", max_length=600)


class SynthesizedFinding(BaseModel):
    finding: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=3)
    category: str = Field(min_length=1, max_length=80)
    polarity: str = "mixed"


class SynthesisDraft(BaseModel):
    # A missing narrative does not invalidate fully cited findings from a small
    # local model; the graph retains its deterministic narrative in that case.
    overall_assessment: str = Field(default="", max_length=700)
    main_findings: list[SynthesizedFinding] = Field(default_factory=list, max_length=3)
    recommended_next_actions: list[str] = Field(default_factory=list, max_length=3)


class LocalLLM:
    """Small, synchronous adapter that fails closed to deterministic behavior."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def _json_completion(self, system: str, payload: dict[str, Any], max_tokens: int) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key,
                timeout=self.settings.timeout_seconds,
            )
            response = client.chat.completions.create(
                model=self.settings.model,
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            # Defensive cleanup supports models that wrap JSON in a Markdown fence.
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I).strip()
            parsed = json.loads(content)
            # Small local models occasionally mirror the example field name and
            # put the requested object under ``schema``. Accept only this
            # unambiguous wrapper; the normal Pydantic/citation validators still
            # run immediately afterwards.
            if isinstance(parsed, dict) and set(parsed) == {"schema"} and isinstance(parsed["schema"], dict):
                return parsed["schema"]
            return parsed
        except Exception:
            return None

    def extract_product_hints(
        self,
        query: str,
        *,
        allowed_brands: list[str],
        allowed_categories: list[str],
        allowed_materials: list[str],
        allowed_keywords: list[str],
    ) -> ProductHints | None:
        raw = self._json_completion(
            "You extract apparel-product attributes from a user query. Return JSON only. "
            "Use only values from the supplied allowed lists. Never guess a material or brand; use null or [] "
            "when unsupported by the query.",
            {
                "query": query,
                "allowed_brands": allowed_brands,
                "allowed_categories": allowed_categories,
                "allowed_materials": allowed_materials,
                "allowed_sustainability_keywords": allowed_keywords,
                "schema": {"brand": "string|null", "category": "string|null", "materials": ["string"], "sustainability_keywords": ["string"]},
            },
            max_tokens=220,
        )
        try:
            hints = ProductHints.model_validate(raw)
        except (ValidationError, TypeError):
            return None
        allowed = {
            "brand": {item.lower(): item for item in allowed_brands},
            "category": {item.lower(): item for item in allowed_categories},
            "material": {item.lower(): item for item in allowed_materials},
            "keyword": {item.lower(): item for item in allowed_keywords},
        }
        # Reject, rather than repair, any value outside the controlled vocabulary.
        if hints.brand and hints.brand.lower() not in allowed["brand"]:
            return None
        if hints.category and hints.category.lower() not in allowed["category"]:
            return None
        if any(item.lower() not in allowed["material"] for item in hints.materials):
            return None
        if any(item.lower() not in allowed["keyword"] for item in hints.sustainability_keywords):
            return None
        query_lowered = query.lower()
        # Materials and claim keywords affect deterministic scoring. A local
        # model may map "eco-friendly" to "organic" or "recycled" even where
        # the user never supplied that fact, so accept them only when the exact
        # controlled term is present in the query. Brand/category are merely
        # routing attributes and may be inferred from wording such as
        # "Patagonia outerwear" -> Patagonia / jacket.
        explicit_materials = [
            item for item in hints.materials
            if item.lower() in query_lowered
        ]
        explicit_keywords = [
            item for item in hints.sustainability_keywords
            if item.lower() in query_lowered
        ]
        return ProductHints(
            brand=allowed["brand"].get((hints.brand or "").lower()),
            category=allowed["category"].get((hints.category or "").lower()),
            materials=[allowed["material"][item.lower()] for item in dict.fromkeys(explicit_materials)],
            sustainability_keywords=[allowed["keyword"][item.lower()] for item in dict.fromkeys(explicit_keywords)],
        )

    def review_conflict(self, official: dict[str, Any], external: dict[str, Any], allowed_topics: list[str]) -> ConflictReview | None:
        raw = self._json_completion(
            "You review exactly two retrieved sustainability excerpts. Decide whether they make opposing claims "
            "about the same topic. Do not introduce facts. Return JSON only with decision equal to "
            "potential_conflict, no_conflict, or insufficient_evidence; use a topic from allowed_topics.",
            {
                "allowed_topics": allowed_topics,
                "official_evidence": {"id": official["evidence_id"], "date": official.get("published_date"), "text": official["text"][:700]},
                "external_evidence": {"id": external["evidence_id"], "date": external.get("published_date"), "text": external["text"][:700]},
                "schema": {"decision": "string", "topic": "string", "explanation": "short evidence-grounded string"},
            },
            max_tokens=180,
        )
        try:
            review = ConflictReview.model_validate(raw)
        except (ValidationError, TypeError):
            return None
        if review.decision not in {"potential_conflict", "no_conflict", "insufficient_evidence"}:
            return None
        if review.topic not in allowed_topics:
            return None
        return review

    def synthesize(
        self,
        *,
        product: dict[str, Any],
        evidence: list[dict[str, Any]],
        scores: list[dict[str, Any]],
        overall_score: int,
        confidence: float,
        contradictions: list[dict[str, Any]],
        uncertainties: list[str],
        allowed_categories: list[str],
    ) -> SynthesisDraft | None:
        raw = self._json_completion(
            "Write a concise apparel sustainability evidence assessment. Return JSON only with exactly these keys: "
            "overall_assessment, main_findings, recommended_next_actions. Do not output uncertainties or contradictions "
            "because they are supplied separately. Use only the supplied evidence and numeric scores. Do not say a product "
            "is certified or sustainable. Each finding must cite one to three evidence IDs from the supplied evidence list.",
            {
                "product": product,
                "overall_score": overall_score,
                "confidence": confidence,
                "dimension_scores": [
                    {"dimension": item.get("dimension"), "score": item.get("score"), "confidence": item.get("confidence")}
                    for item in scores
                ],
                "contradictions": [
                    {"dimension": item.get("dimension"), "description": item.get("description")}
                    for item in contradictions
                ],
                "uncertainties": uncertainties,
                "allowed_finding_categories": allowed_categories,
                "evidence": [
                    {
                        # Short aliases make it practical for a small local
                        # model to cite evidence. They are mapped back to the
                        # canonical IDs below before the response is accepted.
                        "id": f"E{index}",
                        "canonical_id": item["evidence_id"],
                        "category": item["category"],
                        "source": item.get("source_title") or item.get("source_file"),
                        "published_date": item.get("published_date"),
                        "text": item["text"][:350],
                    }
                    for index, item in enumerate(evidence[:4], start=1)
                ],
                "schema": {
                    "overall_assessment": "string",
                    "main_findings": [{"finding": "string", "evidence_ids": ["id"], "category": "string", "polarity": "positive|negative|mixed"}],
                    "recommended_next_actions": ["string"],
                },
            },
            # qwen2.5:3b runs locally on CPU in this assignment environment;
            # this cap keeps a concise JSON response within interactive latency.
            max_tokens=200,
        )
        try:
            draft = SynthesisDraft.model_validate(raw)
        except (ValidationError, TypeError):
            return None
        valid_ids = {item["evidence_id"] for item in evidence}
        # Small local models sometimes use E1/E2 aliases instead of copying a
        # long stable chunk ID. Resolve only ordinal aliases that correspond to
        # the exact evidence list supplied in the prompt; arbitrary IDs remain
        # rejected.
        aliases = {f"E{index}": item["evidence_id"] for index, item in enumerate(evidence[:4], start=1)}
        normalized_citations: list[SynthesizedFinding] = []
        for finding in draft.main_findings:
            resolved = [aliases.get(evidence_id, evidence_id) for evidence_id in finding.evidence_ids]
            if any(evidence_id not in valid_ids for evidence_id in resolved):
                return None
            normalized_citations.append(finding.model_copy(update={"evidence_ids": resolved}))
        normalized_findings = [
            finding.model_copy(
                update={
                    # A harmless label mismatch must not discard an otherwise
                    # fully cited response from a small local model. Unknown
                    # labels are made explicit as mixed, never treated as a
                    # new assessment dimension.
                    "category": finding.category if finding.category in allowed_categories else "mixed",
                    "polarity": finding.polarity if finding.polarity in {"positive", "negative", "mixed"} else "mixed",
                }
            )
            for finding in draft.main_findings
        ]
        normalized_findings = [
            finding.model_copy(update={"evidence_ids": citations.evidence_ids})
            for finding, citations in zip(normalized_findings, normalized_citations)
        ]
        return draft.model_copy(update={"main_findings": normalized_findings})

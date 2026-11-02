"""Hybrid BM25 + Chroma retrieval over the local sustainability corpus."""

from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


@dataclass
class RetrievedEvidence:
    evidence_id: str
    text: str
    source_file: str
    source_url: str
    source_title: str | None
    publisher: str | None
    published_date: str | None
    captured_at: str | None
    source_type: str
    category: str
    entity: str
    section: str
    page: int | None
    credibility_tier: str
    recency_score: float
    retrieval_score: float
    retrieval_methods: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class HybridRetriever:
    """Search both local indexes and combine their ranks with weighted RRF."""

    def __init__(
        self,
        index_dir: str | Path = "data/indexes",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "sustainability_corpus",
    ) -> None:
        self.index_dir = Path(index_dir)
        chunks_path = self.index_dir / "chunks.jsonl"
        bm25_path = self.index_dir / "bm25.pkl"
        chroma_path = self.index_dir / "chroma_db"
        missing = [str(path) for path in (chunks_path, bm25_path, chroma_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing retrieval artifacts: {', '.join(missing)}")

        self.chunks: dict[str, dict] = {}
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                self.chunks[item["id"]] = item

        with bm25_path.open("rb") as handle:
            self.bm25_payload = pickle.load(handle)  # noqa: S301 - trusted local build artifact

        try:
            # Inference must be reproducible after ingestion. The embedding model is
            # already cached when the Chroma index is built, so runtime retrieval
            # must not pause while checking the Hugging Face network.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError as exc:
            raise RuntimeError("Retrieval requires chromadb and sentence-transformers") from exc

        self.chroma = chromadb.PersistentClient(path=str(chroma_path))
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection = self.chroma.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @staticmethod
    def _tokens(query: str) -> list[str]:
        return re.findall(r"\b\w+\b", query.lower())

    @staticmethod
    def _as_set(values: str | Iterable[str] | None) -> set[str]:
        if values is None:
            return set()
        if isinstance(values, str):
            return {values.lower()}
        return {str(value).lower() for value in values}

    def _matches(self, item: dict, categories: set[str], entities: set[str]) -> bool:
        if categories and str(item.get("category", "")).lower() not in categories:
            return False
        if entities and str(item.get("entity", "")).lower() not in entities:
            return False
        return True

    @staticmethod
    def _recency_score(published_date: str | None) -> float:
        """Return a neutral score for unknown dates and decay older known sources."""
        if not published_date:
            return 0.6
        try:
            published = datetime.strptime(published_date, "%Y-%m-%d").date()
        except ValueError:
            return 0.6
        age_days = max(0, (date.today() - published).days)
        # Half-life of two years: recency informs ranking but does not erase
        # older authoritative standards or longitudinal evidence.
        return round(0.5 ** (age_days / 730), 4)

    @staticmethod
    def _chroma_where(categories: set[str], entities: set[str]) -> dict | None:
        clauses: list[dict] = []
        if categories:
            clauses.append({"category": {"$in": sorted(categories)}})
        if entities:
            clauses.append({"entity": {"$in": sorted(entities)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _bm25_ranked(self, query: str, categories: set[str], entities: set[str], limit: int) -> list[str]:
        scores = self.bm25_payload["bm25"].get_scores(self._tokens(query))
        pairs = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)
        ranked: list[str] = []
        ids = self.bm25_payload["ids"]
        for index, score in pairs:
            if float(score) <= 0:
                break
            chunk_id = ids[index]
            item = self.chunks.get(chunk_id)
            if item and self._matches(item, categories, entities):
                ranked.append(chunk_id)
            if len(ranked) >= limit:
                break
        return ranked

    def _semantic_ranked(self, query: str, categories: set[str], entities: set[str], limit: int) -> list[str]:
        count = self.collection.count()
        if count == 0:
            return []
        candidate_count = min(max(limit * 8, 60), count)
        where = self._chroma_where(categories, entities)
        try:
            result = self.collection.query(
                query_texts=[query],
                n_results=candidate_count,
                where=where,
                include=["metadatas", "distances"],
            )
        except Exception:
            # The Python-side filter remains a safe fallback for older Chroma
            # versions whose metadata-filter syntax differs.
            result = self.collection.query(
                query_texts=[query],
                n_results=candidate_count,
                include=["metadatas", "distances"],
            )
        ranked: list[str] = []
        for chunk_id in (result.get("ids") or [[]])[0]:
            item = self.chunks.get(chunk_id)
            if item and self._matches(item, categories, entities):
                ranked.append(chunk_id)
            if len(ranked) >= limit:
                break
        return ranked

    def search(
        self,
        query: str,
        *,
        categories: str | Iterable[str] | None = None,
        entities: str | Iterable[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        category_filter = self._as_set(categories)
        entity_filter = self._as_set(entities)
        pool_size = max(top_k * 5, 20)
        bm25_ids = self._bm25_ranked(query, category_filter, entity_filter, pool_size)
        semantic_ids = self._semantic_ranked(query, category_filter, entity_filter, pool_size)

        # Weighted reciprocal-rank fusion rewards agreement without requiring
        # incomparable raw BM25 and cosine scores to share a scale.
        fused: dict[str, float] = {}
        methods: dict[str, list[str]] = {}
        for method, weight, ranked in (
            ("bm25", 1.0, bm25_ids),
            ("semantic", 1.15, semantic_ids),
        ):
            for rank, chunk_id in enumerate(ranked, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (60 + rank)
                methods.setdefault(chunk_id, []).append(method)

        ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
        maximum = max((fused[chunk_id] for chunk_id in ordered), default=1.0)
        output: list[RetrievedEvidence] = []
        for chunk_id in ordered:
            item = self.chunks[chunk_id]
            output.append(
                RetrievedEvidence(
                    evidence_id=chunk_id,
                    text=item["text"],
                    source_file=item["source_file"],
                    source_url=item.get("source_url") or "",
                    source_title=item.get("source_title"),
                    publisher=item.get("publisher"),
                    published_date=item.get("published_date"),
                    captured_at=item.get("captured_at"),
                    source_type=item["source_type"],
                    category=item["category"],
                    entity=item["entity"],
                    section=item.get("section", ""),
                    page=item.get("page"),
                    credibility_tier=item["confidence_tier"],
                    # This is a normalized rank for ordering, not a probability
                    # of factual correctness or evidence truth.
                    retrieval_score=round(
                        (fused[chunk_id] / maximum)
                        * (0.75 + 0.25 * self._recency_score(item.get("published_date"))),
                        4,
                    ),
                    recency_score=self._recency_score(item.get("published_date")),
                    retrieval_methods=methods.get(chunk_id, []),
                )
            )
        return output


def format_results(results: list[RetrievedEvidence]) -> str:
    lines: list[str] = []
    for rank, item in enumerate(results, start=1):
        preview = re.sub(r"\s+", " ", item.text)[:180]
        lines.append(
            f"{rank}. {item.evidence_id} [{item.category}/{item.credibility_tier}] "
            f"rank={item.retrieval_score:.3f} recency={item.recency_score:.3f} methods={','.join(item.retrieval_methods)}\n"
            f"   {preview}\n   {item.source_url}"
        )
    return "\n".join(lines)

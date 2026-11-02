"""FastAPI entry point for the local sustainability assessment agent."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .graph import AssessmentEngine
from .llm import LLMSettings
from .schemas import AssessmentRequest, AssessmentResponse, HealthResponse


app = FastAPI(
    title="AI Sustainability Assessment Agent",
    version="1.0.0",
    description=(
        "A local, evidence-grounded apparel sustainability assessment service. "
        "It uses hybrid RAG and a bounded LangGraph workflow to plan retrieval, "
        "check evidence sufficiency, flag contradictions, and return citations."
    ),
    contact={"name": "AI/ML Engineering Assignment"},
)


@lru_cache(maxsize=1)
def get_engine() -> AssessmentEngine:
    return AssessmentEngine()


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    index_dir = Path("data/indexes")
    indexes = {
        "chunks": (index_dir / "chunks.jsonl").exists(),
        "bm25": (index_dir / "bm25.pkl").exists(),
        "chroma": (index_dir / "chroma_db").exists(),
    }
    try:
        chunks = get_engine().retriever.chunk_count if all(indexes.values()) else 0
        status = "healthy" if all(indexes.values()) and chunks > 0 else "degraded"
    except Exception:
        chunks = 0
        status = "degraded"
    llm = LLMSettings.from_env()
    return HealthResponse(
        status=status,
        chunks=chunks,
        indexes=indexes,
        llm={"enabled": llm.enabled, "model": llm.model, "base_url": llm.base_url},
    )


@app.post(
    "/assessments",
    response_model=AssessmentResponse,
    tags=["assessments"],
    summary="Assess an apparel product using grounded local evidence",
)
def create_assessment(request: AssessmentRequest) -> AssessmentResponse:
    try:
        return get_engine().assess(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Retrieval indexes unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {exc}") from exc


@app.get("/workflow", tags=["operations"], summary="Return the compiled LangGraph workflow")
def workflow() -> dict[str, str]:
    return {"format": "mermaid", "diagram": get_engine().graph_mermaid()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=False)

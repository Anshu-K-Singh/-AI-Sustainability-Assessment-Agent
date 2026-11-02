from __future__ import annotations

import pytest

from src.graph import AssessmentEngine
from src.llm import LLMSettings, LocalLLM


@pytest.fixture(scope="session")
def engine() -> AssessmentEngine:
    # Unit tests exercise deterministic behavior. Dedicated LLM tests inject a
    # fake adapter, so the suite never depends on a running Ollama service.
    return AssessmentEngine(llm=LocalLLM(LLMSettings(False, "http://127.0.0.1:11434/v1", "qwen2.5:3b", "ollama", 1, 3)))

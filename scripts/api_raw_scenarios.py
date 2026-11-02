"""Capture untouched FastAPI request/response bodies for the benchmark cases.

The output is an audit artifact, not a hand-written example: every record
contains the submitted JSON request, HTTP transport metadata, and the exact
JSON response returned by the FastAPI application. The file is checkpointed
after each case so a long local-Ollama run still preserves completed results.
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LLM_ENABLED", "true")
os.environ.setdefault("LLM_MAX_CONFLICT_PAIRS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "api_raw_scenarios.json"
PROGRESS = ROOT / "output" / "api_raw_capture_progress.log"
CASES_PATH = ROOT / "data" / "evaluation_cases.json"


def load_cases() -> list[tuple[str, dict]]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    selected = [(case["id"], case["request"]) for case in cases]
    # Deliberately invalid input verifies FastAPI/Pydantic validation as well.
    selected.append(("invalid_empty_request", {}))
    return selected


def checkpoint(records: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def progress(message: str) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture exact FastAPI API responses")
    parser.add_argument("--resume", action="store_true", help="Keep checkpointed records and run remaining scenarios only")
    parser.add_argument("--llm-timeout", type=float, default=None, help="Optional local-Ollama timeout per request in seconds")
    args = parser.parse_args()
    if args.llm_timeout is not None:
        os.environ["OLLAMA_TIMEOUT_SECONDS"] = str(args.llm_timeout)
    if not args.resume and PROGRESS.exists():
        PROGRESS.unlink()
    progress(f"capture started; resume={args.resume}; llm_timeout={os.getenv('OLLAMA_TIMEOUT_SECONDS')}")

    # Import after parsing options so the local LLM reads the intended runtime
    # configuration while FastAPI constructs its cached engine.
    from fastapi.testclient import TestClient
    from src.main import app
    progress("FastAPI application imported")

    records: list[dict] = json.loads(OUTPUT.read_text(encoding="utf-8")) if args.resume and OUTPUT.exists() else []
    completed = {record["scenario"] for record in records}
    # Avoid entering the TestClient lifespan context here: this lightweight
    # application has no startup hook, and direct construction keeps the
    # capture runner compatible with the local Starlette/httpx versions.
    progress("constructing FastAPI TestClient")
    client = TestClient(app, raise_server_exceptions=False)
    progress("FastAPI TestClient constructed")
    try:
        for scenario, request in load_cases():
            if scenario in completed:
                continue
            progress(f"starting {scenario}")
            started = time.perf_counter()
            response = client.post("/assessments", json=request)
            record = {
                "scenario": scenario,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "request": request,
                "transport": {
                    "http_status": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
                "response_body": response.json(),
            }
            records.append(record)
            checkpoint(records)
            progress(f"completed {scenario}; status={response.status_code}; latency_ms={record['transport']['latency_ms']}")
            print(json.dumps({"scenario": scenario, **record["transport"]}), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        progress(traceback.format_exc())
        raise

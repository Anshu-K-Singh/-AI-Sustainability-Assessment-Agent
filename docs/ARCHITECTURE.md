# Architecture

```mermaid
flowchart TD
    U[User / Swagger] --> API[FastAPI]
    API --> G[LangGraph assessment workflow]
    G --> P[Rules parser<br/>LLM fallback only if ambiguous]
    P --> SP[Bounded source planner]
    SP --> B[Brand tool]
    SP --> M[Material tool<br/>only when composition is known]
    SP --> C[Certification tool]
    SP --> S[Public-signal tool]
    B & M & C & S --> R[Hybrid RAG<br/>BM25 + Chroma + metadata filters]
    R --> I[(Local corpus<br/>405 cited chunks)]
    R --> Q{Evidence sufficient?}
    Q -->|No, budget remains| SP
    Q -->|Yes / budget exhausted| X[Conflict candidate policy]
    X --> L[Local Ollama review<br/>or safe uncertainty fallback]
    L --> SC[Deterministic scores + confidence]
    SC --> O[Validated JSON response<br/>citations, uncertainties, trace]
```

## Boundaries

- **Deterministic:** source selection, retrieval, metadata filters, retry limits, citation validation, scores, confidence, and API schema.
- **LLM-assisted:** ambiguous attribute extraction, review of one pre-selected conflict pair, and grounded wording of findings.
- **Safe behavior:** no material score without known composition; no confirmed contradiction without semantic validation; unavailable/invalid LLM output uses the deterministic response.

## Retrieval flow

The query is decomposed by evidence type. BM25 retrieves exact terms; Chroma retrieves semantic matches; weighted reciprocal-rank fusion combines them. Returned evidence carries source metadata and stable IDs so every response can show its receipts.

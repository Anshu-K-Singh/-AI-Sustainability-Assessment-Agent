"""Build the local chunk, ChromaDB, and BM25 indexes.

Inputs are first normalized into semantic blocks. Chunking therefore works for
Markdown, HTML, text, and PDF sources rather than depending on one file type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

URL_RE = re.compile(r"(?:<!--\s*Source URL:\s*|Source URL:\s*)(https?://[^\s>]+)", re.I)
TOKEN_RE = re.compile(r"\S+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Block:
    kind: str
    text: str
    section: str = ""
    page: int | None = None


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    source_url: str | None
    source_title: str | None
    publisher: str | None
    published_date: str | None
    captured_at: str | None
    source_type: str
    category: str
    entity: str
    section: str
    page: int | None
    block_kinds: str
    confidence_tier: str


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def source_url(text: str) -> str | None:
    match = URL_RE.search(text[:3000])
    return match.group(1).rstrip(".,)") if match else None


def load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load source provenance keyed by corpus-relative file path.

    Snapshotted HTML/PDF documents cannot reliably retain their original URL or
    publication date in extracted text. The manifest is therefore canonical
    provenance, while a document header remains a useful fallback.
    """
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    captured_at = payload.get("captured_at")
    sources: dict[str, dict[str, Any]] = {}
    for item in payload.get("sources", []):
        relative = item.get("source_file")
        if not relative:
            continue
        metadata = dict(item)
        metadata["captured_at"] = metadata.get("captured_at") or captured_at
        sources[relative.replace("\\", "/")] = metadata
    return sources


def parse_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    heading_path: list[str] = []
    paragraph: list[str] = []
    table: list[str] = []

    def section() -> str:
        return " > ".join(heading_path)

    def flush_paragraph() -> None:
        if paragraph:
            value = clean_text(" ".join(paragraph))
            if value:
                blocks.append(Block("paragraph", value, section()))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            blocks.append(Block("table", "\n".join(table), section()))
            table.clear()

    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            flush_paragraph()
            flush_table()
            continue
        if in_fence:
            paragraph.append(line)
            continue
        if not line or line.startswith("<!--"):
            flush_paragraph()
            flush_table()
            continue
        heading = HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            flush_table()
            level = len(heading.group(1))
            heading_path[:] = heading_path[: level - 1]
            heading_path.append(clean_text(heading.group(2)))
            continue
        if line.startswith("|") or (table and "|" in line):
            flush_paragraph()
            table.append(line)
            continue
        if table:
            flush_table()
        paragraph.append(line)
    flush_paragraph()
    flush_table()
    return blocks


def parse_html(text: str) -> list[Block]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("HTML ingestion requires beautifulsoup4") from exc
    soup = BeautifulSoup(text, "html.parser")
    for node in soup(["script", "style", "nav", "footer"]):
        node.decompose()
    blocks: list[Block] = []
    section = ""
    for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        value = clean_text(node.get_text(" ", strip=True))
        if not value:
            continue
        if node.name.startswith("h"):
            section = value
            continue
        blocks.append(Block("table" if node.name == "table" else "paragraph", value, section))
    return blocks


def parse_pdf(path: Path) -> tuple[list[Block], str | None]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf") from exc
    reader = PdfReader(str(path))
    blocks: list[Block] = []
    for number, page in enumerate(reader.pages, start=1):
        value = clean_text(page.extract_text() or "")
        if value:
            blocks.append(Block("page", value, page=number))
    return blocks, None


def extract(path: Path) -> tuple[list[Block], str | None]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".md", ".markdown", ".txt"}:
        return parse_markdown(raw), source_url(raw)
    if suffix in {".html", ".htm"}:
        return parse_html(raw), source_url(raw)
    return [], None


def infer_category(path: Path) -> str:
    return path.parent.name if path.parent.name not in {"corpus", "data"} else "uncategorized"


def infer_entity(path: Path) -> str:
    name = path.stem.lower()
    if "levis" in name or "levi" in name:
        return "Levi's"
    if "patagonia" in name:
        return "Patagonia"
    if name.startswith("hm_") or name.startswith("hm-"):
        return "H&M"
    if "cotton" in name:
        return "Cotton"
    if "polyester" in name:
        return "Polyester"
    if "recycled" in name:
        return "Recycled materials"
    if "gots" in name:
        return "GOTS"
    return path.stem.replace("_", " ").title()


def confidence_tier(category: str) -> str:
    if category in {"brand", "materials", "certifications"}:
        return "primary"
    if category == "sentiment":
        return "public_signal"
    if category == "independent":
        return "secondary"
    return "unknown"


def split_oversized(text: str, max_tokens: int) -> list[str]:
    """Split large blocks by sentences, then token windows."""
    pieces = [clean_text(p) for p in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if clean_text(p)]
    result: list[str] = []
    current: list[str] = []
    for piece in pieces:
        piece_tokens = tokens(piece)
        if len(current) + len(piece_tokens) <= max_tokens:
            current.extend(piece_tokens)
            continue
        if current:
            result.append(" ".join(current))
            current = []
        for start in range(0, len(piece_tokens), max_tokens):
            result.append(" ".join(piece_tokens[start : start + max_tokens]))
    if current:
        result.append(" ".join(current))
    return result


def make_chunks(
    path: Path,
    blocks: Sequence[Block],
    url: str | None,
    root: Path,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    category = infer_category(path)
    entity = infer_entity(path)
    source_type = path.suffix.lower().lstrip(".") or "text"
    relative = path.relative_to(root).as_posix()
    tier = confidence_tier(category)
    metadata = metadata or {}
    canonical_url = metadata.get("source_url") or url
    stable_prefix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
    prepared: list[Block] = []
    for block in blocks:
        value = clean_text(block.text)
        if not value:
            continue
        combined = f"Section: {block.section}\n{value}" if block.section else value
        if len(tokens(combined)) <= max_tokens:
            prepared.append(Block(block.kind, combined, block.section, block.page))
        else:
            prepared.extend(Block(block.kind, part, block.section, block.page) for part in split_oversized(combined, max_tokens))

    output: list[Chunk] = []
    current: list[Block] = []
    current_count = 0

    def emit(items: list[Block]) -> None:
        if not items:
            return
        output.append(Chunk(
            id=f"{path.stem}-{stable_prefix}-{len(output):04d}",
            text="\n\n".join(item.text for item in items),
            source_file=relative,
            source_url=canonical_url,
            source_title=metadata.get("title"),
            publisher=metadata.get("publisher"),
            published_date=metadata.get("published_date"),
            captured_at=metadata.get("captured_at"),
            source_type=source_type,
            category=category,
            entity=entity,
            section=items[0].section,
            page=items[0].page,
            block_kinds=",".join(sorted({item.kind for item in items})),
            confidence_tier=tier,
        ))

    for block in prepared:
        count = len(tokens(block.text))
        if current and current_count + count > target_tokens:
            emit(current)
            tail: list[Block] = []
            tail_count = 0
            for previous in reversed(current):
                if previous.kind == "table":
                    break
                previous_count = len(tokens(previous.text))
                if tail_count + previous_count > overlap_tokens:
                    break
                tail.insert(0, previous)
                tail_count += previous_count
            current, current_count = tail, tail_count
        current.append(block)
        current_count += count
    emit(current)
    return output


def discover(input_dir: Path) -> Iterable[Path]:
    supported = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}
    return (p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in supported)


def build_chunks(
    input_dir: Path,
    output_dir: Path,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    manifest_path: Path | None = None,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    failures: list[dict[str, str]] = []
    manifest = load_manifest(manifest_path)
    for path in discover(input_dir):
        try:
            blocks, url = extract(path)
            relative = path.relative_to(input_dir).as_posix()
            all_chunks.extend(
                make_chunks(
                    path,
                    blocks,
                    url,
                    input_dir,
                    target_tokens,
                    max_tokens,
                    overlap_tokens,
                    manifest.get(relative),
                )
            )
        except Exception as exc:
            failures.append({"file": path.as_posix(), "error": str(exc)})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    if failures:
        (output_dir / "ingestion_errors.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"Warnings: {len(failures)} files could not be ingested; see ingestion_errors.json")
    return all_chunks


def build_bm25(chunks: Sequence[Chunk], output_dir: Path) -> None:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("BM25 indexing requires rank-bm25") from exc
    tokenized = [re.findall(r"\b\w+\b", c.text.lower()) for c in chunks]
    index = {"ids": [c.id for c in chunks], "texts": [c.text for c in chunks], "metadatas": [asdict(c) for c in chunks], "bm25": BM25Okapi(tokenized)}
    with (output_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(index, handle)


def build_chroma(chunks: Sequence[Chunk], output_dir: Path, embedding_model: str) -> None:
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as exc:
        raise RuntimeError("Chroma indexing requires chromadb and sentence-transformers") from exc
    client = chromadb.PersistentClient(path=str(output_dir / "chroma_db"))
    # Recreate the collection on each ingestion so removed/renamed corpus files
    # cannot leave stale vectors searchable.
    try:
        client.delete_collection("sustainability_corpus")
    except Exception:
        pass
    collection = client.create_collection(
        name="sustainability_corpus",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=embedding_model),
        metadata={"hnsw:space": "cosine"},
    )
    if chunks:
        collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{k: v for k, v in asdict(c).items() if k not in {"id", "text"} and v is not None} for c in chunks],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the local sustainability corpus")
    parser.add_argument("--input", type=Path, default=Path("data/corpus"))
    parser.add_argument("--output", type=Path, default=Path("data/indexes"))
    parser.add_argument("--manifest", type=Path, default=Path("data/source_manifest.json"))
    parser.add_argument("--target-tokens", type=int, default=700)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunks-only", action="store_true", help="Write chunks without index dependencies")
    args = parser.parse_args()
    chunks = build_chunks(
        args.input,
        args.output,
        args.target_tokens,
        args.max_tokens,
        args.overlap_tokens,
        args.manifest,
    )
    print(f"Created {len(chunks)} chunks from {args.input}")
    if not args.chunks_only:
        build_bm25(chunks, args.output)
        build_chroma(chunks, args.output, args.embedding_model)
        print(f"Wrote BM25 and ChromaDB indexes to {args.output}")


if __name__ == "__main__":
    main()

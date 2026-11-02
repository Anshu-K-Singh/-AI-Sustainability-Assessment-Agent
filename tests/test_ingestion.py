from src.ingestion import Block, make_chunks


def test_chunk_ids_do_not_collide_for_same_named_files_in_different_folders(tmp_path) -> None:
    root = tmp_path / "corpus"
    first = root / "brand" / "report.md"
    second = root / "independent" / "report.md"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("brand", encoding="utf-8")
    second.write_text("independent", encoding="utf-8")
    blocks = [Block(kind="paragraph", text="Evidence text.")]

    first_chunk = make_chunks(first, blocks, "https://example.com/a", root, 100, 200, 10)[0]
    second_chunk = make_chunks(second, blocks, "https://example.com/b", root, 100, 200, 10)[0]

    assert first_chunk.id != second_chunk.id
    assert first_chunk.source_file != second_chunk.source_file

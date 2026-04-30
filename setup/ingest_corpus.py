#!/usr/bin/env python3
"""
scripts/ingest_corpus.py

Chunks source documents, embeds them with OpenAI, and saves a FAISS index
to disk so the debate agents have a real retrieval corpus.

Supports two modes:
  --sample     Use the bundled sample corpus (local_data/sample_corpus.json).
               Good for testing and demos. No document prep required.

  --docs PATH  Ingest a folder of .txt or .pdf files for production use.
               Each file is chunked, embedded, and added to the index.

The index is always saved to local_data/faiss_index/ and will overwrite
any existing index at that path.

Usage:
    # Sample corpus (fast, for testing):
    python setup/ingest_corpus.py --sample

    # Real documents:
    python setup/ingest_corpus.py --docs path/to/your/documents/

    # Both (sample + real docs merged into one index):
    python setup/ingest_corpus.py --sample --docs path/to/your/documents/
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without pip install
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from argument_lab.core.faiss_index import ChunkRecord, FaissIndex


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_CORPUS_PATH = Path(__file__).resolve().parent.parent / "local_data" / "sample_corpus.json"
INDEX_OUTPUT_PATH  = Path(__file__).resolve().parent.parent / "local_data" / "faiss_index"

CHUNK_SIZE    = 400   # target characters per chunk
CHUNK_OVERLAP = 80    # character overlap between adjacent chunks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source_id_prefix: str) -> list[ChunkRecord]:
    """
    Splits text into overlapping fixed-size chunks. Each chunk becomes one
    row in the FAISS index.

    For the sample corpus, source documents are already pre-chunked at a
    sentence level, so this function is primarily used for raw .txt/.pdf
    ingestion where documents arrive as continuous text.
    """
    text = text.strip()
    chunks: list[ChunkRecord] = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        excerpt = text[start:end].strip()

        # Don't create a chunk that's just whitespace or too short to be useful
        if len(excerpt) >= 40:
            chunks.append(ChunkRecord(
                source_id=f"{source_id_prefix}_chunk_{chunk_idx:03d}",
                excerpt=excerpt,
                doc_title=source_id_prefix,
            ))
            chunk_idx += 1

        start = end - CHUNK_OVERLAP  # overlap for context continuity

    return chunks


# ---------------------------------------------------------------------------
# Sample corpus loader
# ---------------------------------------------------------------------------

def load_sample_corpus() -> list[ChunkRecord]:
    """
    Loads the bundled sample_corpus.json. Each document entry contains a
    title and a list of pre-chunked excerpts, so we don't re-chunk them —
    they're already at the right granularity.
    """
    if not SAMPLE_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Sample corpus not found at {SAMPLE_CORPUS_PATH}. "
            "Ensure local_data/sample_corpus.json is present in the repo."
        )

    with open(SAMPLE_CORPUS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    records: list[ChunkRecord] = []
    for doc_idx, doc in enumerate(documents):
        title = doc.get("title", f"doc_{doc_idx:03d}")
        # Sanitise title for use as a source_id prefix
        prefix = f"doc_{doc_idx:03d}"
        for chunk_idx, excerpt in enumerate(doc.get("chunks", [])):
            records.append(ChunkRecord(
                source_id=f"{prefix}_chunk_{chunk_idx:03d}",
                excerpt=excerpt.strip(),
                doc_title=title,
            ))

    print(f"[ingest] Sample corpus: {len(documents)} documents → {len(records)} chunks")
    return records


# ---------------------------------------------------------------------------
# Real document loader
# ---------------------------------------------------------------------------

def load_docs_folder(docs_path: Path) -> list[ChunkRecord]:
    """
    Ingests all .txt and .pdf files from a folder. Each file is chunked
    with overlapping windows and turned into ChunkRecords.

    PDF support requires pypdf (`pip install pypdf`). If pypdf is not
    installed, PDF files are skipped with a warning.
    """
    records: list[ChunkRecord] = []
    files = sorted(docs_path.glob("*"))
    supported = {".txt", ".pdf"}

    for file_idx, filepath in enumerate(files):
        suffix = filepath.suffix.lower()
        if suffix not in supported:
            print(f"[ingest] Skipping unsupported file type: {filepath.name}")
            continue

        prefix = f"user_{file_idx:03d}_{filepath.stem[:30]}"
        text = _extract_text(filepath)

        if not text:
            print(f"[ingest] Warning: no text extracted from {filepath.name}")
            continue

        file_chunks = chunk_text(text, source_id_prefix=prefix)
        records.extend(file_chunks)
        print(f"[ingest] {filepath.name}: {len(file_chunks)} chunks")

    print(f"[ingest] User docs: {len(files)} files → {len(records)} chunks")
    return records


def _extract_text(filepath: Path) -> str:
    if filepath.suffix.lower() == ".txt":
        return filepath.read_text(encoding="utf-8", errors="ignore")

    if filepath.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(
                f"[ingest] Warning: pypdf not installed — skipping {filepath.name}. "
                "Install with: pip install pypdf"
            )
            return ""
        reader = PdfReader(str(filepath))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    return ""


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[ChunkRecord]) -> list[ChunkRecord]:
    """
    Removes exact-duplicate excerpts that arise when sample corpus and
    user docs overlap. Keeps the first occurrence.
    """
    seen: set[str] = set()
    unique: list[ChunkRecord] = []
    for record in records:
        key = record.excerpt.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(record)
    removed = len(records) - len(unique)
    if removed:
        print(f"[ingest] Removed {removed} duplicate chunks")
    return unique


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ArgumentLab FAISS retrieval index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python setup/ingest_corpus.py --sample
          python setup/ingest_corpus.py --docs ./my_documents/
          python setup/ingest_corpus.py --sample --docs ./my_documents/
        """),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Include the bundled sample corpus (local_data/sample_corpus.json)",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        metavar="PATH",
        help="Path to a folder of .txt or .pdf files to ingest",
    )
    args = parser.parse_args()

    if not args.sample and not args.docs:
        parser.error("Specify at least one of --sample or --docs PATH")

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "[ingest] Warning: OPENAI_API_KEY is not set. "
            "Embedding calls will fail unless you export it first.\n"
            "  export OPENAI_API_KEY=sk-..."
        )

    # Collect all chunks
    all_records: list[ChunkRecord] = []

    if args.sample:
        all_records.extend(load_sample_corpus())

    if args.docs:
        docs_path = args.docs
        if not docs_path.is_dir():
            print(f"[ingest] Error: '{docs_path}' is not a directory.")
            sys.exit(1)
        all_records.extend(load_docs_folder(docs_path))

    if not all_records:
        print("[ingest] No chunks produced. Check your inputs.")
        sys.exit(1)

    all_records = deduplicate(all_records)
    print(f"[ingest] Total chunks to embed: {len(all_records)}")

    # Build and save the index
    index = FaissIndex.build(all_records)
    index.save(INDEX_OUTPUT_PATH)

    print(f"\n[ingest] Done. Index saved to: {INDEX_OUTPUT_PATH}")
    print(f"[ingest] Run a debate with: python setup/debate.py --proposition \"...\"")


if __name__ == "__main__":
    main()
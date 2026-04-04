"""
Step 1 - Corpus Preprocessing for Legal Inheritance RAG
========================================================
Handles: PDF and TXT files
Outputs:
  - chunks_fixed.json       (Fixed-size chunking)
  - chunks_recursive.json   (Recursive chunking)
  - bm25_corpus.json        (Plain text list for BM25 index)

Usage:
  pip install pypdf langchain-text-splitters tiktoken
  python step1_preprocess.py
"""

import os
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict

# ── pip install pypdf langchain-text-splitters tiktoken ──
from pypdf import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


# ─────────────────────────────────────────────
# CONFIG  (edit these paths before running)
# ─────────────────────────────────────────────
CORPUS_DIR   = "./corpus"       # folder containing your .pdf and .txt files
OUTPUT_DIR   = "./output"       # where chunk JSONs will be saved

# Chunking parameters
FIXED_CHUNK_SIZE    = 512       # tokens  (fixed strategy)
FIXED_OVERLAP       = 50        # tokens
RECURSIVE_CHUNK_SIZE = 600      # chars   (recursive strategy)
RECURSIVE_OVERLAP   = 100       # chars
# ─────────────────────────────────────────────


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    """Extract raw text from a PDF, page by page."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_txt(path: str) -> str:
    """Read a plain-text file (tries UTF-8, falls back to latin-1)."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="latin-1")


def load_corpus(corpus_dir: str) -> List[Dict]:
    """
    Walk corpus_dir, extract text from every .pdf and .txt file.
    Returns a list of dicts:
      { "source": filename, "text": raw_text }
    """
    documents = []
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    files = list(corpus_path.rglob("*.pdf")) + list(corpus_path.rglob("*.txt"))

    if not files:
        raise ValueError(f"No .pdf or .txt files found in {corpus_dir}")

    print(f"Found {len(files)} files in '{corpus_dir}'")

    for fp in sorted(files):
        print(f"  Reading: {fp.name} ...", end=" ")
        try:
            if fp.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(str(fp))
            else:
                text = extract_text_from_txt(str(fp))

            text = clean_text(text)

            if len(text.strip()) < 50:
                print("SKIPPED (too short after cleaning)")
                continue

            documents.append({"source": fp.name, "text": text})
            print(f"OK ({len(text):,} chars)")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nLoaded {len(documents)} documents successfully.\n")
    return documents


# ─────────────────────────────────────────────
# 2.  TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw extracted text:
    - Collapse 3+ newlines into 2
    - Strip page numbers like  "Page 1 of 10"  or  "- 5 -"
    - Remove soft hyphens and zero-width spaces
    - Normalize whitespace
    """
    # Remove zero-width / soft-hyphen chars
    text = text.replace("\u00ad", "").replace("\u200b", "")

    # Strip common page-number patterns
    text = re.sub(r"(?im)^\s*[-–]\s*\d+\s*[-–]\s*$", "", text)        # - 5 -
    text = re.sub(r"(?im)^\s*page\s+\d+\s+(of\s+\d+)?\s*$", "", text) # Page 1 of 10

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize whitespace within lines (keep newlines)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


# ─────────────────────────────────────────────
# 3.  CHUNKING  (two strategies for ablation)
# ─────────────────────────────────────────────

def chunk_id(source: str, index: int, text: str) -> str:
    """Generate a stable unique ID for each chunk."""
    h = hashlib.md5(f"{source}_{index}_{text[:50]}".encode()).hexdigest()[:8]
    return f"{Path(source).stem}_{index:04d}_{h}"


def fixed_size_chunking(documents: List[Dict]) -> List[Dict]:
    """
    Fixed-size chunking by token count using tiktoken.
    Good baseline for the ablation study.
    """
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_OVERLAP,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            if len(split.strip()) < 30:
                continue
            chunks.append({
                "id":       chunk_id(doc["source"], i, split),
                "source":   doc["source"],
                "strategy": "fixed",
                "chunk_index": i,
                "text":     split.strip(),
            })

    return chunks


def recursive_chunking(documents: List[Dict]) -> List[Dict]:
    """
    Recursive character-based chunking.
    Splits on ['\n\n', '\n', ' ', ''] in order — preserves paragraph
    boundaries better, which matters for legal prose.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            if len(split.strip()) < 30:
                continue
            chunks.append({
                "id":       chunk_id(doc["source"], i, split),
                "source":   doc["source"],
                "strategy": "recursive",
                "chunk_index": i,
                "text":     split.strip(),
            })

    return chunks


# ─────────────────────────────────────────────
# 4.  SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_chunks(chunks: List[Dict], filename: str):
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(chunks):,} chunks → {out_path}")


def save_bm25_corpus(chunks: List[Dict], filename: str):
    """
    Save a plain list of chunk texts for BM25 indexing.
    Also saves the matching IDs so results map back to full metadata.
    """
    bm25_data = {
        "ids":   [c["id"]   for c in chunks],
        "texts": [c["text"] for c in chunks],
    }
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved BM25 corpus ({len(bm25_data['texts']):,} entries) → {out_path}")


# ─────────────────────────────────────────────
# 5.  STATS  (for your report)
# ─────────────────────────────────────────────

def print_stats(label: str, chunks: List[Dict]):
    lengths = [len(c["text"]) for c in chunks]
    avg  = sum(lengths) / len(lengths) if lengths else 0
    mn   = min(lengths) if lengths else 0
    mx   = max(lengths) if lengths else 0
    sources = len(set(c["source"] for c in chunks))

    print(f"\n  [{label}]")
    print(f"    Total chunks : {len(chunks):,}")
    print(f"    Source docs  : {sources}")
    print(f"    Avg length   : {avg:.0f} chars")
    print(f"    Min / Max    : {mn} / {mx} chars")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Legal Inheritance RAG — Step 1: Preprocessing")
    print("=" * 55)

    # 1. Load all documents
    documents = load_corpus(CORPUS_DIR)

    # 2. Chunk with both strategies
    print("Chunking with fixed-size strategy...")
    fixed_chunks = fixed_size_chunking(documents)

    print("Chunking with recursive strategy...")
    recursive_chunks = recursive_chunking(documents)

    # 3. Print stats (copy these into your ablation study table)
    print("\n── Chunking Stats ──")
    print_stats("Fixed-size (512 tokens, 50 overlap)", fixed_chunks)
    print_stats("Recursive  (600 chars,  100 overlap)", recursive_chunks)

    # 4. Save outputs
    print("\nSaving outputs...")
    save_chunks(fixed_chunks,     "chunks_fixed.json")
    save_chunks(recursive_chunks, "chunks_recursive.json")

    # BM25 corpus uses the recursive chunks by default
    # (you'll compare both strategies in your ablation study later)
    save_bm25_corpus(recursive_chunks, "bm25_corpus.json")

    print("\nDone! Next step: run step2_embed_upsert.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
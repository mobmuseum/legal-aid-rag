"""
Document Ingestion Pipeline
============================
Extracts text from PDFs and TXT files in the Source directory,
chunks the text using configurable strategies, generates embeddings,
upserts to Pinecone, and builds a BM25 index.
"""

import os
import re
import json
import pickle
import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

import config


# ── Text Extraction ──────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_txt(txt_path: str) -> str:
    """Read a plain text file."""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_all_documents(source_dir: str) -> list[dict]:
    """
    Walk the Source directory and extract text from every PDF and TXT file.
    Returns a list of dicts: {filename, category, text, filepath}
    """
    documents = []
    for root, _, files in os.walk(source_dir):
        category = os.path.basename(root)
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext == ".pdf":
                text = extract_text_from_pdf(fpath)
            elif ext == ".txt":
                text = extract_text_from_txt(fpath)
            else:
                continue

            if text.strip():
                documents.append({
                    "filename": fname,
                    "category": category,
                    "text": text,
                    "filepath": fpath,
                })
                print(f"  ✓ Loaded: {fname} ({len(text):,} chars)")

    print(f"\n📄 Total documents loaded: {len(documents)}")
    return documents


# ── Text Cleaning ────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page numbers like "Page X of Y"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ── Chunking Strategies ─────────────────────────────────

def chunk_fixed_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size chunks with overlap (character-based)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def chunk_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Recursively split text using a hierarchy of separators:
    \\n\\n → \\n → . → (space)
    Tries to keep chunks semantically coherent.
    """
    separators = ["\n\n", "\n", ". ", " "]
    return _recursive_split(text, separators, chunk_size, overlap)


def _recursive_split(
    text: str, separators: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """Internal recursive splitting implementation."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Find the best separator that produces splits
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            break
    else:
        # No separator found, force-split at chunk_size
        return chunk_fixed_size(text, chunk_size, overlap)

    chunks = []
    current_chunk = ""

    for part in parts:
        candidate = current_chunk + sep + part if current_chunk else part

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # If this single part is still too large, recurse
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(sep) + 1 :]
                if remaining_seps:
                    sub_chunks = _recursive_split(
                        part, remaining_seps, chunk_size, overlap
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    chunks.extend(chunk_fixed_size(part, chunk_size, overlap))
                    current_chunk = ""
            else:
                current_chunk = part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Apply overlap by including tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            merged = prev_tail + " " + chunks[i]
            overlapped.append(merged.strip())
        chunks = overlapped

    return chunks


def create_chunks(
    documents: list[dict], strategy: str = "recursive"
) -> list[dict]:
    """
    Chunk all documents using the specified strategy.
    Returns list of dicts: {chunk_id, text, source, category, strategy}
    """
    if strategy == "fixed":
        chunk_size = config.CHUNK_SIZE_FIXED
        overlap = config.CHUNK_OVERLAP_FIXED
        chunk_fn = chunk_fixed_size
    else:
        chunk_size = config.CHUNK_SIZE_RECURSIVE
        overlap = config.CHUNK_OVERLAP_RECURSIVE
        chunk_fn = chunk_recursive

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        cleaned = clean_text(doc["text"])
        pieces = chunk_fn(cleaned, chunk_size, overlap)

        for piece in pieces:
            all_chunks.append({
                "chunk_id": f"{strategy}_{chunk_id}",
                "text": piece,
                "source": doc["filename"],
                "category": doc["category"],
                "strategy": strategy,
            })
            chunk_id += 1

    print(f"📦 Created {len(all_chunks)} chunks using '{strategy}' strategy")
    return all_chunks


# ── Embedding Generation ────────────────────────────────

def generate_embeddings(
    chunks: list[dict], model: SentenceTransformer
) -> list[dict]:
    """Generate embeddings for all chunks and attach them."""
    texts = [c["text"] for c in chunks]
    print("🔢 Generating embeddings...")
    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    return chunks


# ── Pinecone Upsert ─────────────────────────────────────

def upsert_to_pinecone(chunks: list[dict], index_name: str = None):
    """Create/connect to Pinecone index and upsert all chunk vectors."""
    if not config.PINECONE_API_KEY:
        print("⚠️  No PINECONE_API_KEY found. Skipping Pinecone upsert.")
        print("   Set it in .env file and re-run.")
        return

    idx_name = index_name or config.PINECONE_INDEX_NAME
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing = [i.name for i in pc.list_indexes()]
    if idx_name not in existing:
        print(f"🔨 Creating Pinecone index '{idx_name}'...")
        pc.create_index(
            name=idx_name,
            dimension=config.EMBEDDING_DIMENSION,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,
                region=config.PINECONE_REGION,
            ),
        )
        print(f"   ✓ Index '{idx_name}' created")
    else:
        print(f"   Index '{idx_name}' already exists")

    index = pc.Index(idx_name)

    # Upsert in batches of 100
    batch_size = 100
    vectors = []
    for chunk in chunks:
        vectors.append({
            "id": chunk["chunk_id"],
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "source": chunk["source"],
                "category": chunk["category"],
                "strategy": chunk["strategy"],
            },
        })

    print(f"⬆️  Upserting {len(vectors)} vectors to Pinecone...")
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    print(f"   ✓ Upserted {len(vectors)} vectors successfully")


# ── BM25 Index ──────────────────────────────────────────

def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Build a BM25 index from chunk texts."""
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"🔍 Built BM25 index over {len(chunks)} chunks")
    return bm25


def save_artifacts(chunks: list[dict], bm25: BM25Okapi):
    """Save chunks and BM25 index to disk for reuse."""
    with open(config.CHUNKS_PATH, "wb") as f:
        # Save chunks without embeddings (to save disk space)
        chunks_no_emb = [
            {k: v for k, v in c.items() if k != "embedding"} for c in chunks
        ]
        pickle.dump(chunks_no_emb, f)
    print(f"💾 Saved chunks to {config.CHUNKS_PATH}")

    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"💾 Saved BM25 index to {config.BM25_INDEX_PATH}")


def load_artifacts():
    """Load chunks and BM25 index from disk."""
    with open(config.CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    with open(config.BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    return chunks, bm25


# ── Main Ingestion Pipeline ─────────────────────────────

def run_ingestion(strategy: str = "recursive"):
    """Run the full ingestion pipeline."""
    print("=" * 60)
    print("  LEGAL AID RAG — Document Ingestion Pipeline")
    print("=" * 60)
    print(f"\n📂 Source directory: {config.SOURCE_DIR}")
    print(f"📐 Chunking strategy: {strategy}\n")

    # 1. Load documents
    print("─── Step 1: Loading Documents ───")
    documents = load_all_documents(config.SOURCE_DIR)

    # 2. Create chunks
    print("\n─── Step 2: Chunking Documents ───")
    chunks = create_chunks(documents, strategy=strategy)

    # 3. Generate embeddings
    print("\n─── Step 3: Generating Embeddings ───")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    chunks = generate_embeddings(chunks, model)

    # 4. Upsert to Pinecone
    print("\n─── Step 4: Upserting to Pinecone ───")
    upsert_to_pinecone(chunks)

    # 5. Build BM25 index
    print("\n─── Step 5: Building BM25 Index ───")
    bm25 = build_bm25_index(chunks)

    # 6. Save artifacts
    print("\n─── Step 6: Saving Artifacts ───")
    save_artifacts(chunks, bm25)

    print("\n" + "=" * 60)
    print(f"  ✅ Ingestion complete! {len(chunks)} chunks processed.")
    print("=" * 60)

    return chunks, bm25


if __name__ == "__main__":
    import sys

    strategy = sys.argv[1] if len(sys.argv) > 1 else "recursive"
    run_ingestion(strategy)

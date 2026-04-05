"""
Step 2 - Embedding + Pinecone Upsert for Legal Inheritance RAG
==============================================================
Reads chunks from Step 1 output, generates embeddings locally using
sentence-transformers (all-MiniLM-L6-v2), then upserts to Pinecone.

Run ONCE offline — do not re-embed on every query.

Usage:
  pip install sentence-transformers pinecone tqdm
  python embed_upsert.py

Environment variables required (set these before running):
  PINECONE_API_KEY   — from https://app.pinecone.io → API Keys
"""

import os
import json
import time
import math
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "legal-inheritance-rag"   # must be lowercase, no spaces
PINECONE_CLOUD      = "aws"                     # free tier uses aws
PINECONE_REGION     = "us-east-1"               # free tier region

EMBEDDING_MODEL     = "all-MiniLM-L6-v2"        # 384-dim, fast, good quality
EMBEDDING_DIM       = 384

CHUNKS_FILE         = "./output/chunks_recursive.json"  # from Step 1
BATCH_SIZE          = 100   # Pinecone upsert batch limit is 100 vectors
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1.  LOAD CHUNKS
# ─────────────────────────────────────────────

def load_chunks(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks):,} chunks from '{path}'")
    return chunks


# ─────────────────────────────────────────────
# 2.  GENERATE EMBEDDINGS
# ─────────────────────────────────────────────

def embed_chunks(chunks: List[Dict], model_name: str) -> List[List[float]]:
    """
    Generate embeddings for all chunks locally.
    Returns a list of float vectors, one per chunk.
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts):,} chunks (this may take a few minutes)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity becomes dot product
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.tolist()


# ─────────────────────────────────────────────
# 3.  PINECONE SETUP
# ─────────────────────────────────────────────

def init_pinecone_index(api_key: str, index_name: str) -> object:
    """
    Connect to Pinecone and create the index if it doesn't exist yet.
    Uses the free Serverless tier (no pods needed).
    """
    if api_key == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Set your Pinecone API key:\n"
            "  export PINECONE_API_KEY=your_key_here\n"
            "Get it at: https://app.pinecone.io → API Keys"
        )

    print(f"\nConnecting to Pinecone...")
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing:
        print(f"Creating new index: '{index_name}'")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...", end="")
        while not pc.describe_index(index_name).status["ready"]:
            print(".", end="", flush=True)
            time.sleep(2)
        print(" ready!")
    else:
        print(f"Index '{index_name}' already exists — connecting.")

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Index stats: {stats.total_vector_count:,} vectors currently stored")
    return index


# ─────────────────────────────────────────────
# 4.  UPSERT TO PINECONE
# ─────────────────────────────────────────────

def build_pinecone_vectors(chunks: List[Dict], embeddings: List[List[float]]) -> List[Dict]:
    """
    Format chunks + embeddings into Pinecone upsert format.
    Metadata stored per vector:
      - text     : the chunk text (needed for LLM context at query time)
      - source   : original filename
      - strategy : chunking strategy used
      - chunk_index
    """
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        # Truncate text to stay safely under 40KB metadata limit
        text_for_meta = chunk["text"][:3000]
        
        # Pinecone requires ASCII vector IDs; strip non-ascii characters (like smart quotes)
        safe_id = chunk["id"].encode("ascii", "ignore").decode("ascii")

        vectors.append({
            "id": safe_id,
            "values": embedding,
            "metadata": {
                "text":        text_for_meta,
                "source":      chunk["source"],
                "strategy":    chunk.get("strategy", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
            },
        })
    return vectors


def upsert_to_pinecone(index, vectors: List[Dict]):
    """
    Upsert vectors in batches of BATCH_SIZE.
    Pinecone's upsert is idempotent — safe to re-run if interrupted.
    """
    total_batches = math.ceil(len(vectors) / BATCH_SIZE)
    print(f"\nUpserting {len(vectors):,} vectors in {total_batches} batches...")

    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Upserting", unit="batch"):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)

    print("Upsert complete!")

    # Final stats
    time.sleep(2)  # give Pinecone a moment to update stats
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count:,}")


# ─────────────────────────────────────────────
# 5.  SAVE EMBEDDING CACHE (optional but useful)
# ─────────────────────────────────────────────

def save_embedding_cache(chunks: List[Dict], embeddings: List[List[float]]):
    """
    Save embeddings locally alongside the chunks.
    Lets you re-upsert without re-embedding (saves time if you tweak metadata).
    """
    cache_path = "./output/embeddings_cache.json"
    cache = [
        {"id": c["id"], "embedding": e}
        for c, e in zip(chunks, embeddings)
    ]
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"Embedding cache saved → {cache_path}")


def load_embedding_cache(chunks: List[Dict]):
    """
    Load cached embeddings if available, skipping re-computation.
    Returns None if cache doesn't exist or IDs don't match.
    """
    cache_path = "./output/embeddings_cache.json"
    if not Path(cache_path).exists():
        return None

    print("Found embedding cache — loading...")
    with open(cache_path) as f:
        cache = json.load(f)

    cache_ids = {item["id"]: item["embedding"] for item in cache}
    chunk_ids = {c["id"] for c in chunks}

    if cache_ids.keys() != chunk_ids:
        print("Cache IDs don't match current chunks — will re-embed.")
        return None

    # Return embeddings in same order as chunks
    embeddings = [cache_ids[c["id"]] for c in chunks]
    print(f"Loaded {len(embeddings):,} cached embeddings.")
    return embeddings


# ─────────────────────────────────────────────
# QUICK SANITY CHECK (run after upsert)
# ─────────────────────────────────────────────

def sanity_check(index, model_name: str):
    """
    Run a test query against the index to confirm everything works.
    """
    print("\n── Sanity check ──")
    model = SentenceTransformer(model_name)

    test_query = "Who inherits property when there is no will?"
    query_vec = model.encode(test_query, normalize_embeddings=True).tolist()

    results = index.query(
        vector=query_vec,
        top_k=3,
        include_metadata=True,
    )

    print(f"Query: '{test_query}'")
    print(f"Top {len(results['matches'])} results:")
    for i, match in enumerate(results["matches"], 1):
        score  = match["score"]
        source = match["metadata"].get("source", "?")
        snippet = match["metadata"].get("text", "")[:120].replace("\n", " ")
        print(f"  {i}. [{score:.3f}] {source}")
        print(f"     {snippet}...")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Legal Inheritance RAG — Step 2: Embed + Upsert")
    print("=" * 55)

    # 1. Load chunks from Step 1
    chunks = load_chunks(CHUNKS_FILE)

    # 2. Generate (or load cached) embeddings
    embeddings = load_embedding_cache(chunks)
    if embeddings is None:
        embeddings = embed_chunks(chunks, EMBEDDING_MODEL)
        save_embedding_cache(chunks, embeddings)

    # 3. Connect to / create Pinecone index
    index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX_NAME)

    # 4. Build Pinecone vector format and upsert
    vectors = build_pinecone_vectors(chunks, embeddings)
    upsert_to_pinecone(index, vectors)

    # 5. Quick sanity check
    sanity_check(index, EMBEDDING_MODEL)

    print("Done!")
    print("=" * 55)


if __name__ == "__main__":
    main()
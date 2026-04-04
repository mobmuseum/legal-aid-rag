"""
Step 3 - Hybrid Retrieval: BM25 + Semantic + RRF + Cross-Encoder Re-ranking
=============================================================================
This module is imported by your Streamlit app (Step 6).
It can also be run standalone to test retrieval.

Pipeline per query:
  1. BM25 keyword search       → top-20 candidates
  2. Semantic vector search    → top-20 candidates  (Pinecone)
  3. RRF fusion                → merged ranked list
  4. Cross-encoder re-ranking  → final top-k chunks

Usage (standalone test):
  pip install rank-bm25 sentence-transformers pinecone \
              crossencoder nltk tqdm
  python step3_retrieval.py

Import in your app:
  from step3_retrieval import build_retriever, retrieve
"""

import os
import json
import math
import re
from typing import List, Dict, Tuple

import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "legal-inheritance-rag"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"

# Cross-encoder for re-ranking
# ms-marco is trained on QA pairs — well suited for legal retrieval
RERANKER_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_TOP_K          = 20    # candidates from BM25
SEMANTIC_TOP_K      = 20    # candidates from Pinecone
RRF_K               = 60    # RRF constant (standard value — do not change)
FINAL_TOP_K         = 5     # chunks sent to LLM after re-ranking

BM25_CORPUS_FILE    = "./output/bm25_corpus.json"
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1.  BM25 INDEX
# ─────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer: lowercase, remove punctuation, split on whitespace.
    Legal text benefits from keeping compound terms intact, so we
    avoid aggressive stemming here.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # strip punctuation
    tokens = text.split()
    return tokens


def build_bm25_index(corpus_file: str) -> Tuple[BM25Okapi, List[str], List[str]]:
    """
    Load the BM25 corpus saved in Step 1 and build an in-memory index.
    Returns (bm25, ids, texts) — ids are needed to map scores back to chunks.
    """
    print(f"Loading BM25 corpus from '{corpus_file}'...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids   = data["ids"]
    texts = data["texts"]

    print(f"Tokenizing {len(texts):,} documents for BM25...")
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    print(f"BM25 index built — {len(texts):,} documents.")
    return bm25, ids, texts


# ─────────────────────────────────────────────
# 2.  PINECONE + EMBEDDING MODEL
# ─────────────────────────────────────────────

def init_pinecone(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index


def init_models(embedding_model: str, reranker_model: str):
    print(f"Loading embedding model: {embedding_model}")
    embedder = SentenceTransformer(embedding_model)

    print(f"Loading cross-encoder: {reranker_model}")
    reranker = CrossEncoder(reranker_model, max_length=512)

    return embedder, reranker


# ─────────────────────────────────────────────
# 3.  INDIVIDUAL SEARCH FUNCTIONS
# ─────────────────────────────────────────────

def bm25_search(
    query: str,
    bm25: BM25Okapi,
    bm25_ids: List[str],
    bm25_texts: List[str],
    top_k: int = BM25_TOP_K,
) -> List[Dict]:
    """
    BM25 keyword search.
    Returns list of { id, text, bm25_score } dicts, sorted descending.
    """
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Get top_k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:   # skip zero-score results
            results.append({
                "id":         bm25_ids[idx],
                "text":       bm25_texts[idx],
                "bm25_score": float(scores[idx]),
            })

    return results


def semantic_search(
    query: str,
    embedder: SentenceTransformer,
    pinecone_index,
    top_k: int = SEMANTIC_TOP_K,
) -> List[Dict]:
    """
    Semantic vector search via Pinecone.
    Returns list of { id, text, source, semantic_score } dicts.
    """
    query_vec = embedder.encode(query, normalize_embeddings=True).tolist()

    response = pinecone_index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
    )

    results = []
    for match in response["matches"]:
        results.append({
            "id":             match["id"],
            "text":           match["metadata"].get("text", ""),
            "source":         match["metadata"].get("source", "unknown"),
            "chunk_index":    match["metadata"].get("chunk_index", 0),
            "semantic_score": float(match["score"]),
        })

    return results


# ─────────────────────────────────────────────
# 4.  RECIPROCAL RANK FUSION (RRF)
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: List[Dict],
    semantic_results: List[Dict],
    k: int = RRF_K,
) -> List[Dict]:
    """
    Merge BM25 and semantic ranked lists using Reciprocal Rank Fusion.

    Formula:  RRF(d) = Σ  1 / (k + rank(d))
                      lists

    k=60 is the standard constant from the original RRF paper
    (Cormack et al., 2009). It dampens the impact of top ranks,
    making the fusion robust to outliers in either list.

    Returns merged list sorted by RRF score (descending),
    with full metadata from whichever list had the richer entry.
    """
    rrf_scores: Dict[str, float] = {}
    metadata:   Dict[str, Dict]  = {}

    # Score from BM25 ranking
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in metadata:
            metadata[doc_id] = result

    # Score from semantic ranking
    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        # Prefer semantic metadata (has source/chunk_index from Pinecone)
        if doc_id not in metadata or "source" not in metadata[doc_id]:
            metadata[doc_id] = result

    # Sort by fused score
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, rrf_score in fused:
        entry = dict(metadata[doc_id])
        entry["rrf_score"] = round(rrf_score, 6)
        results.append(entry)

    return results


# ─────────────────────────────────────────────
# 5.  CROSS-ENCODER RE-RANKING
# ─────────────────────────────────────────────

def rerank(
    query: str,
    candidates: List[Dict],
    reranker: CrossEncoder,
    top_k: int = FINAL_TOP_K,
) -> List[Dict]:
    """
    Re-rank the RRF-fused candidates using a cross-encoder.

    Unlike bi-encoders (which encode query and doc independently),
    a cross-encoder sees the query and document together — giving it
    much richer interaction signal. Slower, but only runs on the
    fused top candidates (not the full corpus).

    Returns top_k chunks with a 'rerank_score' field added.
    """
    if not candidates:
        return []

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ─────────────────────────────────────────────
# 6.  FULL HYBRID PIPELINE
# ─────────────────────────────────────────────

def retrieve(
    query: str,
    bm25: BM25Okapi,
    bm25_ids: List[str],
    bm25_texts: List[str],
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    pinecone_index,
    top_k: int = FINAL_TOP_K,
    semantic_only: bool = False,   # for ablation study
) -> Dict:
    """
    Full hybrid retrieval pipeline.

    Args:
        query          : user's question
        semantic_only  : if True, skips BM25 and RRF (ablation baseline)
        top_k          : number of final chunks to return

    Returns dict with:
        chunks         : list of final top_k chunks with all scores
        debug          : intermediate results for the UI / report
    """
    # Step A: BM25 search
    if not semantic_only:
        bm25_results = bm25_search(query, bm25, bm25_ids, bm25_texts)
    else:
        bm25_results = []

    # Step B: Semantic search
    sem_results = semantic_search(query, embedder, pinecone_index)

    # Step C: RRF fusion (or just use semantic if ablation)
    if not semantic_only and bm25_results:
        fused = reciprocal_rank_fusion(bm25_results, sem_results)
    else:
        fused = sem_results  # semantic-only baseline

    # Step D: Cross-encoder re-ranking
    final_chunks = rerank(query, fused, reranker, top_k=top_k)

    return {
        "chunks": final_chunks,
        "debug": {
            "bm25_count":     len(bm25_results),
            "semantic_count": len(sem_results),
            "fused_count":    len(fused),
            "final_count":    len(final_chunks),
            "mode":           "semantic_only" if semantic_only else "hybrid",
        }
    }


# ─────────────────────────────────────────────
# 7.  BUILDER FUNCTION (for Streamlit app)
# ─────────────────────────────────────────────

def build_retriever():
    """
    Initialize all retrieval components.
    Call this ONCE at app startup (use st.cache_resource in Streamlit).

    Returns a dict of components to pass into retrieve().
    """
    bm25, bm25_ids, bm25_texts = build_bm25_index(BM25_CORPUS_FILE)
    pinecone_index = init_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME)
    embedder, reranker = init_models(EMBEDDING_MODEL, RERANKER_MODEL)

    return {
        "bm25":          bm25,
        "bm25_ids":      bm25_ids,
        "bm25_texts":    bm25_texts,
        "embedder":      embedder,
        "reranker":      reranker,
        "pinecone_index": pinecone_index,
    }


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

def print_results(result: Dict, query: str):
    debug = result["debug"]
    chunks = result["chunks"]

    print(f"\n{'='*55}")
    print(f"Query : {query}")
    print(f"Mode  : {debug['mode']}")
    print(f"BM25 candidates     : {debug['bm25_count']}")
    print(f"Semantic candidates : {debug['semantic_count']}")
    print(f"After RRF fusion    : {debug['fused_count']}")
    print(f"After re-ranking    : {debug['final_count']}")
    print(f"{'='*55}")

    for i, chunk in enumerate(chunks, 1):
        source      = chunk.get("source", "?")
        rerank_score = chunk.get("rerank_score", 0)
        rrf_score   = chunk.get("rrf_score", 0)
        snippet     = chunk["text"][:200].replace("\n", " ")
        print(f"\n[{i}] source: {source}")
        print(f"    rerank_score: {rerank_score:.4f}  |  rrf_score: {rrf_score:.6f}")
        print(f"    {snippet}...")


def main():
    print("=" * 55)
    print("  Legal Inheritance RAG — Step 3: Retrieval Test")
    print("=" * 55)

    # Build all components
    components = build_retriever()

    test_queries = [
        "Who inherits property when there is no will?",
        "What is the share of a daughter in inheritance under Islamic law?",
        "Can a non-Muslim inherit from a Muslim estate?",
    ]

    for query in test_queries:
        # Hybrid retrieval
        result_hybrid = retrieve(query, **components, semantic_only=False)
        print_results(result_hybrid, f"[HYBRID] {query}")

        # Semantic-only (for ablation comparison)
        result_semantic = retrieve(query, **components, semantic_only=True)
        print_results(result_semantic, f"[SEMANTIC-ONLY] {query}")


if __name__ == "__main__":
    main()
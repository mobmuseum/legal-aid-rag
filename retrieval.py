"""
Retrieval Engine
=================
Implements hybrid search (BM25 + Semantic via Pinecone),
Reciprocal Rank Fusion (RRF), and Cross-Encoder re-ranking.
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

import config


class RetrievalEngine:
    """
    Hybrid retrieval engine combining BM25 keyword search
    with Pinecone semantic search, fused via RRF and re-ranked
    with a Cross-Encoder.
    """

    def __init__(self, use_reranker: bool = True):
        self.use_reranker = use_reranker

        # Load embedding model (lightweight, runs on-app)
        print("🔢 Loading embedding model...")
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        # Load Cross-Encoder for re-ranking
        if self.use_reranker:
            print("🔀 Loading Cross-Encoder re-ranker...")
            self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)

        # Load chunks and BM25 index from disk
        print("📦 Loading chunks and BM25 index...")
        self.chunks, self.bm25 = self._load_artifacts()
        print(f"   ✓ {len(self.chunks)} chunks loaded")

        # Connect to Pinecone
        if config.PINECONE_API_KEY:
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
            print(f"   ✓ Connected to Pinecone index '{config.PINECONE_INDEX_NAME}'")
        else:
            self.pc = None
            self.index = None
            print("   ⚠️  No Pinecone API key — semantic search disabled")

    def _load_artifacts(self):
        """Load chunks and BM25 index from disk."""
        with open(config.CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        with open(config.BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)
        return chunks, bm25

    # ── Semantic Search (Pinecone) ───────────────────────

    def semantic_search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Embed the query and search Pinecone for similar chunks.
        Returns list of {chunk_id, text, source, category, score}.
        """
        top_k = top_k or config.SEMANTIC_TOP_K

        if not self.index:
            return []

        query_embedding = self.embed_model.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        hits = []
        for match in results.get("matches", []):
            hits.append({
                "chunk_id": match["id"],
                "text": match["metadata"].get("text", ""),
                "source": match["metadata"].get("source", ""),
                "category": match["metadata"].get("category", ""),
                "score": match["score"],
            })

        return hits

    # ── BM25 Keyword Search ──────────────────────────────

    def bm25_search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Keyword-based search using BM25 over the chunk corpus.
        Returns list of {chunk_id, text, source, category, score}.
        """
        top_k = top_k or config.BM25_TOP_K

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                hits.append({
                    "chunk_id": self.chunks[idx].get("chunk_id", str(idx)),
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx].get("source", ""),
                    "category": self.chunks[idx].get("category", ""),
                    "score": float(scores[idx]),
                })

        return hits

    # ── Reciprocal Rank Fusion (RRF) ─────────────────────

    def reciprocal_rank_fusion(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict],
        k: int = None,
    ) -> list[dict]:
        """
        Merge semantic and BM25 results using Reciprocal Rank Fusion.
        RRF score = sum( 1 / (k + rank) ) across all result lists.
        """
        k = k or config.RRF_K
        fused_scores = {}
        chunk_data = {}

        # Score semantic results
        for rank, hit in enumerate(semantic_results):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            chunk_data[cid] = hit

        # Score BM25 results
        for rank, hit in enumerate(bm25_results):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = hit

        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        results = []
        for cid in sorted_ids:
            entry = chunk_data[cid].copy()
            entry["rrf_score"] = fused_scores[cid]
            results.append(entry)

        return results

    # ── Cross-Encoder Re-ranking ─────────────────────────

    def rerank(
        self, query: str, candidates: list[dict], top_k: int = None
    ) -> list[dict]:
        """
        Re-rank candidates using a Cross-Encoder model for
        more accurate query-document relevance scoring.
        """
        top_k = top_k or config.RERANK_TOP_K

        if not self.use_reranker or not candidates:
            return candidates[:top_k]

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ── Full Retrieval Pipeline ──────────────────────────

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = None,
    ) -> list[dict]:
        """
        Full retrieval pipeline.

        Modes:
          - "semantic": Pinecone semantic search only
          - "bm25": BM25 keyword search only
          - "hybrid": BM25 + Semantic with RRF + Cross-Encoder re-ranking

        Returns top-k relevant chunks.
        """
        top_k = top_k or config.RERANK_TOP_K

        if mode == "semantic":
            results = self.semantic_search(query)
            if self.use_reranker:
                results = self.rerank(query, results, top_k)
            else:
                results = results[:top_k]

        elif mode == "bm25":
            results = self.bm25_search(query)
            results = results[:top_k]

        elif mode == "hybrid":
            semantic_hits = self.semantic_search(query)
            bm25_hits = self.bm25_search(query)
            fused = self.reciprocal_rank_fusion(semantic_hits, bm25_hits)
            if self.use_reranker:
                results = self.rerank(query, fused, top_k)
            else:
                results = fused[:top_k]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        return results


if __name__ == "__main__":
    # Quick test
    engine = RetrievalEngine()
    query = "What is the inheritance share of a daughter in Sunni law?"
    print(f"\n🔍 Query: {query}\n")

    results = engine.retrieve(query, mode="hybrid")
    for i, r in enumerate(results):
        print(f"  [{i+1}] Score: {r.get('rerank_score', r.get('rrf_score', 'N/A')):.4f}")
        print(f"      Source: {r['source']}")
        print(f"      Text: {r['text'][:150]}...\n")

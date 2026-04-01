"""
Ablation Study
===============
Compares RAG system performance across different configurations:
- Chunking: Fixed-size vs Recursive
- Retrieval: Semantic-only vs Hybrid (BM25 + Semantic + RRF + Re-ranking)
- Metrics: Faithfulness, Relevancy, and latency for each variation
"""

import json
import time
import os
import pickle
import numpy as np
from tqdm import tqdm

import config
from ingest import load_all_documents, create_chunks, generate_embeddings, build_bm25_index, upsert_to_pinecone
from retrieval import RetrievalEngine
from generator import generate_answer
from evaluation import Evaluator
from sentence_transformers import SentenceTransformer


def run_ablation_study(test_queries_path: str = "test_queries.json"):
    """
    Run the ablation study comparing 4 configurations:
    
    | ID | Chunking   | Retrieval            | Re-ranking     |
    |----|------------|----------------------|----------------|
    | A  | Fixed-512  | Semantic Only        | None           |
    | B  | Recursive  | Semantic Only        | None           |
    | C  | Fixed-512  | Hybrid (BM25+Sem)    | Cross-Encoder  |
    | D  | Recursive  | Hybrid (BM25+Sem)    | Cross-Encoder  |
    """
    with open(test_queries_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Use a subset for ablation to save API calls (first 5 queries)
    test_subset = test_data[:5]

    configurations = [
        {"id": "A", "chunking": "fixed",     "retrieval": "semantic", "reranker": False},
        {"id": "B", "chunking": "recursive", "retrieval": "semantic", "reranker": False},
        {"id": "C", "chunking": "fixed",     "retrieval": "hybrid",  "reranker": True},
        {"id": "D", "chunking": "recursive", "retrieval": "hybrid",  "reranker": True},
    ]

    embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    evaluator = Evaluator()
    all_results = []

    print("=" * 70)
    print("  ABLATION STUDY — Comparing RAG Configurations")
    print("=" * 70)

    for cfg in configurations:
        print(f"\n{'─' * 70}")
        print(f"  Configuration {cfg['id']}: {cfg['chunking']} chunking | {cfg['retrieval']} retrieval | reranker={cfg['reranker']}")
        print(f"{'─' * 70}")

        # 1. Load and chunk documents with this strategy
        print(f"\n  Step 1: Chunking with '{cfg['chunking']}' strategy...")
        documents = load_all_documents(config.SOURCE_DIR)
        chunks = create_chunks(documents, strategy=cfg["chunking"])

        # 2. Generate embeddings
        print(f"  Step 2: Generating embeddings...")
        chunks = generate_embeddings(chunks, embed_model)

        # 3. Upsert to Pinecone (use a unique index name per config)
        print(f"  Step 3: Upserting to Pinecone...")
        idx_name = f"{config.PINECONE_INDEX_NAME}"
        upsert_to_pinecone(chunks, index_name=idx_name)

        # 4. Build BM25 index
        from rank_bm25 import BM25Okapi
        bm25 = build_bm25_index(chunks)

        # Save artifacts temporarily
        chunks_no_emb = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
        with open(config.CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks_no_emb, f)
        with open(config.BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25, f)

        # 5. Create retrieval engine with this config
        engine = RetrievalEngine(use_reranker=cfg["reranker"])

        # 6. Evaluate on test queries
        print(f"\n  Step 4: Evaluating on {len(test_subset)} test queries...")
        config_results = {
            "config_id": cfg["id"],
            "chunking": cfg["chunking"],
            "retrieval": cfg["retrieval"],
            "reranker": cfg["reranker"],
            "num_chunks": len(chunks),
            "queries": [],
        }

        for qi, item in enumerate(test_subset):
            query = item["query"]
            print(f"    [{qi+1}/{len(test_subset)}] {query[:60]}...")

            # Retrieve
            t0 = time.time()
            retrieved = engine.retrieve(query, mode=cfg["retrieval"])
            retrieval_time = time.time() - t0

            # Generate
            t0 = time.time()
            answer = generate_answer(query, retrieved)
            generation_time = time.time() - t0

            # Evaluate
            t0 = time.time()
            eval_result = evaluator.evaluate(query, answer, retrieved)
            eval_time = time.time() - t0

            config_results["queries"].append({
                "query": query,
                "answer": answer[:200] + "...",
                "faithfulness": eval_result["faithfulness"]["score"],
                "relevancy": eval_result["relevancy"]["score"],
                "retrieval_time_s": round(retrieval_time, 3),
                "generation_time_s": round(generation_time, 3),
                "total_time_s": round(retrieval_time + generation_time + eval_time, 3),
            })

            time.sleep(1)  # Rate limit buffer

        # Aggregate
        faith_scores = [q["faithfulness"] for q in config_results["queries"]]
        rel_scores = [q["relevancy"] for q in config_results["queries"]]
        ret_times = [q["retrieval_time_s"] for q in config_results["queries"]]
        gen_times = [q["generation_time_s"] for q in config_results["queries"]]
        total_times = [q["total_time_s"] for q in config_results["queries"]]

        config_results["avg_faithfulness"] = round(float(np.mean(faith_scores)), 4)
        config_results["avg_relevancy"] = round(float(np.mean(rel_scores)), 4)
        config_results["avg_retrieval_time_s"] = round(float(np.mean(ret_times)), 3)
        config_results["avg_generation_time_s"] = round(float(np.mean(gen_times)), 3)
        config_results["avg_total_time_s"] = round(float(np.mean(total_times)), 3)

        all_results.append(config_results)

        print(f"\n  ✅ Config {cfg['id']} — Faithfulness: {config_results['avg_faithfulness']:.2%} | Relevancy: {config_results['avg_relevancy']:.4f}")

    # Final comparison table
    print("\n" + "=" * 70)
    print("  ABLATION STUDY — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<8} {'Chunking':<12} {'Retrieval':<18} {'Reranker':<10} {'Faithfulness':<14} {'Relevancy':<12} {'Avg Time':<10}")
    print("  " + "─" * 84)

    for r in all_results:
        print(
            f"  {r['config_id']:<8} {r['chunking']:<12} {r['retrieval']:<18} "
            f"{'Yes' if r['reranker'] else 'No':<10} "
            f"{r['avg_faithfulness']:.2%}{'':<8} "
            f"{r['avg_relevancy']:.4f}{'':<6} "
            f"{r['avg_total_time_s']:.2f}s"
        )

    # Save results
    output_path = "ablation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  📊 Full results saved to: {output_path}")
    return all_results


if __name__ == "__main__":
    run_ablation_study()

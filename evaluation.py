"""
Evaluation Pipeline — LLM-as-a-Judge
======================================
Implements Faithfulness (claim extraction + verification)
and Relevancy (alternate query generation + cosine similarity) scoring.
"""

import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from generator import extract_claims, verify_claim, generate_questions


class Evaluator:
    """
    Evaluates RAG system outputs using LLM-as-a-Judge methodology.
    
    Faithfulness: What fraction of claims in the answer are supported by context?
    Relevancy: How well does the answer address the original query?
    """

    def __init__(self):
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def compute_faithfulness(
        self, answer: str, context_chunks: list[dict], verbose: bool = False
    ) -> dict:
        """
        Compute Faithfulness Score using claim extraction & verification.
        
        Steps:
        1. Extract factual claims from the generated answer.
        2. Verify each claim against the retrieved context.
        3. Score = (number of supported claims) / (total claims).
        
        Returns dict with score, claims, and verification details.
        """
        # 1. Extract claims
        claims = extract_claims(answer)

        if not claims:
            return {
                "score": 0.0,
                "num_claims": 0,
                "num_supported": 0,
                "claims": [],
                "details": "No claims could be extracted from the answer.",
            }

        # 2. Build combined context
        combined_context = "\n\n".join(
            chunk.get("text", "") for chunk in context_chunks
        )

        # 3. Verify each claim
        results = []
        supported_count = 0

        for claim in claims:
            is_supported = verify_claim(claim, combined_context)
            if is_supported:
                supported_count += 1
            results.append({
                "claim": claim,
                "supported": is_supported,
            })

            if verbose:
                status = "✅" if is_supported else "❌"
                print(f"  {status} {claim}")

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        score = supported_count / len(claims) if claims else 0.0

        return {
            "score": round(score, 4),
            "num_claims": len(claims),
            "num_supported": supported_count,
            "claims": results,
        }

    def compute_relevancy(
        self, query: str, answer: str, verbose: bool = False
    ) -> dict:
        """
        Compute Relevancy Score using alternate query generation.
        
        Steps:
        1. Generate N questions from the generated answer.
        2. Encode original query + generated questions.
        3. Compute cosine similarity between each generated question and the original.
        4. Score = mean similarity.
        
        Returns dict with score, generated questions, and individual similarities.
        """
        # 1. Generate alternate questions
        generated_questions = generate_questions(answer)

        if not generated_questions:
            return {
                "score": 0.0,
                "generated_questions": [],
                "similarities": [],
                "details": "No questions could be generated from the answer.",
            }

        # 2. Encode all queries
        all_texts = [query] + generated_questions
        embeddings = self.embed_model.encode(all_texts, normalize_embeddings=True)

        query_emb = embeddings[0]
        question_embs = embeddings[1:]

        # 3. Compute cosine similarities
        similarities = []
        for i, q_emb in enumerate(question_embs):
            sim = float(np.dot(query_emb, q_emb))
            similarities.append(round(sim, 4))

            if verbose:
                print(f"  📐 Q{i+1}: {generated_questions[i]}")
                print(f"      Similarity: {sim:.4f}")

        # 4. Mean similarity
        avg_score = float(np.mean(similarities))

        return {
            "score": round(avg_score, 4),
            "generated_questions": generated_questions,
            "similarities": similarities,
        }

    def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: list[dict],
        verbose: bool = False,
    ) -> dict:
        """
        Run full evaluation (faithfulness + relevancy) for a single Q/A pair.
        """
        if verbose:
            print("\n─── Faithfulness Evaluation ───")

        faithfulness = self.compute_faithfulness(answer, context_chunks, verbose)

        if verbose:
            print(f"\n  Faithfulness Score: {faithfulness['score']:.2%}")
            print(f"  ({faithfulness['num_supported']}/{faithfulness['num_claims']} claims supported)")
            print("\n─── Relevancy Evaluation ───")

        relevancy = self.compute_relevancy(query, answer, verbose)

        if verbose:
            print(f"\n  Relevancy Score: {relevancy['score']:.4f}")

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
        }


def run_evaluation_on_test_set(retrieval_engine, test_queries_path: str = "test_queries.json"):
    """
    Run evaluation on the full test query set and produce a summary report.
    """
    from generator import generate_answer

    with open(test_queries_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    evaluator = Evaluator()
    results = []

    print("=" * 60)
    print("  RAG SYSTEM EVALUATION — LLM-as-a-Judge")
    print("=" * 60)

    for i, item in enumerate(test_data):
        query = item["query"]
        print(f"\n[{i+1}/{len(test_data)}] {query}")

        # Retrieve
        t_start = time.time()
        chunks = retrieval_engine.retrieve(query, mode="hybrid")
        retrieval_time = time.time() - t_start

        # Generate
        t_start = time.time()
        answer = generate_answer(query, chunks)
        generation_time = time.time() - t_start

        # Evaluate
        t_start = time.time()
        eval_result = evaluator.evaluate(query, answer, chunks, verbose=True)
        eval_time = time.time() - t_start

        results.append({
            "query": query,
            "answer": answer,
            "chunks_used": len(chunks),
            "faithfulness": eval_result["faithfulness"],
            "relevancy": eval_result["relevancy"],
            "timing": {
                "retrieval_s": round(retrieval_time, 3),
                "generation_s": round(generation_time, 3),
                "evaluation_s": round(eval_time, 3),
                "total_s": round(retrieval_time + generation_time + eval_time, 3),
            },
        })

    # Summary
    faith_scores = [r["faithfulness"]["score"] for r in results]
    rel_scores = [r["relevancy"]["score"] for r in results]
    retrieval_times = [r["timing"]["retrieval_s"] for r in results]
    generation_times = [r["timing"]["generation_s"] for r in results]

    summary = {
        "num_queries": len(results),
        "avg_faithfulness": round(float(np.mean(faith_scores)), 4),
        "avg_relevancy": round(float(np.mean(rel_scores)), 4),
        "avg_retrieval_time_s": round(float(np.mean(retrieval_times)), 3),
        "avg_generation_time_s": round(float(np.mean(generation_times)), 3),
        "results": results,
    }

    # Save results
    output_path = "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Queries evaluated:      {summary['num_queries']}")
    print(f"  Avg Faithfulness:       {summary['avg_faithfulness']:.2%}")
    print(f"  Avg Relevancy:          {summary['avg_relevancy']:.4f}")
    print(f"  Avg Retrieval Time:     {summary['avg_retrieval_time_s']:.3f}s")
    print(f"  Avg Generation Time:    {summary['avg_generation_time_s']:.3f}s")
    print(f"  Results saved to:       {output_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    from retrieval import RetrievalEngine
    engine = RetrievalEngine()
    run_evaluation_on_test_set(engine)

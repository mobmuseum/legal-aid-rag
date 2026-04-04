"""
Step 5 - LLM-as-a-Judge Evaluation (Faithfulness + Relevancy)
==============================================================
Implements the two evaluation metrics required by the assignment:

  Faithfulness  : Claim extraction → verify each claim against context
                  Score = verified_claims / total_claims

  Relevancy     : Generate 3 questions from answer → cosine similarity
                  with original query
                  Score = mean similarity of 3 generated questions

Also runs the full ablation study across:
  - Chunking strategies  (fixed vs recursive)
  - Retrieval modes      (semantic-only vs hybrid + re-ranking)

Outputs:
  ./output/evaluation_results.json   — per-query scores
  ./output/ablation_table.json       — ablation study summary

Usage:
  pip install requests sentence-transformers numpy
  python step5_evaluation.py
"""

import os
import json
import time
import re
import numpy as np
import requests
import json
from typing import List, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HF_API_TOKEN        = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
if HF_API_TOKEN:
    os.environ["HF_TOKEN"] = HF_API_TOKEN

GROQ_API_KEY        = os.getenv("GROQ_API_KEY")

# Groq model for ultra-fast, unmetered evaluation
JUDGE_MODEL         = "llama-3.1-8b-instant"


SIMILARITY_MODEL    = "all-MiniLM-L6-v2"   # same model used in retrieval

OUTPUT_DIR          = "./output"

# Test set — 15 queries covering key inheritance law topics
# Replace / extend with queries relevant to YOUR corpus
TEST_QUERIES = [
    "Who inherits property when a person dies without a will?",
    "What is the share of a daughter in Islamic inheritance?",
    "Can a non-Muslim inherit from a Muslim's estate?",
    "How is the estate divided when there are multiple heirs?",
    "What rights does a spouse have over inherited property?",
    "Can a person be disinherited under Pakistani inheritance law?",
    "What is the difference between a will and intestate succession?",
    "How are debts handled before distributing an inheritance?",
    "What happens to property if all direct heirs are deceased?",
    "Can a minor child be an heir?",
    "What is the role of an executor in inheritance proceedings?",
    "How is agricultural land treated differently in inheritance?",
    "What documents are required to claim inheritance?",
    "Can a gift given during lifetime affect inheritance share?",
    "What is the waiting period before inheritance can be claimed?",
]
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1.  LLM JUDGE HELPER
# ─────────────────────────────────────────────

def call_judge(prompt: str, max_tokens: int = 512, max_retries: int = 4) -> str:
    """
    Call the LLM judge with a prompt. Returns the raw generated text.
    Uses Groq API for blazing-fast inference using Llama 3.
    """
    if not GROQ_API_KEY:
        print("    Judge error: GROQ_API_KEY not set in .env")
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if text:
                return text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                wait_time = 10 * (attempt + 1)
                print(f"    [Groq Rate Limit] Sleeping for {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                print(f"    Judge error on Groq: {e}")
                if hasattr(e, "response") and hasattr(e.response, "text"):
                     print(f"    Groq Details: {e.response.text}")
                break

    return ""


# ─────────────────────────────────────────────
# 2.  FAITHFULNESS EVALUATION
# ─────────────────────────────────────────────

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the answer below.
A claim is a single, verifiable statement of fact.
Return ONLY a numbered list of claims, one per line.
Do not include opinions, hedges, or meta-statements.

Answer:
{answer}

Numbered list of claims:"""


CLAIM_VERIFICATION_PROMPT = """You are a strict legal fact-checker.
Determine whether the following claim is fully supported by the provided context.

Context:
{context}

Claim: {claim}

Reply with ONLY one word: SUPPORTED or NOT_SUPPORTED"""


def extract_claims(answer: str) -> List[str]:
    """Ask the LLM to extract atomic factual claims from the answer."""
    prompt   = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    response = call_judge(prompt, max_tokens=300)
    
    print(f"\n  [DEBUG] Judge Model (Claim Extraction) Response:\n{response}\n")

    claims = []
    for line in response.splitlines():
        line = line.strip()
        # Match numbered list items or bullet points (e.g. "1.", "- ", "* ")
        match = re.match(r"^(?:\d+[.)]|-|\*)\s+(.+)$", line)
        if match:
            claim = match.group(1).strip()
            if len(claim) > 10:   # skip trivially short extractions
                claims.append(claim)

    return claims


def verify_claim(claim: str, context: str) -> bool:
    """Ask the LLM judge whether a single claim is supported by the context."""
    prompt   = CLAIM_VERIFICATION_PROMPT.format(context=context[:3000], claim=claim)
    response = call_judge(prompt, max_tokens=10)
    return "SUPPORTED" in response.upper() and "NOT_SUPPORTED" not in response.upper()


def evaluate_faithfulness(
    answer: str,
    context: str,
    verbose: bool = False,
) -> Dict:
    """
    Full faithfulness pipeline:
      1. Extract claims from answer
      2. Verify each claim against context
      3. Score = verified / total

    Returns:
      score           : float 0.0 – 1.0
      claims          : list of extracted claims
      verifications   : list of (claim, supported) pairs
      total_claims    : int
      supported_claims: int
    """
    if not answer or not context:
        return _empty_faithfulness()

    print("    Extracting claims...", end=" ", flush=True)
    claims = extract_claims(answer)
    print(f"{len(claims)} claims found.")

    if not claims:
        return _empty_faithfulness()

    verifications = []
    supported     = 0

    for i, claim in enumerate(claims):
        if verbose:
            print(f"      Verifying [{i+1}/{len(claims)}]: {claim[:60]}...", end=" ")
        is_supported = verify_claim(claim, context)
        verifications.append({"claim": claim, "supported": is_supported})
        if is_supported:
            supported += 1
        if verbose:
            print("✓" if is_supported else "✗")
        time.sleep(0.3)   # small delay to avoid rate limiting

    score = supported / len(claims) if claims else 0.0

    return {
        "score":             round(score, 4),
        "total_claims":      len(claims),
        "supported_claims":  supported,
        "claims":            claims,
        "verifications":     verifications,
    }


def _empty_faithfulness() -> Dict:
    return {
        "score": 0.0, "total_claims": 0, "supported_claims": 0,
        "claims": [], "verifications": [],
    }


# ─────────────────────────────────────────────
# 3.  RELEVANCY EVALUATION
# ─────────────────────────────────────────────

QUESTION_GEN_PROMPT = """Given the answer below, generate exactly 3 questions that this answer could be responding to.
Each question should be a standalone question that the answer addresses.
Return ONLY a numbered list of 3 questions, one per line.

Answer:
{answer}

3 questions this answer addresses:"""


def generate_questions_from_answer(answer: str) -> List[str]:
    """Ask the LLM to generate 3 questions that the answer responds to."""
    prompt   = QUESTION_GEN_PROMPT.format(answer=answer)
    response = call_judge(prompt, max_tokens=200)

    print(f"\n  [DEBUG] Judge Model (Question Generation) Response:\n{response}\n")

    questions = []
    for line in response.splitlines():
        line  = line.strip()
        # Match numbered list or bullet points ending in a question mark
        match = re.match(r"^(?:\d+[.)]|-|\*)\s+(.+\?)\s*$", line)
        if match:
            q = match.group(1).strip()
            if len(q) > 10:
                questions.append(q)

    # Fallback: any line ending in "?" if numbered parse failed
    if not questions:
        questions = [
            line.strip() for line in response.splitlines()
            if line.strip().endswith("?") and len(line.strip()) > 10
        ]

    return questions[:3]   # ensure max 3


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def evaluate_relevancy(
    original_query: str,
    answer: str,
    similarity_model: SentenceTransformer,
) -> Dict:
    """
    Full relevancy pipeline:
      1. Generate 3 questions from the answer
      2. Compute cosine similarity of each vs. the original query
      3. Score = mean of 3 similarities

    Returns:
      score              : float 0.0 – 1.0
      generated_questions: list of 3 generated questions
      similarities       : per-question similarity scores
    """
    if not answer:
        return _empty_relevancy()

    print("    Generating questions from answer...", end=" ", flush=True)
    questions = generate_questions_from_answer(answer)
    print(f"{len(questions)} questions generated.")

    if not questions:
        return _empty_relevancy()

    # Encode original query and generated questions
    all_texts    = [original_query] + questions
    embeddings   = similarity_model.encode(all_texts, normalize_embeddings=True)
    query_vec    = embeddings[0]
    question_vecs = embeddings[1:]

    similarities = []
    for q_vec in question_vecs:
        sim = cosine_similarity(query_vec, q_vec)
        similarities.append(round(float(sim), 4))

    score = float(np.mean(similarities)) if similarities else 0.0

    return {
        "score":               round(score, 4),
        "generated_questions": questions,
        "similarities":        similarities,
    }


def _empty_relevancy() -> Dict:
    return {"score": 0.0, "generated_questions": [], "similarities": []}


# ─────────────────────────────────────────────
# 4.  FULL EVALUATION RUNNER
# ─────────────────────────────────────────────

def run_evaluation(
    test_queries: List[str],
    retriever_components: Dict,
    similarity_model: SentenceTransformer,
    semantic_only: bool = False,
    chunks_strategy: str = "recursive",
    verbose: bool = True,
) -> List[Dict]:
    """
    Run the full evaluation pipeline on all test queries.

    Args:
        semantic_only    : if True, uses semantic-only retrieval (ablation)
        chunks_strategy  : label for the chunking strategy used
        verbose          : print per-claim verification

    Returns list of per-query result dicts.
    """
    from retrieval import retrieve
    from generation import generate_answer

    results = []
    mode    = "semantic_only" if semantic_only else "hybrid"

    print(f"\nRunning evaluation — mode: {mode}, chunks: {chunks_strategy}")
    print(f"Queries: {len(test_queries)}")
    print("-" * 55)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {query[:70]}")

        t_start = time.time()

        # 1. Retrieve
        print("  Retrieving...", end=" ", flush=True)
        t0 = time.time()
        retrieval = retrieve(query, **retriever_components, semantic_only=semantic_only)
        retrieval_ms = round((time.time() - t0) * 1000)
        print(f"{retrieval_ms} ms — {len(retrieval['chunks'])} chunks")

        chunks  = retrieval["chunks"]
        context = "\n\n".join(c["text"] for c in chunks)

        # 2. Generate
        print("  Generating answer...", end=" ", flush=True)
        t0 = time.time()
        gen = generate_answer(query, chunks)
        generation_ms = round((time.time() - t0) * 1000)
        print(f"{generation_ms} ms")

        answer = gen["answer"]
        print(f"\n  [DEBUG] Generator Model Answer:\n{answer}\n")

        if gen["error"]:
            print(f"  Generation failed: {gen['error']}")
            results.append({
                "query": query, "error": gen["error"],
                "faithfulness": _empty_faithfulness(),
                "relevancy":    _empty_relevancy(),
            })
            continue

        # 3. Faithfulness
        print("  Evaluating faithfulness...")
        faith = evaluate_faithfulness(answer, context, verbose=verbose)
        print(f"  Faithfulness: {faith['score']:.2%} "
              f"({faith['supported_claims']}/{faith['total_claims']} claims supported)")

        # 4. Relevancy
        print("  Evaluating relevancy...")
        rel = evaluate_relevancy(query, answer, similarity_model)
        print(f"  Relevancy: {rel['score']:.4f} "
              f"(similarities: {rel['similarities']})")

        total_ms = round((time.time() - t_start) * 1000)

        results.append({
            "query":           query,
            "answer":          answer,
            "model_used":      gen["model_used"],
            "context_chunks":  len(chunks),
            "faithfulness":    faith,
            "relevancy":       rel,
            "timings": {
                "retrieval_ms":  retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms":      total_ms,
            },
            "config": {
                "mode":             mode,
                "chunks_strategy":  chunks_strategy,
            },
            "error": None,
        })

        # Small pause between queries to stay within HF rate limits
        time.sleep(1)

    return results


# ─────────────────────────────────────────────
# 5.  ABLATION STUDY
# ─────────────────────────────────────────────

def run_ablation_study(
    test_queries: List[str],
    retriever_components_recursive: Dict,
    retriever_components_fixed: Dict,
    similarity_model: SentenceTransformer,
) -> Dict:
    """
    Run all four combinations for the ablation study table:

      A: Recursive chunks  + Semantic-only retrieval
      B: Recursive chunks  + Hybrid + Re-ranking
      C: Fixed chunks      + Semantic-only retrieval
      D: Fixed chunks      + Hybrid + Re-ranking   ← expected best

    Returns a summary dict with mean scores per configuration.
    """
    configurations = [
        ("A", retriever_components_recursive, True,  "recursive"),
        ("B", retriever_components_recursive, False, "recursive"),
        ("C", retriever_components_fixed,     True,  "fixed"),
        ("D", retriever_components_fixed,     False, "fixed"),
    ]

    ablation_results = {}

    for label, components, sem_only, strategy in configurations:
        config_name = f"{label}: {strategy} + {'semantic_only' if sem_only else 'hybrid+rerank'}"
        print(f"\n{'='*55}")
        print(f"  Ablation config {config_name}")
        print(f"{'='*55}")

        results = run_evaluation(
            test_queries[:5],   # use first 5 queries for speed
            components,
            similarity_model,
            semantic_only=sem_only,
            chunks_strategy=strategy,
            verbose=False,
        )

        ablation_results[label] = summarize_results(results, config_name)

    return ablation_results


def summarize_results(results: List[Dict], label: str = "") -> Dict:
    """Compute mean faithfulness and relevancy from a list of query results."""
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {"label": label, "faithfulness": 0.0, "relevancy": 0.0, "n": 0}

    faith_scores = [r["faithfulness"]["score"] for r in valid]
    rel_scores   = [r["relevancy"]["score"]    for r in valid]
    ret_times    = [r["timings"]["retrieval_ms"]  for r in valid]
    gen_times    = [r["timings"]["generation_ms"] for r in valid]

    return {
        "label":             label,
        "n_queries":         len(valid),
        "faithfulness_mean": round(float(np.mean(faith_scores)), 4),
        "faithfulness_std":  round(float(np.std(faith_scores)),  4),
        "relevancy_mean":    round(float(np.mean(rel_scores)),   4),
        "relevancy_std":     round(float(np.std(rel_scores)),    4),
        "avg_retrieval_ms":  round(float(np.mean(ret_times)),    1),
        "avg_generation_ms": round(float(np.mean(gen_times)),    1),
    }


# ─────────────────────────────────────────────
# 6.  REPORT HELPERS
# ─────────────────────────────────────────────

def print_ablation_table(ablation: Dict):
    """Pretty-print the ablation study table for your report."""
    print("\n" + "=" * 75)
    print("  ABLATION STUDY RESULTS")
    print("=" * 75)
    header = f"{'Config':<45} {'Faith':>7} {'Relev':>7} {'Ret ms':>8} {'Gen ms':>8}"
    print(header)
    print("-" * 75)
    for key in sorted(ablation.keys()):
        r = ablation[key]
        row = (
            f"{r['label']:<45} "
            f"{r['faithfulness_mean']:>6.2%} "
            f"{r['relevancy_mean']:>7.4f} "
            f"{r['avg_retrieval_ms']:>8.0f} "
            f"{r['avg_generation_ms']:>8.0f}"
        )
        print(row)
    print("=" * 75)


def print_example_faithfulness(results: List[Dict], n: int = 3):
    """
    Print claim extraction + verification for 14bn example queries.
    Required by the assignment: show at least 3 example queries.
    """
    print("\n" + "=" * 55)
    print("  FAITHFULNESS EXAMPLES (for your report)")
    print("=" * 55)

    shown = 0
    for r in results:
        if r.get("error") or shown >= n:
            continue
        faith = r["faithfulness"]
        if not faith["claims"]:
            continue

        print(f"\nQuery: {r['query']}")
        print(f"Answer snippet: {r['answer'][:200]}...")
        print(f"Faithfulness score: {faith['score']:.2%} "
              f"({faith['supported_claims']}/{faith['total_claims']})")
        print("Claims:")
        for v in faith["verifications"]:
            mark = "✓" if v["supported"] else "✗"
            print(f"  {mark} {v['claim']}")
        shown += 1


def save_results(results: List[Dict], ablation: Dict):
    """Save all evaluation results to JSON for your report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    eval_path    = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    ablation_path = os.path.join(OUTPUT_DIR, "ablation_table.json")

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation results saved → {eval_path}")

    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation, f, ensure_ascii=False, indent=2)
    print(f"Ablation table saved    → {ablation_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Legal Inheritance RAG — Step 5: Evaluation")
    print("=" * 55)

    if HF_API_TOKEN == "YOUR_HF_TOKEN_HERE":
        print("ERROR: Set HF_API_TOKEN environment variable first.")
        print("  export HF_API_TOKEN=hf_...")
        return

    # Load similarity model (used for relevancy scoring)
    print(f"\nLoading similarity model: {SIMILARITY_MODEL}")
    sim_model = SentenceTransformer(SIMILARITY_MODEL)

    # Build retriever components for both chunking strategies
    # Recursive chunks (default — used for main evaluation)
    from retrieval import build_retriever
    print("\nBuilding retriever (recursive chunks)...")
    components_recursive = build_retriever()   # uses chunks_recursive.json by default

    # For ablation on fixed chunks, point BM25_CORPUS_FILE to chunks_fixed.json
    # The simplest way: temporarily patch the config in step3_retrieval
    import retrieval as s3
    original_bm25_file = s3.BM25_CORPUS_FILE
    s3.BM25_CORPUS_FILE = "./output/bm25_corpus_fixed.json"
    print("Building retriever (fixed chunks)...")
    # NOTE: You need to save a separate BM25 corpus for fixed chunks.
    # In step1_preprocess.py, call save_bm25_corpus(fixed_chunks, "bm25_corpus_fixed.json")
    try:
        components_fixed = build_retriever()
    except FileNotFoundError:
        print("  bm25_corpus_fixed.json not found — using recursive for both.")
        components_fixed = components_recursive
    finally:
        s3.BM25_CORPUS_FILE = original_bm25_file   # restore

    # ── Main evaluation (hybrid + recursive) ──
    print("\nRunning main evaluation on full test set...")
    main_results = run_evaluation(
        TEST_QUERIES,
        components_recursive,
        sim_model,
        semantic_only=False,
        chunks_strategy="recursive",
        verbose=True,
    )

    # Print summary
    summary = summarize_results(main_results, "Hybrid + Recursive (Best)")
    print(f"\nOverall Faithfulness : {summary['faithfulness_mean']:.2%} "
          f"± {summary['faithfulness_std']:.2%}")
    print(f"Overall Relevancy    : {summary['relevancy_mean']:.4f} "
          f"± {summary['relevancy_std']:.4f}")
    print(f"Avg Retrieval time   : {summary['avg_retrieval_ms']:.0f} ms")
    print(f"Avg Generation time  : {summary['avg_generation_ms']:.0f} ms")

    # Show example faithfulness breakdowns (required by assignment)
    print_example_faithfulness(main_results, n=3)

    # ── Ablation study ──
    print("\n\nRunning ablation study (4 configurations × 5 queries)...")
    ablation = run_ablation_study(
        TEST_QUERIES,
        components_recursive,
        components_fixed,
        sim_model,
    )
    # Add main result to ablation table
    ablation["BEST"] = summary

    print_ablation_table(ablation)

    # Save everything
    save_results(main_results, ablation)

    print("\nDone! Next step: build the Streamlit app (step6_app.py)")
    print("=" * 55)


if __name__ == "__main__":
    main()
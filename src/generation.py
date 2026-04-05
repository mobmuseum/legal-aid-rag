"""
Step 4 - LLM Generation via HuggingFace Inference API
======================================================
Takes retrieved chunks from Step 3 and generates a grounded answer
using a hosted LLM (Mistral-7B-Instruct or Llama-3-8B).

Also handles:
  - Prompt construction  (system + context + query)
  - Timing measurement   (for latency reporting in your report)
  - Graceful fallback    (if primary model is overloaded, try backup)

Usage (standalone test):
  pip install requests
  python generation.py

Import in your app:
  from generation import generate_answer
"""

import os
import time
from typing import List, Dict, Optional

from huggingface_hub import InferenceClient

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Primary model — Strong instruction following
PRIMARY_MODEL   = "Qwen/Qwen2.5-72B-Instruct"

# Fallback model — smaller, almost always available on free tier
FALLBACK_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.2     # low = more factual, less creative (good for legal)
TOP_P           = 0.9
CONTEXT_CHUNKS  = 5       # how many retrieved chunks to include in prompt
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1.  PROMPT CONSTRUCTION
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable legal assistant specializing in inheritance law. \
Your role is to answer questions based strictly on the provided legal context. \

Rules you must follow:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the context does not contain enough information, say: \
  "The provided documents do not contain sufficient information to answer this question."
- Cite the source document name when referencing specific rules or sections.
- Be precise and use correct legal terminology.
- Do not give personal legal advice — remind users to consult a qualified lawyer for their specific situation."""


def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    Each chunk is prefixed with its source document so the LLM
    can cite it and the judge can verify faithfulness.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source  = chunk.get("source", "Unknown source")
        text    = chunk.get("text", "").strip()
        context_parts.append(f"[Context {i} — Source: {source}]\n{text}")

    return "\n\n".join(context_parts)


def build_messages(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Build the full prompt in standard message format used by HuggingFace Chat Completion API.
    """
    context = format_context(chunks[:CONTEXT_CHUNKS])

    user_message = f"""Using the legal context provided below, answer the following question.

--- LEGAL CONTEXT ---
{context}
--- END CONTEXT ---

Question: {query}

Answer:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]



# ─────────────────────────────────────────────
# 3.  MAIN GENERATION FUNCTION
# ─────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: List[Dict],
    hf_token: Optional[str]  = None,
    model: Optional[str]     = None,
    max_new_tokens: int       = MAX_NEW_TOKENS,
    temperature: float        = TEMPERATURE,
) -> Dict:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        query          : the user's question
        chunks         : retrieved + re-ranked chunks from Step 3
        hf_token       : HF API token (falls back to env var)
        model          : override model (default: PRIMARY_MODEL)
        max_new_tokens : max length of generated answer
        temperature    : sampling temperature (lower = more factual)

    Returns dict:
        answer         : generated answer string
        model_used     : which model actually answered
        prompt         : the full prompt sent (useful for debugging)
        context        : formatted context string shown to LLM
        latency_ms     : total generation time in milliseconds
        error          : error message if generation failed
    """
    token         = hf_token or HF_API_TOKEN
    model_to_use  = model or PRIMARY_MODEL

    if token == "YOUR_HF_TOKEN_HERE":
        return _error_result(
            "HF_API_TOKEN not set.\n"
            "Get your token at https://huggingface.co/settings/tokens\n"
            "Then: export HF_API_TOKEN=hf_..."
        )

    if not chunks:
        return _error_result("No chunks retrieved — cannot generate an answer.")

    prompt_messages  = build_messages(query, chunks)
    context = format_context(chunks[:CONTEXT_CHUNKS])

    client = InferenceClient(token=token)

    # Try primary model, fall back if it fails
    for attempt_model in [model_to_use, FALLBACK_MODEL]:
        try:
            print(f"  Calling model: {attempt_model} ...", end=" ", flush=True)
            t0       = time.time()
            response = client.chat_completion(
                messages       = prompt_messages,
                model          = attempt_model,
                max_tokens     = max_new_tokens,
                temperature    = temperature,
                top_p          = TOP_P,
            )
            latency_ms = round((time.time() - t0) * 1000)
            print(f"done ({latency_ms} ms)")

            answer = response.choices[0].message.content.strip() if response.choices else ""

            if not answer:
                print(f"  Empty response from {attempt_model}, trying fallback...")
                continue

            return {
                "answer":      answer,
                "model_used":  attempt_model,
                "prompt":      str(prompt_messages),
                "context":     context,
                "latency_ms":  latency_ms,
                "error":       None,
            }

        except Exception as e:
            print(f"\n  Error from {attempt_model}: {e}")
            if attempt_model == FALLBACK_MODEL:
                return _error_result(f"Both models failed. Last error: {e}")
            print(f"  Trying fallback...")

    return _error_result("Generation failed after all attempts.")


def _error_result(message: str) -> Dict:
    return {
        "answer":     f"[Generation error] {message}",
        "model_used": None,
        "prompt":     None,
        "context":    None,
        "latency_ms": 0,
        "error":      message,
    }


# ─────────────────────────────────────────────
# 4.  LATENCY BREAKDOWN HELPER
#     (for your report's performance section)
# ─────────────────────────────────────────────

def measure_full_pipeline(
    query: str,
    retriever_components: Dict,
    hf_token: Optional[str] = None,
) -> Dict:
    """
    Run the full pipeline and record timing at each stage.
    Import retrieve from step3_retrieval to use this.

    Returns timing dict suitable for your report's latency table.
    """
    from step3_retrieval import retrieve

    timings = {}

    # Retrieval timing
    t0 = time.time()
    retrieval_result = retrieve(query, **retriever_components)
    timings["retrieval_ms"] = round((time.time() - t0) * 1000)

    chunks = retrieval_result["chunks"]

    # Generation timing
    t0 = time.time()
    gen_result = generate_answer(query, chunks, hf_token=hf_token)
    timings["generation_ms"] = round((time.time() - t0) * 1000)

    timings["total_ms"] = timings["retrieval_ms"] + timings["generation_ms"]

    return {
        "query":        query,
        "answer":       gen_result["answer"],
        "model_used":   gen_result["model_used"],
        "chunks":       chunks,
        "context":      gen_result["context"],
        "timings":      timings,
        "error":        gen_result["error"],
    }


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Legal Inheritance RAG — Step 4: Generation Test")
    print("=" * 55)

    # Dummy chunks so you can test generation without running Step 3
    dummy_chunks = [
        {
            "source": "inheritance_act.pdf",
            "text": (
                "Under the Succession Act, when a person dies intestate (without a will), "
                "their estate is distributed according to the rules of intestate succession. "
                "The spouse receives one-third of the estate, and the remainder is divided "
                "equally among the children. If there are no children, the spouse inherits "
                "the entire estate."
            ),
            "rerank_score": 0.92,
        },
        {
            "source": "islamic_inheritance_rules.txt",
            "text": (
                "In Islamic inheritance law (Fara'id), a daughter receives half the share "
                "of a son. If there are no sons, two or more daughters together receive "
                "two-thirds of the estate. A single daughter alone receives one-half."
            ),
            "rerank_score": 0.85,
        },
    ]

    test_query = "What share does a daughter receive in Islamic inheritance law?"

    print(f"\nQuery: {test_query}")
    print("\nGenerating answer...")

    result = generate_answer(
        query  = test_query,
        chunks = dummy_chunks,
    )

    if result["error"]:
        print(f"\nError: {result['error']}")
    else:
        print(f"\nModel used  : {result['model_used']}")
        print(f"Latency     : {result['latency_ms']} ms")
        print(f"\nAnswer:\n{result['answer']}")

    print("\n" + "=" * 55)
    print("Done!")


if __name__ == "__main__":
    main()
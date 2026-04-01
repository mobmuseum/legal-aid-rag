"""
LLM Generation Module
======================
Handles answer generation using HuggingFace Inference API
with Mistral-7B-Instruct-v0.3.
"""

import time
import requests

import config


# ── Prompt Templates ─────────────────────────────────────

SYSTEM_PROMPT = """You are an expert legal assistant specializing in Pakistan's inheritance and succession laws. You have deep knowledge of:
- Islamic inheritance law (both Sunni/Hanafi and Shia schools)
- The Succession Act of 1925
- The Muslim Family Laws Ordinance (MFLO) 1961
- Provincial legislation on women's property rights
- Family courts and succession certificate procedures

Your task is to answer the user's question accurately and comprehensively based ONLY on the provided context. Follow these rules:
1. Base your answer strictly on the retrieved context passages.
2. If the context does not contain sufficient information, clearly state that.
3. Cite specific laws, sections, or verses when mentioned in the context.
4. Provide clear, structured answers using bullet points or numbered lists when appropriate.
5. If the question involves comparing different schools of thought (Sunni vs Shia), present both perspectives."""

QA_PROMPT_TEMPLATE = """<s>[INST] {system_prompt}

Context (retrieved from legal documents):
---
{context}
---

Question: {question}

Provide a detailed, well-structured answer based on the context above. [/INST]"""


CLAIM_EXTRACTION_PROMPT = """<s>[INST] Extract all factual claims from the following answer as a numbered list. Each claim should be a single, verifiable statement.

Answer:
{answer}

List each claim on a new line, numbered 1, 2, 3, etc. Only extract factual claims, not opinions or hedging statements. [/INST]"""


CLAIM_VERIFICATION_PROMPT = """<s>[INST] Determine whether the following claim is supported by the given context. Respond with exactly "SUPPORTED" or "NOT SUPPORTED".

Claim: {claim}

Context:
{context}

Is this claim supported by the context? Answer with exactly one word: SUPPORTED or NOT SUPPORTED. [/INST]"""


QUESTION_GENERATION_PROMPT = """<s>[INST] Generate exactly 3 questions that the following answer is responding to. Each question should be different but related to the content.

Answer:
{answer}

List exactly 3 questions, one per line, numbered 1, 2, 3. [/INST]"""


# ── HuggingFace Inference API ────────────────────────────

def call_hf_inference(
    prompt: str,
    max_new_tokens: int = None,
    temperature: float = None,
    api_url: str = None,
) -> str:
    """
    Call the HuggingFace Inference API with the given prompt.
    Retries up to 3 times on failure with exponential backoff.
    """
    api_url = api_url or config.LLM_API_URL
    max_new_tokens = max_new_tokens or config.LLM_MAX_NEW_TOKENS
    temperature = temperature or config.LLM_TEMPERATURE

    headers = {
        "Authorization": f"Bearer {config.HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
            "do_sample": True,
        },
    }

    for attempt in range(3):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                return str(result).strip()

            elif response.status_code == 503:
                # Model is loading
                wait_time = response.json().get("estimated_time", 30)
                print(f"   ⏳ Model loading, waiting {wait_time:.0f}s...")
                time.sleep(min(wait_time, 60))
                continue

            elif response.status_code == 429:
                # Rate limited
                wait_time = 2 ** (attempt + 1)
                print(f"   ⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            else:
                print(f"   ❌ API error {response.status_code}: {response.text[:200]}")
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return f"[Error: API returned status {response.status_code}]"

        except requests.exceptions.Timeout:
            print(f"   ⏳ Request timed out (attempt {attempt + 1}/3)")
            if attempt < 2:
                time.sleep(5)
                continue
            return "[Error: Request timed out after 3 attempts]"

        except Exception as e:
            print(f"   ❌ Exception: {e}")
            if attempt < 2:
                time.sleep(2)
                continue
            return f"[Error: {str(e)}]"

    return "[Error: Failed after 3 attempts]"


# ── Answer Generation ────────────────────────────────────

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Generate an answer for the given query using the retrieved context chunks.
    """
    # Format context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk.get("source", "Unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Source: {source}]\n{text}")

    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = QA_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context,
        question=query,
    )

    # Call LLM
    answer = call_hf_inference(prompt)
    return answer


def extract_claims(answer: str) -> list[str]:
    """Extract factual claims from an answer using the LLM."""
    prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    response = call_hf_inference(prompt, max_new_tokens=300)

    # Parse numbered claims
    claims = []
    for line in response.strip().split("\n"):
        line = line.strip()
        # Remove numbering like "1.", "1)", "1:"
        if line and line[0].isdigit():
            claim = line.lstrip("0123456789.):-– ").strip()
            if claim:
                claims.append(claim)

    return claims


def verify_claim(claim: str, context: str) -> bool:
    """Verify a single claim against the context using the LLM."""
    prompt = CLAIM_VERIFICATION_PROMPT.format(claim=claim, context=context)
    response = call_hf_inference(prompt, max_new_tokens=20, temperature=0.1)
    return "SUPPORTED" in response.upper() and "NOT SUPPORTED" not in response.upper()


def generate_questions(answer: str) -> list[str]:
    """Generate alternate questions from an answer for relevancy evaluation."""
    prompt = QUESTION_GENERATION_PROMPT.format(answer=answer)
    response = call_hf_inference(prompt, max_new_tokens=200)

    questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            q = line.lstrip("0123456789.):-– ").strip()
            if q:
                questions.append(q)

    return questions[:config.EVAL_NUM_ALTERNATE_QUERIES]


if __name__ == "__main__":
    # Quick test
    test_chunks = [
        {
            "text": "Under Sunni Law, a daughter is entitled to a share of 1/2 of the inheritance if she is the sole daughter. If there are two or more daughters, they share 2/3 of the inheritance.",
            "source": "zallpinheritancelaw.txt",
        }
    ]
    query = "What share does a daughter get in Sunni inheritance law?"
    print(f"🔍 Query: {query}\n")
    answer = generate_answer(query, test_chunks)
    print(f"💬 Answer:\n{answer}")

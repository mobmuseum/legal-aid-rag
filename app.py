"""
Legal Aid RAG — Gradio Web Interface
======================================
A polished Gradio app for the Pakistan Inheritance Law Q&A system.
Displays generated answers, retrieved context, and evaluation scores.
"""

import time
import gradio as gr

import config
from retrieval import RetrievalEngine
from generator import generate_answer
from evaluation import Evaluator


# ── Global Initialization ────────────────────────────────
print("🚀 Initializing Legal Aid RAG system...")
retrieval_engine = RetrievalEngine(use_reranker=True)
evaluator = Evaluator()
print("✅ System ready!\n")


# ── Custom CSS ───────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%) !important;
}

#app-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.2rem !important;
    letter-spacing: -0.5px;
}

#app-subtitle {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 1rem !important;
    margin-bottom: 1.5rem !important;
    font-weight: 300;
}

.query-input textarea {
    background: rgba(30, 30, 60, 0.8) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1.05rem !important;
    padding: 16px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.query-input textarea:focus {
    border-color: rgba(102, 126, 234, 0.7) !important;
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.15) !important;
}

#submit-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 12px 32px !important;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    min-height: 48px !important;
}

#submit-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35) !important;
}

.answer-box {
    background: rgba(30, 30, 60, 0.6) !important;
    border: 1px solid rgba(102, 126, 234, 0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    padding: 20px !important;
}

.answer-box .prose {
    color: #e2e8f0 !important;
}

.score-card {
    background: rgba(30, 30, 60, 0.7) !important;
    border: 1px solid rgba(102, 126, 234, 0.25) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    text-align: center;
}

.context-accordion {
    background: rgba(20, 20, 50, 0.6) !important;
    border: 1px solid rgba(102, 126, 234, 0.15) !important;
    border-radius: 12px !important;
    margin-top: 8px !important;
}

.context-chunk {
    background: rgba(40, 40, 80, 0.5) !important;
    border-left: 3px solid #667eea !important;
    border-radius: 4px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
    font-size: 0.9rem !important;
    color: #cbd5e1 !important;
}

.timing-info {
    color: #64748b !important;
    font-size: 0.85rem !important;
    text-align: center;
    margin-top: 8px;
}

footer {
    display: none !important;
}
"""


# ── Core Q&A Function ───────────────────────────────────

def ask_question(query: str):
    """
    Process a user query through the full RAG pipeline:
    1. Retrieve relevant chunks (hybrid search + re-ranking)
    2. Generate answer using Mistral-7B
    3. Evaluate faithfulness and relevancy
    """
    if not query or not query.strip():
        return (
            "⚠️ Please enter a question.",
            "",
            "",
            "",
            "",
        )

    # ── Retrieve ──
    t_retrieval_start = time.time()
    chunks = retrieval_engine.retrieve(query, mode="hybrid")
    retrieval_time = time.time() - t_retrieval_start

    # ── Generate ──
    t_gen_start = time.time()
    answer = generate_answer(query, chunks)
    generation_time = time.time() - t_gen_start

    # ── Evaluate ──
    t_eval_start = time.time()
    eval_result = evaluator.evaluate(query, answer, chunks)
    eval_time = time.time() - t_eval_start

    total_time = retrieval_time + generation_time + eval_time

    # ── Format Context ──
    context_md = ""
    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "Unknown")
        category = chunk.get("category", "")
        score = chunk.get("rerank_score", chunk.get("rrf_score", "N/A"))
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        text = chunk.get("text", "")

        context_md += f"### 📄 Chunk {i+1} — *{source}*\n"
        context_md += f"**Category:** {category} | **Score:** {score_str}\n\n"
        context_md += f"> {text}\n\n"
        context_md += "---\n\n"

    # ── Format Scores ──
    faith_score = eval_result["faithfulness"]["score"]
    faith_claims = eval_result["faithfulness"].get("num_claims", 0)
    faith_supported = eval_result["faithfulness"].get("num_supported", 0)

    rel_score = eval_result["relevancy"]["score"]
    gen_questions = eval_result["relevancy"].get("generated_questions", [])
    similarities = eval_result["relevancy"].get("similarities", [])

    # Faithfulness details
    faith_md = f"## Faithfulness: {faith_score:.0%}\n\n"
    faith_md += f"**{faith_supported}/{faith_claims}** claims supported by context\n\n"
    claims = eval_result["faithfulness"].get("claims", [])
    if claims:
        faith_md += "| # | Claim | Supported |\n|---|---|---|\n"
        for j, c in enumerate(claims):
            status = "✅" if c["supported"] else "❌"
            claim_text = c["claim"][:100] + "..." if len(c["claim"]) > 100 else c["claim"]
            faith_md += f"| {j+1} | {claim_text} | {status} |\n"

    # Relevancy details
    rel_md = f"## Relevancy: {rel_score:.4f}\n\n"
    if gen_questions:
        rel_md += "**Generated Questions & Similarities:**\n\n"
        for j, (q, s) in enumerate(zip(gen_questions, similarities)):
            rel_md += f"{j+1}. {q} — *sim: {s:.4f}*\n"

    # Timing
    timing_md = (
        f"⏱ Retrieval: {retrieval_time:.2f}s | "
        f"Generation: {generation_time:.2f}s | "
        f"Evaluation: {eval_time:.2f}s | "
        f"**Total: {total_time:.2f}s**"
    )

    return answer, context_md, faith_md, rel_md, timing_md


# ── Example Queries ──────────────────────────────────────

EXAMPLES = [
    "What is the share of a daughter in inheritance under Sunni law?",
    "How does Shia inheritance law differ from Sunni law?",
    "What is a succession certificate and how do I obtain one in Pakistan?",
    "What are the rights of orphaned grandchildren under MFLO 1961?",
    "What does the Quran say about inheritance in Surah An-Nisa?",
    "What is the Enforcement of Women's Property Rights Act 2020?",
    "What documents are needed to claim inheritance in Pakistan?",
    "Who are the Quranic heirs and what are their shares?",
]


# ── Build Gradio UI ──────────────────────────────────────

def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="Legal Aid RAG — Pakistan Inheritance Law", theme=gr.themes.Base()) as app:

        # Header
        gr.Markdown("# ⚖️ Legal Aid RAG", elem_id="app-title")
        gr.Markdown(
            "Pakistan Inheritance & Succession Law — AI-Powered Q&A System",
            elem_id="app-subtitle",
        )

        # Query input
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    placeholder="Ask a question about Pakistan's inheritance law...",
                    label="Your Question",
                    lines=2,
                    elem_classes=["query-input"],
                )
            with gr.Column(scale=1, min_width=150):
                submit_btn = gr.Button(
                    "🔍 Ask", elem_id="submit-btn", variant="primary"
                )

        # Examples
        gr.Examples(
            examples=EXAMPLES,
            inputs=query_input,
            label="💡 Example Questions",
        )

        # Answer section
        gr.Markdown("### 💬 Generated Answer")
        answer_output = gr.Markdown(elem_classes=["answer-box"])

        # Timing
        timing_output = gr.Markdown(elem_classes=["timing-info"])

        # Scores side by side
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Faithfulness")
                faithfulness_output = gr.Markdown(elem_classes=["score-card"])
            with gr.Column():
                gr.Markdown("### 📐 Relevancy")
                relevancy_output = gr.Markdown(elem_classes=["score-card"])

        # Retrieved context (collapsible)
        with gr.Accordion("📚 Retrieved Context Chunks", open=False, elem_classes=["context-accordion"]):
            context_output = gr.Markdown()

        # Wire up the submit action
        submit_btn.click(
            fn=ask_question,
            inputs=[query_input],
            outputs=[answer_output, context_output, faithfulness_output, relevancy_output, timing_output],
        )

        query_input.submit(
            fn=ask_question,
            inputs=[query_input],
            outputs=[answer_output, context_output, faithfulness_output, relevancy_output, timing_output],
        )

    return app


# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

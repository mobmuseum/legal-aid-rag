"""
Step 6 - Legal Inheritance RAG: Streamlit Web Application
==========================================================
Deploy this on HuggingFace Spaces (Streamlit SDK).

Required Secrets in HF Spaces (Settings → Variables and Secrets):
  PINECONE_API_KEY   — your Pinecone key
  HF_API_TOKEN       — your HuggingFace token

File structure expected on HF Spaces:
  app.py                        ← this file
  requirements.txt
  output/
    chunks_recursive.json       ← upload from local Step 1 output
    bm25_corpus.json            ← upload from local Step 1 output
"""

import os
import time
import json
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Miras — Inheritance Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --ink:       #1a1208;
    --parchment: #f5f0e8;
    --gold:      #b8860b;
    --gold-lt:   #d4a843;
    --rust:      #8b3a1a;
    --sage:      #4a6741;
    --cream:     #faf7f0;
    --border:    #d9cdb8;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--ink);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Header banner ── */
.miras-header {
    background: var(--ink);
    color: var(--parchment);
    padding: 2.2rem 3rem 1.8rem;
    border-radius: 4px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.miras-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        45deg,
        transparent,
        transparent 40px,
        rgba(184,134,11,0.04) 40px,
        rgba(184,134,11,0.04) 41px
    );
}
.miras-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--parchment);
    margin: 0 0 0.3rem;
}
.miras-header .tagline {
    font-size: 0.95rem;
    color: var(--gold-lt);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 300;
}

/* ── Input area ── */
.stTextArea textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 4px !important;
    background: white !important;
    color: var(--ink) !important;
    padding: 1rem !important;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(184,134,11,0.12) !important;
}

/* ── Primary button ── */
.stButton > button {
    background: var(--ink) !important;
    color: var(--parchment) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.65rem 2rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: var(--rust) !important;
}

/* ── Answer card ── */
.answer-card {
    background: white !important;
    color: var(--ink) !important;
    border: 1.5px solid var(--border);
    border-left: 4px solid var(--gold);
    border-radius: 4px;
    padding: 1.6rem 2rem;
    margin: 1.2rem 0;
    line-height: 1.8;
    font-size: 1.02rem;
}
.answer-card * {
    color: var(--ink) !important;
}

/* ── Score pills ── */
.score-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.score-pill {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: var(--parchment);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.7rem 1.4rem;
    min-width: 120px;
}
.score-pill .label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.score-pill .value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--ink);
}
.score-pill.good .value  { color: var(--sage); }
.score-pill.mid  .value  { color: var(--gold); }
.score-pill.low  .value  { color: var(--rust); }

/* ── Context chunks ── */
.chunk-card {
    background: var(--parchment);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
    line-height: 1.7;
}
.chunk-card p, .chunk-card strong, .chunk-card em {
    color: var(--ink) !important;
}
.chunk-meta {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--gold) !important;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--ink) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--parchment) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: var(--gold-lt) !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Section headings ── */
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--ink);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.6rem 0 1rem;
}

/* ── Disclaimer ── */
.disclaimer {
    font-size: 0.78rem;
    color: #999;
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 3px;
    padding: 0.7rem 1rem;
    margin-top: 2rem;
    line-height: 1.6;
}

/* ── Latency bar ── */
.latency-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    font-size: 0.78rem;
    color: #888;
    margin-top: 0.8rem;
}
.latency-chip {
    background: #f0ede6;
    border-radius: 2px;
    padding: 0.2rem 0.6rem;
}

/* ── Spinner tweak ── */
.stSpinner > div { border-top-color: var(--gold) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD COMPONENTS (cached — runs once at startup)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading retrieval models…")
def load_retriever():
    from src.retrieval import build_retriever
    return build_retriever()


@st.cache_resource(show_spinner="Loading similarity model…")
def load_similarity_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="miras-header">
    <h1>⚖️ Miras</h1>
    <div class="tagline">Inheritance Law · Intelligent Research Assistant</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR — settings
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    retrieval_mode = st.selectbox(
        "Retrieval mode",
        ["Hybrid + Re-ranking (recommended)", "Semantic only"],
        index=0,
    )
    semantic_only = retrieval_mode == "Semantic only"

    top_k = st.slider("Chunks retrieved", min_value=3, max_value=10, value=5)

    temperature = st.slider(
        "LLM temperature",
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Lower = more factual. Recommended: 0.1–0.3 for legal answers."
    )

    show_scores = st.checkbox("Show evaluation scores", value=True)
    show_chunks = st.checkbox("Show retrieved context", value=True)

    st.markdown("---")
    st.markdown("### 📚 Example queries")
    examples = [
        "Who inherits when there is no will?",
        "What is a daughter's share in Islamic inheritance?",
        "Can a non-Muslim inherit from a Muslim estate?",
        "How are debts settled before distributing the estate?",
        "Can a minor be an heir?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#888;line-height:1.6'>"
        "Powered by Mistral-7B · Pinecone · HuggingFace"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN — query input
# ─────────────────────────────────────────────

query = st.text_area(
    "Your legal question",
    key="query_input",
    placeholder="e.g. What share does a wife receive from her deceased husband's estate?",
    height=100,
    label_visibility="collapsed",
)

col_btn, col_mode = st.columns([2, 5])
with col_btn:
    search_clicked = st.button("Ask Miras →", use_container_width=True)
with col_mode:
    st.markdown(
        f"<div style='padding:0.6rem 0;font-size:0.82rem;color:#888'>"
        f"Mode: <strong>{retrieval_mode}</strong> · Top-{top_k} chunks"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN — run pipeline on query
# ─────────────────────────────────────────────

if search_clicked and query.strip():

    # Load components (cached after first run)
    try:
        components  = load_retriever()
        sim_model   = load_similarity_model()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    # ── Step 1: Retrieve ──────────────────────
    with st.spinner("Searching legal corpus…"):
        t0 = time.time()
        from src.retrieval import retrieve
        retrieval = retrieve(
            query,
            **components,
            top_k=top_k,
            semantic_only=semantic_only,
        )
        retrieval_ms = round((time.time() - t0) * 1000)

    chunks = retrieval["chunks"]

    if not chunks:
        st.warning("No relevant documents found. Try rephrasing your question.")
        st.stop()

    # ── Step 2: Generate ─────────────────────
    with st.spinner("Generating answer…"):
        t0 = time.time()
        from src.generation import generate_answer
        gen = generate_answer(
            query,
            chunks,
            temperature=temperature,
        )
        generation_ms = round((time.time() - t0) * 1000)

    if gen["error"]:
        st.error(f"Generation failed: {gen['error']}")
        st.stop()

    answer = gen["answer"]
    context_text = "\n\n".join(c["text"] for c in chunks)

    # ── Step 3: Evaluate ─────────────────────
    faith_result = {"score": None}
    rel_result   = {"score": None}

    if show_scores:
        with st.spinner("Evaluating answer quality…"):
            from src.evaluation import evaluate_faithfulness, evaluate_relevancy
            faith_result = evaluate_faithfulness(answer, context_text, verbose=False)
            rel_result   = evaluate_relevancy(query, answer, sim_model)

    # ── Display: answer ──────────────────────
    st.markdown('<div class="section-heading">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    # ── Display: scores ──────────────────────
    if show_scores and faith_result["score"] is not None:
        f_score = faith_result["score"]
        r_score = rel_result["score"]

        def score_class(v):
            if v >= 0.75: return "good"
            if v >= 0.50: return "mid"
            return "low"

        st.markdown('<div class="section-heading">Evaluation Scores</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-row">
            <div class="score-pill {score_class(f_score)}">
                <span class="label">Faithfulness</span>
                <span class="value">{f_score:.0%}</span>
            </div>
            <div class="score-pill {score_class(r_score)}">
                <span class="label">Relevancy</span>
                <span class="value">{r_score:.2f}</span>
            </div>
            <div class="score-pill">
                <span class="label">Claims verified</span>
                <span class="value">{faith_result.get('supported_claims', '—')}/{faith_result.get('total_claims', '—')}</span>
            </div>
            <div class="score-pill">
                <span class="label">Chunks used</span>
                <span class="value">{len(chunks)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable: claim breakdown
        with st.expander("View claim verification details"):
            for v in faith_result.get("verifications", []):
                icon = "✅" if v["supported"] else "❌"
                st.markdown(f"{icon} {v['claim']}")

        # Expandable: generated questions
        with st.expander("View relevancy question breakdown"):
            for q, sim in zip(
                rel_result.get("generated_questions", []),
                rel_result.get("similarities", []),
            ):
                st.markdown(f"- **{q}** — similarity: `{sim:.4f}`")

    # ── Display: latency ─────────────────────
    total_ms = retrieval_ms + generation_ms
    st.markdown(f"""
    <div class="latency-row">
        <span class="latency-chip">🔍 Retrieval: {retrieval_ms} ms</span>
        <span class="latency-chip">🤖 Generation: {generation_ms} ms</span>
        <span class="latency-chip">⏱ Total: {total_ms} ms</span>
        <span class="latency-chip">📦 Model: {gen['model_used'].split('/')[-1] if gen['model_used'] else '—'}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Display: retrieved context ───────────
    if show_chunks:
        st.markdown('<div class="section-heading">Retrieved Context</div>', unsafe_allow_html=True)
        for i, chunk in enumerate(chunks, 1):
            source      = chunk.get("source", "Unknown")
            rerank_score = chunk.get("rerank_score", 0)
            rrf_score   = chunk.get("rrf_score", 0)
            text        = chunk.get("text", "")

            st.markdown(f"""
            <div class="chunk-card">
                <div class="chunk-meta">
                    [{i}] {source} &nbsp;·&nbsp;
                    rerank: {rerank_score:.3f} &nbsp;·&nbsp;
                    rrf: {rrf_score:.5f}
                </div>
                {text}
            </div>
            """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Legal Disclaimer:</strong> Miras is a research tool and does not constitute
        legal advice. Information provided is for educational purposes only. For specific legal
        matters, please consult a qualified lawyer.
    </div>
    """, unsafe_allow_html=True)

elif search_clicked and not query.strip():
    st.warning("Please enter a question before clicking Ask.")


# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────

if not search_clicked:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0 2rem;color:#aaa">
        <div style="font-size:3rem;margin-bottom:1rem">⚖️</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#888;margin-bottom:0.5rem">
            Ask a question about inheritance law
        </div>
        <div style="font-size:0.85rem">
            Try the example queries in the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)
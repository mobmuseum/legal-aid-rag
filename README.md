# ⚖️ Legal Aid RAG — Pakistan Inheritance Law Q&A System

A Retrieval-Augmented Generation (RAG) system for answering questions about inheritance and succession law in Pakistan. Built with hybrid search, cross-encoder re-ranking, and LLM-as-a-Judge evaluation.

## 🏗️ Architecture

```
User Query
    │
    ├── BM25 Keyword Search ──────┐
    │                             ├── Reciprocal Rank Fusion (RRF)
    └── Pinecone Semantic Search ─┘
                                  │
                          Cross-Encoder Re-ranking
                                  │
                          Top-5 Context Chunks
                                  │
                      Mistral-7B-Instruct (HF API)
                                  │
                          Generated Answer
                                  │
                    ┌─────────────┴─────────────┐
              Faithfulness              Relevancy
           (Claim Verification)    (Query Similarity)
```

## 📁 Project Structure

```
legal-aid-rag/
├── Source/                     # Raw legal documents (PDFs + TXT)
│   ├── 1 - Core statutory laws/
│   ├── 2 - Islamic fiqh based/
│   ├── 3 - research paper and legal analysis/
│   └── 4 - govt and legal repos/
├── app.py                     # Gradio web interface
├── config.py                  # Centralized configuration
├── ingest.py                  # Document ingestion pipeline
├── retrieval.py               # Hybrid retrieval engine
├── generator.py               # LLM generation (Mistral-7B)
├── evaluation.py              # LLM-as-a-Judge evaluation
├── ablation.py                # Ablation study runner
├── test_queries.json          # Test query set (15 queries)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
copy .env.example .env
```

- **Pinecone**: https://app.pinecone.io → API Keys
- **HuggingFace**: https://huggingface.co/settings/tokens → New Token (Read)

### 3. Run Document Ingestion

```bash
python ingest.py recursive
```

This extracts text from all source documents, chunks them, generates embeddings, uploads to Pinecone, and builds the BM25 index.

### 4. Launch the Web App

```bash
python app.py
```

Open http://localhost:7860 in your browser.

### 5. Run Evaluation

```bash
python evaluation.py
```

### 6. Run Ablation Study

```bash
python ablation.py
```

## 📊 Evaluation Metrics

| Metric | Description |
|--------|------------|
| **Faithfulness** | % of claims in the answer supported by retrieved context |
| **Relevancy** | Cosine similarity between generated questions and original query |

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `all-MiniLM-L6-v2` (384-dim) |
| Vector DB | Pinecone (Serverless) |
| Keyword Search | BM25 (rank-bm25) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Mistral-7B-Instruct v0.3 (HF Inference API) |
| Web UI | Gradio |
| Hosting | HuggingFace Spaces |

## 📚 Corpus

27 documents covering Pakistan's inheritance law:
- Core statutory laws (Succession Act 1925, Family Laws Ordinance)
- Islamic jurisprudence (Hanafi, Shia schools, Quranic verses)
- Research papers on MFLO Section 4, orphaned grandchildren rights
- Government legislation (Marriage Acts, Property Rights Acts, Family Courts)

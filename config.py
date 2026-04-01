"""
Centralized configuration for the Legal Aid RAG system.
All model names, chunk sizes, API settings, and paths are defined here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# Fix for "not using authenticated requests" warning in huggingface_hub
if HF_API_TOKEN:
    os.environ["HF_TOKEN"] = HF_API_TOKEN

# Disable hf_transfer as it might cause stuck downloads on some networks
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-aid-rag")

# ── Paths ─────────────────────────────────────────────────
SOURCE_DIR = os.path.join(os.path.dirname(__file__), "Source")
BM25_INDEX_PATH = os.path.join(os.path.dirname(__file__), "bm25_index.pkl")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.pkl")

# ── Embedding Model ──────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Cross-Encoder (Re-ranking) ───────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM Generation ───────────────────────────────────────
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL_NAME}"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.3

# ── Chunking Settings ────────────────────────────────────
CHUNK_SIZE_FIXED = 512          # characters for fixed-size chunking
CHUNK_OVERLAP_FIXED = 50        # overlap for fixed-size chunking

CHUNK_SIZE_RECURSIVE = 512      # max characters for recursive chunking
CHUNK_OVERLAP_RECURSIVE = 50    # overlap for recursive chunking

# ── Retrieval Settings ───────────────────────────────────
SEMANTIC_TOP_K = 20             # top-k results from Pinecone
BM25_TOP_K = 20                 # top-k results from BM25
RRF_K = 60                      # RRF constant
RERANK_TOP_K = 5                # final top-k after re-ranking

# ── Pinecone Settings ───────────────────────────────────
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# ── Evaluation ───────────────────────────────────────────
EVAL_NUM_ALTERNATE_QUERIES = 3  # for relevancy scoring

# Legal Inheritance RAG Architecture

This repository contains the source code, offline evaluation scripts, and deployment resources for a specialized Retrieval-Augmented Generation (RAG) pipeline. The architecture is engineered to query complex legal inheritance laws using a Hybrid Retrieval framework, subsequently validating generated responses through an automated LLM-as-a-Judge system.

## Project Structure

The codebase is logically split into modular stages corresponding to the data ingestion pipeline and final deployment.

- `src/preprocess.py`: Extracts and recursively chunks raw legal documents from PDF formats.
- `src/embed_upsert.py`: Processes continuous text chunks into dense mathematical vectors locally using Sentence Transformers and upserts them to a remote Pinecone Serverless matrix.
- `src/retrieval.py`: Houses the mathematical logic for querying. It executes parallel sparse (BM25) and dense (Pinecone) queries, merging their outputs systematically via Reciprocal Rank Fusion (RRF). 
- `src/generation.py`: Dispatches retrieved context arrays securely to the Hugging Face Inference API for strict logical generation using Qwen models.
- `src/evaluation.py`: An autonomous testing suite that verifies Faithfulness and Relevancy algorithmically via the unmetered Groq Llama-3.1 API.
- `app.py`: The interactive graphical user interface built entirely natively within Streamlit.

## Setup and Dependencies

To execute the data injection and compilation stages locally, you must provide explicit environmental variables.

1. Clone the repository and navigate into the root directory.
2. Create a `.env` file containing the following access keys:
   - `HF_TOKEN=your_huggingface_token`
   - `PINECONE_API_KEY=your_pinecone_key`
   - `GROQ_API_KEY=your_groq_key`

3. Install the fundamental Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Note: The `requirements.txt` includes advanced dependencies natively required for both the Streamlit UI and local offline dense embedding (`sentence-transformers`, `pinecone[grpc]`, `rank-bm25`). 

## Execution Sequence

The pipeline must be engaged consecutively to ensure structural integrity across the Vector DB and local sparse matrices before opening the application loop.

1. **Preprocessing Phase**:
   Combiles raw input structures.
   ```bash
   python3 src/preprocess.py
   ```

2. **Embedding and Upsert Phase**:
   Transforms and uploads processed textual boundaries.
   ```bash
   python3 src/embed_upsert.py
   ```

3. **Application Interface**:
   Boots the interactive frontend layer locally.
   ```bash
   streamlit run app.py
   ```

## Evaluation and Validation

The implementation was systematically benchmarked using `src/evaluation.py`. The evaluation isolated varying configurations (fixed vs. recursive chunking, semantic-only vs. hybrid retrieval) verifying their latency properties and synthetic metrics natively. 

The evaluation framework automatically aggregates these properties into statistical tables mapping Faithfulness (claims supported by source documents) and Relevancy (cosine stability against generated inverse questions) specifically using the Llama-3.1-8B model natively on Groq processors.

## Deployment Notes

The resultant interface is designed to migrate beyond local boundaries and is actively configured sequentially for Hugging Face Spaces. Using the provided `Dockerfile`, the Streamlit application containerizes efficiently, circumventing rigid SDK requirements. The interface automatically pulls variables continuously, operating directly under the mapped 7860 network port.

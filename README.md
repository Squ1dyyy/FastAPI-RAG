# RAG FastAPI Documentation Assistant

Asynchronous GPU-accelerated **Hybrid RAG** pipeline for FastAPI reference documentation, powered by **BM25-Fast (`bm25s`)**, **`intfloat/multilingual-e5-small`** (CUDA FP16), **TinyBERT Cross-Encoder Reranker** (CUDA FP16), **OpenRouter LLM Generation**, and **PostgreSQL + SQLAlchemy 2.0 (`asyncpg`)**.

## Project Structure

```text
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── __init__.py
│   └── config.py              # Centralized settings loaded from .env
├── src/
│   ├── pipeline.py            # Async GPU HybridRAGPipeline (Search + Rerank + OpenRouter + Eval)
│   ├── database/              # PostgreSQL + SQLAlchemy 2.0 (asyncpg)
│   │   ├── connection.py      # AsyncEngine, sessionmaker, init_db, get_async_session
│   │   ├── models/            # Document (with toc_sections), DocumentChunk, QueryLog
│   │   └── repository/        # DocumentRepository, QueryLogRepository
│   ├── grabber/               # Async concurrent web scraper and HTML cache for FastAPI docs
│   ├── preprocessing/         # Async structural HTML parser (md-sidebar__scrollwrap + md-content__inner)
│   ├── embeddings/            # Async GPU MultilingualE5Embedder (intfloat/multilingual-e5-small)
│   ├── retrieval/             # Async Hybrid Retrieval: Dense (E5 GPU) + Sparse (BM25-Fast)
│   │   ├── base.py            # BaseRetriever interface
│   │   ├── sparse.py          # BM25Retriever (bm25s + identifier tokenization)
│   │   ├── dense.py           # DenseRetriever (GPU E5 cosine similarity)
│   │   └── hybrid.py          # HybridRetriever (concurrent asyncio.gather + RRF / Linear fusion)
│   ├── reranking/             # Async GPU CrossEncoderReranker (cross-encoder/ms-marco-TinyBERT-L-2-v2)
│   ├── generation/            # Async OpenRouterGenerator (Chat Completions API)
│   └── evaluation/            # Async RAGEvaluator (context coverage, faithfulness, citations, F1)
├── data/                      # Cached raw HTML documentation files
├── experiments/               # Async DB ingestion and Hybrid RAG demos
└── tests/                     # Async unit and integration test suite
```

## Quick Start

### 1. Install GPU Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`, `POSTGRES_URL`, and `DEVICE=cuda`.

### 3. Grab Documentation HTML

```bash
python scraper.py
```

### 4. Ingest Documents into PostgreSQL

```bash
python experiments/database_ingest_demo.py
```

### 5. Run Full Async GPU Hybrid RAG Demo

```bash
python experiments/hybrid_search_demo.py
```

### 6. Run Tests

```bash
python -m unittest discover tests
```

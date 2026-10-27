# RAG FastAPI Documentation Assistant

RAG (Retrieval-Augmented Generation) pipeline for FastAPI documentation with Hybrid Retrieval (Dense + BM25-Fast) and PostgreSQL / SQLAlchemy 2.0.

## Project Structure

```text
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   └── config.py              # Centralized application settings (Pydantic BaseSettings)
├── src/
│   ├── database/              # PostgreSQL + SQLAlchemy 2.0 (asyncpg)
│   │   ├── connection.py      # AsyncEngine, sessionmaker, init_db, get_async_session
│   │   ├── models/            # SQLAlchemy Declarative Models (Document, DocumentChunk, QueryLog)
│   │   └── repository/        # DocumentRepository, QueryLogRepository
│   ├── grabber/               # Web scraper and HTML cache for FastAPI docs
│   ├── preprocessing/         # HTML cleaning, text extraction, and chunking
│   ├── embeddings/            # Embedding interfaces and vectorizers
│   ├── retrieval/             # Hybrid Retrieval: Dense + BM25-Fast
│   │   ├── base.py            # BaseRetriever abstract class
│   │   ├── sparse.py          # BM25-Fast (inverted index BM25 retriever)
│   │   ├── dense.py           # Dense vector similarity retriever
│   │   └── hybrid.py          # Hybrid fusion (RRF / Linear combination)
│   ├── reranking/             # Cross-encoder candidate rerankers
│   ├── generation/            # LLM answer synthesis and prompt templates
│   └── evaluation/            # Evaluation metrics (faithfulness, relevance)
├── data/                      # Cached raw HTML files and processed documents
├── experiments/               # Benchmarks, experiment scripts, and notebooks
└── tests/                     # Unit and integration test suite
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file (or configure environment variables):

```env
POSTGRES_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_fastapi
POSTGRES_ECHO=false
HYBRID_ALPHA=0.5
HYBRID_FUSION_METHOD=rrf
```

### 3. Grab Documentation Data

Run the grabber to download and cache the FastAPI reference documentation:

```bash
python -m src.grabber.grabber
# or
python scraper.py
```

### 4. Ingest Scraped Documents into Database

```bash
python experiments/database_ingest_demo.py
```

### 5. Run Hybrid Search Demo

```bash
python experiments/hybrid_search_demo.py
```

### 6. Run Tests

```bash
python -m unittest discover tests
```

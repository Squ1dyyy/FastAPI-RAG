# RAG FastAPI Documentation Assistant

End-to-end **Hybrid RAG** pipeline for FastAPI reference documentation, powered by **BM25-Fast (`bm25s`)**, **`intfloat/multilingual-e5-small`**, **TinyBERT Cross-Encoder Reranker**, and **PostgreSQL + SQLAlchemy 2.0 (`asyncpg`)**.

## Project Structure

```text
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── __init__.py
│   └── config.py              # Centralized settings (Pydantic BaseSettings)
├── src/
│   ├── pipeline.py            # End-to-end HybridRAGPipeline
│   ├── database/              # PostgreSQL + SQLAlchemy 2.0 (asyncpg)
│   │   ├── connection.py      # AsyncEngine, sessionmaker, init_db, get_async_session
│   │   ├── models/            # Document (with toc_sections), DocumentChunk, QueryLog
│   │   └── repository/        # DocumentRepository, QueryLogRepository
│   ├── grabber/               # Async web scraper and HTML cache for FastAPI docs
│   ├── preprocessing/         # Structural HTML parser (md-sidebar__scrollwrap + md-content__inner)
│   ├── embeddings/            # MultilingualE5Embedder (intfloat/multilingual-e5-small)
│   ├── retrieval/             # Hybrid Retrieval: Dense (E5) + Sparse (BM25-Fast)
│   │   ├── base.py            # BaseRetriever interface
│   │   ├── sparse.py          # BM25Retriever (bm25s + identifier tokenization)
│   │   ├── dense.py           # DenseRetriever (E5 cosine similarity)
│   │   └── hybrid.py          # HybridRetriever (RRF & Linear score fusion)
│   ├── reranking/             # CrossEncoderReranker (cross-encoder/ms-marco-TinyBERT-L-2-v2)
│   ├── generation/            # LLM answer synthesis interfaces
│   └── evaluation/            # RAG evaluation metrics interfaces
├── data/                      # Cached raw HTML documentation files
├── experiments/               # Ingestion and Hybrid RAG search demos
└── tests/                     # Unit and integration test suite
```

## HTML Parsing & Database Schema Design

FastAPI reference HTML pages (`data/*.html`) are parsed structurally by `FastAPIHTMLParser`:
1. **Document Title (`Document.title`)**: Extracted from `<article class="md-content__inner md-typeset"> -> <h1>`.
2. **Table of Contents (`Document.toc_sections`)**: Extracted from `<div class="md-sidebar--secondary"> -> <div class="md-sidebar__scrollwrap">`, preserving section anchors (`#...`), titles, and nesting hierarchy (`breadcrumb`).
3. **Section Chunks (`DocumentChunk`)**: `<article class="md-content__inner md-typeset">` is split along section headings (`h1..h6[id]`) matching the TOC, storing:
   - `section_title`, `anchor`, `section_url`, `breadcrumb`, `symbol_type` (`class`, `function`, `attribute`, `section`)
   - Clean markdown-formatted `text` (signatures, descriptions, formatted parameter tables, code blocks)
   - Precomputed 384-d `embedding` vector from `intfloat/multilingual-e5-small`
   - `metadata_json` (`JSONB` in PostgreSQL)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Grab Documentation HTML

```bash
python scraper.py
```

### 3. Ingest Documents into PostgreSQL (or SQLite fallback)

```bash
python experiments/database_ingest_demo.py
```

### 4. Run Full Hybrid RAG Search + Reranking Demo

```bash
python experiments/hybrid_search_demo.py
```

### 5. Run Tests

```bash
python -m unittest discover tests
```

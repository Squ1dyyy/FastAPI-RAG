# FastAPI Hybrid RAG

An asynchronous, GPU-accelerated, multilingual **Hybrid RAG** system for the official **FastAPI Reference** documentation (`https://fastapi.tiangolo.com/reference/`), featuring an interactive web chat UI, persistent multi-turn conversation sessions, and zero-token structural **DOM LCA Graph Expansion**.

![FastAPI Hybrid RAG Web Chat UI](docs/chat_ui.png)

---

## Key Features

- **Multilingual Asynchronous Scraper (13 Languages)**:
  Automatically scrapes and caches all 23 FastAPI Reference pages across 13 language localizations (`en`, `de`, `es`, `fr`, `hi`, `ja`, `ko`, `pt`, `ru`, `tr`, `uk`, `zh`, `zh-hant` — **299 HTML documents**, **14,534 structural chunks**) into `data/{lang}/` using `curl_cffi.AsyncSession`.
- **Structural HTML Parser (`FastAPIHTMLParser`)**:
  - Extracts the Table of Contents hierarchy (`anchor`, `title`, `breadcrumb`, `level`) from `md-sidebar--secondary` -> `md-sidebar__scrollwrap`.
  - Parses `<article class="md-content__inner md-typeset">` along the `h1 -> h2 -> h3` DOM hierarchy, preserving class/function signatures, parameter tables, and code blocks.
- **Hybrid Retrieval**:
  - **Sparse (Lexical)**: `BM25-Fast` powered by `bm25s` with a custom Python identifier tokenizer (`CamelCase` and `snake_case` splitting).
  - **Dense (GPU Semantic)**: `intfloat/multilingual-e5-small` (`CUDA`, `FP16`, 384-d) with standard `"query: "` and `"passage: "` prefixes.
  - **Fusion**: Reciprocal Rank Fusion (`RRF`) and Linear Combination (`Linear`).
- **Multilingual GPU Cross-Encoder Reranker**:
  Lightweight multilingual cross-encoder `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (`CUDA`, `FP16`) for high-precision re-scoring of hybrid candidates.
- **Zero-Token DOM Tree LCA GraphRAG (`h2` & `h1` Expansion)**:
  - **`h2` Level (`h3 -> h2`)**: When a child `h3` attribute or method is retrieved, the pipeline automatically expands its parent `h2` class/function definition, `Example` sections, and methods.
  - **`h1` Level (`h1 -> all h2`)**: When retrieved chunks span 2+ distinct `h2` subtrees of the same `h1` document (or hit the `h1` overview directly), their Lowest Common Ancestor (`LCA`) in the DOM tree resolves to `h1`, expanding all `h2` sections of that document without spending any LLM tokens at indexing time.
- **Provider-Agnostic `AsyncOpenAI` Generation**:
  Works with Google AI Studio (`gemini-3.1-flash-lite`), OpenRouter, OpenAI, or local vLLM/Ollama endpoints via `LLM_BASE_URL` and `LLM_API_KEY`. Enforces strict context grounding and mandatory code snippet inclusion from retrieved chunks.
- **Database & Migrations**:
  Asynchronous **PostgreSQL (`asyncpg`)** with automatic fallback to **SQLite (`aiosqlite`)** via SQLAlchemy 2.0 and **Alembic** migrations.
- **FastAPI Web Service & Multi-Turn Chat Sessions**:
  Built-in web chat UI with language switching, persistent sessions, adaptive two-step session query resolution, Markdown/code rendering, and clickable reference badges.

---

## Pipeline Architecture

```mermaid
flowchart LR
    U["User Query + Session"] --> S["FastAPI /api/chat"]
    S --> H["Hybrid Retriever (BM25-Fast + GPU E5-small)"]
    H --> R["GPU Reranker (mmarco-mMiniLMv2)"]
    R --> G["DOM LCA Graph Expansion (h3 -> h2 / h1 -> all h2)"]
    G --> L["AsyncOpenAI LLM (gemini-3.1-flash-lite)"]
    L --> DB[("PostgreSQL / SQLite + Alembic")]
```

---

## Project Structure

```text
RAGFastAPI/
├── alembic/                    # Alembic database migrations (0001..0003)
├── configs/
│   └── config.py               # Application settings via Pydantic Settings (.env)
├── data/
│   ├── en/, de/, ru/, ...      # Scraped HTML reference pages across 13 languages (299 files)
│   └── rag.db                  # Local database with precomputed 384-d embeddings
├── docs/
│   └── chat_ui.png             # Web chat interface screenshot
├── src/
│   ├── api/                    # FastAPI app, Pydantic schemas, and web chat UI
│   ├── database/               # SQLAlchemy 2.0 models and async repositories
│   ├── embeddings/             # GPU embedder (intfloat/multilingual-e5-small)
│   ├── evaluation/             # RAG evaluation metrics (Context Coverage, Citations, Token F1)
│   ├── generation/             # AsyncOpenAI generator with strict context & code enforcement
│   ├── grabber/                # Async multilingual FastAPI documentation scraper
│   ├── preprocessing/          # Structural HTML parser (TOC + h1..h6 sections)
│   ├── reranking/              # GPU Cross-Encoder reranker
│   ├── retrieval/              # BM25-Fast, Dense E5, and Hybrid (RRF / Linear) retrievers
│   └── pipeline.py             # HybridRAGPipeline orchestrator
├── tests/                      # 30 unit and integration tests
├── .env.example                # Environment variables template
├── alembic.ini                 # Alembic configuration
├── requirements.txt            # Project dependencies (CUDA 12.6 enabled)
└── run_server.py               # Entry point to start the FastAPI server
```

---

## Setup & Quickstart

### 1. Install Dependencies

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and set your API key and configuration parameters:

```env
POSTGRES_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_fastapi
FALLBACK_SQLITE_URL=sqlite+aiosqlite:///data/rag.db

DEVICE=cuda
USE_FP16=true

EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
RERANKER_MODEL_NAME=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-3.1-flash-lite

SUPPORTED_LANGUAGES=en,de,es,fr,hi,ja,ko,pt,ru,tr,uk,zh,zh-hant
DEFAULT_LANGUAGE=en

APP_HOST=127.0.0.1
APP_PORT=8000
```

### 3. Run Database Migrations

```powershell
.\venv\Scripts\alembic upgrade head
```

### 4. Start the FastAPI Chat Server

```powershell
.\venv\Scripts\python run_server.py
```

Once started, open in your browser:
- **Web Chat UI**: `http://127.0.0.1:8000/`
- **Swagger UI (OpenAPI)**: `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Interactive web chat UI with language selector and session management |
| `GET` | `/api/health` | Service status, active device (`cuda`/`cpu`), loaded models, and chunk count |
| `GET` | `/api/languages` | List of all 13 supported languages and indexed chunk counts |
| `POST` | `/api/sessions` | Create a new chat session |
| `GET` | `/api/sessions` | List existing chat sessions |
| `GET` | `/api/sessions/{session_id}` | Retrieve full message history for a session |
| `DELETE` | `/api/sessions/{session_id}` | Delete a chat session and its history |
| `POST` | `/api/chat` | Execute a Hybrid RAG query within a session |

---

## Running Tests

Run the full suite of 30 unit and integration tests (scraper, structural HTML parser, DB repositories, `BM25-Fast`, `Dense E5`, `RRF`/`Linear`, `DOM LCA h1->h2`, LLM generator, and FastAPI endpoints):

```powershell
.\venv\Scripts\python -m unittest discover tests -v
```

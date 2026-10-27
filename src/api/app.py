from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from configs.config import config
from src.api.schemas import (
	ChatRequest,
	ChatResponse,
	LanguageOut,
	MessageOut,
	SessionCreateRequest,
	SessionDetailOut,
	SessionSummaryOut,
	SourceItem,
)
from src.api.ui import CHAT_HTML
from src.database.connection import create_engine_and_session, init_db
from src.database.repository import ChatSessionRepository
from src.pipeline import HybridRAGPipeline


async def _resolve_engine_and_session() -> Tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
	"""Connect to PostgreSQL or fall back to local async SQLite database."""
	engine, factory = create_engine_and_session(
		database_url=config.postgres_url,
		echo=config.postgres_echo,
	)
	try:
		await init_db(engine)
		return engine, factory
	except Exception:
		await engine.dispose()
		fallback_engine, fallback_factory = create_engine_and_session(
			database_url=config.fallback_sqlite_url,
			echo=False,
		)
		await init_db(fallback_engine)
		return fallback_engine, fallback_factory


def create_app(
	pipeline: Optional[HybridRAGPipeline] = None,
	session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
) -> FastAPI:
	"""Create and configure the FastAPI Hybrid RAG chat application."""

	@asynccontextmanager
	async def lifespan(app: FastAPI):
		owned_engine: Optional[AsyncEngine] = None
		if session_factory is not None:
			app.state.session_factory = session_factory
		else:
			owned_engine, resolved_factory = await _resolve_engine_and_session()
			app.state.session_factory = resolved_factory

		if pipeline is not None:
			app.state.pipeline = pipeline
		else:
			active_device = config.resolve_device()
			rag_pipeline = HybridRAGPipeline(device=active_device)
			async with app.state.session_factory() as db_session:
				loaded = await rag_pipeline.load_index_from_db(db_session)
				if loaded == 0:
					await rag_pipeline.build_index_from_files_async()
			app.state.pipeline = rag_pipeline

		try:
			yield
		finally:
			if owned_engine is not None:
				await owned_engine.dispose()

	app = FastAPI(
		title="FastAPI Hybrid RAG Chat Service",
		version="1.0.0",
		lifespan=lifespan,
	)
	app.state.pipeline = pipeline
	app.state.session_factory = session_factory

	async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
		"""Provide an async SQLAlchemy session for the current request."""
		factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
		async with factory() as session:
			yield session

	def get_pipeline(request: Request) -> HybridRAGPipeline:
		"""Return the initialized HybridRAGPipeline instance."""
		return request.app.state.pipeline

	@app.get("/", response_class=HTMLResponse)
	async def index() -> HTMLResponse:
		"""Serve the single-page RAG chat interface."""
		return HTMLResponse(content=CHAT_HTML)

	@app.get("/.well-known/appspecific/com.chrome.devtools.json")
	async def chrome_devtools_manifest() -> dict:
		"""Return empty JSON to suppress Chrome DevTools 404 log noise."""
		return {}

	@app.get("/api/health")
	async def health(
		rag_pipeline: HybridRAGPipeline = Depends(get_pipeline),
	) -> dict:
		"""Return service health, active compute device, languages, and indexed chunk count."""
		return {
			"status": "ok",
			"device": config.resolve_device(),
			"indexed_chunks": rag_pipeline.get_total_indexed_chunks(),
			"languages": rag_pipeline.get_available_languages(),
			"default_language": config.default_language,
			"embedding_model": config.embedding_model_name,
			"reranker_model": config.reranker_model_name,
			"llm_model": config.llm_model,
		}

	@app.get("/api/languages", response_model=List[LanguageOut])
	async def list_languages(
		rag_pipeline: HybridRAGPipeline = Depends(get_pipeline),
	) -> List[LanguageOut]:
		"""List all 13 supported documentation language variations and their chunk counts."""
		return [
			LanguageOut(**item)
			for item in rag_pipeline.get_available_languages()
		]

	@app.post("/api/sessions", response_model=SessionSummaryOut)
	async def create_session(
		payload: SessionCreateRequest,
		db: AsyncSession = Depends(get_db),
	) -> SessionSummaryOut:
		"""Create a new chat session."""
		repo = ChatSessionRepository(db)
		chat_session = await repo.create_session(
			session_id=payload.session_id,
			title=payload.title,
			language=payload.language,
		)
		await db.commit()
		return SessionSummaryOut(
			id=chat_session.id,
			title=chat_session.title,
			language=chat_session.language,
			message_count=len(chat_session.messages),
			created_at=chat_session.created_at.isoformat(),
			updated_at=chat_session.updated_at.isoformat(),
		)

	@app.get("/api/sessions", response_model=List[SessionSummaryOut])
	async def list_sessions(
		db: AsyncSession = Depends(get_db),
	) -> List[SessionSummaryOut]:
		"""List all existing chat sessions ordered by most recently updated."""
		repo = ChatSessionRepository(db)
		sessions = await repo.list_sessions()
		return [
			SessionSummaryOut(
				id=s.id,
				title=s.title,
				language=s.language,
				message_count=len(s.messages),
				created_at=s.created_at.isoformat(),
				updated_at=s.updated_at.isoformat(),
			)
			for s in sessions
		]

	@app.get("/api/sessions/{session_id}", response_model=SessionDetailOut)
	async def get_session(
		session_id: str,
		db: AsyncSession = Depends(get_db),
	) -> SessionDetailOut:
		"""Retrieve a chat session along with its full conversation history."""
		repo = ChatSessionRepository(db)
		chat_session = await repo.get_session(session_id)
		if chat_session is None:
			raise HTTPException(status_code=404, detail="Chat session not found")
		return SessionDetailOut(
			id=chat_session.id,
			title=chat_session.title,
			language=chat_session.language,
			messages=[
				MessageOut(
					id=m.id,
					role=m.role,
					content=m.content,
					sources=m.sources or [],
					metrics=m.metrics or {},
					created_at=m.created_at.isoformat(),
				)
				for m in chat_session.messages
			],
			created_at=chat_session.created_at.isoformat(),
			updated_at=chat_session.updated_at.isoformat(),
		)

	@app.delete("/api/sessions/{session_id}")
	async def delete_session(
		session_id: str,
		db: AsyncSession = Depends(get_db),
	) -> dict:
		"""Delete a chat session and all its messages."""
		repo = ChatSessionRepository(db)
		deleted = await repo.delete_session(session_id)
		if not deleted:
			raise HTTPException(status_code=404, detail="Chat session not found")
		await db.commit()
		return {"deleted": True, "session_id": session_id}

	@app.post("/api/chat", response_model=ChatResponse)
	async def chat(
		payload: ChatRequest,
		db: AsyncSession = Depends(get_db),
		rag_pipeline: HybridRAGPipeline = Depends(get_pipeline),
	) -> ChatResponse:
		"""Run Hybrid RAG search + reranking + Gemini LLM generation inside a persistent chat session."""
		repo = ChatSessionRepository(db)
		chat_session = await repo.get_or_create_session(
			session_id=payload.session_id,
			language=payload.language,
		)

		result = await rag_pipeline.answer(
			query=payload.query,
			top_k=payload.top_k,
			rerank=payload.rerank,
			session=db,
			session_id=chat_session.id,
			language=payload.language or chat_session.language,
		)

		return ChatResponse(
			session_id=result["session_id"] or chat_session.id,
			language=result.get("language") or chat_session.language,
			query=result["query"],
			answer=result["answer"],
			sources=[SourceItem(**src) for src in result.get("sources", [])],
			metrics=result.get("metrics", {}),
			latency_seconds=result["latency_seconds"],
		)

	return app


app = create_app()

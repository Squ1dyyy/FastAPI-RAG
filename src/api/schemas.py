from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LanguageOut(BaseModel):
	"""Metadata and chunk count for a supported documentation language."""

	code: str
	label: str
	chunk_count: int = 0


class SessionCreateRequest(BaseModel):
	"""Request payload for creating a new chat session."""

	session_id: Optional[str] = Field(default=None)
	title: str = Field(default="New Chat", max_length=255)
	language: Optional[str] = Field(default=None, max_length=16)


class MessageOut(BaseModel):
	"""Serialized chat message inside a session."""

	id: int
	role: str
	content: str
	sources: List[Dict[str, Any]] = Field(default_factory=list)
	metrics: Dict[str, Any] = Field(default_factory=dict)
	created_at: str


class SessionSummaryOut(BaseModel):
	"""Summary item for listing chat sessions."""

	id: str
	title: str
	language: str = "ru"
	message_count: int
	created_at: str
	updated_at: str


class SessionDetailOut(BaseModel):
	"""Detailed chat session including full message history."""

	id: str
	title: str
	language: str = "ru"
	messages: List[MessageOut] = Field(default_factory=list)
	created_at: str
	updated_at: str


class ChatRequest(BaseModel):
	"""Request payload for asking a question in the RAG chat."""

	query: str = Field(..., min_length=1)
	session_id: Optional[str] = Field(default=None)
	language: Optional[str] = Field(default=None, max_length=16)
	top_k: Optional[int] = Field(default=None, ge=1, le=20)
	rerank: Optional[bool] = Field(default=True)


class SourceItem(BaseModel):
	"""Retrieved and reranked FastAPI reference documentation chunk metadata."""

	id: Optional[str] = None
	language: Optional[str] = None
	doc_title: Optional[str] = None
	section_title: Optional[str] = None
	breadcrumb: Optional[str] = None
	section_url: Optional[str] = None
	hybrid_score: Optional[float] = None
	rerank_score: Optional[float] = None


class ChatResponse(BaseModel):
	"""Response payload returned by the RAG chat endpoint."""

	session_id: str
	language: str = "ru"
	query: str
	answer: str
	sources: List[SourceItem] = Field(default_factory=list)
	metrics: Dict[str, Any] = Field(default_factory=dict)
	latency_seconds: float

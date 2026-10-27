import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from configs.config import config
from src.database.models.chat_session import ChatMessage, ChatSession


class ChatSessionRepository:
	"""Repository handling asynchronous CRUD operations for chat sessions and messages."""

	def __init__(self, session: AsyncSession):
		self.session = session

	async def get_session(self, session_id: str) -> Optional[ChatSession]:
		"""Fetch a chat session by ID with its messages eagerly loaded."""
		stmt = (
			select(ChatSession)
			.where(ChatSession.id == session_id)
			.options(selectinload(ChatSession.messages))
		)
		result = await self.session.execute(stmt)
		return result.scalar_one_or_none()

	async def create_session(
		self,
		session_id: Optional[str] = None,
		title: str = "New Chat",
		language: Optional[str] = None,
	) -> ChatSession:
		"""Create a new chat session and flush it to the database."""
		active_lang = (language or config.default_language or "ru").strip().lower()
		chat_session = ChatSession(
			id=session_id or str(uuid.uuid4()),
			title=title,
			language=active_lang,
			messages=[],
		)
		self.session.add(chat_session)
		await self.session.flush()
		return chat_session

	async def get_or_create_session(
		self,
		session_id: Optional[str] = None,
		title: str = "New Chat",
		language: Optional[str] = None,
	) -> ChatSession:
		"""Return an existing chat session by ID or create a new one if not found."""
		if session_id:
			existing = await self.get_session(session_id)
			if existing is not None:
				if language:
					existing.language = language.strip().lower()
					await self.session.flush()
				return existing
		return await self.create_session(
			session_id=session_id,
			title=title,
			language=language,
		)

	async def list_sessions(self, limit: int = 50) -> List[ChatSession]:
		"""Return the most recent chat sessions ordered by update time descending."""
		stmt = (
			select(ChatSession)
			.options(selectinload(ChatSession.messages))
			.order_by(desc(ChatSession.updated_at))
			.limit(limit)
		)
		result = await self.session.execute(stmt)
		return list(result.scalars().all())

	async def add_message(
		self,
		session_id: str,
		role: str,
		content: str,
		sources: Optional[List[Dict[str, Any]]] = None,
		metrics: Optional[Dict[str, Any]] = None,
	) -> ChatMessage:
		"""Add a message to a session, auto-updating session title on the first user message."""
		chat_session = await self.get_or_create_session(session_id=session_id)
		if role == "user" and chat_session.title == "New Chat" and content.strip():
			short_title = content.strip().replace("\n", " ")[:60]
			chat_session.title = short_title

		msg = ChatMessage(
			session_id=chat_session.id,
			role=role,
			content=content,
			sources=sources or [],
			metrics=metrics or {},
		)
		self.session.add(msg)
		await self.session.flush()
		return msg

	async def get_history(
		self,
		session_id: str,
		limit: Optional[int] = None,
	) -> List[Dict[str, str]]:
		"""Retrieve conversation history for a session formatted as OpenAI role/content dicts."""
		chat_session = await self.get_session(session_id)
		if chat_session is None or not chat_session.messages:
			return []

		msgs = list(chat_session.messages)
		if limit is not None and limit > 0:
			msgs = msgs[-limit:]

		return [{"role": m.role, "content": m.content} for m in msgs]

	async def delete_session(self, session_id: str) -> bool:
		"""Delete a chat session and its messages by session ID."""
		existing = await self.get_session(session_id)
		if existing is None:
			return False
		await self.session.execute(
			delete(ChatSession).where(ChatSession.id == session_id)
		)
		await self.session.flush()
		return True

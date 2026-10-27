import uuid
from typing import Any, Dict, List
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from .base import Base, TimestampMixin


class ChatSession(Base, TimestampMixin):
	"""Represents a persistent user chat session with conversation history."""

	__tablename__ = "chat_sessions"

	id: Mapped[str] = mapped_column(
		String(64),
		primary_key=True,
		default=lambda: str(uuid.uuid4()),
	)
	title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Chat")

	messages: Mapped[List["ChatMessage"]] = relationship(
		"ChatMessage",
		back_populates="session",
		cascade="all, delete-orphan",
		order_by="ChatMessage.id",
		lazy="selectin",
	)

	def __repr__(self) -> str:
		return f"<ChatSession(id='{self.id}', title='{self.title}', messages={len(self.messages)})>"


class ChatMessage(Base, TimestampMixin):
	"""Represents a single user or assistant message inside a chat session."""

	__tablename__ = "chat_messages"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	session_id: Mapped[str] = mapped_column(
		String(64),
		ForeignKey("chat_sessions.id", ondelete="CASCADE"),
		nullable=False,
		index=True,
	)
	role: Mapped[str] = mapped_column(String(32), nullable=False)
	content: Mapped[str] = mapped_column(Text, nullable=False)

	sources: Mapped[List[Dict[str, Any]]] = mapped_column(
		JSON().with_variant(JSONB(), "postgresql"),
		nullable=False,
		default=list,
	)
	metrics: Mapped[Dict[str, Any]] = mapped_column(
		JSON().with_variant(JSONB(), "postgresql"),
		nullable=False,
		default=dict,
	)

	session: Mapped["ChatSession"] = relationship(
		"ChatSession",
		back_populates="messages",
		lazy="selectin",
	)

	def __repr__(self) -> str:
		return f"<ChatMessage(id={self.id}, session_id='{self.session_id}', role='{self.role}')>"

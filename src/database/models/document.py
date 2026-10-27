from typing import Any, Dict, List, Optional
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class Document(Base, TimestampMixin):
	"""Represents a scraped reference documentation page."""

	__tablename__ = "documents"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	title: Mapped[str] = mapped_column(String(255), nullable=False)
	url: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False, index=True)
	file_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

	# Relation to chunks with selectin eager loading for async compatibility
	chunks: Mapped[List["DocumentChunk"]] = relationship(
		"DocumentChunk",
		back_populates="document",
		cascade="all, delete-orphan",
		order_by="DocumentChunk.chunk_index",
		lazy="selectin",
	)

	def __repr__(self) -> str:
		return f"<Document(id={self.id}, title='{self.title}', url='{self.url}')>"


class DocumentChunk(Base, TimestampMixin):
	"""Represents an individual text chunk extracted from a document for RAG."""

	__tablename__ = "document_chunks"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	document_id: Mapped[int] = mapped_column(
		Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
	)
	chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
	text: Mapped[str] = mapped_column(Text, nullable=False)

	# JSON / JSONB metadata (start/end offsets, tags, headers, etc.)
	metadata_json: Mapped[Dict[str, Any]] = mapped_column(
		JSON().with_variant(JSONB(), "postgresql"),
		nullable=False,
		default=dict,
	)

	# Relation to parent document
	document: Mapped["Document"] = relationship(
		"Document",
		back_populates="chunks",
		lazy="selectin",
	)

	def __repr__(self) -> str:
		return f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"

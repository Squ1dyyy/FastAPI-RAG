from typing import Any, Dict, Optional
from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class QueryLog(Base, TimestampMixin):
	"""Logs user queries, retrieved context, generated answers, and evaluation metrics."""

	__tablename__ = "query_logs"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	query: Mapped[str] = mapped_column(Text, nullable=False, index=True)
	response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

	retriever_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

	retrieved_chunks: Mapped[Dict[str, Any]] = mapped_column(
		JSON().with_variant(JSONB(), "postgresql"),
		nullable=False,
		default=dict,
	)

	evaluation_metrics: Mapped[Dict[str, Any]] = mapped_column(
		JSON().with_variant(JSONB(), "postgresql"),
		nullable=False,
		default=dict,
	)

	latency_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

	def __repr__(self) -> str:
		return f"<QueryLog(id={self.id}, query='{self.query[:30]}...', retriever='{self.retriever_used}')>"

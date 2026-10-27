from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.query_log import QueryLog


class QueryLogRepository:
	"""Repository handling persistence of RAG queries, responses, and evaluation logs."""

	def __init__(self, session: AsyncSession):
		self.session = session

	async def log_query(
		self,
		query: str,
		response: Optional[str] = None,
		retriever_used: Optional[str] = "hybrid_rrf",
		retrieved_chunks: Optional[Dict[str, Any]] = None,
		evaluation_metrics: Optional[Dict[str, Any]] = None,
		latency_seconds: Optional[float] = None,
	) -> QueryLog:
		"""Save a user query and RAG pipeline outputs to PostgreSQL."""
		log_entry = QueryLog(
			query=query,
			response=response,
			retriever_used=retriever_used,
			retrieved_chunks=retrieved_chunks or {},
			evaluation_metrics=evaluation_metrics or {},
			latency_seconds=latency_seconds,
		)
		self.session.add(log_entry)
		await self.session.flush()
		return log_entry

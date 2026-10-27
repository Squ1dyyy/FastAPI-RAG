from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseReranker(ABC):
	"""Abstract base class for reranking retrieved documents."""

	@abstractmethod
	def rerank(
		self, query: str, documents: List[Dict[str, Any]], top_k: int = 3
	) -> List[Dict[str, Any]]:
		"""Rerank candidates based on cross-encoder similarity with query."""
		pass

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRetriever(ABC):
	"""Abstract base class for document retrieval systems."""

	@abstractmethod
	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Retrieve top_k most relevant document chunks for the given query.

		:param query: Search query string.
		:param top_k: Maximum number of results to return.
		:return: List of candidate dictionaries with text, score, and metadata.
		"""
		pass

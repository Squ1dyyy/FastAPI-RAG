import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRetriever(ABC):
	"""Abstract base class for synchronous and asynchronous document retrieval systems."""

	@abstractmethod
	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Retrieve top_k most relevant document chunks for the given query."""
		pass

	async def retrieve_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Asynchronously retrieve top_k most relevant document chunks."""
		return await asyncio.to_thread(self.retrieve, query, top_k)

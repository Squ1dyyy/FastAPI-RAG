from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
	"""Abstract base class for document and query embedding models."""

	@abstractmethod
	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		"""Embed a list of text documents into dense vectors."""
		pass

	@abstractmethod
	def embed_query(self, text: str) -> List[float]:
		"""Embed a single search query into a dense vector."""
		pass

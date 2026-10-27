from abc import ABC, abstractmethod
from typing import List, Optional

from configs.config import config


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


class MultilingualE5Embedder(BaseEmbedder):
	"""Dense embedder powered by `intfloat/multilingual-e5-small` (384-d).

	Automatically applies E5 instruction prefixes:
	- `"passage: "` for indexed document chunks
	- `"query: "` for search queries
	"""

	def __init__(
		self,
		model_name: Optional[str] = None,
		batch_size: Optional[int] = None,
		device: Optional[str] = None,
	):
		self.model_name = model_name or config.embedding_model_name
		self.batch_size = batch_size or config.embedding_batch_size
		self.device = device
		self._model = None

	@property
	def model(self):
		"""Lazy-load SentenceTransformer model on first use."""
		if self._model is None:
			from sentence_transformers import SentenceTransformer

			self._model = SentenceTransformer(self.model_name, device=self.device)
		return self._model

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		"""Encode document chunks with 'passage: ' prefix and L2 normalization."""
		if not texts:
			return []

		prefixed = [
			t if t.startswith("passage: ") else f"passage: {t}"
			for t in texts
		]
		embeddings = self.model.encode(
			prefixed,
			batch_size=self.batch_size,
			normalize_embeddings=True,
			show_progress_bar=False,
		)
		return embeddings.tolist()

	def embed_query(self, text: str) -> List[float]:
		"""Encode search query with 'query: ' prefix and L2 normalization."""
		prefixed = text if text.startswith("query: ") else f"query: {text}"
		embedding = self.model.encode(
			prefixed,
			normalize_embeddings=True,
			show_progress_bar=False,
		)
		return embedding.tolist()

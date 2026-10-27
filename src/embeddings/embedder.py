import asyncio
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

	async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
		"""Asynchronously embed a list of text documents."""
		return await asyncio.to_thread(self.embed_documents, texts)

	async def embed_query_async(self, text: str) -> List[float]:
		"""Asynchronously embed a single search query."""
		return await asyncio.to_thread(self.embed_query, text)


class MultilingualE5Embedder(BaseEmbedder):
	"""GPU-accelerated dense embedder powered by `intfloat/multilingual-e5-small`."""

	def __init__(
		self,
		model_name: Optional[str] = None,
		batch_size: Optional[int] = None,
		device: Optional[str] = None,
		use_fp16: Optional[bool] = None,
		passage_prefix: Optional[str] = None,
		query_prefix: Optional[str] = None,
	):
		self.model_name = model_name or config.embedding_model_name
		self.batch_size = batch_size or config.embedding_batch_size
		self.device = device or config.resolve_device()
		self.use_fp16 = use_fp16 if use_fp16 is not None else config.use_fp16
		self.passage_prefix = (
			passage_prefix if passage_prefix is not None else config.embedding_passage_prefix
		)
		self.query_prefix = (
			query_prefix if query_prefix is not None else config.embedding_query_prefix
		)
		self._model = None

	@property
	def model(self):
		"""Lazy-load SentenceTransformer model onto target device (CUDA GPU if available)."""
		if self._model is None:
			import torch
			from sentence_transformers import SentenceTransformer

			target_device = self.device
			if target_device.startswith("cuda") and not torch.cuda.is_available():
				target_device = "cpu"

			model_kwargs = {}
			if target_device.startswith("cuda") and self.use_fp16:
				model_kwargs["torch_dtype"] = torch.float16

			self._model = SentenceTransformer(
				self.model_name,
				device=target_device,
				model_kwargs=model_kwargs,
			)
		return self._model

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		"""Encode document chunks on GPU/CPU with passage prefix and L2 normalization."""
		if not texts:
			return []

		prefixed = [
			t if t.startswith(self.passage_prefix) else f"{self.passage_prefix}{t}"
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
		"""Encode search query on GPU/CPU with query prefix and L2 normalization."""
		prefixed = text if text.startswith(self.query_prefix) else f"{self.query_prefix}{text}"
		embedding = self.model.encode(
			prefixed,
			normalize_embeddings=True,
			show_progress_bar=False,
		)
		return embedding.tolist()

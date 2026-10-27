import asyncio
import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.embeddings.embedder import BaseEmbedder, MultilingualE5Embedder
from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
	"""GPU-accelerated dense retriever using `multilingual-e5-small` and cosine similarity."""

	def __init__(self, embedder: Optional[BaseEmbedder] = None):
		self.embedder = embedder or MultilingualE5Embedder()
		self.corpus: List[Dict[str, Any]] = []
		self.embeddings: List[List[float]] = []
		self._matrix: Optional[np.ndarray] = None

	@staticmethod
	def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
		"""Compute cosine similarity between two vectors."""
		dot = sum(a * b for a, b in zip(vec_a, vec_b))
		norm_a = math.sqrt(sum(a * a for a in vec_a))
		norm_b = math.sqrt(sum(b * b for b in vec_b))
		if norm_a == 0.0 or norm_b == 0.0:
			return 0.0
		return dot / (norm_a * norm_b)

	def _build_matrix(self):
		"""Normalize and store numpy embedding matrix."""
		mat = np.asarray(self.embeddings, dtype=np.float32)
		norms = np.linalg.norm(mat, axis=1, keepdims=True)
		norms[norms == 0.0] = 1.0
		self._matrix = mat / norms

	def index(self, corpus: List[Dict[str, Any]]):
		"""Build dense vector index from corpus synchronously."""
		self.corpus = corpus
		self.embeddings = []
		self._matrix = None

		if not corpus:
			return

		missing_indices = [
			i for i, doc in enumerate(corpus)
			if not doc.get("embedding")
		]

		if missing_indices:
			texts_to_embed = [corpus[i].get("text", "") for i in missing_indices]
			computed = self.embedder.embed_documents(texts_to_embed)
			for idx, emb in zip(missing_indices, computed):
				corpus[idx]["embedding"] = emb

		self.embeddings = [doc["embedding"] for doc in corpus]
		self._build_matrix()

	async def index_async(self, corpus: List[Dict[str, Any]]):
		"""Asynchronously build dense vector index from corpus."""
		self.corpus = corpus
		self.embeddings = []
		self._matrix = None

		if not corpus:
			return

		missing_indices = [
			i for i, doc in enumerate(corpus)
			if not doc.get("embedding")
		]

		if missing_indices:
			texts_to_embed = [corpus[i].get("text", "") for i in missing_indices]
			computed = await self.embedder.embed_documents_async(texts_to_embed)
			for idx, emb in zip(missing_indices, computed):
				corpus[idx]["embedding"] = emb

		self.embeddings = [doc["embedding"] for doc in corpus]
		await asyncio.to_thread(self._build_matrix)

	def _score_query_vector(self, query_vec_list: List[float], top_k: int) -> List[Dict[str, Any]]:
		"""Score normalized query vector against document matrix."""
		if not self.corpus or self._matrix is None:
			return []

		query_vec = np.asarray(query_vec_list, dtype=np.float32)
		q_norm = np.linalg.norm(query_vec)
		if q_norm > 0.0:
			query_vec = query_vec / q_norm

		scores = np.dot(self._matrix, query_vec)
		k = min(top_k, len(self.corpus))
		top_indices = np.argsort(scores)[::-1][:k]

		results = []
		for doc_idx in top_indices:
			idx = int(doc_idx)
			doc = dict(self.corpus[idx])
			doc["score"] = float(scores[idx])
			doc["retriever"] = "dense_e5"
			results.append(doc)

		return results

	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Retrieve top_k documents with highest semantic cosine similarity."""
		if not self.corpus or self._matrix is None:
			return []
		query_vec = self.embedder.embed_query(query)
		return self._score_query_vector(query_vec, top_k=top_k)

	async def retrieve_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Asynchronously retrieve top_k documents with highest semantic similarity."""
		if not self.corpus or self._matrix is None:
			return []
		query_vec = await self.embedder.embed_query_async(query)
		return await asyncio.to_thread(self._score_query_vector, query_vec, top_k)

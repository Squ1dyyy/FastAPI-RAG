import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.embeddings.embedder import BaseEmbedder, MultilingualE5Embedder
from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
	"""Vector-based dense retriever using `multilingual-e5-small` and cosine similarity."""

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

	def index(self, corpus: List[Dict[str, Any]]):
		"""Build dense vector index from corpus (reusing precomputed embeddings if present).

		:param corpus: List of document chunk dicts containing 'text' and optional 'embedding'.
		"""
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

		mat = np.asarray(self.embeddings, dtype=np.float32)
		norms = np.linalg.norm(mat, axis=1, keepdims=True)
		norms[norms == 0.0] = 1.0
		self._matrix = mat / norms

	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Retrieve top_k documents with highest semantic cosine similarity."""
		if not self.corpus or self._matrix is None:
			return []

		query_vec = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
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

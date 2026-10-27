import math
from typing import Any, Dict, List, Optional

from src.embeddings.embedder import BaseEmbedder
from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
	"""Vector-based dense retriever using dense embeddings and cosine similarity."""

	def __init__(self, embedder: BaseEmbedder):
		self.embedder = embedder
		self.corpus: List[Dict[str, Any]] = []
		self.embeddings: List[List[float]] = []

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
		"""Embed documents and store them in the index.

		:param corpus: List of document dicts containing a 'text' field.
		"""
		self.corpus = corpus
		texts = [doc.get("text", "") for doc in corpus]
		self.embeddings = self.embedder.embed_documents(texts)

	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Retrieve top_k documents with highest cosine similarity."""
		if not self.corpus or not self.embeddings:
			return []

		query_vec = self.embedder.embed_query(query)
		scored_docs = []

		for doc_idx, doc_vec in enumerate(self.embeddings):
			sim = self._cosine_similarity(query_vec, doc_vec)
			scored_docs.append((doc_idx, sim))

		sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]

		results = []
		for doc_idx, score in sorted_docs:
			doc = dict(self.corpus[doc_idx])
			doc["score"] = score
			doc["retriever"] = "dense"
			results.append(doc)

		return results

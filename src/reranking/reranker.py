from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from configs.config import config


class BaseReranker(ABC):
	"""Abstract base class for reranking retrieved documents."""

	@abstractmethod
	def rerank(
		self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
	) -> List[Dict[str, Any]]:
		"""Rerank candidates based on cross-encoder similarity with query."""
		pass


class CrossEncoderReranker(BaseReranker):
	"""Ultra-lightweight Cross-Encoder reranker (`cross-encoder/ms-marco-TinyBERT-L-2-v2`).

	Scores `(query, passage)` pairs jointly and re-orders hybrid search candidates.
	"""

	def __init__(
		self,
		model_name: Optional[str] = None,
		device: Optional[str] = None,
	):
		self.model_name = model_name or config.reranker_model_name
		self.device = device
		self._model = None

	@property
	def model(self):
		"""Lazy-load CrossEncoder model on first use."""
		if self._model is None:
			from sentence_transformers import CrossEncoder

			self._model = CrossEncoder(self.model_name, device=self.device)
		return self._model

	def rerank(
		self,
		query: str,
		documents: List[Dict[str, Any]],
		top_k: int = 5,
	) -> List[Dict[str, Any]]:
		"""Score and sort candidate documents for the query."""
		if not documents:
			return []

		pairs = [(query, doc.get("text", "")) for doc in documents]
		scores = self.model.predict(pairs, show_progress_bar=False)

		reranked = []
		for idx, (doc, score) in enumerate(zip(documents, scores), start=1):
			item = dict(doc)
			item["pre_rerank_rank"] = idx
			item["rerank_score"] = float(score)
			item["score"] = float(score)
			reranked.append(item)

		reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
		return reranked[:top_k]

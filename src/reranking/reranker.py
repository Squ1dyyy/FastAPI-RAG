import asyncio
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

	async def rerank_async(
		self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
	) -> List[Dict[str, Any]]:
		"""Asynchronously rerank candidate documents."""
		return await asyncio.to_thread(self.rerank, query, documents, top_k)


class CrossEncoderReranker(BaseReranker):
	"""GPU-accelerated lightweight Cross-Encoder reranker (`cross-encoder/ms-marco-TinyBERT-L-2-v2`)."""

	def __init__(
		self,
		model_name: Optional[str] = None,
		batch_size: Optional[int] = None,
		device: Optional[str] = None,
		use_fp16: Optional[bool] = None,
	):
		self.model_name = model_name or config.reranker_model_name
		self.batch_size = batch_size or config.reranker_batch_size
		self.device = device or config.resolve_device()
		self.use_fp16 = use_fp16 if use_fp16 is not None else config.use_fp16
		self._model = None

	@property
	def model(self):
		"""Lazy-load CrossEncoder model onto target device (CUDA GPU if available)."""
		if self._model is None:
			import torch
			from sentence_transformers import CrossEncoder

			target_device = self.device
			if target_device.startswith("cuda") and not torch.cuda.is_available():
				target_device = "cpu"

			model_kwargs = {}
			if target_device.startswith("cuda") and self.use_fp16:
				model_kwargs["torch_dtype"] = torch.float16

			self._model = CrossEncoder(
				self.model_name,
				device=target_device,
				model_kwargs=model_kwargs,
			)
		return self._model

	def rerank(
		self,
		query: str,
		documents: List[Dict[str, Any]],
		top_k: int = 5,
	) -> List[Dict[str, Any]]:
		"""Score and sort candidate documents on GPU/CPU for the query."""
		if not documents:
			return []

		pairs = [(query, doc.get("text", "")) for doc in documents]
		scores = [
			float(s)
			for s in self.model.predict(
				pairs,
				batch_size=self.batch_size,
				show_progress_bar=False,
			)
		]

		max_ce = max(scores) if scores else 0.0
		low_ce_confidence = max_ce < -3.0

		reranked = []
		for idx, (doc, score) in enumerate(zip(documents, scores), start=1):
			item = dict(doc)
			item["pre_rerank_rank"] = idx
			item["rerank_score"] = score
			if low_ce_confidence:
				hybrid_prior = float(doc.get("hybrid_score") or doc.get("score") or (1.0 / idx))
				item["score"] = hybrid_prior + (score * 1e-4)
			else:
				item["score"] = score
			reranked.append(item)

		reranked.sort(key=lambda x: x["score"], reverse=True)
		return reranked[:top_k]

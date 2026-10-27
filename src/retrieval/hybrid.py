import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional

from configs.config import config
from src.retrieval.base import BaseRetriever


class HybridRetriever(BaseRetriever):
	"""Hybrid retriever combining Dense (E5 GPU) and Sparse (BM25-Fast) retrieval."""

	def __init__(
		self,
		dense_retriever: BaseRetriever,
		sparse_retriever: BaseRetriever,
		alpha: Optional[float] = None,
		fusion_method: Optional[str] = None,
		rrf_k: Optional[int] = None,
	):
		self.dense_retriever = dense_retriever
		self.sparse_retriever = sparse_retriever
		self.alpha = alpha if alpha is not None else config.hybrid_alpha
		self.fusion_method = fusion_method or config.hybrid_fusion_method
		self.rrf_k = rrf_k if rrf_k is not None else config.rrf_k

	def index(self, corpus: List[Dict[str, Any]]):
		"""Build indices for both sparse (BM25-Fast) and dense retrievers."""
		if hasattr(self.sparse_retriever, "index"):
			self.sparse_retriever.index(corpus)
		if hasattr(self.dense_retriever, "index"):
			self.dense_retriever.index(corpus)

	async def index_async(self, corpus: List[Dict[str, Any]]):
		"""Asynchronously build indices for both sparse and dense retrievers concurrently."""
		tasks = []
		if hasattr(self.sparse_retriever, "index_async"):
			tasks.append(self.sparse_retriever.index_async(corpus))
		elif hasattr(self.sparse_retriever, "index"):
			tasks.append(asyncio.to_thread(self.sparse_retriever.index, corpus))

		if hasattr(self.dense_retriever, "index_async"):
			tasks.append(self.dense_retriever.index_async(corpus))
		elif hasattr(self.dense_retriever, "index"):
			tasks.append(asyncio.to_thread(self.dense_retriever.index, corpus))

		if tasks:
			await asyncio.gather(*tasks)

	def _rrf_fusion(
		self,
		dense_results: List[Dict[str, Any]],
		sparse_results: List[Dict[str, Any]],
		top_k: int,
	) -> List[Dict[str, Any]]:
		"""Fuse results using Reciprocal Rank Fusion (RRF)."""
		doc_store: Dict[str, Dict[str, Any]] = {}
		rrf_scores: Dict[str, float] = defaultdict(float)
		dense_ranks: Dict[str, int] = {}
		sparse_ranks: Dict[str, int] = {}
		dense_scores: Dict[str, float] = {}
		sparse_scores: Dict[str, float] = {}

		dense_weight = self.alpha
		sparse_weight = 1.0 - self.alpha

		for rank, doc in enumerate(dense_results, start=1):
			doc_id = doc.get("id") or doc.get("text", "")
			doc_store[doc_id] = doc
			dense_ranks[doc_id] = rank
			dense_scores[doc_id] = doc.get("score", 0.0)
			rrf_scores[doc_id] += dense_weight * (1.0 / (self.rrf_k + rank))

		for rank, doc in enumerate(sparse_results, start=1):
			doc_id = doc.get("id") or doc.get("text", "")
			doc_store[doc_id] = doc
			sparse_ranks[doc_id] = rank
			sparse_scores[doc_id] = doc.get("score", 0.0)
			rrf_scores[doc_id] += sparse_weight * (1.0 / (self.rrf_k + rank))

		sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

		results = []
		for doc_id, hybrid_score in sorted_items:
			item = dict(doc_store[doc_id])
			item["score"] = hybrid_score
			item["hybrid_score"] = hybrid_score
			item["dense_score"] = dense_scores.get(doc_id)
			item["sparse_score"] = sparse_scores.get(doc_id)
			item["dense_rank"] = dense_ranks.get(doc_id)
			item["sparse_rank"] = sparse_ranks.get(doc_id)
			item["retriever"] = "hybrid_rrf"
			results.append(item)

		return results

	def _linear_fusion(
		self,
		dense_results: List[Dict[str, Any]],
		sparse_results: List[Dict[str, Any]],
		top_k: int,
	) -> List[Dict[str, Any]]:
		"""Fuse results using min-max normalized weighted linear combination."""
		doc_store: Dict[str, Dict[str, Any]] = {}

		def normalize(results: List[Dict[str, Any]]) -> Dict[str, float]:
			if not results:
				return {}
			scores = [doc.get("score", 0.0) for doc in results]
			min_s, max_s = min(scores), max(scores)
			diff = max_s - min_s
			normalized = {}
			for doc in results:
				doc_id = doc.get("id") or doc.get("text", "")
				doc_store[doc_id] = doc
				raw = doc.get("score", 0.0)
				normalized[doc_id] = (raw - min_s) / diff if diff > 0 else 1.0
			return normalized

		norm_dense = normalize(dense_results)
		norm_sparse = normalize(sparse_results)

		all_doc_ids = set(norm_dense.keys()).union(set(norm_sparse.keys()))
		combined: Dict[str, float] = {}

		for doc_id in all_doc_ids:
			d_score = norm_dense.get(doc_id, 0.0)
			s_score = norm_sparse.get(doc_id, 0.0)
			combined[doc_id] = (self.alpha * d_score) + ((1.0 - self.alpha) * s_score)

		sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

		results = []
		for doc_id, final_score in sorted_items:
			item = dict(doc_store[doc_id])
			item["score"] = final_score
			item["hybrid_score"] = final_score
			item["norm_dense_score"] = norm_dense.get(doc_id, 0.0)
			item["norm_sparse_score"] = norm_sparse.get(doc_id, 0.0)
			item["retriever"] = "hybrid_linear"
			results.append(item)

		return results

	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Synchronously retrieve and fuse candidate documents from both retrievers."""
		candidate_k = max(top_k * 2, config.retrieval_candidate_k)

		dense_candidates = self.dense_retriever.retrieve(query, top_k=candidate_k)
		sparse_candidates = self.sparse_retriever.retrieve(query, top_k=candidate_k)

		if self.fusion_method == "linear":
			return self._linear_fusion(dense_candidates, sparse_candidates, top_k=top_k)
		return self._rrf_fusion(dense_candidates, sparse_candidates, top_k=top_k)

	async def retrieve_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Asynchronously retrieve and fuse candidate documents concurrently via asyncio.gather."""
		candidate_k = max(top_k * 2, config.retrieval_candidate_k)

		dense_candidates, sparse_candidates = await asyncio.gather(
			self.dense_retriever.retrieve_async(query, top_k=candidate_k),
			self.sparse_retriever.retrieve_async(query, top_k=candidate_k),
		)

		if self.fusion_method == "linear":
			return await asyncio.to_thread(
				self._linear_fusion, dense_candidates, sparse_candidates, top_k
			)
		return await asyncio.to_thread(
			self._rrf_fusion, dense_candidates, sparse_candidates, top_k
		)

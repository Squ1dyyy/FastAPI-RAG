import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEvaluator(ABC):
	"""Abstract base class for asynchronous RAG pipeline evaluation metrics."""

	@abstractmethod
	async def evaluate(
		self,
		query: str,
		response: str,
		context: List[Dict[str, Any]],
		ground_truth: Optional[str] = None,
	) -> Dict[str, float]:
		"""Calculate evaluation metrics asynchronously."""
		pass


class RAGEvaluator(BaseEvaluator):
	"""Asynchronous evaluator computing context coverage, faithfulness overlap, and F1."""

	@staticmethod
	def _tokenize(text: str) -> List[str]:
		return [t.lower() for t in re.findall(r"\b\w+\b", text) if len(t) > 2]

	def _compute_metrics(
		self,
		query: str,
		response: str,
		context: List[Dict[str, Any]],
		ground_truth: Optional[str] = None,
	) -> Dict[str, float]:
		q_tokens = set(self._tokenize(query))
		resp_tokens = self._tokenize(response)
		ctx_text = " ".join(c.get("text", "") for c in context)
		ctx_tokens = set(self._tokenize(ctx_text))

		context_coverage = (
			len(q_tokens.intersection(ctx_tokens)) / len(q_tokens)
			if q_tokens
			else 0.0
		)

		faithfulness = (
			sum(1 for t in resp_tokens if t in ctx_tokens) / len(resp_tokens)
			if resp_tokens
			else 0.0
		)

		has_citation = 0.0
		for c in context:
			sec_url = c.get("section_url") or c.get("metadata", {}).get("section_url")
			sec_title = c.get("section_title") or c.get("metadata", {}).get("section_title")
			if (sec_url and sec_url in response) or (sec_title and sec_title in response):
				has_citation = 1.0
				break

		metrics = {
			"context_coverage": round(context_coverage, 4),
			"faithfulness_overlap": round(faithfulness, 4),
			"citation_present": has_citation,
		}

		if ground_truth:
			gt_tokens = set(self._tokenize(ground_truth))
			resp_set = set(resp_tokens)
			overlap = len(gt_tokens.intersection(resp_set))
			precision = overlap / len(resp_set) if resp_set else 0.0
			recall = overlap / len(gt_tokens) if gt_tokens else 0.0
			f1 = (
				(2 * precision * recall) / (precision + recall)
				if (precision + recall) > 0
				else 0.0
			)
			metrics["ground_truth_f1"] = round(f1, 4)

		return metrics

	async def evaluate(
		self,
		query: str,
		response: str,
		context: List[Dict[str, Any]],
		ground_truth: Optional[str] = None,
	) -> Dict[str, float]:
		"""Asynchronously evaluate RAG response and retrieved context."""
		return await asyncio.to_thread(
			self._compute_metrics, query, response, context, ground_truth
		)

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEvaluator(ABC):
	"""Abstract base class for RAG pipeline evaluation metrics."""

	@abstractmethod
	def evaluate(
		self, query: str, response: str, ground_truth: str, context: List[Dict[str, Any]]
	) -> Dict[str, float]:
		"""Calculate evaluation metrics (e.g. faithfulness, context relevance)."""
		pass

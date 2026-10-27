from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseGenerator(ABC):
	"""Abstract base class for LLM response generation."""

	@abstractmethod
	async def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
		"""Generate a synthesized response given query and retrieved context."""
		pass

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from curl_cffi import AsyncSession

from configs.config import config


class BaseGenerator(ABC):
	"""Abstract base class for asynchronous LLM response generation."""

	@abstractmethod
	async def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
		"""Generate a synthesized response given query and retrieved context."""
		pass


class OpenRouterGenerator(BaseGenerator):
	"""Asynchronous RAG response generator powered by the OpenRouter Chat Completions API."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		model: Optional[str] = None,
		temperature: Optional[float] = None,
		max_tokens: Optional[int] = None,
		timeout: Optional[float] = None,
		site_url: Optional[str] = None,
		app_name: Optional[str] = None,
		system_prompt: Optional[str] = None,
	):
		self.api_key = api_key if api_key is not None else config.openrouter_api_key
		self.base_url = (base_url or config.openrouter_base_url).rstrip("/")
		self.model = model or config.openrouter_model
		self.temperature = (
			temperature if temperature is not None else config.openrouter_temperature
		)
		self.max_tokens = max_tokens if max_tokens is not None else config.openrouter_max_tokens
		self.timeout = timeout if timeout is not None else config.openrouter_timeout
		self.site_url = site_url or config.openrouter_site_url
		self.app_name = app_name or config.openrouter_app_name
		self.system_prompt = system_prompt or config.rag_system_prompt

	@staticmethod
	def format_context(context: List[Dict[str, Any]]) -> str:
		"""Format retrieved documentation chunks into a structured context block."""
		if not context:
			return "No relevant documentation context found."

		blocks = []
		for idx, chunk in enumerate(context, start=1):
			meta = chunk.get("metadata", {})
			breadcrumb = (
				chunk.get("breadcrumb")
				or meta.get("breadcrumb")
				or chunk.get("section_title")
				or "FastAPI Reference"
			)
			sec_url = (
				chunk.get("section_url")
				or meta.get("section_url")
				or chunk.get("doc_url")
				or ""
			)
			sym_type = chunk.get("symbol_type") or meta.get("symbol_type") or "section"
			text = chunk.get("text", "").strip()
			blocks.append(
				f"--- Source [{idx}] ---\n"
				f"Section: {breadcrumb} ({sym_type})\n"
				f"URL: {sec_url}\n"
				f"Content:\n{text}"
			)
		return "\n\n".join(blocks)

	def build_messages(
		self, query: str, context: List[Dict[str, Any]]
	) -> List[Dict[str, str]]:
		"""Construct system and user messages for the OpenRouter Chat Completions API."""
		formatted_context = self.format_context(context)
		user_prompt = (
			f"FastAPI Reference Context:\n{formatted_context}\n\n"
			f"User Question:\n{query}\n\n"
			"Provide a clear, accurate answer based on the context above and include "
			"the relevant documentation URLs."
		)
		return [
			{"role": "system", "content": self.system_prompt},
			{"role": "user", "content": user_prompt},
		]

	def _build_headers(self) -> Dict[str, str]:
		"""Construct HTTP headers required by OpenRouter."""
		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}
		if self.site_url:
			headers["HTTP-Referer"] = self.site_url
		if self.app_name:
			headers["X-Title"] = self.app_name
		return headers

	async def generate(
		self,
		query: str,
		context: List[Dict[str, Any]],
		session: Optional[AsyncSession] = None,
	) -> str:
		"""Asynchronously generate an answer from OpenRouter using retrieved context."""
		if not self.api_key or self.api_key == "your_openrouter_api_key_here":
			raise ValueError(
				"OPENROUTER_API_KEY is not configured in .env. "
				"Set a valid OpenRouter API key to generate LLM responses."
			)

		endpoint = f"{self.base_url}/chat/completions"
		payload = {
			"model": self.model,
			"messages": self.build_messages(query, context),
			"temperature": self.temperature,
			"max_tokens": self.max_tokens,
		}
		headers = self._build_headers()

		if session is not None:
			res = await session.post(
				endpoint,
				json=payload,
				headers=headers,
				timeout=self.timeout,
			)
			res.raise_for_status()
			data = res.json()
			return data["choices"][0]["message"]["content"].strip()

		async with AsyncSession() as local_session:
			res = await local_session.post(
				endpoint,
				json=payload,
				headers=headers,
				timeout=self.timeout,
			)
			res.raise_for_status()
			data = res.json()
			return data["choices"][0]["message"]["content"].strip()

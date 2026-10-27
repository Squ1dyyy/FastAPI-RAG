from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI

from configs.config import config


class BaseGenerator(ABC):
	"""Abstract base class for asynchronous LLM response generation."""

	@abstractmethod
	async def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
		"""Generate a synthesized response given query and retrieved context."""
		pass


class LLMGenerator(BaseGenerator):
	"""Asynchronous RAG response generator using `AsyncOpenAI` for any OpenAI-compatible provider."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		model: Optional[str] = None,
		temperature: Optional[float] = None,
		max_tokens: Optional[int] = None,
		timeout: Optional[float] = None,
		system_prompt: Optional[str] = None,
	):
		self.api_key = api_key if api_key is not None else config.llm_api_key
		self.base_url = (base_url or config.llm_base_url).rstrip("/")
		self.model = model or config.llm_model
		self.temperature = (
			temperature if temperature is not None else config.llm_temperature
		)
		self.max_tokens = max_tokens if max_tokens is not None else config.llm_max_tokens
		self.timeout = timeout if timeout is not None else config.llm_timeout
		self.system_prompt = system_prompt or config.rag_system_prompt
		self._client: Optional[AsyncOpenAI] = None

	@property
	def client(self) -> AsyncOpenAI:
		"""Lazy-initialize AsyncOpenAI client."""
		if self._client is None:
			if not self.api_key or self.api_key == "your_api_key_here":
				raise ValueError(
					"LLM_API_KEY is not configured in .env. "
					"Set a valid API key for your LLM provider."
				)
			self._client = AsyncOpenAI(
				api_key=self.api_key,
				base_url=self.base_url,
				timeout=self.timeout,
			)
		return self._client

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
		"""Construct system and user messages for chat completion."""
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

	async def generate(
		self,
		query: str,
		context: List[Dict[str, Any]],
	) -> str:
		"""Asynchronously generate an answer via AsyncOpenAI chat completions."""
		response = await self.client.chat.completions.create(
			model=self.model,
			messages=self.build_messages(query, context),
			temperature=self.temperature,
			max_tokens=self.max_tokens,
		)
		content = response.choices[0].message.content
		return content.strip() if content else ""

	async def generate_stream(
		self,
		query: str,
		context: List[Dict[str, Any]],
	) -> AsyncGenerator[str, None]:
		"""Asynchronously stream response tokens via AsyncOpenAI chat completions."""
		stream = await self.client.chat.completions.create(
			model=self.model,
			messages=self.build_messages(query, context),
			temperature=self.temperature,
			max_tokens=self.max_tokens,
			stream=True,
		)
		async for chunk in stream:
			if chunk.choices and chunk.choices[0].delta.content:
				yield chunk.choices[0].delta.content


OpenAIGenerator = LLMGenerator

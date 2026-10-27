import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI

from configs.config import Config, config
from src.grabber import LANGUAGE_LABELS


class BaseGenerator(ABC):
	"""Abstract base class for asynchronous LLM response generation."""

	@abstractmethod
	async def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
		"""Generate a synthesized response given query and retrieved context."""
		pass

	async def generate_with_history(
		self,
		query: str,
		context: List[Dict[str, Any]],
		history: Optional[List[Dict[str, str]]] = None,
		language: Optional[str] = None,
	) -> str:
		"""Generate a response taking optional multi-turn chat session history into account."""
		return await self.generate(query=query, context=context)


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
		self._explicit_api_key = api_key
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

	def has_valid_api_key(self) -> bool:
		"""Return True when a non-placeholder LLM API key is configured."""
		if self._explicit_api_key is None and (
			not self.api_key or self.api_key == "your_api_key_here"
		):
			fresh_cfg = Config()
			if fresh_cfg.llm_api_key and fresh_cfg.llm_api_key != "your_api_key_here":
				self.api_key = fresh_cfg.llm_api_key
				self.base_url = fresh_cfg.llm_base_url.rstrip("/")
				self.model = fresh_cfg.llm_model
				self._client = None
		return bool(self.api_key and self.api_key != "your_api_key_here")

	@property
	def client(self) -> AsyncOpenAI:
		"""Lazy-initialize AsyncOpenAI client."""
		if self._client is None:
			if not self.has_valid_api_key():
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
	def clean_chunk_text(raw_text: str) -> str:
		"""Strip internal indexing prefixes and source-code footer tags from chunk text."""
		lines = raw_text.strip().splitlines()
		cleaned: List[str] = []
		for idx, line in enumerate(lines):
			stripped = line.strip()
			if stripped.startswith("[") and stripped.endswith("]") and (idx == 0 or " > " in stripped):
				continue
			if stripped.startswith("[Source code in ") and stripped.endswith("]"):
				continue
			if re.match(r"^#{1,6}\s+", stripped):
				heading_text = re.sub(r"^#{1,6}\s+", "", stripped)
				cleaned.append(f"**{heading_text}**")
			else:
				cleaned.append(line)
		return "\n".join(cleaned).strip()

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
			lang = chunk.get("language") or meta.get("language") or "en"
			sym_type = chunk.get("symbol_type") or meta.get("symbol_type") or "section"
			text = LLMGenerator.clean_chunk_text(chunk.get("text", ""))
			blocks.append(
				f"--- Source [{idx}] (lang={lang}) ---\n"
				f"Section: {breadcrumb} ({sym_type})\n"
				f"URL: {sec_url}\n"
				f"Content:\n{text}"
			)
		return "\n\n".join(blocks)

	@staticmethod
	def format_extractive_fallback(query: str, context: List[Dict[str, Any]]) -> str:
		"""Build a clean, readable documentation answer from top reranked chunks when LLM is unavailable."""
		if not context:
			return f"По запросу «{query}» ничего не найдено в документации FastAPI."

		sections: List[str] = []
		for idx, chunk in enumerate(context[:3], start=1):
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
			body = LLMGenerator.clean_chunk_text(chunk.get("text", ""))
			header = f"### {idx}. {breadcrumb}"
			if sec_url:
				header += f"\n[Открыть в документации]({sec_url})"
			sections.append(f"{header}\n\n{body}")

		return "\n\n---\n\n".join(sections)

	@staticmethod
	def extract_code_blocks(context: List[Dict[str, Any]]) -> List[str]:
		"""Extract fenced Python code blocks from retrieved chunks, prioritizing multi-line examples."""
		blocks: List[str] = []
		seen: set[str] = set()
		pattern = re.compile(r"```(?:python)?\n([\s\S]*?)```")

		for chunk in context:
			raw = chunk.get("text", "")
			for match in pattern.finditer(raw):
				code = match.group(1).strip()
				if not code or code in seen:
					continue
				if "=" in code and "\n" not in code:
					parts = [p.strip() for p in code.split("=", 1)]
					if len(parts) == 2 and parts[0] == parts[1]:
						continue
				seen.add(code)
				blocks.append(code)

		blocks.sort(
			key=lambda c: (
				1 if ("from fastapi" in c or "@app" in c or "def " in c) else 0,
				c.count("\n"),
				len(c),
			),
			reverse=True,
		)
		return blocks

	@classmethod
	def ensure_code_snippet_in_answer(
		cls,
		answer: str,
		context: List[Dict[str, Any]],
	) -> str:
		"""Guarantee that a code block from the retrieved chunks is present in the answer if available."""
		if not answer:
			return answer

		available_codes = cls.extract_code_blocks(context)
		if not available_codes:
			return answer

		best_code = available_codes[0]
		if "```" not in answer:
			return f"{answer.rstrip()}\n\n```python\n{best_code}\n```"

		if best_code.count("\n") >= 2 and best_code not in answer:
			existing_blocks = re.findall(r"```(?:python)?\n([\s\S]*?)```", answer)
			max_existing_lines = max((b.strip().count("\n") for b in existing_blocks), default=0)
			if max_existing_lines < 2:
				return f"{answer.rstrip()}\n\n```python\n{best_code}\n```"

		return answer

	def build_messages(
		self,
		query: str,
		context: List[Dict[str, Any]],
		history: Optional[List[Dict[str, str]]] = None,
		language: Optional[str] = None,
	) -> List[Dict[str, str]]:
		"""Construct system, conversation history, and user messages for chat completion."""
		formatted_context = self.format_context(context)
		lang_hint = ""
		if language and language.lower() != "all":
			lang_label = LANGUAGE_LABELS.get(language.lower(), language)
			lang_hint = f"\nSelected documentation language: {lang_label}. Respond in the language of the user's question."

		user_prompt = (
			f"CURRENT USER QUESTION:\n{query}\n\n"
			f"FastAPI Reference Context:\n{formatted_context}{lang_hint}\n\n"
			"Instructions:\n"
			f"1. Answer ONLY the CURRENT USER QUESTION ('{query}') using exclusively the FastAPI Reference Context above.\n"
			"2. Use prior conversation history ONLY if the current question is a direct follow-up (such as asking for an example or clarification of the previous topic). If the current question is about a new topic, completely ignore previous topics.\n"
			"3. Do NOT use any external knowledge outside the provided context.\n"
			"4. MANDATORY: If the FastAPI Reference Context contains any code snippets or signatures (```python ... ```), you MUST include at least one verbatim code block (especially full usage examples if present) from the context in your response.\n"
			"5. Format the answer in clean Markdown and cite the exact documentation URLs from the context."
		)
		messages: List[Dict[str, str]] = [
			{"role": "system", "content": self.system_prompt}
		]
		if history:
			for msg in history[-4:]:
				role = msg.get("role", "")
				content = msg.get("content", "")
				if role in ("user", "assistant") and content:
					trimmed = content[:600] + "..." if role == "assistant" and len(content) > 600 else content
					messages.append({"role": role, "content": trimmed})
		messages.append({"role": "user", "content": user_prompt})
		return messages

	async def generate(
		self,
		query: str,
		context: List[Dict[str, Any]],
		history: Optional[List[Dict[str, str]]] = None,
		language: Optional[str] = None,
	) -> str:
		"""Asynchronously generate an answer via AsyncOpenAI chat completions."""
		response = await self.client.chat.completions.create(
			model=self.model,
			messages=self.build_messages(
				query, context, history=history, language=language
			),
			temperature=self.temperature,
			max_tokens=self.max_tokens,
		)
		content = response.choices[0].message.content
		raw_answer = content.strip() if content else ""
		return self.ensure_code_snippet_in_answer(raw_answer, context)

	async def generate_with_history(
		self,
		query: str,
		context: List[Dict[str, Any]],
		history: Optional[List[Dict[str, str]]] = None,
		language: Optional[str] = None,
	) -> str:
		"""Generate an answer with session history, falling back to extractive output if no API key."""
		if not self.has_valid_api_key():
			return self.format_extractive_fallback(query=query, context=context)
		try:
			return await self.generate(
				query=query,
				context=context,
				history=history,
				language=language,
			)
		except Exception:
			return self.format_extractive_fallback(query=query, context=context)

	async def generate_stream(
		self,
		query: str,
		context: List[Dict[str, Any]],
		history: Optional[List[Dict[str, str]]] = None,
		language: Optional[str] = None,
	) -> AsyncGenerator[str, None]:
		"""Asynchronously stream response tokens via AsyncOpenAI chat completions."""
		stream = await self.client.chat.completions.create(
			model=self.model,
			messages=self.build_messages(
				query, context, history=history, language=language
			),
			temperature=self.temperature,
			max_tokens=self.max_tokens,
			stream=True,
		)
		async for chunk in stream:
			if chunk.choices and chunk.choices[0].delta.content:
				yield chunk.choices[0].delta.content


OpenAIGenerator = LLMGenerator

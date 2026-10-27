from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
	"""Application and RAG pipeline configuration loaded from .env."""

	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		extra="ignore",
	)

	postgres_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_fastapi"
	postgres_echo: bool = False
	fallback_sqlite_url: str = "sqlite+aiosqlite:///data/rag.db"

	reference_url: str = "https://fastapi.tiangolo.com/reference/"
	data_dir: Path = Path("data")
	cached_routers_filename: str = "cached_routers.html"
	browser_impersonate: str = "safari15_5"
	grabber_concurrency: int = 5

	chunk_size: int = 1200
	chunk_overlap: int = 200

	device: str = "cuda"
	use_fp16: bool = True

	embedding_model_name: str = "intfloat/multilingual-e5-small"
	embedding_batch_size: int = 64
	embedding_dim: int = 384
	embedding_passage_prefix: str = "passage: "
	embedding_query_prefix: str = "query: "

	retrieval_top_k: int = 5
	retrieval_candidate_k: int = 15
	hybrid_alpha: float = 0.5
	hybrid_fusion_method: str = "rrf"
	rrf_k: int = 60

	bm25_k1: float = 1.5
	bm25_b: float = 0.75
	use_bm25s_backend: bool = True

	reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
	reranker_batch_size: int = 32
	use_reranker: bool = True
	small_to_big: bool = True

	llm_api_key: str = ""
	llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
	llm_model: str = "gemini-3.1-flash-lite"
	llm_temperature: float = 0.2
	llm_max_tokens: int = 1024
	llm_timeout: float = 60.0
	rag_system_prompt: str = (
		"You are an expert FastAPI documentation assistant. You MUST answer the user's "
		"question strictly and exclusively using ONLY the provided FastAPI Reference Context "
		"chunks. Do NOT use external knowledge, do NOT invent API parameters, and do NOT make "
		"assumptions beyond the provided context. If the provided context contains any code "
		"blocks or signatures, you MUST include the exact code snippet(s) from the context "
		"inside ```python ... ``` blocks in your answer. If the provided context does not "
		"contain enough information to answer the question, explicitly state that this "
		"information is not present in the retrieved documentation context. Always cite the "
		"exact section titles and URLs from the provided context."
	)

	supported_languages: str = "en,de,es,fr,hi,ja,ko,pt,ru,tr,uk,zh,zh-hant"
	default_language: str = "ru"

	app_host: str = "127.0.0.1"
	app_port: int = 8000
	chat_history_limit: int = 10

	def get_languages(self) -> list[str]:
		"""Return list of configured documentation language codes."""
		return [lang.strip() for lang in self.supported_languages.split(",") if lang.strip()]

	def resolve_device(self) -> str:
		"""Return configured device ('cuda' when CUDA GPU is available, otherwise 'cpu')."""
		requested = self.device.strip().lower()
		if requested.startswith("cuda"):
			try:
				import torch

				if torch.cuda.is_available():
					return self.device
			except ImportError:
				pass
			return "cpu"
		return self.device


config = Config()

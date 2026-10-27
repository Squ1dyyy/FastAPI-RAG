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

	reranker_model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
	reranker_batch_size: int = 32
	use_reranker: bool = True

	llm_api_key: str = ""
	llm_base_url: str = "https://openrouter.ai/api/v1"
	llm_model: str = "openai/gpt-4o-mini"
	llm_temperature: float = 0.2
	llm_max_tokens: int = 1024
	llm_timeout: float = 60.0
	rag_system_prompt: str = (
		"You are an expert FastAPI assistant. Answer the user's question accurately "
		"using ONLY the provided FastAPI reference documentation context. "
		"Always cite the exact section title and URL for your claims."
	)

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

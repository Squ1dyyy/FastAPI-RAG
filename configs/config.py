from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
	"""Application and RAG pipeline configuration."""

	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		extra="ignore",
	)

	# PostgreSQL Database Settings
	postgres_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_fastapi"
	postgres_echo: bool = False

	# Documentation Grabber Settings
	reference_url: str = "https://fastapi.tiangolo.com/reference/"
	data_dir: Path = Path("data")

	# Text Processing & Chunking
	chunk_size: int = 1200
	chunk_overlap: int = 200

	# Embedding Model Settings
	embedding_model_name: str = "intfloat/multilingual-e5-small"
	embedding_batch_size: int = 32
	embedding_dim: int = 384

	# Hybrid Retrieval Settings (Dense E5 + Fast BM25)
	retrieval_top_k: int = 5
	retrieval_candidate_k: int = 15
	hybrid_alpha: float = 0.5  # 0.0 = pure BM25-Fast, 1.0 = pure Dense (E5)
	hybrid_fusion_method: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "linear"
	rrf_k: int = 60

	# BM25-Fast Parameters
	bm25_k1: float = 1.5
	bm25_b: float = 0.75
	use_bm25s_backend: bool = True

	# Lightweight Reranker Settings
	reranker_model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
	use_reranker: bool = True


config = Config()

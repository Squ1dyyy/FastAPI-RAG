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
	chunk_size: int = 1000
	chunk_overlap: int = 200

	# Hybrid Retrieval Settings (Dense + Fast BM25)
	retrieval_top_k: int = 5
	hybrid_alpha: float = 0.5  # 0.0 = pure BM25, 1.0 = pure Dense
	hybrid_fusion_method: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "linear"
	rrf_k: int = 60

	# BM25 Parameters
	bm25_k1: float = 1.5
	bm25_b: float = 0.75


config = Config()

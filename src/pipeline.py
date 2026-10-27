import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from configs.config import config
from src.database.repository import DocumentRepository, QueryLogRepository
from src.embeddings import BaseEmbedder, MultilingualE5Embedder
from src.preprocessing import FastAPIHTMLParser
from src.reranking import BaseReranker, CrossEncoderReranker
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class HybridRAGPipeline:
	"""End-to-end Hybrid RAG pipeline for FastAPI documentation.

	Combines:
	1. Structural HTML parsing (`md-sidebar__scrollwrap` TOC + `md-content__inner md-typeset` sections)
	2. Sparse retrieval via `BM25-Fast` (`bm25s` + identifier tokenization)
	3. Dense semantic retrieval via `intfloat/multilingual-e5-small`
	4. Hybrid fusion via Reciprocal Rank Fusion (RRF) or Linear combination
	5. Lightweight Cross-Encoder reranking (`cross-encoder/ms-marco-TinyBERT-L-2-v2`)
	6. Persistence and query logging in PostgreSQL via SQLAlchemy 2.0
	"""

	def __init__(
		self,
		embedder: Optional[BaseEmbedder] = None,
		reranker: Optional[BaseReranker] = None,
		alpha: Optional[float] = None,
		fusion_method: Optional[str] = None,
		use_reranker: Optional[bool] = None,
	):
		self.parser = FastAPIHTMLParser()
		self.embedder = embedder or MultilingualE5Embedder()
		self.sparse_retriever = BM25Retriever()
		self.dense_retriever = DenseRetriever(embedder=self.embedder)
		self.hybrid_retriever = HybridRetriever(
			dense_retriever=self.dense_retriever,
			sparse_retriever=self.sparse_retriever,
			alpha=alpha,
			fusion_method=fusion_method,
		)
		self.use_reranker = use_reranker if use_reranker is not None else config.use_reranker
		self.reranker = reranker or (CrossEncoderReranker() if self.use_reranker else None)
		self.corpus: List[Dict[str, Any]] = []

	def parse_data_directory(self, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
		"""Parse all cached HTML documentation files from `data/` into structured documents."""
		target_dir = Path(data_dir) if data_dir is not None else config.data_dir
		html_files = sorted(target_dir.glob("*.html"))

		parsed_docs: List[Dict[str, Any]] = []
		for file_path in html_files:
			if file_path.name == "cached_routers.html":
				continue
			parsed = self.parser.parse_file(file_path)
			parsed_docs.append(parsed)

		return parsed_docs

	async def ingest_to_db(
		self,
		session: AsyncSession,
		data_dir: Optional[Path] = None,
		compute_embeddings: bool = True,
	) -> int:
		"""Parse HTML files, compute `multilingual-e5-small` embeddings, and save to DB.

		:return: Total number of chunks saved in the database.
		"""
		parsed_docs = self.parse_data_directory(data_dir=data_dir)
		doc_repo = DocumentRepository(session)
		total_chunks = 0

		for doc_data in parsed_docs:
			chunks = doc_data["chunks"]
			if compute_embeddings and chunks:
				texts = [c["text"] for c in chunks]
				vectors = self.embedder.embed_documents(texts)
				for chunk_dict, vec in zip(chunks, vectors):
					chunk_dict["embedding"] = vec

			await doc_repo.save_document(
				title=doc_data["title"],
				url=doc_data["url"],
				file_path=doc_data.get("file_path"),
				toc_sections=doc_data["toc"],
				chunks_data=chunks,
			)
			total_chunks += len(chunks)

		await session.commit()
		return total_chunks

	async def load_index_from_db(self, session: AsyncSession) -> int:
		"""Load all structural chunks and embeddings from DB and build BM25-Fast + Dense indices."""
		doc_repo = DocumentRepository(session)
		self.corpus = await doc_repo.get_all_chunks_for_indexing()
		self.hybrid_retriever.index(self.corpus)
		return len(self.corpus)

	def build_index_from_files(self, data_dir: Optional[Path] = None) -> int:
		"""Parse HTML files directly into memory and build BM25-Fast + Dense E5 indices."""
		parsed_docs = self.parse_data_directory(data_dir=data_dir)
		corpus: List[Dict[str, Any]] = []

		for doc_idx, doc_data in enumerate(parsed_docs, start=1):
			for chunk_idx, ch in enumerate(doc_data["chunks"]):
				item = dict(ch)
				item["id"] = f"doc_{doc_idx}_chunk_{chunk_idx}"
				item["doc_title"] = doc_data["title"]
				item["doc_url"] = doc_data["url"]
				corpus.append(item)

		self.corpus = corpus
		self.hybrid_retriever.index(self.corpus)
		return len(self.corpus)

	def search(
		self,
		query: str,
		top_k: Optional[int] = None,
		candidate_k: Optional[int] = None,
		rerank: Optional[bool] = None,
	) -> List[Dict[str, Any]]:
		"""Execute hybrid search (BM25-Fast + E5-small) followed by lightweight reranking."""
		final_k = top_k or config.retrieval_top_k
		pool_k = candidate_k or max(config.retrieval_candidate_k, final_k * 3)
		do_rerank = self.use_reranker if rerank is None else rerank

		candidates = self.hybrid_retriever.retrieve(query, top_k=pool_k)
		if do_rerank and self.reranker is not None and candidates:
			return self.reranker.rerank(query, candidates, top_k=final_k)

		return candidates[:final_k]

	async def search_and_log(
		self,
		query: str,
		session: AsyncSession,
		top_k: Optional[int] = None,
		candidate_k: Optional[int] = None,
		rerank: Optional[bool] = None,
	) -> List[Dict[str, Any]]:
		"""Run hybrid search + reranking and record the execution in `query_logs` table."""
		start_time = time.perf_counter()
		results = self.search(
			query=query,
			top_k=top_k,
			candidate_k=candidate_k,
			rerank=rerank,
		)
		elapsed = time.perf_counter() - start_time

		query_repo = QueryLogRepository(session)
		await query_repo.log_query(
			query=query,
			retriever_used="hybrid_e5_bm25fast_reranked" if (rerank or self.use_reranker) else "hybrid_e5_bm25fast",
			retrieved_chunks={
				"results": [
					{
						"id": r.get("id"),
						"doc_title": r.get("doc_title") or r.get("metadata", {}).get("doc_title"),
						"section_title": r.get("section_title"),
						"section_url": r.get("section_url"),
						"hybrid_score": r.get("hybrid_score"),
						"rerank_score": r.get("rerank_score"),
					}
					for r in results
				]
			},
			latency_seconds=elapsed,
		)
		await session.commit()
		return results

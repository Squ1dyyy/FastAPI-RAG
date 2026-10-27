import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from configs.config import config
from src.database.repository import DocumentRepository, QueryLogRepository
from src.embeddings import BaseEmbedder, MultilingualE5Embedder
from src.evaluation import BaseEvaluator, RAGEvaluator
from src.generation import BaseGenerator, OpenRouterGenerator
from src.preprocessing import FastAPIHTMLParser
from src.reranking import BaseReranker, CrossEncoderReranker
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class HybridRAGPipeline:
	"""Asynchronous GPU-accelerated Hybrid RAG pipeline with OpenRouter generation."""

	def __init__(
		self,
		embedder: Optional[BaseEmbedder] = None,
		reranker: Optional[BaseReranker] = None,
		generator: Optional[BaseGenerator] = None,
		evaluator: Optional[BaseEvaluator] = None,
		alpha: Optional[float] = None,
		fusion_method: Optional[str] = None,
		use_reranker: Optional[bool] = None,
		device: Optional[str] = None,
	):
		target_device = device or config.resolve_device()
		self.parser = FastAPIHTMLParser()
		self.embedder = embedder or MultilingualE5Embedder(device=target_device)
		self.sparse_retriever = BM25Retriever()
		self.dense_retriever = DenseRetriever(embedder=self.embedder)
		self.hybrid_retriever = HybridRetriever(
			dense_retriever=self.dense_retriever,
			sparse_retriever=self.sparse_retriever,
			alpha=alpha,
			fusion_method=fusion_method,
		)
		self.use_reranker = use_reranker if use_reranker is not None else config.use_reranker
		self.reranker = reranker or (
			CrossEncoderReranker(device=target_device) if self.use_reranker else None
		)
		self.generator = generator or OpenRouterGenerator()
		self.evaluator = evaluator or RAGEvaluator()
		self.corpus: List[Dict[str, Any]] = []

	def parse_data_directory(self, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
		"""Synchronously parse all cached HTML documentation files from `data/`."""
		target_dir = Path(data_dir) if data_dir is not None else config.data_dir
		html_files = [
			f for f in sorted(target_dir.glob("*.html"))
			if f.name != config.cached_routers_filename
		]
		return [self.parser.parse_file(f) for f in html_files]

	async def parse_data_directory_async(
		self, data_dir: Optional[Path] = None
	) -> List[Dict[str, Any]]:
		"""Asynchronously parse all cached HTML documentation files concurrently."""
		target_dir = Path(data_dir) if data_dir is not None else config.data_dir
		html_files = [
			f for f in sorted(target_dir.glob("*.html"))
			if f.name != config.cached_routers_filename
		]
		tasks = [self.parser.parse_file_async(f) for f in html_files]
		return list(await asyncio.gather(*tasks))

	async def ingest_to_db(
		self,
		session: AsyncSession,
		data_dir: Optional[Path] = None,
		compute_embeddings: bool = True,
	) -> int:
		"""Asynchronously parse HTML files, compute GPU E5 embeddings, and save to DB."""
		parsed_docs = await self.parse_data_directory_async(data_dir=data_dir)
		doc_repo = DocumentRepository(session)
		total_chunks = 0

		for doc_data in parsed_docs:
			chunks = doc_data["chunks"]
			if compute_embeddings and chunks:
				texts = [c["text"] for c in chunks]
				vectors = await self.embedder.embed_documents_async(texts)
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
		"""Load all structural chunks and embeddings from DB and build indices asynchronously."""
		doc_repo = DocumentRepository(session)
		self.corpus = await doc_repo.get_all_chunks_for_indexing()
		await self.hybrid_retriever.index_async(self.corpus)
		return len(self.corpus)

	def build_index_from_files(self, data_dir: Optional[Path] = None) -> int:
		"""Synchronously parse HTML files and build BM25-Fast + Dense E5 indices."""
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

	async def build_index_from_files_async(
		self, data_dir: Optional[Path] = None
	) -> int:
		"""Asynchronously parse HTML files and build BM25-Fast + Dense GPU E5 indices."""
		parsed_docs = await self.parse_data_directory_async(data_dir=data_dir)
		corpus: List[Dict[str, Any]] = []

		for doc_idx, doc_data in enumerate(parsed_docs, start=1):
			for chunk_idx, ch in enumerate(doc_data["chunks"]):
				item = dict(ch)
				item["id"] = f"doc_{doc_idx}_chunk_{chunk_idx}"
				item["doc_title"] = doc_data["title"]
				item["doc_url"] = doc_data["url"]
				corpus.append(item)

		self.corpus = corpus
		await self.hybrid_retriever.index_async(self.corpus)
		return len(self.corpus)

	def search(
		self,
		query: str,
		top_k: Optional[int] = None,
		candidate_k: Optional[int] = None,
		rerank: Optional[bool] = None,
	) -> List[Dict[str, Any]]:
		"""Synchronously execute hybrid search followed by lightweight GPU reranking."""
		final_k = top_k or config.retrieval_top_k
		pool_k = candidate_k or max(config.retrieval_candidate_k, final_k * 3)
		do_rerank = self.use_reranker if rerank is None else rerank

		candidates = self.hybrid_retriever.retrieve(query, top_k=pool_k)
		if do_rerank and self.reranker is not None and candidates:
			return self.reranker.rerank(query, candidates, top_k=final_k)

		return candidates[:final_k]

	async def search_async(
		self,
		query: str,
		top_k: Optional[int] = None,
		candidate_k: Optional[int] = None,
		rerank: Optional[bool] = None,
	) -> List[Dict[str, Any]]:
		"""Asynchronously execute concurrent hybrid search and GPU Cross-Encoder reranking."""
		final_k = top_k or config.retrieval_top_k
		pool_k = candidate_k or max(config.retrieval_candidate_k, final_k * 3)
		do_rerank = self.use_reranker if rerank is None else rerank

		candidates = await self.hybrid_retriever.retrieve_async(query, top_k=pool_k)
		if do_rerank and self.reranker is not None and candidates:
			return await self.reranker.rerank_async(query, candidates, top_k=final_k)

		return candidates[:final_k]

	async def answer(
		self,
		query: str,
		top_k: Optional[int] = None,
		candidate_k: Optional[int] = None,
		rerank: Optional[bool] = None,
		ground_truth: Optional[str] = None,
		session: Optional[AsyncSession] = None,
	) -> Dict[str, Any]:
		"""Execute full end-to-end RAG: Hybrid Search -> GPU Rerank -> OpenRouter LLM -> Evaluation -> DB Log."""
		start_time = time.perf_counter()

		context_chunks = await self.search_async(
			query=query,
			top_k=top_k,
			candidate_k=candidate_k,
			rerank=rerank,
		)

		llm_answer = await self.generator.generate(query=query, context=context_chunks)
		metrics = await self.evaluator.evaluate(
			query=query,
			response=llm_answer,
			context=context_chunks,
			ground_truth=ground_truth,
		)
		elapsed = time.perf_counter() - start_time

		if session is not None:
			query_repo = QueryLogRepository(session)
			await query_repo.log_query(
				query=query,
				response=llm_answer,
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
						for r in context_chunks
					]
				},
				evaluation_metrics=metrics,
				latency_seconds=elapsed,
			)
			await session.commit()

		return {
			"query": query,
			"answer": llm_answer,
			"context": context_chunks,
			"metrics": metrics,
			"latency_seconds": round(elapsed, 4),
		}

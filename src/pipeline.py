from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from configs.config import config
from src.database.repository import (
	ChatSessionRepository,
	DocumentRepository,
	QueryLogRepository,
)
from src.embeddings import BaseEmbedder, MultilingualE5Embedder
from src.evaluation import BaseEvaluator, RAGEvaluator
from src.generation import BaseGenerator, LLMGenerator
from src.grabber import LANGUAGE_LABELS
from src.preprocessing import FastAPIHTMLParser
from src.reranking import BaseReranker, CrossEncoderReranker
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class HybridRAGPipeline:
	"""Asynchronous GPU-accelerated Multi-Language Hybrid RAG pipeline."""

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
		self.device = target_device
		self.alpha = alpha
		self.fusion_method = fusion_method
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
		self.generator = generator or LLMGenerator()
		self.evaluator = evaluator or RAGEvaluator()
		self.corpus: List[Dict[str, Any]] = []
		self.language_corpora: Dict[str, List[Dict[str, Any]]] = {}
		self.language_retrievers: Dict[str, HybridRetriever] = {}
		self.language_counts: Dict[str, int] = {}

	def _create_hybrid_retriever(self) -> HybridRetriever:
		"""Create a fresh HybridRetriever sharing the GPU embedder."""
		return HybridRetriever(
			dense_retriever=DenseRetriever(embedder=self.embedder),
			sparse_retriever=BM25Retriever(),
			alpha=self.alpha,
			fusion_method=self.fusion_method,
		)

	@staticmethod
	def _collect_html_files(
		data_dir: Optional[Path] = None,
		languages: Optional[List[str]] = None,
	) -> List[tuple[Path, str]]:
		"""Discover HTML files organized in `data/{lang}/*.html` or flat `data/*.html`."""
		target_dir = Path(data_dir) if data_dir is not None else config.data_dir
		lang_codes = languages or config.get_languages()
		collected: List[tuple[Path, str]] = []

		for lang in lang_codes:
			lang_dir = target_dir / lang
			if lang_dir.is_dir():
				for f in sorted(lang_dir.glob("*.html")):
					if f.name != config.cached_routers_filename:
						collected.append((f, lang))

		if not collected:
			for f in sorted(target_dir.glob("*.html")):
				if f.name != config.cached_routers_filename:
					collected.append((f, "en"))

		return collected

	def parse_data_directory(
		self,
		data_dir: Optional[Path] = None,
		languages: Optional[List[str]] = None,
	) -> List[Dict[str, Any]]:
		"""Synchronously parse cached HTML documentation files across configured languages."""
		file_pairs = self._collect_html_files(data_dir=data_dir, languages=languages)
		return [self.parser.parse_file(f, language=lang) for f, lang in file_pairs]

	async def parse_data_directory_async(
		self,
		data_dir: Optional[Path] = None,
		languages: Optional[List[str]] = None,
	) -> List[Dict[str, Any]]:
		"""Asynchronously parse cached HTML documentation files across configured languages."""
		return await asyncio.to_thread(
			self.parse_data_directory, data_dir, languages
		)

	async def ingest_to_db(
		self,
		session: AsyncSession,
		data_dir: Optional[Path] = None,
		compute_embeddings: bool = True,
		languages: Optional[List[str]] = None,
	) -> int:
		"""Asynchronously parse multi-language HTML files, compute GPU E5 embeddings, and save to DB."""
		parsed_docs = await self.parse_data_directory_async(
			data_dir=data_dir,
			languages=languages,
		)
		doc_repo = DocumentRepository(session)
		total_chunks = 0

		embedding_cache: Dict[str, List[float]] = {}
		if compute_embeddings:
			unique_texts: List[str] = []
			seen_texts = set()
			for doc_data in parsed_docs:
				for c in doc_data["chunks"]:
					t = c["text"]
					if t not in seen_texts:
						seen_texts.add(t)
						unique_texts.append(t)
			if unique_texts:
				unique_vectors = await self.embedder.embed_documents_async(unique_texts)
				embedding_cache = dict(zip(unique_texts, unique_vectors))

		for idx, doc_data in enumerate(parsed_docs, start=1):
			chunks = doc_data["chunks"]
			doc_lang = doc_data.get("language", "en")
			if compute_embeddings and chunks:
				for chunk_dict in chunks:
					vec = embedding_cache.get(chunk_dict["text"])
					if vec is not None:
						chunk_dict["embedding"] = vec

			await doc_repo.save_document(
				title=doc_data["title"],
				url=doc_data["url"],
				file_path=doc_data.get("file_path"),
				toc_sections=doc_data["toc"],
				chunks_data=chunks,
				language=doc_lang,
			)
			total_chunks += len(chunks)
			if idx % 25 == 0:
				await session.commit()

		await session.commit()
		return total_chunks

	async def _build_language_indices_async(self, all_chunks: List[Dict[str, Any]]) -> None:
		"""Partition corpus by language and pre-build primary hybrid indices."""
		self.corpus = all_chunks
		self.language_corpora = {}
		for ch in all_chunks:
			lang = (ch.get("language") or ch.get("metadata", {}).get("language") or "en").lower()
			self.language_corpora.setdefault(lang, []).append(ch)

		self.language_retrievers = {}
		eager_langs = {"en", config.default_language.lower()}
		for lang in eager_langs:
			if lang in self.language_corpora:
				retr = self._create_hybrid_retriever()
				await retr.index_async(self.language_corpora[lang])
				self.language_retrievers[lang] = retr

		default_corpus = (
			self.language_corpora.get(config.default_language.lower())
			or self.language_corpora.get("en")
			or all_chunks
		)
		if default_corpus:
			await self.hybrid_retriever.index_async(default_corpus)

	def _build_language_indices_sync(self, all_chunks: List[Dict[str, Any]]) -> None:
		"""Synchronously partition corpus by language and build primary hybrid indices."""
		self.corpus = all_chunks
		self.language_corpora = {}
		for ch in all_chunks:
			lang = (ch.get("language") or ch.get("metadata", {}).get("language") or "en").lower()
			self.language_corpora.setdefault(lang, []).append(ch)

		self.language_retrievers = {}
		eager_langs = {"en", config.default_language.lower()}
		for lang in eager_langs:
			if lang in self.language_corpora:
				retr = self._create_hybrid_retriever()
				retr.index(self.language_corpora[lang])
				self.language_retrievers[lang] = retr

		default_corpus = (
			self.language_corpora.get(config.default_language.lower())
			or self.language_corpora.get("en")
			or all_chunks
		)
		if default_corpus:
			self.hybrid_retriever.index(default_corpus)

	async def load_index_from_db(
		self,
		session: AsyncSession,
		language: Optional[str] = None,
	) -> int:
		"""Load structural chunks and embeddings from DB and build language indices."""
		doc_repo = DocumentRepository(session)
		self.language_counts = await doc_repo.get_language_counts()
		if not self.language_counts:
			return 0

		if language and language.lower() != "all":
			chunks = await doc_repo.get_all_chunks_for_indexing(language=language)
			if chunks:
				await self._build_language_indices_async(chunks)
			return len(chunks)

		primary_lang = (
			config.default_language.lower()
			if config.default_language.lower() in self.language_counts
			else next(iter(self.language_counts.keys()))
		)
		primary_chunks = await doc_repo.get_all_chunks_for_indexing(language=primary_lang)
		if primary_chunks:
			await self._build_language_indices_async(primary_chunks)
		return sum(self.language_counts.values())

	def build_index_from_files(
		self,
		data_dir: Optional[Path] = None,
		languages: Optional[List[str]] = None,
	) -> int:
		"""Synchronously parse HTML files and build BM25-Fast + Dense E5 indices."""
		parsed_docs = self.parse_data_directory(data_dir=data_dir, languages=languages)
		corpus: List[Dict[str, Any]] = []

		for doc_idx, doc_data in enumerate(parsed_docs, start=1):
			doc_lang = doc_data.get("language", "en")
			for chunk_idx, ch in enumerate(doc_data["chunks"]):
				item = dict(ch)
				item["id"] = f"doc_{doc_idx}_chunk_{chunk_idx}"
				item["language"] = doc_lang
				item["doc_title"] = doc_data["title"]
				item["doc_url"] = doc_data["url"]
				corpus.append(item)

		self._build_language_indices_sync(corpus)
		return len(self.corpus)

	async def build_index_from_files_async(
		self,
		data_dir: Optional[Path] = None,
		languages: Optional[List[str]] = None,
	) -> int:
		"""Asynchronously parse HTML files and build BM25-Fast + Dense GPU E5 indices."""
		parsed_docs = await self.parse_data_directory_async(
			data_dir=data_dir,
			languages=languages,
		)
		corpus: List[Dict[str, Any]] = []

		embedding_cache: Dict[str, List[float]] = {}
		unique_texts: List[str] = []
		seen_texts = set()
		for doc_data in parsed_docs:
			for c in doc_data["chunks"]:
				t = c["text"]
				if t not in seen_texts:
					seen_texts.add(t)
					unique_texts.append(t)
		if unique_texts:
			unique_vectors = await self.embedder.embed_documents_async(unique_texts)
			embedding_cache = dict(zip(unique_texts, unique_vectors))

		for doc_idx, doc_data in enumerate(parsed_docs, start=1):
			doc_lang = doc_data.get("language", "en")
			for chunk_idx, ch in enumerate(doc_data["chunks"]):
				item = dict(ch)
				item["id"] = f"doc_{doc_idx}_chunk_{chunk_idx}"
				item["language"] = doc_lang
				item["doc_title"] = doc_data["title"]
				item["doc_url"] = doc_data["url"]
				if item["text"] in embedding_cache:
					item["embedding"] = embedding_cache[item["text"]]
				corpus.append(item)

		await self._build_language_indices_async(corpus)
		return len(self.corpus)

	def _select_retriever(self, language: Optional[str] = None) -> HybridRetriever:
		"""Select or lazily build the language-specific HybridRetriever."""
		if language:
			lang_key = language.strip().lower()
			if lang_key in self.language_retrievers:
				return self.language_retrievers[lang_key]
			if lang_key in self.language_corpora:
				retr = self._create_hybrid_retriever()
				retr.index(self.language_corpora[lang_key])
				self.language_retrievers[lang_key] = retr
				return retr
		return self.hybrid_retriever

	async def _select_retriever_async(self, language: Optional[str] = None) -> HybridRetriever:
		"""Asynchronously select or lazily build the language-specific HybridRetriever."""
		if language:
			lang_key = language.strip().lower()
			if lang_key in self.language_retrievers:
				return self.language_retrievers[lang_key]
			if lang_key in self.language_corpora:
				retr = self._create_hybrid_retriever()
				await retr.index_async(self.language_corpora[lang_key])
				self.language_retrievers[lang_key] = retr
				return retr
		return self.hybrid_retriever

	def get_available_languages(self) -> List[Dict[str, Any]]:
		"""Return metadata and chunk counts for all configured and indexed languages."""
		out: List[Dict[str, Any]] = []
		for code in config.get_languages():
			label = LANGUAGE_LABELS.get(code, code)
			count = self.language_counts.get(
				code, len(self.language_corpora.get(code, []))
			)
			out.append({"code": code, "label": label, "chunk_count": count})
		return out

	def get_total_indexed_chunks(self) -> int:
		"""Return total number of indexed chunks across all languages."""
		if self.language_counts:
			return sum(self.language_counts.values())
		return len(self.corpus)

	def search(
		self,
		query: str,
		top_k: Optional[int] = None,
		candidate_k: Optional[int] = None,
		rerank: Optional[bool] = None,
		language: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		"""Synchronously execute hybrid search followed by lightweight GPU reranking."""
		final_k = top_k or config.retrieval_top_k
		pool_k = candidate_k or max(config.retrieval_candidate_k, final_k * 3)
		do_rerank = self.use_reranker if rerank is None else rerank
		active_retriever = self._select_retriever(language)

		candidates = active_retriever.retrieve(query, top_k=pool_k)
		if do_rerank and self.reranker is not None and candidates:
			return self.reranker.rerank(query, candidates, top_k=final_k)

		return candidates[:final_k]

	async def search_async(
		self,
		query: str,
		top_k: Optional[int] = None,
		candidate_k: Optional[int] = None,
		rerank: Optional[bool] = None,
		language: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		"""Asynchronously execute concurrent hybrid search and GPU Cross-Encoder reranking."""
		final_k = top_k or config.retrieval_top_k
		pool_k = candidate_k or max(config.retrieval_candidate_k, final_k * 3)
		do_rerank = self.use_reranker if rerank is None else rerank
		active_retriever = await self._select_retriever_async(language)

		candidates = await active_retriever.retrieve_async(query, top_k=pool_k)
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
		session_id: Optional[str] = None,
		history: Optional[List[Dict[str, str]]] = None,
		language: Optional[str] = None,
	) -> Dict[str, Any]:
		"""Execute full end-to-end RAG: Hybrid Search -> GPU Rerank -> Gemini LLM -> Evaluation -> Session & DB Log."""
		start_time = time.perf_counter()

		active_session_id = session_id
		active_language = (language or config.default_language or "ru").strip().lower()
		chat_repo: Optional[ChatSessionRepository] = None
		conversation_history: List[Dict[str, str]] = list(history or [])

		if session is not None and session_id is not None:
			chat_repo = ChatSessionRepository(session)
			chat_session = await chat_repo.get_or_create_session(
				session_id=session_id,
				language=language,
			)
			active_session_id = chat_session.id
			active_language = language or chat_session.language or active_language
			if not conversation_history:
				conversation_history = await chat_repo.get_history(
					active_session_id,
					limit=config.chat_history_limit,
				)

		if (
			session is not None
			and active_language not in self.language_retrievers
			and active_language in self.language_counts
		):
			doc_repo = DocumentRepository(session)
			lang_chunks = await doc_repo.get_all_chunks_for_indexing(
				language=active_language
			)
			if lang_chunks:
				self.language_corpora[active_language] = lang_chunks
				retr = self._create_hybrid_retriever()
				await retr.index_async(lang_chunks)
				self.language_retrievers[active_language] = retr

		context_chunks = await self.search_async(
			query=query,
			top_k=top_k,
			candidate_k=candidate_k,
			rerank=rerank,
			language=active_language,
		)

		llm_answer = await self.generator.generate_with_history(
			query=query,
			context=context_chunks,
			history=conversation_history,
			language=active_language,
		)
		metrics = await self.evaluator.evaluate(
			query=query,
			response=llm_answer,
			context=context_chunks,
			ground_truth=ground_truth,
		)
		elapsed = time.perf_counter() - start_time

		sources_summary = [
			{
				"id": r.get("id"),
				"language": r.get("language") or r.get("metadata", {}).get("language") or active_language,
				"doc_title": r.get("doc_title") or r.get("metadata", {}).get("doc_title"),
				"section_title": r.get("section_title"),
				"breadcrumb": r.get("breadcrumb") or r.get("section_title"),
				"section_url": r.get("section_url"),
				"hybrid_score": r.get("hybrid_score"),
				"rerank_score": r.get("rerank_score"),
			}
			for r in context_chunks
		]

		if session is not None:
			if chat_repo is not None and active_session_id is not None:
				await chat_repo.add_message(
					session_id=active_session_id,
					role="user",
					content=query,
				)
				await chat_repo.add_message(
					session_id=active_session_id,
					role="assistant",
					content=llm_answer,
					sources=sources_summary,
					metrics={**metrics, "latency_seconds": round(elapsed, 4)},
				)

			query_repo = QueryLogRepository(session)
			await query_repo.log_query(
				query=query,
				response=llm_answer,
				retriever_used=(
					f"hybrid_e5_bm25fast_reranked_{active_language}"
					if (rerank or self.use_reranker)
					else f"hybrid_e5_bm25fast_{active_language}"
				),
				retrieved_chunks={"results": sources_summary, "language": active_language},
				evaluation_metrics=metrics,
				latency_seconds=elapsed,
			)
			await session.commit()

		return {
			"session_id": active_session_id,
			"language": active_language,
			"query": query,
			"answer": llm_answer,
			"context": context_chunks,
			"sources": sources_summary,
			"metrics": metrics,
			"latency_seconds": round(elapsed, 4),
		}

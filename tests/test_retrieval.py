import unittest
from typing import Any, Dict, List
from src.database import (
	QueryLog,
	create_engine_and_session,
	init_db,
)
from src.embeddings.embedder import BaseEmbedder
from src.evaluation import RAGEvaluator
from src.generation import BaseGenerator, LLMGenerator
from src.pipeline import HybridRAGPipeline
from src.reranking import BaseReranker
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from sqlalchemy import select


class DummyEmbedder(BaseEmbedder):
	"""Deterministic dummy embedder for unit testing."""

	def __init__(self):
		self.doc_embed_calls = 0

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		self.doc_embed_calls += 1
		return [self.embed_query(t) for t in texts]

	def embed_query(self, text: str) -> List[float]:
		v1 = 1.0 if "fastapi" in text.lower() else 0.0
		v2 = 1.0 if "route" in text.lower() or "router" in text.lower() else 0.0
		v3 = 1.0 if "response" in text.lower() else 0.0
		norm = (v1 * v1 + v2 * v2 + v3 * v3) ** 0.5
		if norm == 0:
			return [0.33, 0.33, 0.33]
		return [v1 / norm, v2 / norm, v3 / norm]


class DummyReranker(BaseReranker):
	"""Deterministic dummy reranker for unit testing."""

	def rerank(
		self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
	) -> List[Dict[str, Any]]:
		results = []
		for idx, doc in enumerate(documents[:top_k], start=1):
			item = dict(doc)
			item["pre_rerank_rank"] = idx
			item["rerank_score"] = float(doc.get("score", 0.0)) + 1.0
			results.append(item)
		return results


class DummyGenerator(BaseGenerator):
	"""Deterministic dummy generator for unit testing."""

	async def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
		if not context:
			return "No context."
		top = context[0]
		return f"Answer for '{query}': {top.get('text', '')} ({top.get('section_url', '')})"


class TestHybridRetrievalAndAsyncRAG(unittest.IsolatedAsyncioTestCase):

	def setUp(self):
		self.corpus = [
			{
				"id": "doc1",
				"section_title": "FastAPI",
				"section_url": "https://fastapi.tiangolo.com/reference/fastapi/#fastapi.FastAPI",
				"text": "FastAPI is a modern web framework for Python.",
			},
			{
				"id": "doc2",
				"section_title": "APIRouter",
				"section_url": "https://fastapi.tiangolo.com/reference/apirouter/#fastapi.APIRouter",
				"text": "APIRouter allows structuring large applications with routes and status_code.",
			},
			{
				"id": "doc3",
				"section_title": "JSONResponse",
				"section_url": "https://fastapi.tiangolo.com/reference/responses/#fastapi.responses.JSONResponse",
				"text": "Custom response classes include JSONResponse and HTMLResponse.",
			},
		]

	def test_bm25_tokenizer_snake_and_camel_case(self):
		tokens = BM25Retriever.tokenize("HTTPException status_code WebSocketException")
		self.assertIn("httpexception", tokens)
		self.assertIn("http", tokens)
		self.assertIn("exception", tokens)
		self.assertIn("status_code", tokens)
		self.assertIn("status", tokens)
		self.assertIn("code", tokens)
		self.assertIn("web", tokens)
		self.assertIn("socket", tokens)

	async def test_bm25_fast_async(self):
		bm25 = BM25Retriever(use_bm25s=True)
		await bm25.index_async(self.corpus)

		results = await bm25.retrieve_async("APIRouter routes", top_k=2)
		self.assertGreater(len(results), 0)
		self.assertEqual(results[0]["id"], "doc2")

	def test_bm25_inverted_index_fallback(self):
		bm25 = BM25Retriever(use_bm25s=False)
		bm25.index(self.corpus)

		results = bm25.retrieve("status code", top_k=1)
		self.assertEqual(len(results), 1)
		self.assertEqual(results[0]["id"], "doc2")

	def test_bm25_empty_corpus_or_query(self):
		bm25 = BM25Retriever()
		self.assertEqual(bm25.retrieve("FastAPI"), [])
		bm25.index(self.corpus)
		self.assertEqual(bm25.retrieve("   "), [])

	async def test_dense_retriever_reuses_precomputed_embeddings(self):
		embedder = DummyEmbedder()
		dense = DenseRetriever(embedder=embedder)
		precomputed_corpus = [
			{"id": "d1", "text": "FastAPI framework", "embedding": [1.0, 0.0, 0.0]},
			{"id": "d2", "text": "APIRouter routes", "embedding": [0.0, 1.0, 0.0]},
		]
		await dense.index_async(precomputed_corpus)
		self.assertEqual(embedder.doc_embed_calls, 0)

		results = await dense.retrieve_async("FastAPI", top_k=1)
		self.assertEqual(results[0]["id"], "d1")
		self.assertAlmostEqual(results[0]["score"], 1.0, places=4)

	async def test_hybrid_rrf_async(self):
		bm25 = BM25Retriever()
		dense = DenseRetriever(embedder=DummyEmbedder())
		hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=bm25, fusion_method="rrf")

		await hybrid.index_async(self.corpus)
		results = await hybrid.retrieve_async("FastAPI", top_k=2)

		self.assertGreater(len(results), 0)
		self.assertIn("hybrid_score", results[0])
		self.assertEqual(results[0]["id"], "doc1")

	async def test_hybrid_linear_async(self):
		bm25 = BM25Retriever()
		dense = DenseRetriever(embedder=DummyEmbedder())
		hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=bm25, fusion_method="linear")

		await hybrid.index_async(self.corpus)
		results = await hybrid.retrieve_async("response", top_k=2)

		self.assertGreater(len(results), 0)
		self.assertEqual(results[0]["id"], "doc3")

	async def test_hybrid_alpha_pure_dense_vs_pure_sparse(self):
		bm25 = BM25Retriever(use_bm25s=False)
		dense = DenseRetriever(embedder=DummyEmbedder())

		pure_sparse = HybridRetriever(dense_retriever=dense, sparse_retriever=bm25, alpha=0.0)
		await pure_sparse.index_async(self.corpus)
		res_sparse = await pure_sparse.retrieve_async("HTMLResponse", top_k=1)
		self.assertEqual(res_sparse[0]["id"], "doc3")

		pure_dense = HybridRetriever(dense_retriever=dense, sparse_retriever=bm25, alpha=1.0)
		await pure_dense.index_async(self.corpus)
		res_dense = await pure_dense.retrieve_async("router", top_k=1)
		self.assertEqual(res_dense[0]["id"], "doc2")

	def test_llm_generator_prompt_formatting(self):
		llm_gen = LLMGenerator(api_key="test_key", base_url="https://api.openai.com/v1", model="gpt-4o-mini")
		messages = llm_gen.build_messages("How to use APIRouter?", self.corpus[:2])
		self.assertEqual(len(messages), 2)
		self.assertEqual(messages[0]["role"], "system")
		self.assertIn("APIRouter", messages[1]["content"])
		self.assertIsNotNone(llm_gen.client)

	def test_llm_generator_missing_api_key_raises(self):
		llm_gen = LLMGenerator(api_key="")
		with self.assertRaises(ValueError):
			_ = llm_gen.client

	def test_ensure_code_snippet_in_answer(self):
		chunks_with_code = [
			{
				"id": "bg_1",
				"text": (
					"## BackgroundTasks\n```python\nBackgroundTasks(tasks=None)\n```\n\n"
					"## Example\n```python\nfrom fastapi import BackgroundTasks, FastAPI\n"
					"app = FastAPI()\n@app.post('/send')\nasync def send(bt: BackgroundTasks):\n"
					"    bt.add_task(write_log, 'hello')\n```"
				),
			}
		]
		answer_without_code = "Вы можете использовать BackgroundTasks для запуска фоновых задач."
		enforced = LLMGenerator.ensure_code_snippet_in_answer(answer_without_code, chunks_with_code)
		self.assertIn("```python", enforced)
		self.assertIn("from fastapi import BackgroundTasks, FastAPI", enforced)
		self.assertIn("bt.add_task(write_log, 'hello')", enforced)

	def test_small_to_big_h2_expansion(self):
		pipeline = HybridRAGPipeline(
			embedder=DummyEmbedder(),
			reranker=DummyReranker(),
			generator=DummyGenerator(),
			evaluator=RAGEvaluator(),
		)
		h2_chunk = {
			"id": "doc_mw_0",
			"doc_title": "Middleware",
			"doc_url": "https://fastapi.tiangolo.com/reference/middleware/",
			"section_title": "GZipMiddleware",
			"anchor": "fastapi.middleware.gzip.GZipMiddleware",
			"breadcrumb": "Middleware > GZipMiddleware",
			"section_url": "https://fastapi.tiangolo.com/reference/middleware/#fastapi.middleware.gzip.GZipMiddleware",
			"text": "## fastapi.middleware.gzip.GZipMiddleware\nGZipMiddleware(app, minimum_size=500, compresslevel=9)",
			"metadata": {"heading_level": 2, "parent_anchor": ""},
		}
		h3_chunk = {
			"id": "doc_mw_1",
			"doc_title": "Middleware",
			"doc_url": "https://fastapi.tiangolo.com/reference/middleware/",
			"section_title": "compresslevel",
			"anchor": "fastapi.middleware.gzip.GZipMiddleware.compresslevel",
			"breadcrumb": "Middleware > GZipMiddleware > compresslevel",
			"section_url": "https://fastapi.tiangolo.com/reference/middleware/#fastapi.middleware.gzip.GZipMiddleware.compresslevel",
			"text": "## compresslevel instance-attribute\ncompresslevel = compresslevel",
			"metadata": {"heading_level": 3, "parent_anchor": "fastapi.middleware.gzip.GZipMiddleware"},
		}
		pipeline.corpus = [h2_chunk, h3_chunk]
		expanded = pipeline.expand_small_to_big([h3_chunk])
		self.assertEqual(len(expanded), 1)
		self.assertTrue(expanded[0].get("small_to_big"))
		self.assertEqual(expanded[0].get("parent_anchor"), "fastapi.middleware.gzip.GZipMiddleware")
		self.assertIn("GZipMiddleware(app, minimum_size=500, compresslevel=9)", expanded[0]["text"])
		self.assertIn("compresslevel = compresslevel", expanded[0]["text"])

	def test_lca_h1_graph_expansion_returns_all_h2_sections(self):
		pipeline = HybridRAGPipeline(
			embedder=DummyEmbedder(),
			reranker=DummyReranker(),
			generator=DummyGenerator(),
			evaluator=RAGEvaluator(),
		)
		doc_url = "https://fastapi.tiangolo.com/reference/middleware/"
		h1_overview = {
			"id": "mw_h1",
			"doc_title": "Middleware",
			"doc_url": doc_url,
			"section_title": "Middleware",
			"anchor": "middleware",
			"breadcrumb": "Middleware",
			"symbol_type": "overview",
			"section_url": f"{doc_url}#middleware",
			"text": "# Middleware\nSeveral middlewares are available in FastAPI.",
			"metadata": {"heading_level": 1},
		}
		h2_cors = {
			"id": "mw_cors",
			"doc_title": "Middleware",
			"doc_url": doc_url,
			"section_title": "CORSMiddleware",
			"anchor": "fastapi.middleware.cors.CORSMiddleware",
			"breadcrumb": "Middleware > CORSMiddleware",
			"symbol_type": "class",
			"section_url": f"{doc_url}#fastapi.middleware.cors.CORSMiddleware",
			"text": "## CORSMiddleware\nCORSMiddleware(app, allow_origins=...)",
			"metadata": {"heading_level": 2},
		}
		h2_gzip = {
			"id": "mw_gzip",
			"doc_title": "Middleware",
			"doc_url": doc_url,
			"section_title": "GZipMiddleware",
			"anchor": "fastapi.middleware.gzip.GZipMiddleware",
			"breadcrumb": "Middleware > GZipMiddleware",
			"symbol_type": "class",
			"section_url": f"{doc_url}#fastapi.middleware.gzip.GZipMiddleware",
			"text": "## GZipMiddleware\nGZipMiddleware(app, minimum_size=500, compresslevel=9)",
			"metadata": {"heading_level": 2},
		}
		h3_gzip_attr = {
			"id": "mw_gzip_attr",
			"doc_title": "Middleware",
			"doc_url": doc_url,
			"section_title": "compresslevel",
			"anchor": "fastapi.middleware.gzip.GZipMiddleware.compresslevel",
			"breadcrumb": "Middleware > GZipMiddleware > compresslevel",
			"symbol_type": "attribute",
			"section_url": f"{doc_url}#fastapi.middleware.gzip.GZipMiddleware.compresslevel",
			"text": "## compresslevel\ncompresslevel = compresslevel",
			"metadata": {"heading_level": 3, "parent_anchor": "fastapi.middleware.gzip.GZipMiddleware"},
		}
		h2_trusted = {
			"id": "mw_trusted",
			"doc_title": "Middleware",
			"doc_url": doc_url,
			"section_title": "TrustedHostMiddleware",
			"anchor": "fastapi.middleware.trustedhost.TrustedHostMiddleware",
			"breadcrumb": "Middleware > TrustedHostMiddleware",
			"symbol_type": "class",
			"section_url": f"{doc_url}#fastapi.middleware.trustedhost.TrustedHostMiddleware",
			"text": "## TrustedHostMiddleware\nTrustedHostMiddleware(app, allowed_hosts=None)",
			"metadata": {"heading_level": 2},
		}
		pipeline.corpus = [h1_overview, h2_cors, h2_gzip, h3_gzip_attr, h2_trusted]

		expanded = pipeline.expand_small_to_big([h3_gzip_attr, h2_gzip, h2_trusted])
		self.assertEqual(expanded[0].get("lca_level"), "h1")
		self.assertIn("CORSMiddleware", expanded[0]["text"])
		self.assertIn("GZipMiddleware", expanded[0]["text"])
		self.assertIn("TrustedHostMiddleware", expanded[0]["text"])
		section_titles = [c.get("section_title") for c in expanded]
		self.assertIn("CORSMiddleware", section_titles)
		self.assertNotIn("compresslevel", section_titles)

	async def test_rag_evaluator_metrics_and_ground_truth_f1(self):
		evaluator = RAGEvaluator()
		metrics = await evaluator.evaluate(
			query="FastAPI modern framework",
			response="FastAPI is a modern web framework (https://fastapi.tiangolo.com/reference/fastapi/#fastapi.FastAPI)",
			context=[self.corpus[0]],
			ground_truth="FastAPI is a modern web framework for Python",
		)
		self.assertEqual(metrics["context_coverage"], 1.0)
		self.assertEqual(metrics["citation_present"], 1.0)
		self.assertGreater(metrics["ground_truth_f1"], 0.5)

	async def test_end_to_end_hybrid_rag_pipeline_with_db_logging(self):
		engine, session_factory = create_engine_and_session(
			database_url="sqlite+aiosqlite:///:memory:",
			echo=False,
		)
		await init_db(engine)

		try:
			pipeline = HybridRAGPipeline(
				embedder=DummyEmbedder(),
				reranker=DummyReranker(),
				generator=DummyGenerator(),
				evaluator=RAGEvaluator(),
			)
			pipeline.corpus = [dict(c) for c in self.corpus]
			await pipeline.hybrid_retriever.index_async(pipeline.corpus)

			async with session_factory() as session:
				rag_out = await pipeline.answer(
					"APIRouter routes",
					top_k=2,
					ground_truth="APIRouter allows structuring large applications with routes",
					session=session,
				)
				self.assertIn("APIRouter", rag_out["answer"])
				self.assertEqual(rag_out["metrics"]["citation_present"], 1.0)
				self.assertIn("ground_truth_f1", rag_out["metrics"])

				logs = (await session.execute(select(QueryLog))).scalars().all()
				self.assertEqual(len(logs), 1)
				self.assertEqual(logs[0].query, "APIRouter routes")
		finally:
			await engine.dispose()


if __name__ == "__main__":
	unittest.main()

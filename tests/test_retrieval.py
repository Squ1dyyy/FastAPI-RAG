import unittest
from typing import Any, Dict, List
from src.embeddings.embedder import BaseEmbedder
from src.evaluation import RAGEvaluator
from src.generation import BaseGenerator, OpenRouterGenerator
from src.pipeline import HybridRAGPipeline
from src.reranking import BaseReranker
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class DummyEmbedder(BaseEmbedder):
	"""Deterministic dummy embedder for unit testing."""

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
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
				"text": "APIRouter allows structuring large applications with routes.",
			},
			{
				"id": "doc3",
				"section_title": "JSONResponse",
				"section_url": "https://fastapi.tiangolo.com/reference/responses/#fastapi.responses.JSONResponse",
				"text": "Custom response classes include JSONResponse and HTMLResponse.",
			},
		]

	async def test_bm25_fast_async(self):
		bm25 = BM25Retriever()
		await bm25.index_async(self.corpus)

		results = await bm25.retrieve_async("APIRouter routes", top_k=2)
		self.assertGreater(len(results), 0)
		self.assertEqual(results[0]["id"], "doc2")

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

	async def test_openrouter_prompt_and_async_rag_pipeline(self):
		openrouter = OpenRouterGenerator(api_key="test_key", model="openai/gpt-4o-mini")
		messages = openrouter.build_messages("How to use APIRouter?", self.corpus[:2])
		self.assertEqual(len(messages), 2)
		self.assertEqual(messages[0]["role"], "system")
		self.assertIn("APIRouter", messages[1]["content"])

		pipeline = HybridRAGPipeline(
			embedder=DummyEmbedder(),
			reranker=DummyReranker(),
			generator=DummyGenerator(),
			evaluator=RAGEvaluator(),
		)
		pipeline.corpus = self.corpus
		await pipeline.hybrid_retriever.index_async(self.corpus)

		rag_out = await pipeline.answer("APIRouter routes", top_k=2)
		self.assertIn("APIRouter", rag_out["answer"])
		self.assertIn("context_coverage", rag_out["metrics"])
		self.assertEqual(rag_out["metrics"]["citation_present"], 1.0)


if __name__ == "__main__":
	unittest.main()

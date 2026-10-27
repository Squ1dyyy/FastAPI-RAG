import unittest
from typing import Any, Dict, List, Optional
import numpy as np
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app
from src.database.connection import create_engine_and_session, init_db
from src.embeddings import BaseEmbedder
from src.evaluation import RAGEvaluator
from src.generation import BaseGenerator
from src.pipeline import HybridRAGPipeline
from src.reranking import BaseReranker
from src.retrieval import BM25Retriever


class MockAPIEmbedder(BaseEmbedder):
	"""Deterministic embedder for FastAPI service unit testing."""

	VOCAB = ["fastapi", "router", "httpexception", "backgroundtasks", "status_code"]

	def _vec(self, text: str) -> List[float]:
		tokens = BM25Retriever.tokenize(text)
		vec = np.array([float(tokens.count(w)) for w in self.VOCAB], dtype=np.float32)
		norm = np.linalg.norm(vec)
		if norm > 0:
			vec = vec / norm
		else:
			vec[0] = 1.0
		return vec.tolist()

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		return [self._vec(t) for t in texts]

	def embed_query(self, query: str) -> List[float]:
		return self._vec(query)


class MockAPIReranker(BaseReranker):
	"""Deterministic reranker for FastAPI service unit testing."""

	def rerank(
		self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
	) -> List[Dict[str, Any]]:
		out = []
		for idx, doc in enumerate(documents[:top_k], start=1):
			item = dict(doc)
			item["pre_rerank_rank"] = idx
			item["rerank_score"] = 5.0 - idx * 0.5
			out.append(item)
		return out


class MockAPIGenerator(BaseGenerator):
	"""Session-aware mock generator for FastAPI service unit testing."""

	async def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
		top = context[0] if context else {}
		return f"Generated answer for '{query}' from {top.get('section_url', '')}"

	async def generate_with_history(
		self,
		query: str,
		context: List[Dict[str, Any]],
		history: Optional[List[Dict[str, str]]] = None,
	) -> str:
		top = context[0] if context else {}
		turns = len(history) if history else 0
		return (
			f"[history_turns={turns}] Answer for '{query}' "
			f"citing {top.get('section_url', '')}"
		)


class TestFastAPIChatService(unittest.IsolatedAsyncioTestCase):

	async def asyncSetUp(self):
		self.engine, self.session_factory = create_engine_and_session(
			database_url="sqlite+aiosqlite:///:memory:",
			echo=False,
		)
		await init_db(self.engine)

		self.pipeline = HybridRAGPipeline(
			embedder=MockAPIEmbedder(),
			reranker=MockAPIReranker(),
			generator=MockAPIGenerator(),
			evaluator=RAGEvaluator(),
		)
		self.pipeline.corpus = [
			{
				"id": "doc_1_chunk_0",
				"doc_title": "Exceptions",
				"section_title": "HTTPException",
				"breadcrumb": "Exceptions > HTTPException",
				"section_url": "https://fastapi.tiangolo.com/reference/exceptions/#fastapi.HTTPException",
				"text": "HTTPException(status_code, detail=None, headers=None) raises HTTP errors.",
			},
			{
				"id": "doc_2_chunk_0",
				"doc_title": "APIRouter",
				"section_title": "APIRouter",
				"breadcrumb": "APIRouter > include_router",
				"section_url": "https://fastapi.tiangolo.com/reference/apirouter/#fastapi.APIRouter",
				"text": "APIRouter groups path operations and status_code configurations.",
			},
		]
		await self.pipeline.hybrid_retriever.index_async(self.pipeline.corpus)
		self.app = create_app(
			pipeline=self.pipeline,
			session_factory=self.session_factory,
		)

	async def asyncTearDown(self):
		await self.engine.dispose()

	async def test_ui_and_health_endpoints(self):
		transport = ASGITransport(app=self.app)
		async with AsyncClient(transport=transport, base_url="http://test") as client:
			ui_res = await client.get("/")
			self.assertEqual(ui_res.status_code, 200)
			self.assertIn("FastAPI Hybrid RAG", ui_res.text)

			health_res = await client.get("/api/health")
			self.assertEqual(health_res.status_code, 200)
			data = health_res.json()
			self.assertEqual(data["status"], "ok")
			self.assertEqual(data["indexed_chunks"], 2)

	async def test_session_lifecycle_crud(self):
		transport = ASGITransport(app=self.app)
		async with AsyncClient(transport=transport, base_url="http://test") as client:
			create_res = await client.post(
				"/api/sessions",
				json={"title": "FastAPI Testing Session"},
			)
			self.assertEqual(create_res.status_code, 200)
			session_data = create_res.json()
			session_id = session_data["id"]
			self.assertEqual(session_data["title"], "FastAPI Testing Session")

			list_res = await client.get("/api/sessions")
			self.assertEqual(list_res.status_code, 200)
			self.assertEqual(len(list_res.json()), 1)

			detail_res = await client.get(f"/api/sessions/{session_id}")
			self.assertEqual(detail_res.status_code, 200)
			self.assertEqual(detail_res.json()["id"], session_id)

			del_res = await client.delete(f"/api/sessions/{session_id}")
			self.assertEqual(del_res.status_code, 200)
			self.assertTrue(del_res.json()["deleted"])

			missing_res = await client.get(f"/api/sessions/{session_id}")
			self.assertEqual(missing_res.status_code, 404)

	async def test_multi_turn_rag_chat_with_session_history(self):
		transport = ASGITransport(app=self.app)
		async with AsyncClient(transport=transport, base_url="http://test") as client:
			first_res = await client.post(
				"/api/chat",
				json={"query": "How to use HTTPException status_code?", "top_k": 2},
			)
			self.assertEqual(first_res.status_code, 200)
			first_payload = first_res.json()
			session_id = first_payload["session_id"]
			self.assertIn("[history_turns=0]", first_payload["answer"])
			self.assertGreater(len(first_payload["sources"]), 0)

			second_res = await client.post(
				"/api/chat",
				json={
					"query": "And what about APIRouter?",
					"session_id": session_id,
					"top_k": 2,
				},
			)
			self.assertEqual(second_res.status_code, 200)
			second_payload = second_res.json()
			self.assertEqual(second_payload["session_id"], session_id)
			self.assertIn("[history_turns=2]", second_payload["answer"])

			session_detail = (await client.get(f"/api/sessions/{session_id}")).json()
			self.assertEqual(len(session_detail["messages"]), 4)
			self.assertEqual(session_detail["messages"][0]["role"], "user")
			self.assertEqual(session_detail["messages"][1]["role"], "assistant")
			self.assertEqual(session_detail["messages"][2]["role"], "user")
			self.assertEqual(session_detail["messages"][3]["role"], "assistant")


if __name__ == "__main__":
	unittest.main()

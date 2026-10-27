import unittest
from typing import List
from src.embeddings.embedder import BaseEmbedder
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class DummyEmbedder(BaseEmbedder):
	"""Deterministic dummy embedder for unit testing."""

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		return [self.embed_query(t) for t in texts]

	def embed_query(self, text: str) -> List[float]:
		# Dummy 3-dimensional vector based on word presence
		v1 = 1.0 if "fastapi" in text.lower() else 0.0
		v2 = 1.0 if "route" in text.lower() or "router" in text.lower() else 0.0
		v3 = 1.0 if "response" in text.lower() else 0.0
		norm = (v1*v1 + v2*v2 + v3*v3) ** 0.5
		if norm == 0:
			return [0.33, 0.33, 0.33]
		return [v1 / norm, v2 / norm, v3 / norm]


class TestHybridRetrieval(unittest.TestCase):

	def setUp(self):
		self.corpus = [
			{"id": "doc1", "text": "FastAPI is a modern web framework for Python."},
			{"id": "doc2", "text": "APIRouter allows structuring large applications with routes."},
			{"id": "doc3", "text": "Custom response classes include JSONResponse and HTMLResponse."},
		]

	def test_bm25_fast(self):
		bm25 = BM25Retriever()
		bm25.index(self.corpus)

		results = bm25.retrieve("APIRouter routes", top_k=2)
		self.assertGreater(len(results), 0)
		self.assertEqual(results[0]["id"], "doc2")

	def test_hybrid_rrf(self):
		bm25 = BM25Retriever()
		dense = DenseRetriever(embedder=DummyEmbedder())
		hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=bm25, fusion_method="rrf")

		hybrid.index(self.corpus)
		results = hybrid.retrieve("FastAPI", top_k=2)

		self.assertGreater(len(results), 0)
		self.assertIn("hybrid_score", results[0])
		self.assertEqual(results[0]["id"], "doc1")

	def test_hybrid_linear(self):
		bm25 = BM25Retriever()
		dense = DenseRetriever(embedder=DummyEmbedder())
		hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=bm25, fusion_method="linear")

		hybrid.index(self.corpus)
		results = hybrid.retrieve("response", top_k=2)

		self.assertGreater(len(results), 0)
		self.assertEqual(results[0]["id"], "doc3")


if __name__ == "__main__":
	unittest.main()

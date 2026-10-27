import asyncio
import unittest
from src.database import (
	Base,
	DocumentRepository,
	QueryLogRepository,
	create_engine_and_session,
	init_db,
)


class TestDatabaseModelsAndRepositories(unittest.IsolatedAsyncioTestCase):

	async def asyncSetUp(self):
		# Use in-memory SQLite database for isolated unit testing
		self.engine, self.session_factory = create_engine_and_session(
			database_url="sqlite+aiosqlite:///:memory:",
			echo=False,
		)
		await init_db(self.engine)

	async def asyncTearDown(self):
		await self.engine.dispose()

	async def test_save_and_retrieve_document(self):
		async with self.session_factory() as session:
			repo = DocumentRepository(session)
			chunks_data = [
				{"text": "FastAPI is fast.", "metadata": {"section": "intro"}},
				{"text": "APIRouter is modular.", "metadata": {"section": "routing"}},
			]
			doc = await repo.save_document(
				title="FastAPI Guide",
				url="https://fastapi.tiangolo.com/reference/fastapi/",
				file_path="data/fastapi.html",
				chunks_data=chunks_data,
			)
			await session.commit()
			self.assertIsNotNone(doc.id)
			self.assertEqual(len(doc.chunks), 2)

		# Verify retrieval in a new session
		async with self.session_factory() as session:
			repo = DocumentRepository(session)
			fetched = await repo.get_by_url("https://fastapi.tiangolo.com/reference/fastapi/")
			self.assertIsNotNone(fetched)
			self.assertEqual(fetched.title, "FastAPI Guide")
			self.assertEqual(len(fetched.chunks), 2)

	async def test_get_all_chunks_for_indexing(self):
		async with self.session_factory() as session:
			repo = DocumentRepository(session)
			await repo.save_document(
				title="FastAPI Guide",
				url="https://fastapi.tiangolo.com/test/",
				chunks_data=[{"text": "Chunk 1", "metadata": {"idx": 0}}],
			)
			await session.commit()

			corpus = await repo.get_all_chunks_for_indexing()
			self.assertGreaterEqual(len(corpus), 1)
			self.assertEqual(corpus[0]["text"], "Chunk 1")
			self.assertEqual(corpus[0]["metadata"]["doc_title"], "FastAPI Guide")

	async def test_query_logging(self):
		async with self.session_factory() as session:
			repo = QueryLogRepository(session)
			log = await repo.log_query(
				query="What is APIRouter?",
				response="APIRouter is a mini FastAPI app.",
				retriever_used="hybrid_rrf",
				retrieved_chunks={"top_1": "doc_1_chunk_0"},
				evaluation_metrics={"faithfulness": 0.95},
				latency_seconds=0.25,
			)
			await session.commit()
			self.assertIsNotNone(log.id)
			self.assertEqual(log.retriever_used, "hybrid_rrf")
			self.assertEqual(log.evaluation_metrics["faithfulness"], 0.95)


if __name__ == "__main__":
	unittest.main()

import unittest
from src.database import (
	DocumentRepository,
	QueryLogRepository,
	create_engine_and_session,
	init_db,
)


class TestDatabaseModelsAndRepositories(unittest.IsolatedAsyncioTestCase):

	async def asyncSetUp(self):
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
			toc = [{"anchor": "fastapi.FastAPI", "title": "FastAPI", "breadcrumb": "FastAPI", "level": 1}]
			chunks_data = [
				{
					"section_title": "FastAPI",
					"anchor": "fastapi.FastAPI",
					"section_url": "https://fastapi.tiangolo.com/reference/fastapi/#fastapi.FastAPI",
					"breadcrumb": "FastAPI class > FastAPI",
					"symbol_type": "class",
					"text": "FastAPI is fast.",
					"embedding": [0.1, 0.2, 0.3],
					"metadata": {"section": "intro"},
				},
				{
					"section_title": "APIRouter",
					"anchor": "fastapi.APIRouter",
					"section_url": "https://fastapi.tiangolo.com/reference/fastapi/#fastapi.APIRouter",
					"breadcrumb": "FastAPI class > APIRouter",
					"symbol_type": "class",
					"text": "APIRouter is modular.",
					"embedding": [0.4, 0.5, 0.6],
					"metadata": {"section": "routing"},
				},
			]
			doc = await repo.save_document(
				title="FastAPI class",
				url="https://fastapi.tiangolo.com/reference/fastapi/",
				file_path="data/fastapi.html",
				toc_sections=toc,
				chunks_data=chunks_data,
			)
			await session.commit()
			self.assertIsNotNone(doc.id)
			self.assertEqual(len(doc.toc_sections), 1)
			self.assertEqual(len(doc.chunks), 2)

		async with self.session_factory() as session:
			repo = DocumentRepository(session)
			fetched = await repo.get_by_url("https://fastapi.tiangolo.com/reference/fastapi/")
			self.assertIsNotNone(fetched)
			self.assertEqual(fetched.title, "FastAPI class")
			self.assertEqual(len(fetched.chunks), 2)
			self.assertEqual(fetched.chunks[0].section_title, "FastAPI")
			self.assertEqual(fetched.chunks[0].symbol_type, "class")
			self.assertEqual(fetched.chunks[0].embedding, [0.1, 0.2, 0.3])

	async def test_update_existing_document_replaces_chunks(self):
		url = "https://fastapi.tiangolo.com/reference/status/"
		async with self.session_factory() as session:
			repo = DocumentRepository(session)
			await repo.save_document(
				title="Old Status Title",
				url=url,
				chunks_data=[
					{"section_title": "Old1", "text": "Text 1"},
					{"section_title": "Old2", "text": "Text 2"},
				],
			)
			await session.commit()

		async with self.session_factory() as session:
			repo = DocumentRepository(session)
			updated = await repo.save_document(
				title="New Status Codes",
				url=url,
				toc_sections=[{"anchor": "http_200", "title": "200 OK"}],
				chunks_data=[{"section_title": "HTTP 200", "text": "HTTP_200_OK"}],
			)
			await session.commit()
			self.assertEqual(updated.title, "New Status Codes")
			self.assertEqual(len(updated.chunks), 1)
			self.assertEqual(updated.chunks[0].section_title, "HTTP 200")

	async def test_get_all_chunks_for_indexing(self):
		async with self.session_factory() as session:
			repo = DocumentRepository(session)
			await repo.save_document(
				title="FastAPI Guide",
				url="https://fastapi.tiangolo.com/test/",
				chunks_data=[
					{
						"section_title": "HTTPException",
						"anchor": "fastapi.HTTPException",
						"text": "Chunk 1",
						"embedding": [0.5, 0.5],
						"metadata": {"idx": 0},
					}
				],
			)
			await session.commit()

			corpus = await repo.get_all_chunks_for_indexing()
			self.assertGreaterEqual(len(corpus), 1)
			self.assertEqual(corpus[0]["text"], "Chunk 1")
			self.assertEqual(corpus[0]["section_title"], "HTTPException")
			self.assertEqual(corpus[0]["embedding"], [0.5, 0.5])
			self.assertEqual(corpus[0]["metadata"]["doc_title"], "FastAPI Guide")

	async def test_get_nonexistent_document_returns_none(self):
		async with self.session_factory() as session:
			repo = DocumentRepository(session)
			doc = await repo.get_by_url("https://fastapi.tiangolo.com/nonexistent/")
			self.assertIsNone(doc)

	async def test_query_logging(self):
		async with self.session_factory() as session:
			repo = QueryLogRepository(session)
			log = await repo.log_query(
				query="What is APIRouter?",
				response="APIRouter is a mini FastAPI app.",
				retriever_used="hybrid_e5_bm25fast_reranked",
				retrieved_chunks={"top_1": "doc_1_chunk_0"},
				evaluation_metrics={"faithfulness": 0.95},
				latency_seconds=0.25,
			)
			await session.commit()
			self.assertIsNotNone(log.id)
			self.assertEqual(log.retriever_used, "hybrid_e5_bm25fast_reranked")
			self.assertEqual(log.evaluation_metrics["faithfulness"], 0.95)


if __name__ == "__main__":
	unittest.main()

"""Asynchronously ingest structural HTML sections, TOC, and GPU E5 embeddings into the database."""
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.config import config
from src.database import (
	DocumentRepository,
	create_engine_and_session,
	init_db,
)
from src.pipeline import HybridRAGPipeline


async def main():
	db_url = config.postgres_url
	print(f"Connecting to database: {db_url}")

	try:
		engine, session_factory = create_engine_and_session(db_url)
		await init_db(engine)
		print("Connected to PostgreSQL successfully.")
	except Exception as exc:
		print(f"PostgreSQL connection failed ({exc}). Falling back to {config.fallback_sqlite_url}.")
		engine, session_factory = create_engine_and_session(config.fallback_sqlite_url)
		await init_db(engine)

	active_device = config.resolve_device()
	print(f"Initializing HybridRAGPipeline on device='{active_device}'...")
	pipeline = HybridRAGPipeline(device=active_device)

	async with session_factory() as session:
		total_chunks = await pipeline.ingest_to_db(
			session=session,
			data_dir=config.data_dir,
			compute_embeddings=True,
		)
		print(f"Successfully ingested {total_chunks} chunks with GPU E5 embeddings into the database.")

		repo = DocumentRepository(session)
		corpus = await repo.get_all_chunks_for_indexing()
		has_emb = sum(1 for c in corpus if c.get("embedding"))
		print(f"Verified DB records: {len(corpus)} chunks loaded ({has_emb} with 384-d E5 vectors).")

	await engine.dispose()


if __name__ == "__main__":
	asyncio.run(main())

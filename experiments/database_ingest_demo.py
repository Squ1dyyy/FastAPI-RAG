"""Demo showing how to ingest scraped HTML documents into PostgreSQL with SQLAlchemy."""
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
from src.preprocessing import HTMLCleaner, TextChunker


async def main():
	# Connect to PostgreSQL (or in-memory SQLite if Postgres isn't running)
	db_url = config.postgres_url
	print(f"Connecting to database: {db_url}")

	try:
		engine, session_factory = create_engine_and_session(db_url)
		await init_db(engine)
		print("Connected to PostgreSQL successfully.")
	except Exception as exc:
		print(f"PostgreSQL connection failed ({exc}). Falling back to local SQLite demo database.")
		engine, session_factory = create_engine_and_session("sqlite+aiosqlite:///data/rag.db")
		await init_db(engine)

	cleaner = HTMLCleaner()
	chunker = TextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

	data_files = list(config.data_dir.glob("*.html"))
	print(f"Ingesting {len(data_files)} documentation files into database...")

	async with session_factory() as session:
		repo = DocumentRepository(session)
		count = 0
		for f in data_files:
			if f.name == "cached_routers.html":
				continue
			text = cleaner.clean_file(f)
			chunks = chunker.split_text(text, metadata={"file": f.name})
			title = f.stem.replace("_", " ").title()
			url = f"https://fastapi.tiangolo.com/reference/{f.stem}/"

			await repo.save_document(
				title=title,
				url=url,
				file_path=str(f),
				chunks_data=chunks,
			)
			count += 1
		await session.commit()
		print(f"Successfully ingested {count} documents into database.")

		# Fetch all chunks back from DB for BM25 indexing
		corpus = await repo.get_all_chunks_for_indexing()
		print(f"Retrieved {len(corpus)} chunks ready for BM25-Fast and Hybrid Retrieval indexing.")

	await engine.dispose()


if __name__ == "__main__":
	asyncio.run(main())

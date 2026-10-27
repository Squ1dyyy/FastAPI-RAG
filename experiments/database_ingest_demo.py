"""Asynchronously ingest structural HTML sections and TOC from `data/` into PostgreSQL with SQLAlchemy."""
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
from src.preprocessing import FastAPIHTMLParser


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

	parser = FastAPIHTMLParser()
	data_files = [
		f for f in sorted(config.data_dir.glob("*.html"))
		if f.name != config.cached_routers_filename
	]
	print(f"Asynchronously parsing structural sections from {len(data_files)} files...")

	parsed_docs = await asyncio.gather(*[parser.parse_file_async(f) for f in data_files])

	async with session_factory() as session:
		repo = DocumentRepository(session)
		total_toc = 0
		for parsed in parsed_docs:
			await repo.save_document(
				title=parsed["title"],
				url=parsed["url"],
				file_path=parsed["file_path"],
				toc_sections=parsed["toc"],
				chunks_data=parsed["chunks"],
			)
			total_toc += len(parsed["toc"])

		await session.commit()
		print(f"Ingested {len(parsed_docs)} documents ({total_toc} TOC items from md-sidebar__scrollwrap).")

		corpus = await repo.get_all_chunks_for_indexing()
		print(f"Retrieved {len(corpus)} structural section chunks ready for Hybrid RAG indexing.")

	await engine.dispose()


if __name__ == "__main__":
	asyncio.run(main())

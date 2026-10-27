"""Ingest structural HTML sections and TOC from `data/` into PostgreSQL with SQLAlchemy."""
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
		print(f"PostgreSQL connection failed ({exc}). Falling back to local SQLite database.")
		engine, session_factory = create_engine_and_session("sqlite+aiosqlite:///data/rag.db")
		await init_db(engine)

	parser = FastAPIHTMLParser()
	data_files = sorted(config.data_dir.glob("*.html"))
	print(f"Parsing structural sections (md-sidebar__scrollwrap + md-content__inner) from {len(data_files)} files...")

	async with session_factory() as session:
		repo = DocumentRepository(session)
		doc_count = 0
		total_toc = 0
		for f in data_files:
			if f.name == "cached_routers.html":
				continue
			parsed = parser.parse_file(f)
			await repo.save_document(
				title=parsed["title"],
				url=parsed["url"],
				file_path=parsed["file_path"],
				toc_sections=parsed["toc"],
				chunks_data=parsed["chunks"],
			)
			doc_count += 1
			total_toc += len(parsed["toc"])

		await session.commit()
		print(f"Ingested {doc_count} documents ({total_toc} TOC items from md-sidebar__scrollwrap).")

		corpus = await repo.get_all_chunks_for_indexing()
		print(f"Retrieved {len(corpus)} structural section chunks ready for Hybrid RAG indexing.")

	await engine.dispose()


if __name__ == "__main__":
	asyncio.run(main())

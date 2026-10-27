from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
	AsyncEngine,
	AsyncSession,
	async_sessionmaker,
	create_async_engine,
)

from configs.config import config
from src.database.models.base import Base


def create_engine_and_session(
	database_url: Optional[str] = None,
	echo: Optional[bool] = None,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
	"""Create an async SQLAlchemy engine and session factory."""
	url = database_url or config.postgres_url
	is_echo = echo if echo is not None else config.postgres_echo

	engine = create_async_engine(
		url,
		echo=is_echo,
		future=True,
		pool_pre_ping=True,
	)

	session_factory = async_sessionmaker(
		bind=engine,
		class_=AsyncSession,
		expire_on_commit=False,
		autoflush=False,
	)

	return engine, session_factory


# Default application engine and session factory
engine, async_session_factory = create_engine_and_session()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
	"""Dependency that yields an async database session with automatic commit/rollback."""
	async with async_session_factory() as session:
		try:
			yield session
			await session.commit()
		except Exception:
			await session.rollback()
			raise


async def init_db(target_engine: Optional[AsyncEngine] = None):
	"""Create all database tables based on registered models."""
	eng = target_engine or engine
	async with eng.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)

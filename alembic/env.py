import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from configs.config import config as app_config
from src.database.models import Base

alembic_cfg = context.config

if alembic_cfg.config_file_name is not None:
	fileConfig(alembic_cfg.config_file_name)

target_metadata = Base.metadata


def get_database_url() -> str:
	configured_url = alembic_cfg.get_main_option("sqlalchemy.url")
	return configured_url or app_config.postgres_url


def run_migrations_offline() -> None:
	"""Run migrations in 'offline' mode."""
	url = get_database_url()
	context.configure(
		url=url,
		target_metadata=target_metadata,
		literal_binds=True,
		dialect_opts={"paramstyle": "named"},
	)

	with context.begin_transaction():
		context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
	context.configure(connection=connection, target_metadata=target_metadata)

	with context.begin_transaction():
		context.run_migrations()


async def run_async_migrations() -> None:
	"""Create an async engine and run migrations over the connection."""
	db_url = get_database_url()
	connectable = create_async_engine(db_url, poolclass=pool.NullPool)

	try:
		async with connectable.connect() as connection:
			await connection.run_sync(do_run_migrations)
	except Exception:
		await connectable.dispose()
		fallback_engine = create_async_engine(
			app_config.fallback_sqlite_url,
			poolclass=pool.NullPool,
		)
		async with fallback_engine.connect() as connection:
			await connection.run_sync(do_run_migrations)
		await fallback_engine.dispose()
		return

	await connectable.dispose()


def run_migrations_online() -> None:
	"""Run migrations in 'online' mode."""
	asyncio.run(run_async_migrations())


if context.is_offline_mode():
	run_migrations_offline()
else:
	run_migrations_online()

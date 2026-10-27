"""Add language column to documents, document_chunks, and chat_sessions.

Revision ID: 0003_language_support
Revises: 0002_chat_sessions
Create Date: 2026-10-27 18:28:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003_language_support"
down_revision: Union[str, Sequence[str], None] = "0002_chat_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
	op.add_column(
		"documents",
		sa.Column("language", sa.String(length=16), server_default="en", nullable=False),
	)
	op.create_index(
		op.f("ix_documents_language"), "documents", ["language"], unique=False
	)

	op.add_column(
		"document_chunks",
		sa.Column("language", sa.String(length=16), server_default="en", nullable=False),
	)
	op.create_index(
		op.f("ix_document_chunks_language"),
		"document_chunks",
		["language"],
		unique=False,
	)

	op.add_column(
		"chat_sessions",
		sa.Column("language", sa.String(length=16), server_default="ru", nullable=False),
	)
	op.create_index(
		op.f("ix_chat_sessions_language"),
		"chat_sessions",
		["language"],
		unique=False,
	)


def downgrade() -> None:
	op.drop_index(op.f("ix_chat_sessions_language"), table_name="chat_sessions")
	op.drop_column("chat_sessions", "language")

	op.drop_index(op.f("ix_document_chunks_language"), table_name="document_chunks")
	op.drop_column("document_chunks", "language")

	op.drop_index(op.f("ix_documents_language"), table_name="documents")
	op.drop_column("documents", "language")

"""Add chat_sessions and chat_messages tables for conversational RAG sessions.

Revision ID: 0002_chat_sessions
Revises: 0001_initial
Create Date: 2026-10-27 18:07:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002_chat_sessions"
down_revision: Union[str, Sequence[str], None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
	json_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

	op.create_table(
		"chat_sessions",
		sa.Column("id", sa.String(length=64), nullable=False),
		sa.Column("title", sa.String(length=255), nullable=False),
		sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
		sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
		sa.PrimaryKeyConstraint("id"),
	)

	op.create_table(
		"chat_messages",
		sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
		sa.Column("session_id", sa.String(length=64), nullable=False),
		sa.Column("role", sa.String(length=32), nullable=False),
		sa.Column("content", sa.Text(), nullable=False),
		sa.Column("sources", json_type, nullable=False),
		sa.Column("metrics", json_type, nullable=False),
		sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
		sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
		sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
		sa.PrimaryKeyConstraint("id"),
	)
	op.create_index(
		op.f("ix_chat_messages_session_id"),
		"chat_messages",
		["session_id"],
		unique=False,
	)


def downgrade() -> None:
	op.drop_index(op.f("ix_chat_messages_session_id"), table_name="chat_messages")
	op.drop_table("chat_messages")
	op.drop_table("chat_sessions")

"""Initial database schema for documents, document_chunks, and query_logs.

Revision ID: 0001_initial
Revises: None
Create Date: 2026-10-27 17:59:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
	json_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

	op.create_table(
		"documents",
		sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
		sa.Column("title", sa.String(length=255), nullable=False),
		sa.Column("url", sa.String(length=1024), nullable=False),
		sa.Column("file_path", sa.String(length=512), nullable=True),
		sa.Column("toc_sections", json_type, nullable=False),
		sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
		sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
		sa.PrimaryKeyConstraint("id"),
	)
	op.create_index(op.f("ix_documents_url"), "documents", ["url"], unique=True)

	op.create_table(
		"document_chunks",
		sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
		sa.Column("document_id", sa.Integer(), nullable=False),
		sa.Column("chunk_index", sa.Integer(), nullable=False),
		sa.Column("section_title", sa.String(length=255), nullable=True),
		sa.Column("anchor", sa.String(length=255), nullable=True),
		sa.Column("section_url", sa.String(length=1024), nullable=True),
		sa.Column("breadcrumb", sa.String(length=1024), nullable=True),
		sa.Column("symbol_type", sa.String(length=64), nullable=True),
		sa.Column("text", sa.Text(), nullable=False),
		sa.Column("embedding", json_type, nullable=True),
		sa.Column("metadata_json", json_type, nullable=False),
		sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
		sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
		sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
		sa.PrimaryKeyConstraint("id"),
	)
	op.create_index(
		op.f("ix_document_chunks_anchor"), "document_chunks", ["anchor"], unique=False
	)
	op.create_index(
		op.f("ix_document_chunks_document_id"),
		"document_chunks",
		["document_id"],
		unique=False,
	)
	op.create_index(
		op.f("ix_document_chunks_section_title"),
		"document_chunks",
		["section_title"],
		unique=False,
	)

	op.create_table(
		"query_logs",
		sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
		sa.Column("query", sa.Text(), nullable=False),
		sa.Column("response", sa.Text(), nullable=True),
		sa.Column("retriever_used", sa.String(length=50), nullable=True),
		sa.Column("retrieved_chunks", json_type, nullable=False),
		sa.Column("evaluation_metrics", json_type, nullable=False),
		sa.Column("latency_seconds", sa.Float(), nullable=True),
		sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
		sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
		sa.PrimaryKeyConstraint("id"),
	)
	op.create_index(op.f("ix_query_logs_query"), "query_logs", ["query"], unique=False)


def downgrade() -> None:
	op.drop_index(op.f("ix_query_logs_query"), table_name="query_logs")
	op.drop_table("query_logs")
	op.drop_index(
		op.f("ix_document_chunks_section_title"), table_name="document_chunks"
	)
	op.drop_index(
		op.f("ix_document_chunks_document_id"), table_name="document_chunks"
	)
	op.drop_index(op.f("ix_document_chunks_anchor"), table_name="document_chunks")
	op.drop_table("document_chunks")
	op.drop_index(op.f("ix_documents_url"), table_name="documents")
	op.drop_table("documents")

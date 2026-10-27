from typing import Any, Dict, List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.models.document import Document, DocumentChunk


class DocumentRepository:
	"""Repository handling database operations for documents and chunks."""

	def __init__(self, session: AsyncSession):
		self.session = session

	async def get_by_url(self, url: str) -> Optional[Document]:
		"""Find a document by its reference URL with chunks eagerly loaded."""
		stmt = (
			select(Document)
			.where(Document.url == url)
			.options(selectinload(Document.chunks))
		)
		result = await self.session.execute(stmt)
		return result.scalar_one_or_none()

	async def save_document(
		self,
		title: str,
		url: str,
		file_path: Optional[str] = None,
		chunks_data: Optional[List[Dict[str, Any]]] = None,
	) -> Document:
		"""Save or update a document and its text chunks."""
		doc = await self.get_by_url(url)

		chunks = [
			DocumentChunk(
				chunk_index=idx,
				text=chunk_item["text"],
				metadata_json=chunk_item.get("metadata", {}),
			)
			for idx, chunk_item in enumerate(chunks_data or [])
		]

		if doc is None:
			doc = Document(
				title=title,
				url=url,
				file_path=file_path,
				chunks=chunks,
			)
			self.session.add(doc)
		else:
			doc.title = title
			doc.file_path = file_path
			await self.session.execute(
				delete(DocumentChunk).where(DocumentChunk.document_id == doc.id)
			)
			for chunk in chunks:
				chunk.document_id = doc.id
			self.session.add_all(chunks)
			doc.chunks = chunks

		await self.session.flush()
		return doc

	async def get_all_chunks_for_indexing(self) -> List[Dict[str, Any]]:
		"""Retrieve all document chunks formatted for BM25 and Dense retrievers.

		:return: List of dicts with 'id', 'text', 'metadata' (including doc_title and doc_url).
		"""
		stmt = (
			select(DocumentChunk, Document)
			.join(Document, DocumentChunk.document_id == Document.id)
			.order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
		)
		result = await self.session.execute(stmt)

		corpus = []
		for chunk, doc in result.all():
			corpus.append({
				"id": f"doc_{doc.id}_chunk_{chunk.chunk_index}",
				"chunk_id": chunk.id,
				"document_id": doc.id,
				"text": chunk.text,
				"metadata": {
					**chunk.metadata_json,
					"doc_title": doc.title,
					"doc_url": doc.url,
					"chunk_index": chunk.chunk_index,
				},
			})
		return corpus

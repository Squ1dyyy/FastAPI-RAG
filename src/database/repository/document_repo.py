from typing import Any, Dict, List, Optional
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.models.document import Document, DocumentChunk


class DocumentRepository:
	"""Repository handling database operations for documents and structural chunks."""

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
		toc_sections: Optional[List[Dict[str, Any]]] = None,
		chunks_data: Optional[List[Dict[str, Any]]] = None,
	) -> Document:
		"""Save or update a document, its TOC sections, and its structural text chunks."""
		doc = await self.get_by_url(url)

		chunks = []
		for idx, chunk_item in enumerate(chunks_data or []):
			meta = chunk_item.get("metadata", {})
			chunks.append(
				DocumentChunk(
					chunk_index=idx,
					section_title=chunk_item.get("section_title") or meta.get("section_title"),
					anchor=chunk_item.get("anchor") or meta.get("anchor"),
					section_url=chunk_item.get("section_url") or meta.get("section_url") or url,
					breadcrumb=chunk_item.get("breadcrumb") or meta.get("breadcrumb") or title,
					symbol_type=chunk_item.get("symbol_type") or meta.get("symbol_type") or "section",
					text=chunk_item["text"],
					embedding=chunk_item.get("embedding"),
					metadata_json=meta,
				)
			)

		if doc is None:
			doc = Document(
				title=title,
				url=url,
				file_path=file_path,
				toc_sections=toc_sections or [],
				chunks=chunks,
			)
			self.session.add(doc)
		else:
			doc.title = title
			doc.file_path = file_path
			if toc_sections is not None:
				doc.toc_sections = toc_sections
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
		"""Retrieve all document chunks formatted for BM25-Fast and Dense E5 retrievers.

		:return: List of dicts with 'id', 'text', 'embedding', 'section_title', 'anchor',
		         'section_url', 'breadcrumb', and 'metadata'.
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
				"doc_title": doc.title,
				"doc_url": doc.url,
				"section_title": chunk.section_title or doc.title,
				"anchor": chunk.anchor or "",
				"section_url": chunk.section_url or doc.url,
				"breadcrumb": chunk.breadcrumb or doc.title,
				"symbol_type": chunk.symbol_type or "section",
				"text": chunk.text,
				"embedding": chunk.embedding,
				"metadata": {
					**chunk.metadata_json,
					"doc_title": doc.title,
					"doc_url": doc.url,
					"section_title": chunk.section_title or doc.title,
					"anchor": chunk.anchor or "",
					"section_url": chunk.section_url or doc.url,
					"breadcrumb": chunk.breadcrumb or doc.title,
					"symbol_type": chunk.symbol_type or "section",
					"chunk_index": chunk.chunk_index,
				},
			})
		return corpus

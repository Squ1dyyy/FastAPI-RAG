from typing import Any, Dict, List, Optional
from sqlalchemy import delete, func, select
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
		language: str = "en",
	) -> Document:
		"""Save or update a document, its language, TOC sections, and structural text chunks."""
		doc = await self.get_by_url(url)
		doc_lang = (language or "en").strip().lower()

		chunks = []
		for idx, chunk_item in enumerate(chunks_data or []):
			meta = chunk_item.get("metadata", {})
			ch_lang = (
				chunk_item.get("language")
				or meta.get("language")
				or doc_lang
			).strip().lower()
			chunks.append(
				DocumentChunk(
					chunk_index=idx,
					language=ch_lang,
					section_title=chunk_item.get("section_title") or meta.get("section_title"),
					anchor=chunk_item.get("anchor") or meta.get("anchor"),
					section_url=chunk_item.get("section_url") or meta.get("section_url") or url,
					breadcrumb=chunk_item.get("breadcrumb") or meta.get("breadcrumb") or title,
					symbol_type=chunk_item.get("symbol_type") or meta.get("symbol_type") or "section",
					text=chunk_item["text"],
					embedding=chunk_item.get("embedding"),
					metadata_json={**meta, "language": ch_lang},
				)
			)

		if doc is None:
			doc = Document(
				title=title,
				url=url,
				language=doc_lang,
				file_path=file_path,
				toc_sections=toc_sections or [],
				chunks=chunks,
			)
			self.session.add(doc)
		else:
			doc.title = title
			doc.language = doc_lang
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

	async def get_language_counts(self) -> Dict[str, int]:
		"""Return a mapping of language code to number of indexed chunks in DB."""
		stmt = select(DocumentChunk.language, func.count(DocumentChunk.id)).group_by(
			DocumentChunk.language
		)
		result = await self.session.execute(stmt)
		return {str(lang): int(cnt) for lang, cnt in result.all()}

	async def get_all_chunks_for_indexing(
		self,
		language: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		"""Retrieve document chunks formatted for BM25-Fast and Dense E5 retrievers."""
		stmt = (
			select(
				DocumentChunk.id,
				DocumentChunk.document_id,
				DocumentChunk.chunk_index,
				DocumentChunk.language,
				DocumentChunk.section_title,
				DocumentChunk.anchor,
				DocumentChunk.section_url,
				DocumentChunk.breadcrumb,
				DocumentChunk.symbol_type,
				DocumentChunk.text,
				DocumentChunk.embedding,
				DocumentChunk.metadata_json,
				Document.title,
				Document.url,
			)
			.join(Document, DocumentChunk.document_id == Document.id)
			.order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
		)
		if language and language.lower() != "all":
			stmt = stmt.where(DocumentChunk.language == language.strip().lower())

		result = await self.session.execute(stmt)

		corpus = []
		for row in result.all():
			(
				ch_id,
				doc_id,
				ch_idx,
				ch_lang_raw,
				sec_title,
				anchor,
				sec_url,
				breadcrumb,
				sym_type,
				text,
				embedding,
				meta_json,
				doc_title,
				doc_url,
			) = row
			ch_lang = ch_lang_raw or "en"
			meta_dict = meta_json if isinstance(meta_json, dict) else {}
			corpus.append({
				"id": f"doc_{doc_id}_chunk_{ch_idx}",
				"chunk_id": ch_id,
				"document_id": doc_id,
				"language": ch_lang,
				"doc_title": doc_title,
				"doc_url": doc_url,
				"section_title": sec_title or doc_title,
				"anchor": anchor or "",
				"section_url": sec_url or doc_url,
				"breadcrumb": breadcrumb or doc_title,
				"symbol_type": sym_type or "section",
				"text": text,
				"embedding": embedding,
				"metadata": {
					**meta_dict,
					"language": ch_lang,
					"doc_title": doc_title,
					"doc_url": doc_url,
					"section_title": sec_title or doc_title,
					"anchor": anchor or "",
					"section_url": sec_url or doc_url,
					"breadcrumb": breadcrumb or doc_title,
					"symbol_type": sym_type or "section",
					"chunk_index": ch_idx,
				},
			})
		return corpus

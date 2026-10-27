from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup as BS


class HTMLCleaner:
	"""Extracts clean text and structural content from HTML pages."""

	def __init__(self, remove_tags: Optional[List[str]] = None):
		self.remove_tags = remove_tags or ["nav", "header", "footer", "script", "style", "aside"]

	def clean_html(self, html_content: str) -> str:
		"""Remove navigational and boilerplate elements, returning clean text."""
		soup = BS(html_content, "html.parser")

		for tag in self.remove_tags:
			for element in soup.find_all(tag):
				element.decompose()

		# Target main documentation content container if present
		content = soup.find("article", class_="md-content__inner") or soup.find("main") or soup.body or soup
		return content.get_text(separator="\n", strip=True)

	def clean_file(self, file_path: Path) -> str:
		"""Read and clean an HTML file from disk."""
		html = file_path.read_text(encoding="utf-8")
		return self.clean_html(html)


class TextChunker:
	"""Splits text into overlapping chunks for embedding and retrieval."""

	def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap

	def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""Split text into chunks with metadata."""
		chunks = []
		start = 0
		text_len = len(text)
		step = self.chunk_size - self.chunk_overlap

		while start < text_len:
			end = min(start + self.chunk_size, text_len)
			chunk_content = text[start:end].strip()

			if chunk_content:
				chunks.append({
					"text": chunk_content,
					"start_idx": start,
					"end_idx": end,
					"metadata": metadata or {},
				})

			start += step

		return chunks

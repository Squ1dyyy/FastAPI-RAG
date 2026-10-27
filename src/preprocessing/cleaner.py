import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup as BS, NavigableString, Tag

from configs.config import config


class TextChunker:
	"""Splits text into overlapping chunks for embedding and retrieval."""

	def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap

	def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""Split text into chunks with metadata."""
		chunks = []
		start = 0
		text_len = len(text)
		step = max(self.chunk_size - self.chunk_overlap, 1)

		while start < text_len:
			end = min(start + self.chunk_size, text_len)
			chunk_content = text[start:end].strip()

			if chunk_content:
				chunks.append({
					"text": chunk_content,
					"start_idx": start,
					"end_idx": end,
					"metadata": dict(metadata) if metadata else {},
				})

			if end >= text_len:
				break
			start += step

		return chunks


class FastAPIHTMLParser:
	"""Structural parser for FastAPI MkDocs / mkdocstrings HTML documentation.

	Uses:
	- `md-sidebar--secondary` -> `md-sidebar__scrollwrap` to extract the document's
	  Table of Contents (sections, anchors, and hierarchy breadcrumbs).
	- `article.md-content__inner.md-typeset` -> `h1` for the document title, and
	  section headings (`h1..h6[id]`) to split the content by logical documentation pieces.
	"""

	HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

	def __init__(
		self,
		chunk_size: Optional[int] = None,
		chunk_overlap: Optional[int] = None,
	):
		self.chunker = TextChunker(
			chunk_size=chunk_size or config.chunk_size,
			chunk_overlap=chunk_overlap or config.chunk_overlap,
		)

	@staticmethod
	def _normalize_whitespace(text: str) -> str:
		"""Collapse excessive horizontal spaces and blank lines."""
		text = re.sub(r"[ \t]+", " ", text)
		text = re.sub(r"\n{3,}", "\n\n", text)
		return text.strip()

	def extract_doc_title(self, soup: BS, fallback_name: str = "") -> str:
		"""Extract the main document title from `md-content__inner md-typeset` h1."""
		content = soup.find("article", class_="md-content__inner") or soup
		h1 = content.find("h1")
		if h1:
			h1_clone = BS(str(h1), "html.parser")
			for link in h1_clone.find_all("a", class_="headerlink"):
				link.decompose()
			title = self._normalize_whitespace(h1_clone.get_text(" ", strip=True))
			if title:
				return title

		if soup.title and soup.title.string:
			return soup.title.string.split(" - ")[0].strip()

		return fallback_name

	@staticmethod
	def extract_canonical_url(soup: BS, file_stem: str = "") -> str:
		"""Extract canonical URL from `<link rel='canonical'>` or build from reference_url."""
		canonical = soup.find("link", rel="canonical")
		if canonical and canonical.get("href"):
			return str(canonical["href"]).strip()
		clean_path = file_stem.replace("_", "/").strip("/")
		return urljoin(config.reference_url, f"{clean_path}/") if clean_path else config.reference_url

	def parse_toc_sidebar(self, soup: BS) -> List[Dict[str, Any]]:
		"""Parse `md-sidebar__scrollwrap` inside `md-sidebar--secondary` to get TOC pieces.

		:return: List of dicts with `anchor`, `title`, `breadcrumb`, `level`.
		"""
		secondary_sidebar = soup.find("div", class_="md-sidebar--secondary")
		if not secondary_sidebar:
			return []

		scrollwrap = secondary_sidebar.find("div", class_="md-sidebar__scrollwrap") or secondary_sidebar
		nav = scrollwrap.find("nav", class_="md-nav--secondary") or scrollwrap
		root_ul = nav.find("ul", class_="md-nav__list")
		if not root_ul:
			return []

		toc_items: List[Dict[str, Any]] = []

		def _walk_list(ul_tag: Tag, parents: List[str], level: int):
			for li in ul_tag.find_all("li", class_="md-nav__item", recursive=False):
				link = li.find("a", class_="md-nav__link", recursive=False)
				current_parents = list(parents)
				if link and link.get("href"):
					href = str(link["href"]).strip()
					if href.startswith("#"):
						anchor = href.lstrip("#")
						span = link.find("span", class_="md-ellipsis")
						title = self._normalize_whitespace(
							(span or link).get_text(" ", strip=True)
						)
						if anchor and title:
							breadcrumb_parts = current_parents + [title]
							toc_items.append({
								"anchor": anchor,
								"title": title,
								"breadcrumb": " > ".join(breadcrumb_parts),
								"level": level,
							})
							current_parents = breadcrumb_parts

				nested_nav = li.find("nav", class_="md-nav", recursive=False)
				if nested_nav:
					nested_ul = nested_nav.find("ul", class_="md-nav__list", recursive=False)
					if nested_ul:
						_walk_list(nested_ul, current_parents, level + 1)

		_walk_list(root_ul, [], 1)
		return toc_items

	def _prepare_article_dom(self, article: Tag) -> Tag:
		"""Clean noisy tags and format code/tables inside `md-content__inner md-typeset`."""
		clone = BS(str(article), "html.parser")
		root = clone.find("article") or clone

		# 1. Remove permanent link symbols (¶)
		for headerlink in root.find_all("a", class_="headerlink"):
			headerlink.decompose()

		# 2. Replace collapsible raw source code (<details class="mkdocstrings-source">)
		# with its concise summary ("Source code in fastapi/...") to avoid duplicating docstrings
		for details in root.find_all("details", class_="mkdocstrings-source"):
			summary = details.find("summary")
			summary_text = self._normalize_whitespace(summary.get_text(" ", strip=True)) if summary else ""
			replacement = clone.new_tag("p")
			replacement.string = f"[{summary_text}]" if summary_text else ""
			details.replace_with(replacement)

		# 3. Remove line number spans inside code blocks
		for linenos in root.find_all("span", class_="linenos"):
			linenos.decompose()

		# 4. Format parameter tables cleanly as structured bullet lines
		# and place them before docstring subheadings (e.g. h4 Example) so they stay with the parent class/function
		for table in root.find_all("table"):
			rows_text = []
			for tr in table.find_all("tr"):
				cells = tr.find_all(["th", "td"])
				if len(cells) >= 2:
					col_name = self._normalize_whitespace(cells[0].get_text(" ", strip=True))
					col_desc = self._normalize_whitespace(cells[1].get_text(" ", strip=True))
					if col_name.upper() == "PARAMETER" and col_desc.upper() == "DESCRIPTION":
						continue
					rows_text.append(f"- {col_name}: {col_desc}")
			if rows_text:
				p_tag = clone.new_tag("p")
				p_tag.string = "Parameters:\n" + "\n".join(rows_text)
				parent_contents = table.find_parent("div", class_="doc-contents")
				first_subhead = (
					parent_contents.find(["h4", "h5", "h6"], recursive=False)
					if parent_contents
					else None
				)
				if first_subhead is not None:
					first_subhead.insert_before(p_tag)
					table.decompose()
				else:
					table.replace_with(p_tag)

		return root

	@staticmethod
	def _extract_symbol_type(heading: Tag) -> str:
		"""Extract symbol type (class, function, method, attribute) from heading classes."""
		code_sym = heading.find("code", class_="doc-symbol")
		if code_sym and code_sym.get("class"):
			for cls_name in code_sym.get("class", []):
				if cls_name.startswith("doc-symbol-") and cls_name not in (
					"doc-symbol",
					"doc-symbol-heading",
				):
					return cls_name.replace("doc-symbol-", "")
		return "section"

	def parse_html_sections(
		self,
		html_content: str,
		doc_url: Optional[str] = None,
		fallback_title: str = "",
	) -> Dict[str, Any]:
		"""Parse an HTML document into metadata, TOC list, and structured section chunks.

		:return: Dict with keys: `title`, `url`, `toc`, `sections`, `chunks`.
		"""
		soup = BS(html_content, "html.parser")
		doc_title = self.extract_doc_title(soup, fallback_name=fallback_title)
		base_url = doc_url or self.extract_canonical_url(soup, file_stem=fallback_title)
		toc_list = self.parse_toc_sidebar(soup)
		toc_map = {item["anchor"]: item for item in toc_list}

		article = soup.find("article", class_="md-content__inner") or soup.find("main") or soup.body or soup
		cleaned_root = self._prepare_article_dom(article)

		sections: List[Dict[str, Any]] = []
		current_section: Dict[str, Any] = {
			"anchor": "",
			"section_title": doc_title,
			"breadcrumb": doc_title,
			"symbol_type": "overview",
			"signature": "",
			"lines": [],
		}

		def _flush_current():
			raw_body = "\n".join(line for line in current_section["lines"] if line.strip())
			body = self._normalize_whitespace(raw_body)
			if body:
				anchor = current_section["anchor"]
				section_url = f"{base_url.rstrip('/')}/#{anchor}" if anchor else base_url
				sections.append({
					"doc_title": doc_title,
					"doc_url": base_url,
					"section_title": current_section["section_title"],
					"anchor": anchor,
					"section_url": section_url,
					"breadcrumb": current_section["breadcrumb"],
					"symbol_type": current_section["symbol_type"],
					"signature": current_section["signature"],
					"text": body,
				})

		def _walk_dom(node: Tag):
			nonlocal current_section
			for child in node.children:
				if isinstance(child, NavigableString):
					txt = str(child).strip()
					if txt:
						current_section["lines"].append(txt)
					continue

				if not isinstance(child, Tag):
					continue

				# Check if this element is a section heading
				if child.name in self.HEADING_TAGS:
					heading_id = child.get("id", "").strip()
					heading_text = self._normalize_whitespace(child.get_text(" ", strip=True))
					if child.name == "h1":
						current_section["anchor"] = heading_id
						current_section["section_title"] = doc_title
						current_section["breadcrumb"] = doc_title
						current_section["lines"].append(f"# {doc_title}")
					elif heading_id:
						_flush_current()
						toc_info = toc_map.get(heading_id)
						sec_title = toc_info["title"] if toc_info else heading_text
						sec_crumb = (
							f"{doc_title} > {toc_info['breadcrumb']}"
							if toc_info
							else f"{doc_title} > {sec_title}"
						)
						sym_type = self._extract_symbol_type(child)
						current_section = {
							"anchor": heading_id,
							"section_title": sec_title,
							"breadcrumb": sec_crumb,
							"symbol_type": sym_type,
							"signature": "",
							"lines": [f"## {heading_text}"],
						}
					else:
						current_section["lines"].append(f"### {heading_text}")
					continue

				# Handle signature block
				classes = child.get("class", [])
				if "doc-signature" in classes:
					sig_text = child.get_text().strip()
					if sig_text:
						current_section["signature"] = self._normalize_whitespace(sig_text)
						current_section["lines"].append(f"```python\n{sig_text}\n```")
					continue

				# Handle code highlight block
				if "highlight" in classes:
					code_text = child.get_text().strip()
					if code_text:
						current_section["lines"].append(f"```python\n{code_text}\n```")
					continue

				# Leaf block elements
				if child.name in {"p", "li", "pre"}:
					# If a <p> or <li> does not contain nested headings, extract its text
					if not child.find(list(self.HEADING_TAGS)):
						blk_text = self._normalize_whitespace(child.get_text(" ", strip=True))
						if blk_text:
							prefix = "- " if child.name == "li" else ""
							current_section["lines"].append(f"{prefix}{blk_text}")
						continue

				# Recurse into containers (divs, articles, ul, ol, doc-contents, doc-children)
				_walk_dom(child)

		_walk_dom(cleaned_root)
		_flush_current()

		# Convert sections into chunks (splitting oversized sections while preserving context header)
		chunks: List[Dict[str, Any]] = []
		for sec in sections:
			header_prefix = f"[{sec['breadcrumb']}]\n"
			full_sec_text = f"{header_prefix}{sec['text']}"
			base_meta = {
				"doc_title": sec["doc_title"],
				"doc_url": sec["doc_url"],
				"section_title": sec["section_title"],
				"anchor": sec["anchor"],
				"section_url": sec["section_url"],
				"breadcrumb": sec["breadcrumb"],
				"symbol_type": sec["symbol_type"],
				"signature": sec["signature"],
			}

			if len(full_sec_text) <= self.chunker.chunk_size:
				chunks.append({
					"section_title": sec["section_title"],
					"anchor": sec["anchor"],
					"section_url": sec["section_url"],
					"breadcrumb": sec["breadcrumb"],
					"symbol_type": sec["symbol_type"],
					"text": full_sec_text,
					"metadata": base_meta,
				})
			else:
				sub_chunks = self.chunker.split_text(sec["text"], metadata=base_meta)
				for sub_idx, sc in enumerate(sub_chunks):
					sc_meta = dict(sc["metadata"])
					sc_meta["sub_chunk_index"] = sub_idx
					sc_meta["start_idx"] = sc["start_idx"]
					sc_meta["end_idx"] = sc["end_idx"]
					chunks.append({
						"section_title": sec["section_title"],
						"anchor": sec["anchor"],
						"section_url": sec["section_url"],
						"breadcrumb": sec["breadcrumb"],
						"symbol_type": sec["symbol_type"],
						"text": f"{header_prefix}{sc['text']}",
						"metadata": sc_meta,
					})

		return {
			"title": doc_title,
			"url": base_url,
			"toc": toc_list,
			"sections": sections,
			"chunks": chunks,
		}

	def parse_file(self, file_path: Path, doc_url: Optional[str] = None) -> Dict[str, Any]:
		"""Parse an HTML file from disk into structured document metadata, TOC, and chunks."""
		html = file_path.read_text(encoding="utf-8")
		parsed = self.parse_html_sections(
			html_content=html,
			doc_url=doc_url,
			fallback_title=file_path.stem,
		)
		parsed["file_path"] = str(file_path)
		for ch in parsed["chunks"]:
			ch["metadata"]["file_path"] = str(file_path)
		return parsed


class HTMLCleaner:
	"""Extracts clean text and structural content from HTML pages."""

	def __init__(self, remove_tags: Optional[List[str]] = None):
		self.remove_tags = remove_tags or ["nav", "header", "footer", "script", "style", "aside"]
		self.parser = FastAPIHTMLParser()

	def clean_html(self, html_content: str) -> str:
		"""Remove navigational and boilerplate elements, returning clean text."""
		soup = BS(html_content, "html.parser")

		for tag in self.remove_tags:
			for element in soup.find_all(tag):
				element.decompose()

		content = soup.find("article", class_="md-content__inner") or soup.find("main") or soup.body or soup
		for headerlink in content.find_all("a", class_="headerlink"):
			headerlink.decompose()
		return content.get_text(separator="\n", strip=True)

	def clean_file(self, file_path: Path) -> str:
		"""Read and clean an HTML file from disk."""
		html = file_path.read_text(encoding="utf-8")
		return self.clean_html(html)

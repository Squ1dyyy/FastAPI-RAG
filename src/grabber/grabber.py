import asyncio
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

from bs4 import BeautifulSoup as BS
from curl_cffi import AsyncSession

from configs.config import config


class Grabber:
	"""Web grabber for FastAPI reference documentation pages."""

	def __init__(
		self,
		path: Optional[Union[Path, str]] = None,
		reference_url: Optional[str] = None,
	):
		self.path_to_data = Path(path) if path is not None else config.data_dir
		self.reference_url = reference_url or config.reference_url
		self.cached_path = self.path_to_data / "cached_routers.html"
		self.soup: Optional[BS] = None

	async def _get_html_site(self, url: str, session: Optional[AsyncSession] = None) -> BS:
		"""Fetch HTML from the specified URL and return a BeautifulSoup object."""
		if session is not None:
			res = await session.get(url)
			res.raise_for_status()
			return BS(res.text, "html.parser")

		async with AsyncSession(impersonate="safari15_5") as local_session:
			res = await local_session.get(url)
			res.raise_for_status()
			return BS(res.text, "html.parser")

	async def _load_soup(self):
		"""Load the main reference page from cache or download it."""
		if self.cached_path.is_file():
			html = self.cached_path.read_text(encoding="utf-8")
			if "<coroutine object" not in html and len(html) > 500:
				self.soup = BS(html, "html.parser")
				return

		self.soup = await self._get_html_site(self.reference_url)
		self.cached_path.parent.mkdir(parents=True, exist_ok=True)
		self.cached_path.write_text(str(self.soup), encoding="utf-8")

	def _get_all_sources(self) -> List[Tuple[str, str]]:
		"""Extract documentation navigation links from the active section.

		:return: List of tuples containing (relative_path, text).
		"""
		active_section = self.soup.find(
			class_="md-nav__item md-nav__item--active md-nav__item--section md-nav__item--nested"
		)

		if not active_section:
			return []

		nav_sources = active_section.find_all(class_="md-nav__item")
		sources = []

		for i in nav_sources:
			link = i.find("a", class_="md-nav__link")
			span = i.find("span", class_="md-ellipsis")

			if link and span and link.get("href"):
				path = link["href"].strip()
				if path.startswith("#"):
					continue
				text = span.get_text(strip=True)
				sources.append((path, text))

		return sources

	async def parsing_all_sources(self) -> AsyncGenerator[Dict[str, Union[str, Path]], None]:
		"""Fetch, cache, and yield all reference documentation sources.

		:yield: Dictionary containing title, url, file_path, and html content.
		"""
		if self.soup is None:
			await self._load_soup()

		sources = self._get_all_sources()
		self.path_to_data.mkdir(parents=True, exist_ok=True)

		async with AsyncSession(impersonate="safari15_5") as session:
			for path, text in sources:
				full_url = urljoin(self.reference_url, path)

				# Convert relative path to filename (e.g. './fastapi/' -> 'fastapi.html')
				file_stem = path.strip("./").strip("/").replace("/", "_") or "index"
				cur_file = self.path_to_data / f"{file_stem}.html"

				if cur_file.is_file():
					html_content = cur_file.read_text(encoding="utf-8")
				else:
					page_soup = await self._get_html_site(full_url, session=session)
					html_content = str(page_soup)
					cur_file.write_text(html_content, encoding="utf-8")

				yield {
					"title": text,
					"url": full_url,
					"file_path": cur_file,
					"html": html_content,
				}


# Backwards compatibility alias
Scraper = Grabber


if __name__ == "__main__":
	async def main():
		grabber = Grabber()
		async for item in grabber.parsing_all_sources():
			print(f"{item['title']} -> {item['file_path']}")

	asyncio.run(main())

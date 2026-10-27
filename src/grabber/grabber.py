import asyncio
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

from bs4 import BeautifulSoup as BS
from curl_cffi import AsyncSession

from configs.config import config


class Grabber:
	"""Asynchronous web grabber for FastAPI reference documentation pages."""

	def __init__(
		self,
		path: Optional[Union[Path, str]] = None,
		reference_url: Optional[str] = None,
		impersonate: Optional[str] = None,
		concurrency: Optional[int] = None,
	):
		self.path_to_data = Path(path) if path is not None else config.data_dir
		self.reference_url = reference_url or config.reference_url
		self.impersonate = impersonate or config.browser_impersonate
		self.concurrency = concurrency or config.grabber_concurrency
		self.cached_path = self.path_to_data / config.cached_routers_filename
		self.soup: Optional[BS] = None

	async def _get_html_site(self, url: str, session: Optional[AsyncSession] = None) -> BS:
		"""Fetch HTML from the specified URL and return a BeautifulSoup object."""
		if session is not None:
			res = await session.get(url)
			res.raise_for_status()
			return await asyncio.to_thread(BS, res.text, "html.parser")

		async with AsyncSession(impersonate=self.impersonate) as local_session:
			res = await local_session.get(url)
			res.raise_for_status()
			return await asyncio.to_thread(BS, res.text, "html.parser")

	async def _load_soup(self):
		"""Load the main reference page from cache or download it asynchronously."""
		if self.cached_path.is_file():
			html = await asyncio.to_thread(self.cached_path.read_text, "utf-8")
			if "<coroutine object" not in html and len(html) > 500:
				self.soup = await asyncio.to_thread(BS, html, "html.parser")
				return

		self.soup = await self._get_html_site(self.reference_url)
		await asyncio.to_thread(self.cached_path.parent.mkdir, parents=True, exist_ok=True)
		await asyncio.to_thread(self.cached_path.write_text, str(self.soup), "utf-8")

	def _get_all_sources(self) -> List[Tuple[str, str]]:
		"""Extract documentation navigation links from the active section."""
		if self.soup is None:
			return []

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

	async def _process_single_source(
		self,
		path: str,
		text: str,
		session: AsyncSession,
		semaphore: asyncio.Semaphore,
	) -> Dict[str, Union[str, Path]]:
		"""Fetch or load a single documentation page from cache with concurrency control."""
		full_url = urljoin(self.reference_url, path)
		file_stem = path.strip("./").strip("/").replace("/", "_") or "index"
		cur_file = self.path_to_data / f"{file_stem}.html"

		if cur_file.is_file():
			html_content = await asyncio.to_thread(cur_file.read_text, "utf-8")
		else:
			async with semaphore:
				page_soup = await self._get_html_site(full_url, session=session)
				html_content = str(page_soup)
				await asyncio.to_thread(cur_file.write_text, html_content, "utf-8")

		return {
			"title": text,
			"url": full_url,
			"file_path": cur_file,
			"html": html_content,
		}

	async def parsing_all_sources(self) -> AsyncGenerator[Dict[str, Union[str, Path]], None]:
		"""Asynchronously fetch, cache, and yield all reference documentation sources."""
		if self.soup is None:
			await self._load_soup()

		sources = self._get_all_sources()
		await asyncio.to_thread(self.path_to_data.mkdir, parents=True, exist_ok=True)
		semaphore = asyncio.Semaphore(self.concurrency)

		async with AsyncSession(impersonate=self.impersonate) as session:
			for path, text in sources:
				item = await self._process_single_source(path, text, session, semaphore)
				yield item

	async def fetch_all_sources_concurrently(self) -> List[Dict[str, Union[str, Path]]]:
		"""Fetch and cache all documentation sources concurrently using asyncio.gather."""
		if self.soup is None:
			await self._load_soup()

		sources = self._get_all_sources()
		await asyncio.to_thread(self.path_to_data.mkdir, parents=True, exist_ok=True)
		semaphore = asyncio.Semaphore(self.concurrency)

		async with AsyncSession(impersonate=self.impersonate) as session:
			tasks = [
				self._process_single_source(path, text, session, semaphore)
				for path, text in sources
			]
			return list(await asyncio.gather(*tasks))


Scraper = Grabber


if __name__ == "__main__":
	async def main():
		grabber = Grabber()
		async for item in grabber.parsing_all_sources():
			print(f"{item['title']} -> {item['file_path']}")

	asyncio.run(main())

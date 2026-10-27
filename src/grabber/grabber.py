from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

from bs4 import BeautifulSoup as BS
from curl_cffi import AsyncSession

from configs.config import config

LANGUAGE_LABELS: Dict[str, str] = {
	"en": "en - English",
	"de": "de - Deutsch",
	"es": "es - español",
	"fr": "fr - français",
	"hi": "hi - हिन्दी",
	"ja": "ja - 日本語",
	"ko": "ko - 한국어",
	"pt": "pt - português",
	"ru": "ru - русский язык",
	"tr": "tr - Türkçe",
	"uk": "uk - українська мова",
	"zh": "zh - 简体中文",
	"zh-hant": "zh-hant - 繁體中文",
}


class Grabber:
	"""Asynchronous multi-language web grabber for FastAPI reference documentation pages."""

	def __init__(
		self,
		path: Optional[Union[Path, str]] = None,
		reference_url: Optional[str] = None,
		impersonate: Optional[str] = None,
		concurrency: Optional[int] = None,
		language: str = "en",
	):
		self.language = language.strip().lower() or "en"
		self.path_to_data = Path(path) if path is not None else config.data_dir
		self.reference_url = reference_url or self.build_reference_url(self.language)
		self.impersonate = impersonate or config.browser_impersonate
		self.concurrency = concurrency or config.grabber_concurrency
		self.cached_path = self.path_to_data / config.cached_routers_filename
		self.soup: Optional[BS] = None

	@staticmethod
	def build_reference_url(language: str) -> str:
		"""Construct the FastAPI reference base URL for a given language code."""
		lang = language.strip().strip("/").lower()
		if not lang or lang == "en":
			return "https://fastapi.tiangolo.com/reference/"
		return f"https://fastapi.tiangolo.com/{lang}/reference/"

	@classmethod
	def for_language(
		cls,
		language: str,
		base_data_dir: Optional[Union[Path, str]] = None,
		impersonate: Optional[str] = None,
		concurrency: Optional[int] = None,
	) -> "Grabber":
		"""Create a Grabber instance targeting a language-specific subdirectory `data/{lang}/`."""
		root_dir = Path(base_data_dir) if base_data_dir is not None else config.data_dir
		lang_code = language.strip().lower() or "en"
		lang_dir = root_dir / lang_code
		return cls(
			path=lang_dir,
			reference_url=cls.build_reference_url(lang_code),
			impersonate=impersonate,
			concurrency=concurrency,
			language=lang_code,
		)

	@staticmethod
	def extract_language_variations(soup: BS) -> List[Dict[str, str]]:
		"""Extract all language variations from `.md-select__link` in FastAPI HTML."""
		variations: List[Dict[str, str]] = []
		seen = set()
		for link in soup.select(".md-select__link"):
			href = (link.get("href") or "").strip()
			label = link.get_text(strip=True)
			code = href.strip("/") or "en"
			if not code and " - " in label:
				code = label.split(" - ", 1)[0].strip()
			if code in seen:
				continue
			seen.add(code)
			variations.append(
				{
					"code": code,
					"label": label or LANGUAGE_LABELS.get(code, code),
					"href": href or "/",
					"reference_url": Grabber.build_reference_url(code),
				}
			)
		return variations

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

	async def _load_soup(self, session: Optional[AsyncSession] = None):
		"""Load the main reference page from cache or download it asynchronously."""
		if self.cached_path.is_file():
			html = await asyncio.to_thread(self.cached_path.read_text, "utf-8")
			if "<coroutine object" not in html and len(html) > 500:
				self.soup = await asyncio.to_thread(BS, html, "html.parser")
				return

		self.soup = await self._get_html_site(self.reference_url, session=session)
		await asyncio.to_thread(self.cached_path.parent.mkdir, parents=True, exist_ok=True)
		await asyncio.to_thread(self.cached_path.write_text, str(self.soup), "html.parser" if False else "utf-8")

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
			"language": self.language,
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

	async def fetch_all_sources_concurrently(
		self,
		session: Optional[AsyncSession] = None,
		semaphore: Optional[asyncio.Semaphore] = None,
	) -> List[Dict[str, Union[str, Path]]]:
		"""Fetch and cache all documentation sources concurrently using asyncio.gather."""
		await asyncio.to_thread(self.path_to_data.mkdir, parents=True, exist_ok=True)
		active_sem = semaphore or asyncio.Semaphore(self.concurrency)

		if session is not None:
			if self.soup is None:
				await self._load_soup(session=session)
			sources = self._get_all_sources()
			tasks = [
				self._process_single_source(path, text, session, active_sem)
				for path, text in sources
			]
			return list(await asyncio.gather(*tasks))

		async with AsyncSession(impersonate=self.impersonate) as own_session:
			if self.soup is None:
				await self._load_soup(session=own_session)
			sources = self._get_all_sources()
			tasks = [
				self._process_single_source(path, text, own_session, active_sem)
				for path, text in sources
			]
			return list(await asyncio.gather(*tasks))

	@classmethod
	async def scrape_all_languages(
		cls,
		languages: Optional[List[str]] = None,
		base_data_dir: Optional[Union[Path, str]] = None,
		concurrency: int = 12,
	) -> Dict[str, List[Dict[str, Union[str, Path]]]]:
		"""Scrape and cache all FastAPI reference pages across all language variations into `data/{lang}/`."""
		target_langs = languages or config.get_languages()
		root_dir = Path(base_data_dir) if base_data_dir is not None else config.data_dir
		sem = asyncio.Semaphore(concurrency)
		results: Dict[str, List[Dict[str, Union[str, Path]]]] = {}

		async with AsyncSession(impersonate=config.browser_impersonate) as shared_session:
			async def _scrape_lang(lang_code: str):
				g = cls.for_language(
					language=lang_code,
					base_data_dir=root_dir,
					concurrency=concurrency,
				)
				pages = await g.fetch_all_sources_concurrently(
					session=shared_session,
					semaphore=sem,
				)
				results[lang_code] = pages

			await asyncio.gather(*[_scrape_lang(lang) for lang in target_langs])

		return results


Scraper = Grabber
AsyncGrabber = Grabber

import unittest
from bs4 import BeautifulSoup as BS
from configs.config import config
from src.grabber import Grabber
from src.preprocessing import FastAPIHTMLParser, HTMLCleaner, TextChunker


class TestGrabberAndPreprocessing(unittest.IsolatedAsyncioTestCase):

	def test_grabber_init(self):
		grabber = Grabber()
		self.assertEqual(grabber.reference_url, config.reference_url)
		self.assertEqual(grabber.path_to_data, config.data_dir)
		self.assertEqual(grabber.concurrency, config.grabber_concurrency)

	def test_grabber_extract_sources_filters_anchors(self):
		grabber = Grabber()
		nav_html = """
		<li class="md-nav__item md-nav__item--active md-nav__item--section md-nav__item--nested">
			<ul class="md-nav__list">
				<li class="md-nav__item">
					<a class="md-nav__link" href="./fastapi/"><span class="md-ellipsis">FastAPI class</span></a>
				</li>
				<li class="md-nav__item">
					<a class="md-nav__link" href="#anchor-only"><span class="md-ellipsis">Anchor</span></a>
				</li>
				<li class="md-nav__item">
					<a class="md-nav__link" href="./exceptions/"><span class="md-ellipsis">Exceptions</span></a>
				</li>
			</ul>
		</li>
		"""
		grabber.soup = BS(nav_html, "html.parser")
		sources = grabber._get_all_sources()
		self.assertEqual(len(sources), 2)
		self.assertEqual(sources[0], ("./fastapi/", "FastAPI class"))
		self.assertEqual(sources[1], ("./exceptions/", "Exceptions"))

	def test_cleaner(self):
		cleaner = HTMLCleaner()
		html = "<nav>Menu</nav><article class='md-content__inner'><p>FastAPI reference</p></article>"
		text = cleaner.clean_html(html)
		self.assertIn("FastAPI reference", text)
		self.assertNotIn("Menu", text)

	async def test_chunker_overlap_and_async(self):
		chunker = TextChunker(chunk_size=15, chunk_overlap=5)
		chunks = await chunker.split_text_async(
			"FastAPI is a modern high-performance web framework",
			metadata={"doc": "test"},
		)
		self.assertGreater(len(chunks), 2)
		self.assertEqual(chunks[0]["metadata"]["doc"], "test")
		self.assertEqual(chunks[1]["start_idx"], 10)

	def test_structural_html_parser_toc_and_sections(self):
		sample_html = """
		<html>
		<head><link rel="canonical" href="https://fastapi.tiangolo.com/reference/exceptions/"></head>
		<body>
			<div class="md-sidebar md-sidebar--secondary">
				<div class="md-sidebar__scrollwrap">
					<nav class="md-nav md-nav--secondary">
						<ul class="md-nav__list">
							<li class="md-nav__item">
								<a class="md-nav__link" href="#fastapi.HTTPException">
									<span class="md-ellipsis">HTTPException</span>
								</a>
								<nav class="md-nav">
									<ul class="md-nav__list">
										<li class="md-nav__item">
											<a class="md-nav__link" href="#fastapi.HTTPException.status_code">
												<span class="md-ellipsis">status_code</span>
											</a>
										</li>
									</ul>
								</nav>
							</li>
						</ul>
					</nav>
				</div>
			</div>
			<article class="md-content__inner md-typeset">
				<h1 id="exceptions">Exceptions<a class="headerlink" href="#exceptions">¶</a></h1>
				<p>Overview text.</p>
				<div class="doc doc-object doc-class">
					<h2 id="fastapi.HTTPException">
						<code class="doc-symbol doc-symbol-heading doc-symbol-class"></code>
						<span>fastapi.HTTPException</span>
					</h2>
					<div class="doc-signature highlight"><pre><code>HTTPException(status_code)</code></pre></div>
					<div class="doc doc-contents">
						<p>Raise HTTP errors.</p>
						<div class="doc doc-children">
							<div class="doc doc-object doc-attribute">
								<h3 id="fastapi.HTTPException.status_code">
									<code class="doc-symbol doc-symbol-heading doc-symbol-attribute"></code>
									<span>status_code</span>
								</h3>
								<div class="doc doc-contents"><p>HTTP status code int.</p></div>
							</div>
						</div>
					</div>
				</div>
			</article>
		</body>
		</html>
		"""
		parser = FastAPIHTMLParser()
		parsed = parser.parse_html_sections(sample_html)
		self.assertEqual(parsed["title"], "Exceptions")
		self.assertEqual(len(parsed["toc"]), 2)
		self.assertEqual(parsed["toc"][1]["breadcrumb"], "HTTPException > status_code")
		self.assertEqual(len(parsed["chunks"]), 3)
		self.assertEqual(parsed["chunks"][1]["symbol_type"], "class")
		self.assertEqual(parsed["chunks"][2]["symbol_type"], "attribute")
		self.assertEqual(
			parsed["chunks"][2]["breadcrumb"],
			"Exceptions > HTTPException > status_code",
		)

	def test_structural_html_parser_parameter_table_and_source_details(self):
		html = """
		<article class="md-content__inner md-typeset">
			<h1 id="doc">Test Doc</h1>
			<div class="doc doc-object doc-function">
				<h2 id="my_func">
					<code class="doc-symbol doc-symbol-heading doc-symbol-function"></code>
					<span>my_func</span>
				</h2>
				<div class="doc doc-contents">
					<p>Function description.</p>
					<h4 id="my_func--example">Example</h4>
					<div class="highlight"><pre><code>my_func(42)</code></pre></div>
					<table>
						<thead><tr><th>PARAMETER</th><th>DESCRIPTION</th></tr></thead>
						<tbody>
							<tr><td><code>val</code></td><td>The integer value TYPE: int</td></tr>
						</tbody>
					</table>
					<details class="mkdocstrings-source">
						<summary>Source code in <code>fastapi/routing.py</code></summary>
						<pre><span class="linenos">10</span>def my_func(val: int): pass</pre>
					</details>
				</div>
			</div>
		</article>
		"""
		parser = FastAPIHTMLParser()
		parsed = parser.parse_html_sections(html)
		func_chunk = next(c for c in parsed["chunks"] if c["anchor"] == "my_func")
		self.assertIn("Parameters:", func_chunk["text"])
		self.assertIn("- val: The integer value TYPE: int", func_chunk["text"])
		example_chunk = next(c for c in parsed["chunks"] if c["anchor"] == "my_func--example")
		self.assertIn("[Source code in fastapi/routing.py]", example_chunk["text"])
		self.assertNotIn("def my_func(val: int): pass", example_chunk["text"])

	def test_structural_html_parser_oversized_section_subchunking(self):
		long_text = " ".join(f"word_{i}" for i in range(150))
		html = f"""
		<article class="md-content__inner md-typeset">
			<h1 id="big">Big Doc</h1>
			<h2 id="sec1">Large Section</h2>
			<p>{long_text}</p>
		</article>
		"""
		parser = FastAPIHTMLParser(chunk_size=200, chunk_overlap=30)
		parsed = parser.parse_html_sections(html)
		sec_chunks = [c for c in parsed["chunks"] if c["anchor"] == "sec1"]
		self.assertGreater(len(sec_chunks), 2)
		for idx, ch in enumerate(sec_chunks):
			self.assertTrue(ch["text"].startswith("[Big Doc > Large Section]"))
			self.assertEqual(ch["metadata"]["sub_chunk_index"], idx)


if __name__ == "__main__":
	unittest.main()

import unittest
from pathlib import Path
from configs.config import config
from src.grabber import Grabber
from src.preprocessing import FastAPIHTMLParser, HTMLCleaner, TextChunker


class TestGrabberAndPreprocessing(unittest.TestCase):

	def test_grabber_init(self):
		grabber = Grabber()
		self.assertEqual(grabber.reference_url, config.reference_url)
		self.assertEqual(grabber.path_to_data, config.data_dir)

	def test_cleaner(self):
		cleaner = HTMLCleaner()
		html = "<article class='md-content__inner'><p>FastAPI reference</p></article>"
		text = cleaner.clean_html(html)
		self.assertIn("FastAPI reference", text)

	def test_chunker(self):
		chunker = TextChunker(chunk_size=10, chunk_overlap=2)
		chunks = chunker.split_text("FastAPI is modern and fast")
		self.assertGreater(len(chunks), 0)

	def test_structural_html_parser(self):
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
					<div class="doc doc-contents"><p>Raise HTTP errors.</p></div>
				</div>
			</article>
		</body>
		</html>
		"""
		parser = FastAPIHTMLParser()
		parsed = parser.parse_html_sections(sample_html)
		self.assertEqual(parsed["title"], "Exceptions")
		self.assertEqual(len(parsed["toc"]), 1)
		self.assertEqual(parsed["toc"][0]["anchor"], "fastapi.HTTPException")
		self.assertEqual(len(parsed["chunks"]), 2)
		self.assertEqual(parsed["chunks"][1]["section_title"], "HTTPException")
		self.assertEqual(parsed["chunks"][1]["symbol_type"], "class")


if __name__ == "__main__":
	unittest.main()

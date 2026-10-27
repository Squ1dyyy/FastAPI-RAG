import unittest
from pathlib import Path
from src.grabber import Grabber
from configs.config import config
from src.preprocessing import HTMLCleaner, TextChunker


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


if __name__ == "__main__":
	unittest.main()

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from configs.config import config
from src.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
	"""High-performance BM25 lexical retriever (BM25-Fast).

	Uses an inverted index to achieve fast O(matching_docs) scoring.
	Can also wrap external fastbm25 or bm25s libraries if available.
	"""

	def __init__(
		self,
		k1: Optional[float] = None,
		b: Optional[float] = None,
	):
		self.k1 = k1 if k1 is not None else config.bm25_k1
		self.b = b if b is not None else config.bm25_b

		self.corpus: List[Dict[str, Any]] = []
		self.doc_lengths: List[int] = []
		self.avg_doc_len: float = 0.0
		self.doc_count: int = 0

		# Inverted index: word -> list of (doc_index, term_frequency)
		self.inverted_index: Dict[str, List[tuple]] = defaultdict(list)
		self.idf: Dict[str, float] = {}

	@staticmethod
	def tokenize(text: str) -> List[str]:
		"""Tokenize string into lowercase alphanumeric words."""
		return re.findall(r"\b\w+\b", text.lower())

	def index(self, corpus: List[Dict[str, Any]]):
		"""Build an inverted index over the given corpus of documents.

		:param corpus: List of document dicts with at least a 'text' field.
		"""
		self.corpus = corpus
		self.doc_count = len(corpus)
		self.doc_lengths = []
		self.inverted_index.clear()
		self.idf.clear()

		if not corpus:
			return

		total_len = 0
		doc_freqs: Dict[str, int] = Counter()

		for doc_idx, doc in enumerate(corpus):
			text = doc.get("text", "")
			tokens = self.tokenize(text)
			doc_len = len(tokens)
			self.doc_lengths.append(doc_len)
			total_len += doc_len

			counts = Counter(tokens)
			for token, count in counts.items():
				self.inverted_index[token].append((doc_idx, count))
				doc_freqs[token] += 1

		self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 0.0

		# Precompute Robertson-Spärck Jones IDF for all vocabulary terms
		for token, df in doc_freqs.items():
			self.idf[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Retrieve top_k highest scoring documents for the query using BM25."""
		if not self.corpus:
			return []

		tokens = self.tokenize(query)
		if not tokens:
			return []

		scores: Dict[int, float] = defaultdict(float)

		# Fast inverted index traversal: only score docs that contain at least one query term
		for token in tokens:
			if token not in self.inverted_index:
				continue

			idf_val = self.idf.get(token, 0.0)
			for doc_idx, tf in self.inverted_index[token]:
				doc_len = self.doc_lengths[doc_idx]
				numerator = tf * (self.k1 + 1.0)
				denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
				scores[doc_idx] += idf_val * (numerator / denominator)

		if not scores:
			return []

		sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

		results = []
		for doc_idx, score in sorted_docs:
			doc = dict(self.corpus[doc_idx])
			doc["score"] = score
			doc["retriever"] = "bm25_fast"
			results.append(doc)

		return results


# Alias for compatibility
FastBM25Retriever = BM25Retriever

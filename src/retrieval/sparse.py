import asyncio
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from configs.config import config
from src.retrieval.base import BaseRetriever

try:
	import bm25s

	HAS_BM25S = True
except ImportError:
	HAS_BM25S = False


class BM25Retriever(BaseRetriever):
	"""High-performance BM25-Fast lexical retriever with async support."""

	def __init__(
		self,
		k1: Optional[float] = None,
		b: Optional[float] = None,
		use_bm25s: Optional[bool] = None,
	):
		self.k1 = k1 if k1 is not None else config.bm25_k1
		self.b = b if b is not None else config.bm25_b
		self.use_bm25s = (
			(use_bm25s if use_bm25s is not None else config.use_bm25s_backend)
			and HAS_BM25S
		)

		self.corpus: List[Dict[str, Any]] = []
		self._bm25s_model = None

		self.doc_lengths: List[int] = []
		self.avg_doc_len: float = 0.0
		self.doc_count: int = 0
		self.inverted_index: Dict[str, List[tuple]] = defaultdict(list)
		self.idf: Dict[str, float] = {}

	@staticmethod
	def tokenize(text: str) -> List[str]:
		"""Tokenize text into lowercase words, splitting snake_case and CamelCase identifiers."""
		raw_tokens = re.findall(r"\b\w+\b", text)
		tokens: List[str] = []
		for tok in raw_tokens:
			low = tok.lower()
			tokens.append(low)
			if "_" in low:
				parts = [p for p in low.split("_") if p]
				if len(parts) > 1:
					tokens.extend(parts)
			camel_parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+", tok)
			if len(camel_parts) > 1:
				tokens.extend(p.lower() for p in camel_parts if p)
		return tokens

	def index(self, corpus: List[Dict[str, Any]]):
		"""Build BM25-Fast index over the given corpus of document chunks."""
		self.corpus = corpus
		self.doc_count = len(corpus)
		self.doc_lengths = []
		self.inverted_index.clear()
		self.idf.clear()
		self._bm25s_model = None

		if not corpus:
			return

		tokenized_corpus: List[List[str]] = []
		total_len = 0
		doc_freqs: Dict[str, int] = Counter()

		for doc_idx, doc in enumerate(corpus):
			parts = [
				str(doc.get("section_title", "")),
				str(doc.get("anchor", "")),
				str(doc.get("breadcrumb", "")),
				str(doc.get("text", "")),
			]
			combined_text = " ".join(p for p in parts if p)
			tokens = self.tokenize(combined_text)
			if not tokens:
				tokens = ["_empty_"]

			tokenized_corpus.append(tokens)
			doc_len = len(tokens)
			self.doc_lengths.append(doc_len)
			total_len += doc_len

			counts = Counter(tokens)
			for token, count in counts.items():
				self.inverted_index[token].append((doc_idx, count))
				doc_freqs[token] += 1

		self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 0.0

		for token, df in doc_freqs.items():
			self.idf[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

		if self.use_bm25s:
			self._bm25s_model = bm25s.BM25(k1=self.k1, b=self.b)
			self._bm25s_model.index(tokenized_corpus, show_progress=False)

	async def index_async(self, corpus: List[Dict[str, Any]]):
		"""Asynchronously build BM25-Fast index."""
		await asyncio.to_thread(self.index, corpus)

	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""Retrieve top_k highest scoring documents for the query using BM25-Fast."""
		if not self.corpus:
			return []

		tokens = self.tokenize(query)
		if not tokens:
			return []

		k = min(top_k, len(self.corpus))

		if self.use_bm25s and self._bm25s_model is not None:
			doc_indices, scores = self._bm25s_model.retrieve(
				[tokens],
				k=k,
				show_progress=False,
			)
			results = []
			for idx_val, score_val in zip(doc_indices[0], scores[0]):
				score = float(score_val)
				if score <= 0.0:
					continue
				doc = dict(self.corpus[int(idx_val)])
				doc["score"] = score
				doc["retriever"] = "bm25_fast"
				results.append(doc)
			if results:
				return results

		scores_map: Dict[int, float] = defaultdict(float)
		for token in tokens:
			if token not in self.inverted_index:
				continue
			idf_val = self.idf.get(token, 0.0)
			for doc_idx, tf in self.inverted_index[token]:
				doc_len = self.doc_lengths[doc_idx]
				numerator = tf * (self.k1 + 1.0)
				denominator = tf + self.k1 * (
					1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
				)
				scores_map[doc_idx] += idf_val * (numerator / denominator)

		if not scores_map:
			return []

		sorted_docs = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)[:k]

		results = []
		for doc_idx, score in sorted_docs:
			doc = dict(self.corpus[doc_idx])
			doc["score"] = float(score)
			doc["retriever"] = "bm25_fast"
			results.append(doc)

		return results


FastBM25Retriever = BM25Retriever

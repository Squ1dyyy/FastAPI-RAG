"""Demonstration of BM25-Fast and Hybrid Search over cached FastAPI docs."""
import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.config import config
from src.preprocessing import HTMLCleaner, TextChunker
from src.retrieval import BM25Retriever


def main():
	cleaner = HTMLCleaner()
	chunker = TextChunker(chunk_size=500, chunk_overlap=100)

	corpus = []
	data_files = list(config.data_dir.glob("*.html"))
	print(f"Loading from {len(data_files)} HTML files in {config.data_dir}...")

	for file_path in data_files:
		if file_path.name == "cached_routers.html":
			continue

		text = cleaner.clean_file(file_path)
		chunks = chunker.split_text(text, metadata={"source": file_path.name})
		corpus.extend(chunks)

	print(f"Total chunks indexed: {len(corpus)}")

	# Initialize BM25-Fast
	bm25 = BM25Retriever()
	bm25.index(corpus)

	queries = [
		"APIRouter prefix and tags",
		"HTTPException status code detail",
		"BackgroundTasks add_task",
	]

	for q in queries:
		print(f"\n--- Query: '{q}' ---")
		results = bm25.retrieve(q, top_k=2)
		for idx, r in enumerate(results, 1):
			source = r.get("metadata", {}).get("source", "unknown")
			score = r.get("score", 0.0)
			preview = r.get("text", "")[:120].replace("\n", " ")
			print(f"[{idx}] (Score: {score:.3f}) [{source}] {preview}...")


if __name__ == "__main__":
	main()

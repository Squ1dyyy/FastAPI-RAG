"""Full Hybrid RAG demonstration: BM25-Fast + multilingual-e5-small + TinyBERT CrossEncoder Reranker."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import HybridRAGPipeline


def main():
	print("Initializing HybridRAGPipeline (BM25-Fast + intfloat/multilingual-e5-small + TinyBERT Reranker)...")
	pipeline = HybridRAGPipeline()

	num_chunks = pipeline.build_index_from_files()
	print(f"Indexed {num_chunks} structural documentation chunks from data/*.html")

	queries = [
		"How to raise HTTPException with custom status_code and headers?",
		"Как добавить фоновую задачу через BackgroundTasks add_task?",
		"APIRouter include_router prefix and tags parameters",
	]

	for q in queries:
		print(f"\n==========================================")
		print(f"Query: '{q}'")
		results = pipeline.search(q, top_k=3, candidate_k=10, rerank=True)
		for idx, r in enumerate(results, 1):
			breadcrumb = r.get("breadcrumb", "unknown")
			sec_url = r.get("section_url", "")
			h_score = r.get("hybrid_score", 0.0)
			r_score = r.get("rerank_score", 0.0)
			preview = r.get("text", "")[:160].replace("\n", " ")
			print(
				f"[{idx}] Rerank: {r_score:.3f} | Hybrid(RRF): {h_score:.4f} | "
				f"{breadcrumb}\n    URL: {sec_url}\n    Text: {preview}..."
			)


if __name__ == "__main__":
	main()

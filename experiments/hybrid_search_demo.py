"""Asynchronous GPU-accelerated Hybrid RAG demo with BM25-Fast, E5-small, TinyBERT Reranker, and AsyncOpenAI."""
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.config import config
from src.pipeline import HybridRAGPipeline


async def main():
	active_device = config.resolve_device()
	print(
		f"Initializing Async HybridRAGPipeline on device='{active_device}' "
		f"(BM25-Fast + {config.embedding_model_name} + {config.reranker_model_name})..."
	)
	pipeline = HybridRAGPipeline(device=active_device)

	num_chunks = await pipeline.build_index_from_files_async()
	print(f"Indexed {num_chunks} structural documentation chunks from {config.data_dir}")

	queries = [
		"How to raise HTTPException with custom status_code and headers?",
		"Как добавить фоновую задачу через BackgroundTasks add_task?",
		"APIRouter include_router prefix and tags parameters",
	]

	has_llm_key = bool(
		config.llm_api_key
		and config.llm_api_key != "your_api_key_here"
	)

	for q in queries:
		print(f"\n==========================================")
		print(f"Query: '{q}'")
		if has_llm_key:
			rag_result = await pipeline.answer(q, top_k=3)
			print(f"\n[LLM Answer ({config.llm_model})]:\n{rag_result['answer']}")
			print(f"Metrics: {rag_result['metrics']} | Latency: {rag_result['latency_seconds']}s")
			results = rag_result["context"]
		else:
			results = await pipeline.search_async(q, top_k=3, candidate_k=10, rerank=True)

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
	asyncio.run(main())

"""Launch the FastAPI Hybrid RAG chat server with Uvicorn."""
import uvicorn

from configs.config import config


if __name__ == "__main__":
	uvicorn.run(
		"src.api.app:app",
		host=config.app_host,
		port=config.app_port,
		reload=False,
	)

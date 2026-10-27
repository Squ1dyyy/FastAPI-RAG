import asyncio
from src.grabber import Grabber


async def main():
	grabber = Grabber()
	async for item in grabber.parsing_all_sources():
		print(f"{item['title']} -> {item['file_path']}")


if __name__ == "__main__":
	asyncio.run(main())

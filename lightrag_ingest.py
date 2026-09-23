import asyncio
import os
import shutil

import config
from document_loader import load_documents
from lightrag_manager import get_lightrag, insert_documents


async def ingest():
    print("=" * 60)
    print("LightRAG Ingestion — Entity/Relation Extraction & Indexing")
    print("=" * 60)

    if config.FRESH_LIGHTRAG_INDEX and os.path.exists(config.LIGHTRAG_WORKING_DIR):
        shutil.rmtree(config.LIGHTRAG_WORKING_DIR)
        print(f"FRESH_LIGHTRAG_INDEX — cleared {config.LIGHTRAG_WORKING_DIR}")

    documents = load_documents()
    if not documents:
        print("No documents found in context folder. Exiting.")
        return

    await get_lightrag()
    track_id = await insert_documents(documents)
    print(f"LightRAG insert completed — track_id: {track_id}")
    print("=" * 60)
    print(f"Ingestion complete — {len(documents)} documents indexed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(ingest())

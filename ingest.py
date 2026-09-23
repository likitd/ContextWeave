import asyncio
import shutil
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from document_loader import load_documents
from hybrid_retriever import create_embedding_function, create_vector_store
from agentfs_manager import AgentFSManager


async def ingest():
    print("=" * 60)
    print("Data Ingestion — Embedding & Indexing")
    print("=" * 60)

    if config.FRESH_INDEX and os.path.exists(config.CHROMA_PERSIST_DIR):
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
        print(f"FRESH_INDEX=True — cleared existing ChromaDB at {config.CHROMA_PERSIST_DIR}")

    documents = load_documents()
    if not documents:
        print("No documents found in context folder. Exiting.")
        return

    if config.USE_CHUNKING:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
    else:
        chunks = documents
        print(f"USE_CHUNKING=False — using {len(chunks)} full documents (no splitting)")

    embedding_fn = create_embedding_function()
    create_vector_store(chunks, embedding_fn)

    agentfs = AgentFSManager()
    await agentfs.initialize()
    await agentfs.ingest_to_agentfs(documents)
    await agentfs.close()

    print("=" * 60)
    print(f"Ingestion complete — {len(documents)} documents, {len(chunks)} chunks indexed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(ingest())

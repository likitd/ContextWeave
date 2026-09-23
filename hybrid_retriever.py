from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
import config


def create_embedding_function() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.EMBEDDING_BASE_URL,
    )


def create_vector_store(chunks: list[Document], embedding_fn: OllamaEmbeddings = None) -> Chroma:
    embedding_fn = embedding_fn or create_embedding_function()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    print(f"ChromaDB vector store created with {len(chunks)} chunks")
    return vector_store


def load_existing_vector_store(embedding_fn: OllamaEmbeddings = None) -> Chroma | None:
    import os
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        return None
    embedding_fn = embedding_fn or create_embedding_function()
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=embedding_fn,
    )


def create_bm25_retriever(chunks: list[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(chunks, k=config.BM25_TOP_K)
    print(f"BM25 retriever created with {len(chunks)} chunks")
    return retriever


def create_ensemble_retriever(
    chunks: list[Document],
    vector_store: Chroma = None,
    embedding_fn: OllamaEmbeddings = None,
) -> EnsembleRetriever:
    embedding_fn = embedding_fn or create_embedding_function()

    if vector_store is None:
        vector_store = create_vector_store(chunks, embedding_fn)

    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.VECTOR_TOP_K},
    )

    bm25_retriever = create_bm25_retriever(chunks)

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[config.BM25_WEIGHT, config.VECTOR_WEIGHT],
    )
    print(f"Ensemble retriever ready (BM25 weight={config.BM25_WEIGHT}, Vector weight={config.VECTOR_WEIGHT})")
    return ensemble

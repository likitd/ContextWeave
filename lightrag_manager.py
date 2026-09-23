import asyncio
import os
import numpy as np
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import config

_rag_instance = None
_token_counter = None
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is not None:
        return _encoding
    tiktoken_name = getattr(config, "LIGHTRAG_TOKENIZER", None)
    if tiktoken_name:
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding(tiktoken_name)
            return _encoding
        except Exception:
            pass
    _encoding = False
    return _encoding


def _get_token_counter():
    global _token_counter
    if _token_counter is not None:
        return _token_counter
    enc = _get_encoding()
    if enc is not False:
        _token_counter = lambda t: len(enc.encode(t))
        return _token_counter
    chars_per_token = getattr(config, "LIGHTRAG_CHARS_PER_TOKEN", 4)
    _token_counter = lambda t: max(1, len(t) // chars_per_token)
    return _token_counter


async def _embed_texts(texts: list[str]) -> np.ndarray:
    token_counter = _get_token_counter()
    max_tokens = getattr(config, "LIGHTRAG_EMBEDDING_MAX_TOKENS", 8100)
    chars_per_token = getattr(config, "LIGHTRAG_CHARS_PER_TOKEN", 4)
    max_chars = max_tokens * chars_per_token
    truncated_texts: list[str] = []
    for t in texts:
        if t is None:
            continue
        if token_counter(t) > max_tokens:
            truncated_texts.append(t[-max_chars:])
        else:
            truncated_texts.append(t)
    if not truncated_texts:
        truncated_texts = [""]
    emb = OllamaEmbeddings(
        model=config.LIGHTRAG_EMBEDDING_MODEL,
        base_url=config.LIGHTRAG_EMBEDDING_BASE_URL,
    )
    vectors = await asyncio.to_thread(emb.embed_documents, truncated_texts)
    return np.array(vectors, dtype=np.float32)


async def _llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    try:
        from lightrag.llm.openai import openai_complete_if_cache
    except ImportError:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.LIGHTRAG_LLM_MODEL,
            api_key=config.LIGHTRAG_LLM_API_KEY,
            base_url=config.LIGHTRAG_LLM_BASE_URL or None,
            temperature=config.LIGHTRAG_LLM_TEMPERATURE,
        )
        if system_prompt:
            msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        else:
            msg = [{"role": "user", "content": prompt}]
        response = await llm.ainvoke(msg)
        return response.content
    response = await openai_complete_if_cache(
        model=config.LIGHTRAG_LLM_MODEL,
        api_key=config.LIGHTRAG_LLM_API_KEY,
        base_url=config.LIGHTRAG_LLM_BASE_URL or "",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        temperature=config.LIGHTRAG_LLM_TEMPERATURE,
        extra_body={"think": False},
        **kwargs,
    )
    return response


async def get_lightrag():
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    embedding_func = EmbeddingFunc(
        embedding_dim=config.LIGHTRAG_EMBEDDING_DIM,
        max_token_size=config.LIGHTRAG_EMBEDDING_MAX_TOKENS,
        func=_embed_texts,
    )
    os.makedirs(config.LIGHTRAG_WORKING_DIR, exist_ok=True)
    _rag_instance = LightRAG(
        working_dir=config.LIGHTRAG_WORKING_DIR,
        embedding_func=embedding_func,
        llm_model_func=_llm_func,
    )
    if hasattr(_rag_instance, "initialize_storages"):
        await _rag_instance.initialize_storages()
    return _rag_instance


def _chunk_one_document(doc: Document, max_tokens: int, enc) -> list[Document]:
    text = doc.page_content
    meta = dict(doc.metadata)
    if enc is not False:
        tokens = enc.encode(text)
        T = len(tokens)
        if T <= max_tokens:
            return [doc]
        k = (T + max_tokens - 1) // max_tokens
        out = []
        for i in range(k):
            start = max(0, T - (k - i) * max_tokens)
            end = min(start + max_tokens, T)
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            out.append(Document(page_content=chunk_text, metadata=meta))
        return out
    token_counter = _get_token_counter()
    T = token_counter(text)
    if T <= max_tokens:
        return [doc]
    chars_per_token = getattr(config, "LIGHTRAG_CHARS_PER_TOKEN", 4)
    M_chars = max_tokens * chars_per_token
    T_chars = len(text)
    k = (T_chars + M_chars - 1) // M_chars
    out = []
    for i in range(k):
        start_c = max(0, T_chars - (k - i) * M_chars)
        end_c = min(start_c + M_chars, T_chars)
        chunk_text = text[start_c:end_c]
        out.append(Document(page_content=chunk_text, metadata=meta))
    return out


def _chunk_documents(documents: list[Document]) -> list[Document]:
    max_tokens = config.LIGHTRAG_EMBEDDING_MAX_TOKENS
    enc = _get_encoding()
    out = []
    for doc in documents:
        out.extend(_chunk_one_document(doc, max_tokens, enc))
    return out


async def insert_documents(documents: list[Document]) -> str:
    rag = await get_lightrag()
    chunks = _chunk_documents(documents)
    texts = [doc.page_content for doc in chunks]
    paths = [doc.metadata.get("source", "") for doc in chunks]
    return await rag.ainsert(
        input=texts,
        file_paths=paths if any(p for p in paths) else None,
    )


async def query(question: str, mode: str | None = None) -> str:
    try:
        from lightrag import QueryParam
    except ImportError:
        from lightrag.base import QueryParam

    rag = await get_lightrag()
    mode = mode or config.LIGHTRAG_MODE
    param = QueryParam(mode=mode)
    return await rag.aquery(question, param=param)

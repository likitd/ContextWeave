import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _path_env(name: str, default: str) -> str:
    value = Path(_env(name, default)).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return str(value)


CONTEXT_FOLDER = _path_env("CONTEXT_FOLDER", "context")

CHROMA_PERSIST_DIR = _path_env("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION_NAME = "hybrid_rag_docs"

FRESH_INDEX = _env_bool("FRESH_INDEX", True)
FRESH_LIGHTRAG_INDEX = _env_bool("FRESH_LIGHTRAG_INDEX", True)

STRATEGY = _env("STRATEGY", "lightrag")  # "auto" | "hybrid" | "agentfs" | "lightrag"

AGENTFS_REQUIRE_OVERVIEW_FIRST = _env_bool("AGENTFS_REQUIRE_OVERVIEW_FIRST", True)

BASE_URL = _env("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = _env("LLM_API_KEY", "ollama")
LLM_MODEL = _env("LLM_MODEL", "gpt-oss:20b-cloud")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.1)

EMBEDDING_BASE_URL = _env("EMBEDDING_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "nomic-embed-text:latest")

LIGHTRAG_EMBEDDING_BASE_URL = _env(
    "LIGHTRAG_EMBEDDING_BASE_URL", EMBEDDING_BASE_URL
)
LIGHTRAG_EMBEDDING_MODEL = _env("LIGHTRAG_EMBEDDING_MODEL", EMBEDDING_MODEL)
LIGHTRAG_EMBEDDING_DIM = _env_int("LIGHTRAG_EMBEDDING_DIM", 768)

LIGHTRAG_WORKING_DIR = _path_env("LIGHTRAG_WORKING_DIR", "lightrag_storage")
LIGHTRAG_MODE = _env("LIGHTRAG_MODE", "hybrid")  # "naive" | "local" | "global" | "hybrid"

LIGHTRAG_LLM_TEMPERATURE = _env_float("LIGHTRAG_LLM_TEMPERATURE", 0.1)
LIGHTRAG_LLM_BASE_URL = _env("LIGHTRAG_LLM_BASE_URL", BASE_URL)
LIGHTRAG_LLM_API_KEY = _env("LIGHTRAG_LLM_API_KEY", API_KEY)
LIGHTRAG_LLM_MODEL = _env("LIGHTRAG_LLM_MODEL", LLM_MODEL)

LIGHTRAG_EMBEDDING_MAX_TOKENS = _env_int("LIGHTRAG_EMBEDDING_MAX_TOKENS", 8100)
LIGHTRAG_TOKENIZER = _env("LIGHTRAG_TOKENIZER", "cl100k_base")
LIGHTRAG_CHARS_PER_TOKEN = _env_int("LIGHTRAG_CHARS_PER_TOKEN", 4)

USE_CHUNKING = _env_bool("USE_CHUNKING", True)

CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 200)

BM25_TOP_K = _env_int("BM25_TOP_K", 10)
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", 10)

BM25_WEIGHT = _env_float("BM25_WEIGHT", 0.4)
VECTOR_WEIGHT = _env_float("VECTOR_WEIGHT", 0.6)

SUBAGENT_BATCH_SIZE = _env_int("SUBAGENT_BATCH_SIZE", 3)
SUBAGENT_MAX_TOKENS_PER_BATCH = _env_int("SUBAGENT_MAX_TOKENS_PER_BATCH", 32000)

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h",
    ".html", ".css", ".json", ".yaml", ".yml", ".xml", ".csv",
    ".sql", ".kt", ".pdf", ".docx", ".pptx", ".xlsx"
}

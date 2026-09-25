import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent

STORAGE_PATH = Path(os.getenv("STORAGE_PATH", BASE_DIR / "storage"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", BASE_DIR / "data" / "documents.json"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "quickstart")
PINECONE_RERANK_MODEL = os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".py"}

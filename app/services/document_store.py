import json
import threading
from datetime import datetime, timezone
from typing import Any

from app.config import METADATA_PATH

lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_store() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def save_store(data: dict[str, Any]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def upsert_document(doc: dict[str, Any]) -> dict[str, Any]:
    with lock:
        data = load_store()
        data[doc["id"]] = doc
        save_store(data)
    return doc


def get_document(doc_id: str) -> dict[str, Any] | None:
    with lock:
        return load_store().get(doc_id)


def list_documents(*, include_deleted: bool = False) -> list[dict[str, Any]]:
    with lock:
        docs = list(load_store().values())
    if not include_deleted:
        docs = [d for d in docs if d.get("status") != "DELETED"]
    return sorted(docs, key=lambda d: d.get("created_at", ""), reverse=True)


def list_soft_deleted() -> list[dict[str, Any]]:
    """Soft-deleted docs that still have vectors in Pinecone (restorable)."""
    with lock:
        docs = list(load_store().values())
    soft = [
        d
        for d in docs
        if d.get("status") == "DELETED" and d.get("vectors_present", False)
    ]
    return sorted(soft, key=lambda d: d.get("deleted_at") or d.get("created_at", ""), reverse=True)


def find_by_content_hash(content_hash: str) -> dict[str, Any] | None:
    """Return the best matching doc for this file hash (prefer vectors still present)."""
    if not content_hash:
        return None
    with lock:
        matches = [
            d for d in load_store().values() if d.get("content_hash") == content_hash
        ]
    if not matches:
        return None
    # Prefer docs that still have Pinecone vectors
    with_vectors = [d for d in matches if d.get("vectors_present")]
    pool = with_vectors or matches
    # Prefer active COMPLETED, then soft-deleted, then others
    status_rank = {"COMPLETED": 0, "PROCESSING": 1, "PENDING": 2, "DELETED": 3, "FAILED": 4}
    pool.sort(key=lambda d: (status_rank.get(d.get("status", ""), 9), d.get("created_at", "")), reverse=False)
    return pool[0]


def update_document(doc_id: str, **fields: Any) -> dict[str, Any] | None:
    with lock:
        data = load_store()
        doc = data.get(doc_id)
        if not doc:
            return None
        doc.update(fields)
        doc["updated_at"] = utc_now()
        data[doc_id] = doc
        save_store(data)
        return doc


def new_document_record(
    *,
    doc_id: str,
    filename: str,
    file_type: str,
    storage_path: str,
    content_hash: str,
    uploaded_by: str | None = None,
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "storage_path": storage_path,
        "content_hash": content_hash,
        "uploaded_by": uploaded_by,
        "status": "PENDING",
        "chunk_count": 0,
        "vectors_present": False,
        "error_message": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "deleted_at": None,
    }

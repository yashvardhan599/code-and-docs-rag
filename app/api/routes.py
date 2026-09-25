import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app import config
from app.services import document_store
from agent_registory.agent import chat as agent_chat
from app.services.rag import delete_document_vectors, ingest_file, search
from app.validation_model.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentResponse,
    RestoreResponse,
    SearchRequest,
    SearchResponse,
    SearchHit,
    UploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def to_doc_response(doc: dict) -> DocumentResponse:
    return DocumentResponse(
        document_id=doc["id"],
        filename=doc["filename"],
        file_type=doc["file_type"],
        status=doc["status"],
        chunk_count=doc.get("chunk_count", 0),
        content_hash=doc.get("content_hash"),
        vectors_present=bool(doc.get("vectors_present", False)),
        error_message=doc.get("error_message"),
        created_at=doc.get("created_at"),
        deleted_at=doc.get("deleted_at"),
    )


def process_document(document_id: str) -> None:
    doc = document_store.get_document(document_id)
    if not doc or doc["status"] == "DELETED":
        return

    document_store.update_document(document_id, status="PROCESSING", error_message=None)
    try:
        count = ingest_file(
            document_id=document_id,
            filename=doc["filename"],
            file_type=doc["file_type"],
            storage_path=doc["storage_path"],
        )
        document_store.update_document(
            document_id,
            status="COMPLETED",
            chunk_count=count,
            vectors_present=True,
            error_message=None,
        )
        logger.info("Ingested %s → %s chunks", document_id, count)
    except Exception as exc:
        logger.exception("Ingest failed for %s", document_id)
        document_store.update_document(
            document_id, status="FAILED", error_message=str(exc)[:2000]
        )


@router.post("/documents", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    uploaded_by: str | None = Form(default=None),
):
    if not file.filename:
        raise HTTPException(400, "filename required")

    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File too large")

    content_hash = hashlib.sha256(data).hexdigest()
    existing = document_store.find_by_content_hash(content_hash)

    # Same file bytes already indexed in Pinecone → reuse, do not retrain
    if existing and existing.get("vectors_present"):
        if existing.get("status") == "COMPLETED":
            return UploadResponse(
                document_id=existing["id"],
                status="COMPLETED",
                reused=True,
                message="Same file already trained — reusing existing Pinecone data (no retrain).",
            )
        if existing.get("status") == "DELETED":
            restored = document_store.update_document(
                existing["id"],
                status="COMPLETED",
                deleted_at=None,
                filename=file.filename,
                error_message=None,
            )
            return UploadResponse(
                document_id=existing["id"],
                status="COMPLETED",
                reused=True,
                message=(
                    "Restored soft-deleted document — reused existing Pinecone data "
                    f"({(restored or existing).get('chunk_count', 0)} chunks, no retrain)."
                ),
            )
        if existing.get("status") in ("PENDING", "PROCESSING"):
            return UploadResponse(
                document_id=existing["id"],
                status=existing["status"],
                reused=True,
                message="Same file is already being processed — skipping duplicate upload.",
            )

    # Retry failed / hard-deleted (no vectors) on the same record when possible
    if existing and existing.get("status") == "FAILED":
        doc_id = existing["id"]
        storage_path = Path(existing["storage_path"])
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.write_bytes(data)
        document_store.update_document(
            doc_id,
            filename=file.filename,
            content_hash=content_hash,
            status="PENDING",
            error_message=None,
            vectors_present=False,
            deleted_at=None,
        )
        background_tasks.add_task(process_document, doc_id)
        return UploadResponse(
            document_id=doc_id,
            status="PENDING",
            message="Retrying training for previously failed document.",
        )

    doc_id = str(uuid.uuid4())
    file_type = ext.lstrip(".")
    config.STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    storage_path = config.STORAGE_PATH / f"{doc_id}_{file.filename}"
    storage_path.write_bytes(data)

    document_store.upsert_document(
        document_store.new_document_record(
            doc_id=doc_id,
            filename=file.filename,
            file_type=file_type,
            storage_path=str(storage_path),
            content_hash=content_hash,
            uploaded_by=uploaded_by,
        )
    )
    background_tasks.add_task(process_document, doc_id)

    return UploadResponse(document_id=doc_id, status="PENDING")


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents():
    return [to_doc_response(d) for d in document_store.list_documents()]


@router.get("/documents/soft-deleted", response_model=list[DocumentResponse])
def list_soft_deleted_documents():
    return [to_doc_response(d) for d in document_store.list_soft_deleted()]


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str):
    doc = document_store.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return to_doc_response(doc)


@router.post("/documents/{document_id}/restore", response_model=RestoreResponse)
def restore_document(document_id: str, background_tasks: BackgroundTasks):
    """Restore a soft-deleted document. Reuses Pinecone vectors when still present."""
    doc = document_store.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc.get("status") != "DELETED":
        raise HTTPException(400, "Document is not soft-deleted")

    if doc.get("vectors_present"):
        document_store.update_document(
            document_id,
            status="COMPLETED",
            deleted_at=None,
            error_message=None,
        )
        return RestoreResponse(
            document_id=document_id,
            status="COMPLETED",
            reused=True,
            message="Restored — existing Pinecone data is active again (no retrain).",
        )

    # Vectors were permanently removed earlier — need a full retrain
    storage = Path(doc.get("storage_path") or "")
    if not storage.exists():
        raise HTTPException(
            400,
            "Cannot restore: Pinecone vectors and local file are both gone. Re-upload the file.",
        )
    document_store.update_document(
        document_id,
        status="PENDING",
        deleted_at=None,
        error_message=None,
    )
    background_tasks.add_task(process_document, document_id)
    return RestoreResponse(
        document_id=document_id,
        status="PENDING",
        reused=False,
        message="Vectors were missing — retraining from stored file.",
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str, hard: bool = False):
    """
    Soft delete (hard=false): mark DELETED in metadata, keep Pinecone vectors + file.
    Permanent delete (hard=true): remove Pinecone vectors and stored file.
    """
    doc = document_store.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    document_store.update_document(
        document_id,
        status="DELETED",
        deleted_at=datetime.now(timezone.utc).isoformat(),
    )

    if hard:
        try:
            delete_document_vectors(document_id)
        except Exception:
            logger.exception(
                "Pinecone delete failed for %s — metadata already marked DELETED",
                document_id,
            )
        document_store.update_document(document_id, vectors_present=False)
        try:
            path = Path(doc["storage_path"])
            if path.exists():
                path.unlink()
        except OSError:
            logger.exception("Could not delete file for %s", document_id)
        message = "Permanently deleted — removed from metadata and Pinecone"
    else:
        # Soft delete: keep Pinecone vectors; mark restorable
        if doc.get("chunk_count", 0) > 0 or doc.get("vectors_present"):
            document_store.update_document(document_id, vectors_present=True)
        message = "Soft deleted — hidden from app, vectors kept in Pinecone"

    return DeleteResponse(
        document_id=document_id,
        status="DELETED",
        message=message,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    result = agent_chat(body.message, thread_id=body.thread_id)
    return ChatResponse(**result)


@router.post("/search", response_model=SearchResponse)
def search_endpoint(body: SearchRequest):
    hits = search(
        body.query,
        top_k=body.top_k,
        document_id=body.document_id,
        file_type=body.file_type,
    )
    return SearchResponse(
        results=[SearchHit(**h) for h in hits],
        metadata={"count": len(hits)},
    )

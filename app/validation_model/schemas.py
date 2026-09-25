from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    status: str
    message: str = "Accepted for async processing"
    reused: bool = False


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int = 0
    content_hash: str | None = None
    vectors_present: bool = False
    error_message: str | None = None
    created_at: str | None = None
    deleted_at: str | None = None


class DeleteResponse(BaseModel):
    document_id: str
    status: str
    message: str


class RestoreResponse(BaseModel):
    document_id: str
    status: str
    message: str
    reused: bool = True


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = Field(default="default", description="Same id = same conversation memory")


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    message_count: int = 0


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_id: str | None = None
    file_type: str | None = None


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str | None = None
    filename: str | None = None
    file_type: str | None = None
    chunk_type: str | None = None
    symbol_name: str | None = None
    parent_class: str | None = None
    content: str
    score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchHit]
    metadata: dict[str, Any] = Field(default_factory=dict)

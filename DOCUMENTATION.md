# Documentation

Architecture, data model, and scaling notes for the Internal AI Knowledge Platform (as built).

---

## 1. System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI (ui/app.py)                     │
│  Upload │ List Documents │ Soft-deleted │ ChatGPT-style Chat     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI (app/api/routes.py)                   │
│  /documents  /documents/soft-deleted  /restore  /chat  /search   │
└───────┬───────────────────┬─────────────────────┬───────────────┘
        │                   │                     │
        ▼                   ▼                     ▼
┌───────────────┐   ┌───────────────┐    ┌───────────────────────┐
│ Local storage │   │ JSON metadata │    │ agent_registory/      │
│ ./storage/    │   │ documents.json│    │ agent.py + llm/prompt │
│ uploaded files│   │ statuses/hash │    │ InMemorySaver memory  │
└───────┬───────┘   └───────┬───────┘    └───────────┬───────────┘
        │                   │                        │
        │     background    │                        │ tool
        │     ingest        │                        ▼
        ▼                   │              search_knowledge()
┌───────────────────────────────────────┐            │
│         app/services/rag.py           │◄───────────┘
│  extract → OCR → chunk → embed        │
│  upsert / query / rerank / delete     │
└───────────┬─────────────┬─────────────┘
            │             │
            ▼             ▼
   ┌─────────────┐  ┌──────────────────┐
   │ Azure OpenAI│  │ Pinecone index   │
   │ chat + emb. │  │ vectors + rerank │
   └─────────────┘  └──────────────────┘
```

### Request flows

**Upload & train**

```
User uploads file
  → SHA256(content)
  → If hash exists + vectors_present → reuse / restore (no embed)
  → Else save file → PENDING → background process_document
       → extract text (OCR if scanned PDF)
       → chunk (.py AST | prose splitter)
       → Azure embeddings → Pinecone upsert
       → COMPLETED + vectors_present=true
```

**Chat**

```
User message + thread_id
  → Agent (system prompt + chat history)
  → search_knowledge(query)
       → embed query
       → Pinecone top-K (filter: COMPLETED doc ids only)
       → Pinecone rerank (top-N)
  → LLM answers from retrieved chunks
```

**Delete**

```
Soft  (hard=false): status=DELETED, keep Pinecone + file → Soft-deleted page
Hard  (hard=true):  status=DELETED, delete Pinecone vectors + file
Restore:            status=COMPLETED again if vectors_present
```

### Component map

| Piece | Role |
|-------|------|
| Streamlit UI | Demo UX: upload progress, delete choice, soft-deleted restore, chat |
| FastAPI (`app/main.py`) | REST boundary; async background ingest |
| `app/api/routes.py` | Document / chat / search endpoints |
| `app/validation_model/schemas.py` | Pydantic request & response models |
| `app/services/document_store.py` | Lightweight metadata + SHA256 dedup (no Postgres) |
| `app/services/rag.py` | Extraction, chunking, embeddings, Pinecone I/O, rerank |
| `agent_registory/agent.py` | Conversational RAG with per-`thread_id` memory |
| `agent_registory/llm.py` | Azure OpenAI chat + embeddings |
| Pinecone | Vector store + `bge-reranker-v2-m3` |

---

## 2. Database schema

There is **no SQL database**. State is split intentionally:

| Store | Technology | Holds |
|-------|------------|--------|
| Document metadata | JSON file (`data/documents.json`) | Status, hash, paths, chunk counts |
| File blobs | Local disk (`storage/`) | Original uploads |
| Vectors | Pinecone | Embeddings + chunk text metadata |
| Chat memory | In-process `InMemorySaver` | Per-`thread_id` message history |

### 2.1 Document metadata (JSON)

Logical record (one object per document id):

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Primary key |
| `filename` | string | Original file name |
| `file_type` | string | `pdf` / `md` / `txt` / `py` |
| `storage_path` | string | Path under `./storage` |
| `content_hash` | string | SHA256 of file bytes (dedup key) |
| `uploaded_by` | string \| null | e.g. `streamlit` |
| `status` | enum | `PENDING` → `PROCESSING` → `COMPLETED` \| `FAILED` \| `DELETED` |
| `chunk_count` | int | Number of vectors upserted |
| `vectors_present` | bool | `true` if Pinecone still holds this doc’s vectors |
| `error_message` | string \| null | Ingest failure detail |
| `created_at` | ISO datetime | Created |
| `updated_at` | ISO datetime | Last change |
| `deleted_at` | ISO datetime \| null | Soft/hard delete time |

**Status lifecycle**

```
PENDING → PROCESSING → COMPLETED
                    ↘ FAILED
COMPLETED → DELETED (soft: vectors_present=true | hard: vectors_present=false)
DELETED (soft) → COMPLETED via restore or re-upload of same SHA256
```

### 2.2 Pinecone vector record

| Field | Description |
|-------|-------------|
| `id` | `{document_id}#{chunk_index}` |
| `values` | Embedding vector (`EMBEDDING_DIMENSIONS`, e.g. 1024) |
| `metadata.document_id` | Link back to JSON record |
| `metadata.filename` | Source file |
| `metadata.file_type` | Extension |
| `metadata.chunk_index` | Order within doc |
| `metadata.chunk_type` | `text` / `function` / `method` / `class` / … |
| `metadata.symbol_name` | For code chunks |
| `metadata.parent_class` | For methods |
| `metadata.text` | Chunk content (used for rerank + citations) |

### 2.3 Chat memory (not persisted to disk)

| Concept | Storage |
|---------|---------|
| `thread_id` | Client-supplied conversation key |
| Message history | LangGraph `InMemorySaver` (lost on API restart) |

### 2.4 ER-style view

```
┌────────────────────┐         1:N          ┌─────────────────────┐
│  documents.json    │─────────────────────▶│  Pinecone vectors   │
│  (document row)    │   document_id        │  id = doc#chunk     │
└────────────────────┘                      └─────────────────────┘
          │
          │ 1:1
          ▼
┌────────────────────┐
│  storage/{id}_file │
└────────────────────┘
```

---

## 3. Scaling strategy and trade-offs

### What we optimized for (now)

- **Fast to build and explain** — JSON metadata + local files, no DB ops  
- **Cheap dedup** — SHA256 avoids re-embedding identical uploads  
- **Soft delete** — restore without re-OCR / re-embed  
- **Two-stage retrieval** — broad recall (top-K) then precise rerank (top-N)  
- **Structure-aware code chunking** — AST for `.py` improves symbol search  

### Scaling path (when traffic / corpus grows)

| Area | Current | Scale-out option | Trade-off |
|------|---------|------------------|-----------|
| Metadata | Single JSON + file lock | Postgres / DynamoDB | More ops; gains concurrency & queryability |
| Files | Local `./storage` | S3 / GCS | Durable, multi-instance; needs signed URLs |
| Ingest | FastAPI `BackgroundTasks` | Celery / SQS / Cloud Tasks worker pool | True horizontal workers; more infra |
| Vectors | One Pinecone index | Namespaces per tenant; larger pods | Isolation & cost control |
| Chat memory | In-memory checkpointer | Redis / Postgres checkpointer | Survives restarts; shared across replicas |
| OCR | Sync RapidOCR in ingest | Separate OCR worker + queue | API stays responsive on large scanned PDFs |
| API / UI | Single process each | Multiple uvicorn workers + CDN UI | Need shared storage + shared memory store |

### Important trade-offs (honest)

1. **JSON store** — Simple and zero-cost, but **not safe for multi-writer production**. One process is fine; several API replicas will race on `documents.json`.  
2. **In-memory chat** — Great for demos; **history dies on restart** and does not share across pods.  
3. **BackgroundTasks** — Easy, but ingest is tied to the API process. Long OCR (multi-page PDF) can still block a worker.  
4. **Embedding dims** — Must match the Pinecone index exactly (`1024` vs `1536` will fail upsert). Changing dims means a new index or full re-index.  
5. **Soft delete** — Saves cost/time on restore, but **Pinecone storage still grows** until a hard delete. Search correctly excludes soft-deleted docs via active `COMPLETED` id filter.  
6. **OCR / images** — We store **text from OCR**, not image embeddings. Diagrams without readable text remain weak in retrieval.  
7. **Dedup by file hash** — Identical bytes skip retrain. A tiny edit → new hash → full retrain (no incremental patch).  

### Recommended next steps (if this becomes production)

1. Move metadata to Postgres; keep Pinecone as the vector layer.  
2. Put files in object storage; run ingest on a worker queue.  
3. Persist agent memory (Redis/Postgres checkpointer).  
4. Add per-tenant Pinecone namespaces and auth on the API.  
5. Metrics: ingest latency, retrieval hit rate, token cost per chat.  

---

## 4. Related reading in-repo

- Setup & API table: [README.md](./README.md)  
- Env template: [.env.example](./.env.example)  
- Retrieval: `app/services/rag.py`  
- Agent + tool: `agent_registory/agent.py`, `agent_registory/prompts.py`  
- API routes: `app/api/routes.py`  
- Schemas: `app/validation_model/schemas.py`  

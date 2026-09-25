# Internal AI Knowledge Platform

Simple RAG system: **upload → extract/OCR → chunk → embed → Pinecone → conversational agent**.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill Azure + Pinecone keys
```

Create a Pinecone index matching `EMBEDDING_DIMENSIONS` in `.env` (default **1024**, cosine).

```bash
# API
uvicorn app.main:app --reload --port 8000

# UI (another terminal)
streamlit run ui/app.py
```

| Surface | URL |
|---------|-----|
| API docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

## Project layout

```
agent_registory/          # Agent layer
  llm.py                  # Azure chat + embeddings
  prompts.py              # System prompt
  agent.py                # LangGraph agent + chat memory

app/
  main.py                 # FastAPI entrypoint
  config.py               # Settings from .env
  api/
    routes.py             # REST endpoints
  validation_model/
    schemas.py            # Pydantic request/response models
  services/
    rag.py                # Extract, OCR, chunk, embed, search, rerank
    document_store.py     # JSON metadata (SHA256, soft/hard delete)

ui/app.py                 # Streamlit UI
```

## What it does

1. **Upload** PDF / MD / TXT / `.py` → async ingest  
2. **Dedup** via SHA256 — same file reuses Pinecone vectors (no retrain)  
3. **Soft delete** keeps vectors; **permanent delete** removes them  
4. **Chat** agent searches the knowledge base (two-stage: vector + rerank)  
5. Scanned PDFs use **local OCR** when there is no selectable text  

## APIs (summary)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/documents` | Upload + train (or reuse by hash) |
| `GET` | `/documents` | Active documents |
| `GET` | `/documents/soft-deleted` | Soft-deleted (restorable) |
| `POST` | `/documents/{id}/restore` | Restore without retrain if vectors exist |
| `DELETE` | `/documents/{id}?hard=false\|true` | Soft or permanent delete |
| `POST` | `/chat` | Conversational agent |
| `POST` | `/search` | Retrieval only (debug) |

## Documentation

See **[DOCUMENTATION.md](./DOCUMENTATION.md)** for:

- System architecture diagram  
- Data / “database” schema  
- Scaling strategy and trade-offs

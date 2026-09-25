import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="Internal AI Knowledge Platform",
    description="Upload docs → Pinecone RAG → conversational agent with memory",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    config.STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    config.METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}

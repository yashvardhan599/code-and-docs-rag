import ast
import io
import logging
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from agent_registory.llm import embeddings
from app import config
from app.services import document_store

logger = logging.getLogger(__name__)

_rapid_ocr = None
pinecone_client: Pinecone | None = None
pinecone_index = None


def _get_rapid_ocr():
    """Lazy-load RapidOCR (local ONNX models — no Tesseract needed)."""
    global _rapid_ocr
    if _rapid_ocr is None:
        from rapidocr_onnxruntime import RapidOCR

        _rapid_ocr = RapidOCR()
    return _rapid_ocr

def get_pinecone() -> Pinecone:
    global pinecone_client
    if pinecone_client is None:
        if not config.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is missing in .env")
        pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
    return pinecone_client


def get_index():
    global pinecone_index
    if pinecone_index is None:
        pinecone_index = get_pinecone().Index(config.PINECONE_INDEX_NAME)
    return pinecone_index


def _ocr_page(page: Any) -> str:
    """Render a PDF page and OCR it (for scanned / image-only PDFs)."""
    import numpy as np
    import pymupdf
    from PIL import Image

    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
    img = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    result, _ = _get_rapid_ocr()(img)
    if not result:
        return ""
    return "\n".join(line[1] for line in result).strip()


def extract_text(path: str | Path, file_type: str) -> str:
    path = Path(path)
    ext = file_type.lower().lstrip(".")

    if ext == "pdf":
        import pymupdf

        doc = pymupdf.open(path)
        try:
            parts: list[str] = []
            for page_num, page in enumerate(doc, start=1):
                text = (page.get_text("text") or "").strip()
                if not text:
                    logger.info(
                        "No selectable text on page %s — running OCR", page_num
                    )
                    try:
                        text = _ocr_page(page)
                    except Exception:
                        logger.exception("OCR failed on page %s", page_num)
                        text = ""
                if text:
                    parts.append(text)
            return "\n\n".join(parts)
        finally:
            doc.close()

    if ext in {"md", "markdown", "txt", "py"}:
        return path.read_text(encoding="utf-8", errors="replace")

    raise ValueError(f"Unsupported file type: {file_type}")


def chunk_document(text: str, *, filename: str, file_type: str) -> list[dict[str, Any]]:
    """
    Returns a list of chunk dicts:
      { text, chunk_type, symbol_name, parent_class }
    """
    ext = file_type.lower().lstrip(".")
    if ext == "py":
        try:
            return chunk_python_ast(text, filename=filename)
        except SyntaxError:
            logger.warning(
                "AST parse failed for %s — falling back to text splitter", filename
            )
            return chunk_prose(text)
    return chunk_prose(text)


def chunk_prose(text: str) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return [
        {
            "text": c,
            "chunk_type": "text",
            "symbol_name": "",
            "parent_class": "",
        }
        for c in splitter.split_text(text)
        if c.strip()
    ]


def context_header(
    filename: str,
    *,
    chunk_type: str,
    symbol_name: str = "",
    parent_class: str = "",
) -> str:
    """Short label prepended before embedding — improves code retrieval."""
    parts = [f"File: {filename}", f"Type: {chunk_type}"]
    if parent_class:
        parts.append(f"Class: {parent_class}")
    if symbol_name:
        parts.append(f"Symbol: {symbol_name}")
    return " | ".join(parts)


def chunk_python_ast(source: str, *, filename: str) -> list[dict[str, Any]]:
    """
    Structure-aware Python chunking (same idea as Tree-sitter, using stdlib ast):
    - module header (imports + docstring)
    - each top-level function
    - each class method (with class name in the header)
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks: list[dict[str, Any]] = []

    def slice_node(node: ast.AST) -> str:
        start = node.lineno
        end = node.end_lineno or node.lineno
        return "\n".join(lines[start - 1 : end])

    def add_chunk(
        content: str,
        *,
        chunk_type: str,
        symbol_name: str = "",
        parent_class: str = "",
    ) -> None:
        if not content.strip():
            return
        header = context_header(
            filename,
            chunk_type=chunk_type,
            symbol_name=symbol_name,
            parent_class=parent_class,
        )
        # What we embed = header + code (better semantic signal)
        chunks.append(
            {
                "text": f"{header}\n\n{content}",
                "chunk_type": chunk_type,
                "symbol_name": symbol_name,
                "parent_class": parent_class,
            }
        )

    # Module header: docstring + import block
    header_parts: list[str] = []
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        header_parts.append(f'"""{mod_doc}"""')

    import_end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_end = max(import_end, node.end_lineno or node.lineno)
        else:
            break
    if import_end > 0:
        header_parts.append("\n".join(lines[:import_end]))
    if header_parts:
        add_chunk("\n\n".join(header_parts), chunk_type="module", symbol_name=filename)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_chunk(
                slice_node(node),
                chunk_type="function",
                symbol_name=node.name,
            )

        elif isinstance(node, ast.ClassDef):
            methods = [
                n
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            class_doc = ast.get_docstring(node) or ""
            class_shell = f"class {node.name}:"
            if class_doc:
                class_shell += f'\n    """{class_doc}"""'
            add_chunk(class_shell, chunk_type="class", symbol_name=node.name)

            if methods:
                for method in methods:
                    add_chunk(
                        slice_node(method),
                        chunk_type="method",
                        symbol_name=method.name,
                        parent_class=node.name,
                    )
            else:
                # Empty / data-only class — store the whole class body
                add_chunk(
                    slice_node(node),
                    chunk_type="class",
                    symbol_name=node.name,
                )

    return chunks or chunk_prose(source)

def upsert_chunks(
    *,
    document_id: str,
    filename: str,
    file_type: str,
    chunks: list[dict[str, Any]],
) -> int:
    """Embed chunks and upsert to Pinecone. Returns number stored."""
    if not chunks:
        return 0

    index = get_index()
    texts = [c["text"][:8000] for c in chunks]
    vectors = embeddings.embed_documents(texts)

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append(
            {
                "id": f"{document_id}#{i}",
                "values": vector,
                "metadata": {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_index": i,
                    "chunk_type": chunk.get("chunk_type") or "text",
                    "symbol_name": chunk.get("symbol_name") or "",
                    "parent_class": chunk.get("parent_class") or "",
                    "text": chunk["text"][:35000],  # Pinecone metadata size limit
                },
            }
        )

    batch_size = 100
    for start in range(0, len(records), batch_size):
        index.upsert(vectors=records[start : start + batch_size])

    return len(records)


def search(
    query: str,
    *,
    top_k: int | None = None,
    top_n: int | None = None,
    document_id: str | None = None,
    file_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Stage 1: vector similarity search (fast, broad recall)
    Stage 2: Pinecone rerank model (accurate, small set)
    """
    top_k = top_k or config.RETRIEVAL_TOP_K
    top_n = top_n or config.RERANK_TOP_N

    logger.info("Inside the search function")

    # Only search active (COMPLETED) docs — soft-deleted stay in Pinecone but are excluded
    active_ids = [
        d["id"]
        for d in document_store.list_documents()
        if d.get("status") == "COMPLETED"
    ]
    if not active_ids:
        return []
    if document_id and document_id not in active_ids:
        return []

    query_vector = embeddings.embed_query(query[:8000])

    pinecone_filter: dict[str, Any] = {
        "document_id": {"$eq": document_id} if document_id else {"$in": active_ids}
    }
    if file_type:
        pinecone_filter["file_type"] = {"$eq": file_type.lstrip(".").lower()}

    result = get_index().query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter,
    )

    matches = result.get("matches") or []
    if not matches:
        return []

    docs_for_rerank = []
    for m in matches:
        meta = m.get("metadata") or {}
        text = meta.get("text") or ""
        docs_for_rerank.append(
            {
                "id": m["id"],
                "text": text,
                "meta": meta,
                "vector_score": m.get("score"),
            }
        )

    try:
        reranked = get_pinecone().inference.rerank(
            model=config.PINECONE_RERANK_MODEL,
            query=query,
            documents=[{"id": d["id"], "text": d["text"]} for d in docs_for_rerank],
            top_n=min(top_n, len(docs_for_rerank)),
            return_documents=True,
        )
        by_id = {d["id"]: d for d in docs_for_rerank}
        ordered: list[dict[str, Any]] = []
        for item in reranked.data:
            hit_id = (
                item.document["id"]
                if hasattr(item, "document")
                else item["document"]["id"]
            )
            score = item.score if hasattr(item, "score") else item.get("score")
            original = by_id[hit_id]
            meta = original["meta"]
            ordered.append(
                {
                    "chunk_id": hit_id,
                    "document_id": meta.get("document_id"),
                    "filename": meta.get("filename"),
                    "file_type": meta.get("file_type"),
                    "chunk_type": meta.get("chunk_type"),
                    "symbol_name": meta.get("symbol_name"),
                    "parent_class": meta.get("parent_class"),
                    "content": original["text"],
                    "score": float(score) if score is not None else None,
                }
            )
        return ordered
    except Exception:
        logger.exception("Pinecone rerank failed — falling back to vector order")
        fallback = []
        for d in docs_for_rerank[:top_n]:
            meta = d["meta"]
            fallback.append(
                {
                    "chunk_id": d["id"],
                    "document_id": meta.get("document_id"),
                    "filename": meta.get("filename"),
                    "file_type": meta.get("file_type"),
                    "chunk_type": meta.get("chunk_type"),
                    "symbol_name": meta.get("symbol_name"),
                    "parent_class": meta.get("parent_class"),
                    "content": d["text"],
                    "score": float(d["vector_score"])
                    if d["vector_score"] is not None
                    else None,
                }
            )
        return fallback


def format_sources_for_tool(hits: list[dict[str, Any]]) -> str:
    """Turn search hits into plain text the agent tool can return."""
    if not hits:
        return "No relevant chunks found in the knowledge base."

    parts = []
    for i, h in enumerate(hits, 1):
        label = h.get("filename") or "unknown"
        if h.get("parent_class") and h.get("symbol_name"):
            label += f"::{h['parent_class']}.{h['symbol_name']}"
        elif h.get("symbol_name"):
            label += f"::{h['symbol_name']}"
        parts.append(
            f"[{i}] {label} (doc={h.get('document_id')})\n{h.get('content')}"
        )
    return "\n\n---\n\n".join(parts)


def delete_document_vectors(document_id: str) -> None:
    """Hard-delete all Pinecone vectors tagged with this document_id."""
    index = get_index()
    index.delete(filter={"document_id": {"$eq": document_id}})


def ingest_file(
    *,
    document_id: str,
    filename: str,
    file_type: str,
    storage_path: str,
) -> int:
    text = extract_text(storage_path, file_type)
    if not text.strip():
        raise ValueError("No extractable text in file")

    chunks = chunk_document(text, filename=filename, file_type=file_type)
    if not chunks:
        raise ValueError("Chunking produced zero chunks")

    return upsert_chunks(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        chunks=chunks,
    )

import time
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="Knowledge Platform", page_icon="📚", layout="wide")

API_BASE = "http://127.0.0.1:8000"

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_upload_id" not in st.session_state:
    st.session_state.pending_upload_id = None
if "pending_upload_name" not in st.session_state:
    st.session_state.pending_upload_name = None
if "upload_notice" not in st.session_state:
    st.session_state.upload_notice = None
if "delete_confirm_id" not in st.session_state:
    st.session_state.delete_confirm_id = None


def fetch_documents() -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def fetch_soft_deleted() -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE}/documents/soft-deleted", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def fetch_document(document_id: str) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}/documents/{document_id}", timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ----- Sidebar: page selection only -----
with st.sidebar:
    st.markdown("### Knowledge Platform")
    page = st.radio(
        "Navigate",
        ["Upload the document", "List of Documents", "Soft deleted documents"],
        label_visibility="collapsed",
    )
    st.divider()

    if page == "Upload the document":
        st.markdown("#### Upload")
        uploaded = st.file_uploader(
            "PDF, Markdown, TXT, or Python",
            type=["pdf", "md", "txt", "py"],
            label_visibility="collapsed",
        )
        if uploaded and st.button("Upload & train", type="primary", use_container_width=True):
            with st.spinner("Uploading…"):
                resp = requests.post(
                    f"{API_BASE}/documents",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    data={"uploaded_by": "streamlit"},
                    timeout=60,
                )
            if resp.ok:
                data = resp.json()
                st.session_state.upload_notice = data.get("message")
                if data.get("reused") or data.get("status") == "COMPLETED":
                    st.session_state.pending_upload_id = None
                    st.session_state.pending_upload_name = None
                else:
                    st.session_state.pending_upload_id = data["document_id"]
                    st.session_state.pending_upload_name = uploaded.name
                st.rerun()
            else:
                st.error(resp.text)

        if st.session_state.upload_notice:
            st.success(st.session_state.upload_notice)
            # Keep notice until next upload; clear after showing once on COMPLETED reuse
            if not st.session_state.pending_upload_id:
                pass

        # Reflect training progress for newly uploaded docs only
        if st.session_state.pending_upload_id:
            doc = fetch_document(st.session_state.pending_upload_id)
            name = st.session_state.pending_upload_name or "document"
            if not doc:
                st.warning(f"Lost track of `{name}` — check List of Documents.")
                st.session_state.pending_upload_id = None
                st.session_state.pending_upload_name = None
            else:
                status = doc.get("status", "UNKNOWN")
                chunks = doc.get("chunk_count", 0)
                if status in ("PENDING", "PROCESSING"):
                    st.info(f"Training `{name}`… ({status})")
                    time.sleep(2)
                    st.rerun()
                elif status == "COMPLETED":
                    st.success(
                        f"Finished processing `{name}` — {chunks} chunks indexed."
                    )
                    st.session_state.pending_upload_id = None
                    st.session_state.pending_upload_name = None
                    st.session_state.upload_notice = None
                elif status == "FAILED":
                    st.error(
                        f"Training failed for `{name}`: {doc.get('error_message') or 'unknown error'}"
                    )
                    st.session_state.pending_upload_id = None
                    st.session_state.pending_upload_name = None
                else:
                    st.warning(f"`{name}` status: {status}")

    elif page == "List of Documents":
        st.markdown("#### Documents")
        if st.button("Refresh", use_container_width=True):
            st.rerun()

        docs = fetch_documents()
        if not docs:
            st.info("No documents yet. Upload one first.")
        else:
            for d in docs:
                doc_id = d["document_id"]
                status = d.get("status", "UNKNOWN")
                chunks = d.get("chunk_count", 0)
                st.markdown(f"**{d['filename']}**")
                hash_short = (d.get("content_hash") or "")[:12]
                st.caption(
                    f"{status} · {chunks} chunks"
                    + (f" · sha {hash_short}…" if hash_short else "")
                )

                if st.session_state.delete_confirm_id == doc_id:
                    st.caption("How do you want to delete?")
                    mode = st.radio(
                        "Delete mode",
                        [
                            "Soft delete (keep in Pinecone)",
                            "Permanent delete (remove from Pinecone)",
                        ],
                        key=f"mode-{doc_id}",
                        label_visibility="collapsed",
                    )
                    c1, c2 = st.columns(2)
                    if c1.button("Confirm", key=f"ok-{doc_id}", use_container_width=True):
                        hard = mode.startswith("Permanent")
                        resp = requests.delete(
                            f"{API_BASE}/documents/{doc_id}",
                            params={"hard": str(hard).lower()},
                            timeout=60,
                        )
                        st.session_state.delete_confirm_id = None
                        if resp.ok:
                            st.toast(resp.json().get("message", "Deleted"))
                        else:
                            st.error(resp.text)
                        st.rerun()
                    if c2.button("Cancel", key=f"cancel-{doc_id}", use_container_width=True):
                        st.session_state.delete_confirm_id = None
                        st.rerun()
                else:
                    if st.button("Delete", key=f"del-{doc_id}", use_container_width=True):
                        st.session_state.delete_confirm_id = doc_id
                        st.rerun()
                st.divider()

    else:
        st.markdown("#### Soft deleted")
        st.caption("Vectors still in Pinecone — restore to use again without retraining.")
        if st.button("Refresh soft-deleted", use_container_width=True):
            st.rerun()

        soft = fetch_soft_deleted()
        if not soft:
            st.info("No soft-deleted documents.")
        else:
            for d in soft:
                doc_id = d["document_id"]
                chunks = d.get("chunk_count", 0)
                st.markdown(f"**{d['filename']}**")
                st.caption(f"soft-deleted · {chunks} chunks kept in Pinecone")
                if st.button("Restore / train again", key=f"restore-{doc_id}", use_container_width=True):
                    resp = requests.post(
                        f"{API_BASE}/documents/{doc_id}/restore",
                        timeout=60,
                    )
                    if resp.ok:
                        st.success(resp.json().get("message", "Restored"))
                    else:
                        st.error(resp.text)
                    st.rerun()
                st.divider()

    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


# ----- Main: ChatGPT-style conversation -----
st.markdown("## Chat")
st.caption("Ask anything — upload docs in the sidebar for knowledge-base answers")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Message…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={"message": prompt, "thread_id": st.session_state.thread_id},
                    timeout=120,
                )
                resp.raise_for_status()
                answer = resp.json()["answer"]
            except Exception as exc:
                answer = f"Error talking to API: {exc}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

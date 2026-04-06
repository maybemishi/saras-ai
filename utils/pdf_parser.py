"""
utils/pdf_parser.py — PDF ingestion pipeline.

Flow:
  uploaded bytes → PyMuPDF → clean text → chunks → embeddings → ChromaDB

The collection used is always the session collection so the Analyst
agent can retrieve document content the same way it retrieves web content.
"""

from __future__ import annotations
import hashlib
import re
from typing import Optional
import fitz  # PyMuPDF
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings


# ── ChromaDB client (shared, lazy-initialised) ─────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GOOGLE_API_KEY,
    )


# ── Text extraction ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


# ── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks of ~CHUNK_SIZE words.
    Simple word-boundary split — no external splitter dependency.
    """
    words = text.split()
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += size - overlap
    return chunks


# ── ChromaDB ingestion ──────────────────────────────────────────────────────

def ingest_pdf(pdf_bytes: bytes, filename: str) -> int:
    """
    Full pipeline: bytes → text → chunks → embed → store in ChromaDB.
    Returns the number of chunks stored.
    Idempotent: chunks are keyed by content hash so re-uploading the
    same file will not create duplicates.
    """
    raw = extract_text_from_pdf(pdf_bytes)
    clean = _clean_text(raw)
    chunks = chunk_text(clean)
    if not chunks:
        return 0

    embedder = _get_embedder()
    collection = _get_collection()

    # Batch embed (Google API supports batch calls)
    embeddings = embedder.embed_documents(chunks)

    ids, docs, metas, embs = [], [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        ids.append(chunk_id)
        docs.append(chunk)
        metas.append({"source": filename, "origin": "document", "chunk_index": i})
        embs.append(emb)

    # Upsert — safe for duplicates
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(chunks)


# ── Retrieval ───────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(query: str, n_results: int = 6) -> list[dict]:
    """
    Query ChromaDB for the most relevant stored chunks.
    Returns list of dicts with keys: text, source, origin.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedder = _get_embedder()
    query_embedding = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append({
            "text": doc,
            "source": meta.get("source", "uploaded_document"),
            "origin": meta.get("origin", "document"),
        })
    return output


def clear_session_collection():
    """Wipe the ChromaDB collection for a fresh research session."""
    global _collection
    if _chroma_client is not None:
        try:
            _chroma_client.delete_collection(settings.COLLECTION_NAME)
        except Exception:
            pass
        _collection = None

# ingest.py
from __future__ import annotations

import os
import re
import shutil
import hashlib
import logging
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

logging.getLogger("pypdf").setLevel(logging.ERROR)

# --------- Config ---------
DOC_ROOT = Path("data")          # your PDFs (already OCR'd) live here
DB_DIR = "chroma_db"             # Chroma persistence folder
COLLECTION = "cslewis"
ALLOWED_GENRES = {"fiction", "nonfiction", "poetry"}  # case-insensitive

# --------- Env / API key ---------
load_dotenv()  # read .env if present
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Put it in your .env (OPENAI_API_KEY=sk-...) "
        "or set it in the environment before running ingest.py."
    )

# --------- Helpers ---------
_ws_collapse = re.compile(r"[ \t\u00A0]+")  # collapse weird intra-line spaces

def normalize_text(t: str) -> str:
    """
    Light cleanup to help chunking & retrieval:
    - Collapse runs of spaces/tabs/nbsp
    - Strip trailing spaces on lines
    - Keep newlines (preserves paragraph structure for better splits)
    """
    if not t:
        return ""
    # Keep newlines; normalize spaces inside lines
    lines = [ _ws_collapse.sub(" ", ln).rstrip() for ln in t.splitlines() ]
    return "\n".join(lines).strip()

def infer_genre(path: Path) -> str:
    """Infer genre from the immediate parent folder name (case-insensitive)."""
    parent = path.parent.name.lower()
    return parent if parent in ALLOWED_GENRES else "unknown"

def content_hash(text: str, meta: Dict) -> str:
    """
    Stable hash of content + a few metadata fields to avoid dupes.
    """
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    h.update(("|" + meta.get("work_title","") + "|" + meta.get("genre","")).encode("utf-8"))
    return h.hexdigest()[:16]

def load_and_split_docs(root: Path) -> List:
    """Load PDFs recursively, attach metadata, and split into chunks."""
    pdf_paths = sorted(root.rglob("*.pdf"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    loaded_count = 0
    skipped = []
    genre_counts = {"fiction": 0, "nonfiction": 0, "poetry": 0, "unknown": 0}

    for pdf in pdf_paths:
        try:
            pages = PyPDFLoader(str(pdf)).load()
            loaded_count += 1
        except Exception as e:
            skipped.append(f"{pdf.name}: {e}")
            continue

        genre = infer_genre(pdf)
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        work_title = pdf.stem

        # enrich page-level metadata + normalize text before splitting
        for d in pages:
            d.page_content = normalize_text(d.page_content or "")
            d.metadata.update(
                {
                    "genre": genre,
                    "author": "C. S. Lewis",
                    "work_title": work_title,
                    "source_path": str(pdf),
                }
            )

        chunks = splitter.split_documents(pages)

        # add chunk_id and drop empty chunks
        kept = []
        for idx, ch in enumerate(chunks):
            if not ch.page_content.strip():
                continue
            ch.metadata["chunk_id"] = f"{work_title}:{idx:05d}"
            kept.append(ch)
        all_chunks.extend(kept)

    # De-duplicate by content hash (common with re-OCR or overlapping pages)
    deduped = []
    seen = set()
    for d in all_chunks:
        h = content_hash(d.page_content, d.metadata)
        if h in seen:
            continue
        d.metadata["content_hash"] = h
        seen.add(h)
        deduped.append(d)

    print(f"[INGEST] Scanned {len(pdf_paths)} PDFs; loaded {loaded_count}, skipped {len(pdf_paths)-loaded_count}.")
    if skipped:
        print("[INGEST] Skipped files:")
        for s in skipped:
            print("  -", s)

    print("[INGEST] Per-genre files:", {k: v for k, v in genre_counts.items() if v})
    print(f"[INGEST] Produced {len(all_chunks)} chunks; kept {len(deduped)} unique after de-dup.")

    return deduped

def rebuild_vectorstore():
    """Wipe and rebuild the persistent Chroma index from /data."""
    # Hard wipe to avoid stale collections
    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR, ignore_errors=True)

    docs = load_and_split_docs(DOC_ROOT)
    if not docs:
        raise RuntimeError("No documents were loaded. Check your /data folder.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",   # upgrade to -3-large if you want higher recall
        api_key=OPENAI_API_KEY,           # important: ingest.py runs outside Streamlit
    )

    _ = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
    )

    print(f"[INGEST] Vector store rebuilt at '{DB_DIR}' (collection '{COLLECTION}').")

if __name__ == "__main__":
    rebuild_vectorstore()

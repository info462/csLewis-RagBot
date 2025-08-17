# main.py — build/persist a FAISS index from PDFs and TXTs, then run a quick test query

from pathlib import Path
import os
import glob
import hashlib

# Optional: load .env locally; Streamlit Cloud will use Secrets
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found.")
    raise SystemExit(1)

print("🔥 Checking vector store...")

# ---------- Config ----------
DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")   # keep this identical in app.py
HASH_FILE = Path("doc_hashes.txt")

# ---------- Deps ----------
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- Helpers ----------
def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# Collect docs
pdf_files = sorted(glob.glob(str(DATA_DIR / "*.pdf")))
txt_files = sorted(glob.glob(str(DATA_DIR / "*.txt")))
print(f"📄 Found {len(pdf_files)} PDFs and {len(txt_files)} TXTs.")

# Load previous hashes
old_hashes = {}
if HASH_FILE.exists():
    for line in HASH_FILE.read_text().splitlines():
        if "|" in line:
            filename, h = line.strip().split("|", 1)
            old_hashes[filename] = h

# Detect changes (both PDFs and TXTs)
all_files = pdf_files + txt_files
changed_files = []
for file in all_files:
    new_hash = file_hash(file)
    if file not in old_hashes or old_hashes[file] != new_hash:
        changed_files.append(file)

rebuild_needed = bool(changed_files) or not (INDEX_DIR.exists() and (INDEX_DIR / "index.faiss").exists())

if not rebuild_needed:
    print("✅ No changes detected. Using existing FAISS index.")
else:
    target_count = len(changed_files) or len(all_files)
    print(f"🔄 Rebuilding vector store for {target_count} file(s)...")

    # Load documents
    docs = []
    for p in txt_files:
        try:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
            print(f"📜 Loaded text: {p}")
        except Exception as e:
            print(f"⚠️ Skipped {p}: {e}")

    for p in pdf_files:
        try:
            loader = PyPDFLoader(str(p))
            pdf_pages = loader.load()
            docs.extend(pdf_pages)
            print(f"📘 Loaded PDF: {p} ({len(pdf_pages)} pages)")
        except Exception as e:
            print(f"⚠️ Skipped {p}: {e}")

    if not docs:
        print("⚠️ No documents loaded. Ensure files exist in ./data")
        raise SystemExit(0)

    # Normalize & chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    for d in docs:
        d.page_content = " ".join(d.page_content.split())

    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks.")

    # Build & persist FAISS
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectordb = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(exist_ok=True)
    vectordb.save_local(str(INDEX_DIR))
    print("💾 Saved FAISS index to faiss_index/")

    # Save new hashes
    with HASH_FILE.open("w") as f:
        for file in all_files:
            f.write(f"{file}|{file_hash(file)}\n")
    print("🧾 Updated doc hashes.")

# ---------- Load index (shared with app.py) ----------
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
vectordb = FAISS.load_local(
    str(INDEX_DIR),
    embeddings,
    allow_dangerous_deserialization=True,  # required by LangChain 0.2.x
)

# ---------- Quick test query ----------
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are C.S. Lewis, writing in the first person. "
        "Use ONLY the excerpts below; do not reference sources explicitly.\n\n"
        "Excerpts:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer as C.S. Lewis:\n"
    ),
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

query = "What does C.S. Lewis say about pain?"
res = qa(query)
print("\n🧠 Answer:\n", res["result"])
print("\n📚 Sources:")
for i, d in enumerate(res.get("source_documents", []) or [], 1):
    meta = d.metadata or {}
    print(f"  {i}. {meta.get('source','?')} — page {meta.get('page','?')}")

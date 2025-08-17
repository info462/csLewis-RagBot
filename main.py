from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


import os
import glob
import hashlib
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("🔥 Checking vector store...")

# Load OpenAI key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("❌ OPENAI_API_KEY not found.")
    exit(1)

# Where we store PDF hash records
HASH_FILE = "pdf_hashes.txt"

def file_hash(path):
    """Return SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# Find all PDFs
pdf_files = sorted(glob.glob("data/*.pdf"))
print(f"📄 Found {len(pdf_files)} PDFs.")

# Load previous hashes
old_hashes = {}
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        for line in f:
            filename, h = line.strip().split("|")
            old_hashes[filename] = h

# Detect changes
changed_files = []
for file in pdf_files:
    new_hash = file_hash(file)
    if file not in old_hashes or old_hashes[file] != new_hash:
        changed_files.append(file)

if not changed_files and os.path.exists("embeddings") and os.listdir("embeddings"):
    print("✅ No changes detected. Using existing vector DB.")
    exit(0)

# If we got here, we need to rebuild
print(f"🔄 Rebuilding vector store for {len(changed_files) or len(pdf_files)} PDFs...")

# Load and split
all_docs = []
for file in pdf_files:
    print(f"🔍 Loading {file}...")
    loader = PyPDFLoader(file)
    raw_docs = loader.load()
    print(f"📄 {file} loaded with {len(raw_docs)} pages.")
    all_docs.extend(raw_docs)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = splitter.split_documents(all_docs)
print(f"✅ Split into {len(docs)} chunks.")

# Build embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="embeddings")
vectordb.persist()

# Save new hashes
with open(HASH_FILE, "w") as f:
    for file in pdf_files:
        f.write(f"{file}|{file_hash(file)}\n")

print("✅ Vector DB updated and saved.")

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

llm = OpenAI(api_key=api_key, temperature=0)

# Ask a question
query = "What does C.S. Lewis say about pain?"
docs = vectordb.similarity_search(query, k=5)

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
    print(doc.page_content[:500] + "...")



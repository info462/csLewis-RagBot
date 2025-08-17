import streamlit as st
from pathlib import Path
from typing import List, Dict

# LangChain / OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    from langchain_community.vectorstores import FAISS  # LC >= 0.1.x
except ImportError:
    from langchain.vectorstores import FAISS            # fallback for older LC
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# Optional for PDF ingest on first run
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # handled later

# --- Page config and header ---
st.set_page_config(page_title="Ask C.S. Lewis Anything", page_icon="📚")
st.title("📚 Ask C.S. Lewis")
st.markdown("Ask questions and receive answers drawn *only* from the writings of C.S. Lewis.")
st.divider()

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Secrets / API key (no .env on Streamlit Cloud) ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing. Add it in Streamlit → Settings → Secrets.")
    st.stop()

# --- Vector store (FAISS) ---
INDEX_DIR = "faiss_index"   # commit this dir if you prebuilt locally
DATA_DIR = Path("data")     # put your PDFs / TXT here for PoC builds

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Load FAISS index if present; otherwise build a lightweight one from /data."""
    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

    # Try loading an existing FAISS index committed with the repo
    try:
        vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
        return vs
    except Exception:
        pass

    # Fallback: build from /data (fast PoC). For production, prebuild locally and commit.
    if not DATA_DIR.exists():
        st.error(
            "No FAISS index found and /data folder does not exist. "
            "Either commit a prebuilt index in 'faiss_index/' or add sources in 'data/'."
        )
        st.stop()

    texts: List[str] = []
    metadatas: List[Dict] = []

    # TXT files
    for p in DATA_DIR.rglob("*.txt"):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                texts.append(content)
                metadatas.append({"source": str(p), "page": "N/A"})
        except Exception:
            continue

    # PDF files (one chunk per page for basic retrieval)
    if PdfReader:
        for p in DATA_DIR.rglob("*.pdf"):
            try:
                pdf = PdfReader(str(p))
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    if page_text.strip():
                        texts.append(page_text)
                        metadatas.append({"source": str(p), "page": i})
            except Exception:
                continue
    else:
        st.warning("pypdf not available; skipping PDF ingest. Add 'pypdf' to requirements.txt.")

    if not texts:
        st.error("Found no ingestible text in /data. Add PDFs/TXTs or commit a prebuilt index.")
        st.stop()

    vs = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metadatas)
    vs.save_local(INDEX_DIR)
    return vs

with st.spinner("Loading knowledge base..."):
    vectordb = load_vectorstore()

# --- LLM & Retriever ---
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.3)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 12})

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are C.S. Lewis, responding in the first person, as if writing a personal message or essay.

Draw your answers only from the excerpts provided below, which are from your published works. Do not summarize or say "the text says"—this is you speaking. Let your tone reflect your style: logical, imaginative, vivid, and grounded in Christian theology. Use metaphor, wit, analogy, and vivid imagery as you often do.

Avoid modern references (e.g., AI, ChatGPT, 21st century events) and do not refer to yourself as dead or historical. Stay in character as if you are writing contemporaneously.

---
Excerpts:
{context}

Question:
{question}

Answer as C.S. Lewis:
""",
)


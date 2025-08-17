# app.py
import os
from pathlib import Path
from typing import List, Dict, Iterable

import streamlit as st
from openai import OpenAI

# ---------- OpenAI client (1.x) ----------
client = OpenAI()  # reads OPENAI_API_KEY from env/Streamlit secrets

# ---------- Minimal LangChain Embeddings adapter ----------
from langchain_core.embeddings import Embeddings
from typing import List, Iterable

class OpenAIEmbedder(Embeddings):
    """LangChain-compatible embeddings using OpenAI 1.x SDK."""
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        BATCH = 256
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            resp = client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# ---------- Optional PDF ingest ----------
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ---------- LangChain bits ----------
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores import FAISS

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Ask C.S. Lewis Anything", page_icon="📚")

# Add custom CSS for cleaner layout
st.markdown(
    """
    <style>
      .block-container { max-width: 900px !important; }
      .stChatMessage { line-height: 1.55; }
      .stExpander { border: 1px solid #e6e6e6; border-radius: 10px; }
      .stExpander > div[role='button'] { font-weight: 600; }
      code, pre { font-size: 0.95em; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 Ask C.S. Lewis")
st.markdown("Answers are composed only from the source texts you provide.")
st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Secrets / API key check
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY. Add it in Streamlit → Settings → Secrets.")
    st.stop()

# Data path
DATA_DIR = Path("data")

# ---------- Build / load the vector store ----------
@st.cache_resource
def load_vectordb():
    texts: List[str] = []
    metadatas: List[Dict] = []

    # TXT files
    if DATA_DIR.exists():
        for p in DATA_DIR.rglob("*.txt"):
            try:
                t = p.read_text(encoding="utf-8", errors="ignore")
                if t.strip():
                    texts.append(t)
                    metadatas.append({"source": str(p), "page": "N/A"})
            except Exception:
                pass

        # PDFs (basic extraction: one chunk per page)
        if PdfReader:
            for p in DATA_DIR.rglob("*.pdf"):
                try:
                    pdf = PdfReader(str(p))
                    for i, page in enumerate(pdf.pages, start=1):
                        try:
                            t = page.extract_text() or ""
                        except Exception:
                            t = ""
                        if t.strip():
                            texts.append(t)
                            metadatas.append({"source": str(p), "page": i})
                except Exception:
                    pass
        else:
            st.info("`pypdf` not available; skipping PDFs. Add `pypdf` to requirements to enable.")

    if not texts:
        # keep the app alive even if data folder is empty
        texts = [""]
        metadatas = [{"source": "none", "page": 0}]

    embedder = OpenAIEmbedder(model="text-embedding-3-small")

    # Build a LangChain FAISS vectorstore
    vectordb = FAISS.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas
    )
    return vectordb

with st.spinner("Loading knowledge base..."):
    vectordb = load_vectordb()

# ---------- LLM, retriever, and prompt ----------
llm: BaseChatModel = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.3)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

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
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,  # return docs so we can display them
)

# ---------- Helper to pretty print sources ----------
def _pretty_source(meta: Dict) -> str:
    src = meta.get("source") or "unknown"
    page = meta.get("page", "?")
    name = Path(src).name
    return f"**{name}** — page {page}"

# ---------- Sidebar dataset info ----------
with st.sidebar:
    st.subheader("Dataset")
    try:
        st.write("FAISS index loaded.")
    except Exception:
        st.write("Vector store ready.")
    st.caption("Files are read from the `data/` folder and chunked for retrieval.")

# ---------- Chat box ----------
user_q = st.chat_input("Ask your question…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                res = qa(user_q)  # returns {"result": str, "source_documents": List[Document]}
                ans = res["result"]
                srcs = res.get("source_documents") or []
            except Exception as e:
                ans, srcs = f"Sorry, I hit an error: `{e}`", []

            st.markdown(ans)

            # Collapsible sources dropdown
            if srcs:
                with st.expander("📚 Sources (click to expand)"):
                    for i, d in enumerate(srcs, 1):
                        st.markdown(f"**{i}.** {_pretty_source(d.metadata)}")
                        excerpt = (d.page_content or "").strip().replace("\n", " ")
                        if excerpt:
                            st.caption(excerpt[:350] + ("…" if len(excerpt) > 350 else ""))

    st.session_state.messages.append({"role": "assistant", "content": ans})

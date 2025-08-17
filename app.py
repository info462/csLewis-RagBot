# app.py
import os
from pathlib import Path
from typing import List, Dict, Iterable

import streamlit as st
from openai import OpenAI

from langchain_community.vectorstores import SKLearnVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI  # only the chat model; embeddings are custom

# --------- Minimal embedder that uses openai>=1 directly (no proxies kwarg issue) ---------
class OpenAIEmbedder:
    """
    Drop-in replacement for LangChain's Embeddings interface.
    Implements `embed_documents` and `embed_query` using the v1 OpenAI SDK.
    """
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI embeddings API accepts up to ~2048 inputs per call; we’ll chunk just in case.
        out: List[List[float]] = []
        BATCH = 512
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# --------- Optional PDF ingest ---------
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # handled later

# --------- Streamlit UI ---------
st.set_page_config(page_title="Ask C.S. Lewis Anything", page_icon="📚")
st.title("📚 Ask C.S. Lewis")
st.markdown("Answers are composed only from the source texts you provide.")
st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY. Add it in Streamlit → Settings → Secrets.")
    st.stop()

# Paths
DATA_DIR = Path("data")
INDEX_DIR = Path("sklearn_index")  # folder where we’ll persist SKLearnVectorStore
INDEX_DIR.mkdir(exist_ok=True)

# --------- Build / load the vector store using our custom embedder ---------
@st.cache_resource(show_spinner=True)
def load_vectordb() -> SKLearnVectorStore:
    embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

    # If you already persisted docs, we can load them by re-embedding (SKLearn needs vectors).
    # To keep things simple and hermetic on Streamlit Cloud, we (re)build quickly from /data.
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

        # PDFs (basic: one chunk per page)
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
            st.info("`pypdf` not available; skipping PDFs. (Add `pypdf` to requirements to enable.)")

    if not texts:
        st.warning("No texts found in /data. The bot will still run, but answers will be generic.")
        # Create an empty store (avoid crash)
        return SKLearnVectorStore.from_texts(
            texts=[""], embedding=embedder, metadatas=[{"source": "none", "page": 0}]
        )

    return SKLearnVectorStore.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)

with st.spinner("Loading knowledge base..."):
    vectordb = load_vectordb()

# --------- LLM, retriever, and prompt ---------
llm: BaseChatModel = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.3)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 12})

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
)

# --------- Chat box ---------
user_q = st.chat_input("Ask your question…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                ans = qa.run(user_q)
            except Exception as e:
                ans = f"Sorry, I hit an error: `{e}`"
            st.markdown(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})

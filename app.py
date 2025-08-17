# app.py
import os
from pathlib import Path
from typing import List, Dict

import streamlit as st

# LangChain + OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Optional: PDF reading (safe to run without it)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # we’ll skip PDFs if pypdf isn’t available


# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Ask C.S. Lewis", page_icon="📚", layout="centered")
st.title("📚 Ask C.S. Lewis")
st.caption("Answers are composed *only* from the source texts you provide.")
st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Secrets / API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY. Add it in Streamlit → Settings → Secrets.")
    st.stop()


# ------------------------- Data ingest -------------------------
DATA_DIR = Path("data")  # put your .txt / .pdf files here

@st.cache_resource(show_spinner=False)
def build_vectorstore() -> SKLearnVectorStore:
    """
    Build a lightweight in-memory vector store using scikit-learn.
    Pure Python: no FAISS, no tiktoken, no native compilers needed.
    """
    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

    if not DATA_DIR.exists():
        st.error("No 'data/' folder found. Add TXT/PDF files under 'data/'.")
        st.stop()

    texts: List[str] = []
    metadatas: List[Dict] = []

    # TXT files
    for p in DATA_DIR.rglob("*.txt"):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""
        if content.strip():
            texts.append(content)
            metadatas.append({"source": str(p), "page": "N/A"})

    # PDF files (optional)
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
                # continue on any single-PDF failure
                continue
    else:
        st.info("`pypdf` not installed, skipping PDFs. Add `pypdf` to requirements.txt to enable.")

    if not texts:
        st.error("No usable text found in 'data/'. Add PDFs/TXTs with content.")
        st.stop()

    # Build the in-memory index
    return SKLearnVectorStore.from_texts(texts=texts, embedding=emb, metadatas=metadatas)


with st.spinner("Loading knowledge base…"):
    vectordb = build_vectorstore()


# ------------------------- LLM + Chain -------------------------
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.3)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are C.S. Lewis, writing in first person.\n"
        "Compose your answer ONLY from the excerpts below (drawn from your published works).\n"
        "Do not say 'the text says'—write as yourself. Avoid modern references and do not refer to yourself as deceased.\n"
        "\n---\nExcerpts:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer as C.S. Lewis:\n"
    ),
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 12}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)


# ------------------------- Chat loop -------------------------
user_input = st.chat_input("Ask C.S. Lewis anything about the texts…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = qa({"query": user_input})
            answer: str = result["result"]
            sources = result.get("source_documents", []) or []
            st.markdown(answer)

            # show simple sources block
            if sources:
                with st.expander("Sources"):
                    for i, doc in enumerate(sources, start=1):
                        meta = doc.metadata or {}
                        st.write(f"**{i}.** {meta.get('source','?')}  (page: {meta.get('page','?')})")

    st.session_state.messages.append({"role": "assistant", "content": answer})

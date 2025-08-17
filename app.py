import streamlit as st
import os
import subprocess
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Page config and header ---
st.set_page_config(page_title="Ask C.S. Lewis Anything")
st.title("📚 Ask C.S. Lewis")
st.markdown("Ask questions and receive answers drawn *only* from the writings of C.S. Lewis.")
st.divider()

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Load environment variables and API key ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in environment variables.")
    st.stop()

# --- Ensure vector store exists ---
def ensure_embeddings():
    embeddings_path = "embeddings"
    if not os.path.exists(embeddings_path) or not os.listdir(embeddings_path):
        st.write("⚙️ Building vector store, please wait...")
        result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
        st.write(result.stdout)
        if result.returncode != 0:
            st.error("❌ Failed to build vector store. Check main.py output.")
            st.stop()

ensure_embeddings()

# --- Load vector DB ---
embeddings = OpenAIEmbeddings(api_key=api_key)
vectordb = Chroma(persist_directory="embeddings", embedding_function=embeddings)

# --- Setup retrieval QA chain ---
llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.5)
retriever = vectordb.as_retriever(
    search_type="mmr",  # Max Marginal Relevance
    search_kwargs={"k": 6, "fetch_k": 12}
)

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are C.S. Lewis, responding in the first person, as if writing a personal message or essay.

Draw your answers only from the excerpts provided below, which are from your published works. Do not summarize or say "the text says"—this is you speaking. Let your tone reflect your style: logical, imaginative, vivid, and grounded in Christian theology. Use the metaphor, wit, analogy and vivid imagery as you often do.

Avoid modern references (e.g., AI, ChatGPT, 21st century events) and do not refer to yourself as dead or historical. Stay in character as if you are writing contemporaneously.

---
Excerpts:
{context}

Question:
{question}

Answer as C.S. Lewis:
""",
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt},
)


# --- Cache response logic ---
@st.cache_data(show_spinner=False)
def get_cached_answer(query):
    return qa_chain.invoke({"query": query})

# --- User input + response logic ---
user_input = st.chat_input("What's on your mind?")
if user_input and user_input.strip():
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.spinner("Pondering..."):
        response = get_cached_answer(user_input)
        source_docs = response.get("source_documents", [])
        answer = response.get("result", "🤔 I couldn’t find anything relevant in the writings of C.S. Lewis.")

    # Show assistant response
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Show sources if any
    if source_docs:
        with st.expander("📚 Sources"):
            seen_sources = set()
            for doc in source_docs:
                source_path = doc.metadata.get("source", "Unknown")
                page_number = doc.metadata.get("page", "N/A")
                book_title = source_path.split("/")[-1].replace("_CSL.pdf", "").replace(".pdf", "").replace("_", " ")
                source_id = f"{book_title} — Page {page_number}"
                if source_id in seen_sources:
                    continue
                seen_sources.add(source_id)
                st.markdown(f"**{book_title} — Page {page_number}**")
                st.text(doc.page_content[:400] + "...")

import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load .env and API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("❌ OPENAI_API_KEY not found.")
    exit(1)

# Load the vector store
vectordb = Chroma(persist_directory="embeddings", embedding_function=OpenAIEmbeddings())

# Set up the LLM and conversation chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    retriever=vectordb.as_retriever()
)

chat_history = []

print("\n🤖 Ask C.S. Lewis a question (type 'exit' to quit):\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    result = qa_chain.run({"question": query, "chat_history": chat_history})
    print(f"C.S. Lewis: {result}\n")
    chat_history.append((query, result))

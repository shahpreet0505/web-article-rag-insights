import os
import time
import pickle
from dotenv import load_dotenv

import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ----------------------------
# 🔹 App Config
# ----------------------------
st.set_page_config(
    page_title="News RAG Research Tool",
    page_icon="📰",
    layout="wide"
)

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
STORE_PATH = "faiss_store_index.pkl"


# ----------------------------
# 🔹 Initialize LLM
# ----------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_KEY
    )


# ----------------------------
# 🔹 Build Vector Index
# ----------------------------
def build_vector_store(urls):
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_index = FAISS.from_documents(chunks, embeddings)

    with open(STORE_PATH, "wb") as f:
        pickle.dump(vector_index, f)

    return vector_index


# ----------------------------
# 🔹 Load Vector Store
# ----------------------------
def load_vector_store():
    with open(STORE_PATH, "rb") as f:
        return pickle.load(f)


# ----------------------------
# 🔹 Build RAG Chain
# ----------------------------
def get_rag_chain(vector_index):
    retriever = vector_index.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using ONLY the context provided below.
    If the answer is not found, reply with: "I don't know."

    Context:
    {context}

    Question:
    {question}
    """)

    return (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | load_llm()
        | StrOutputParser()
    ), retriever


# ----------------------------
# 🔹 UI — Sidebar
# ----------------------------
st.sidebar.title("📰 News Source URLs")
st.sidebar.markdown("Enter up to 3 article links.")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_btn = st.sidebar.button("📥 Process & Index Articles")


# ----------------------------
# 🔹 UI — Main
# ----------------------------
st.title("🔎 News Research & Q/A (RAG Powered)")
st.caption("Ask questions grounded in real news sources — no hallucinations.")


status = st.empty()


# ----------------------------
# 🔹 Process URLs
# ----------------------------
if process_btn:
    valid_urls = [u for u in urls if u.strip()]

    if not valid_urls:
        st.warning("⚠️ Please enter at least one valid URL.")
    else:
        status.info("🔄 Fetching & indexing content...")
        vector_index = build_vector_store(valid_urls)
        time.sleep(1)
        status.success("✅ Index built successfully! You can now ask questions.")


# ----------------------------
# 🔹 Ask Questions
# ----------------------------
query = st.text_input("💬 Ask a question about the articles:")

if query:
    if not os.path.exists(STORE_PATH):
        st.error("⚠️ No index found. Please process URLs first.")
    else:
        vector_index = load_vector_store()
        rag_chain, retriever = get_rag_chain(vector_index)

        with st.spinner("🤖 Generating answer..."):
            answer = rag_chain.invoke(query)

        st.subheader("🟢 Answer")
        st.write(answer)

        # Show Sources
        st.subheader("📚 Sources Used")
        src_docs = retriever.invoke(query)

        for doc in src_docs:
            st.markdown(f"- 🔗 {doc.metadata.get('source', 'Unknown')}")

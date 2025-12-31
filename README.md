📰 News Research RAG Application — LangChain, FAISS, Groq & Streamlit

A Retrieval-Augmented Generation (RAG) system that extracts content from online news articles, chunks and embeds the text, stores it in a FAISS vector index, and answers user questions using semantic retrieval + an LLM.

This project demonstrates:

Web article ingestion and preprocessing

Text chunking with recursive character splitters

Vector database indexing using FAISS

Semantic retrieval + LLM response generation

Prompt-driven question answering

End-to-end RAG pipeline in production-style architecture



🚀 Features

Enter multiple news URLs → system loads and processes content

Document segmentation & embedding generation

Vector index persisted locally (faiss_store_index.pkl)

Query answering based on retrieved context only

“I don’t know” fallback to avoid hallucinations

Built using Groq Llama-3.1-8B-Instant

Streamlit interface for interactive research workflow


🧠 Example Use-Cases

Market / stock news analysis

Company announcements tracking

Public policy / government updates

Product launches research

Competitive intelligence

Tech & startup coverage

ESG / automotive / finance reporting

Interview preparation via article insights

This makes it look like a real-world AI research assistant.
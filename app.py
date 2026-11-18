"""
Multimodal Stateful Hybrid RAG System - Streamlit UI
"""

import os
import streamlit as st
import streamlit.components.v1 as components

from src.ingestion import extract_text, chunk_text
from src.hybrid_search import HybridSearchEngine
from src.knowledge_graph import KnowledgeGraphBuilder
from src.state_graph import build_rag_graph
from src.config import DOCS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP

st.set_page_config(page_title="Multimodal Hybrid RAG System", layout="wide")

os.makedirs(DOCS_FOLDER, exist_ok=True)


@st.cache_resource
def get_search_engine():
    return HybridSearchEngine()


@st.cache_resource
def get_kg_builder():
    return KnowledgeGraphBuilder()


search_engine = get_search_engine()
kg_builder = get_kg_builder()
rag_app = build_rag_graph(search_engine, kg_builder)

# ---------------------------------------------------------------
# Session state - rebuild ingested_files from disk, not just memory,
# so it's accurate even right after an app restart.
# ---------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = search_engine.get_all_sources()

# ---------------------------------------------------------------
# Sidebar: Document upload (multimodal ingestion)
# ---------------------------------------------------------------
st.sidebar.title("📄 Upload Documents")
st.sidebar.caption("Supports PDF, DOCX, images, audio")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=["pdf", "docx", "png", "jpg", "jpeg", "wav", "flac", "mp3", "txt"],
    accept_multiple_files=True,
)

if st.sidebar.button("Ingest Documents") and uploaded_files:
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            save_path = os.path.join(DOCS_FOLDER, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            try:
                text = extract_text(save_path)
                chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                search_engine.add_documents(chunks, source=file.name)
                kg_builder.add_text(text, source=file.name)
                if file.name not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(file.name)
                st.sidebar.success(f"Ingested: {file.name}")
            except Exception as e:
                st.sidebar.error(f"Failed on {file.name}: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Ingested files")
for fname in st.session_state.ingested_files:
    st.sidebar.write(f"- {fname}")

if st.sidebar.button("🗑️ Clear All Documents"):
    search_engine.clear_all()
    kg_builder.clear()

    # Also delete the raw uploaded files saved on disk, so nothing lingers
    # in the project folder after a full reset.
    for fname in os.listdir(DOCS_FOLDER):
        fpath = os.path.join(DOCS_FOLDER, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    st.session_state.ingested_files = []
    st.session_state.history = []
    st.sidebar.success("All documents, chat history, and saved files cleared.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.metric("Knowledge Graph entities", kg_builder.node_count())
st.sidebar.metric("Knowledge Graph relations", kg_builder.edge_count())

if st.sidebar.button("🕸️ View Knowledge Graph"):
    st.session_state.show_kg = True

# ---------------------------------------------------------------
# Main: Chat interface
# ---------------------------------------------------------------
st.title("🔎 Multimodal Stateful Hybrid RAG System")
st.caption("Cross-lingual (English/Tamil/Hindi) · Hybrid Search (BM25 + Vector + RRF) · Knowledge Graph · API by Gemini")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["assistant"])

query = st.chat_input("Ask a question in English, Tamil, or Hindi...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag_app.invoke({
                "query": query,
                "history": st.session_state.history,
            })
            st.write(result["final_answer"])

            with st.expander("🔍 Retrieved sources & knowledge graph facts"):
                st.write("**Hybrid search retrieved chunks (with scores):**")
                for chunk in result["retrieved_chunks"]:
                    st.write(f"- *(BM25: {chunk['bm25_score']} | Vector: {chunk['vector_score']} | RRF: {chunk['rrf_score']} | Re-rank: {chunk['rerank_score']})* {chunk['text'][:200]}...")

                st.write("**Knowledge graph facts:**")
                for fact in result["kg_facts"]:
                    st.write(f"- {fact}")

    st.session_state.history = result["history"]

if st.session_state.get("show_kg"):
    st.subheader("Knowledge Graph Visualization")
    if kg_builder.node_count() == 0:
        st.info("No entities yet — ingest a document first.")
    else:
        html_path = kg_builder.generate_html()
        with open(html_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=570, scrolling=True)
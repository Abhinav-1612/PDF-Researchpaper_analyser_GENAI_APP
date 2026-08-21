"""
Advanced RAG Platform — Streamlit Frontend
Uses LangGraph Agent, Parent-Child Pinecone Retrieval, and Chain-of-Thought parsing.
"""
import logging
import time
import re
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import configure_environment, settings

# Ensure environment variables are loaded
configure_environment()

from app.ingestion.pdf_loader import process_pdf_bytes, process_pdf_bytes_structured
from app.ingestion.chunking import chunk_documents, create_parent_child_chunks
from app.retrieval.vector_store import build_vectorstore_retriever, build_hybrid_retriever
from app.generation.answer_generator import create_rag_chain

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================
st.set_page_config(
    page_title="PDF Intelligence RAG",
    layout="wide",
    page_icon="⚡",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
}
.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# STATE MANAGEMENT
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

AVAILABLE_MODELS = [
    "qwen/qwen3.6-27b",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "deepseek/deepseek-chat:free",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("⚡ PDF Intelligence")
    st.caption("Agentic RAG with Pinecone")
    
    st.markdown("---")
    selected_model = st.selectbox("Select Model", AVAILABLE_MODELS, index=0)
    st.markdown("---")
    
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs to build knowledge base",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button("Index Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing & Indexing PDFs into Pinecone..."):
                all_parent_docs = []
                all_child_docs = []
                fallback_chunks = []
                use_parent_child = True
                
                doc_names = []
                
                for f in uploaded_files:
                    doc_names.append(f.name)
                    content = f.read()
                    st.toast(f"Parsing {f.name}...")
                    
                    # Try structured parsing
                    doc_id, sections = process_pdf_bytes_structured(content, f.name, ocr_engine=None)
                    
                    if sections:
                        parents, children = create_parent_child_chunks(sections, f.name, doc_id)
                        all_parent_docs.extend(parents)
                        all_child_docs.extend(children)
                    else:
                        use_parent_child = False
                        page_chunks = process_pdf_bytes(content, f.name, None)
                        fallback_chunks.extend(chunk_documents(page_chunks))
                        
                st.toast("Building Vector Index...")
                
                if use_parent_child and all_child_docs:
                    retriever = build_hybrid_retriever(
                        parent_docs=all_parent_docs,
                        child_docs=all_child_docs,
                        enable_reranking=True,
                    )
                else:
                    combined = fallback_chunks + all_child_docs
                    retriever = build_vectorstore_retriever(combined or [])
                    
                st.session_state.retriever = retriever
                st.session_state.doc_names = doc_names
                st.success("Indexing Complete!")

    if st.session_state.doc_names:
        st.markdown("### Indexed Files")
        for name in st.session_state.doc_names:
            st.markdown(f"- `{name}`")

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
st.title("Document Chat")

if not st.session_state.retriever:
    st.info("👈 Please upload and index PDF documents in the sidebar to begin.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("think"):
                with st.expander("Chain of Thought"):
                    st.markdown(msg["think"])
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.caption(f"**{s['file']}** - Page {s['page']} | {s['type']}")
                        st.text(s['content'][:300] + "...")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            status_container = st.empty()
            status_container.info("🔍 Retrieving context and thinking...")
            
            # Format chat history for LangChain
            history = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    history.append(HumanMessage(content=m["content"]))
                else:
                    history.append(AIMessage(content=m["content"]))
                    
            try:
                # Run the chain
                chain = create_rag_chain(st.session_state.retriever, model_name=selected_model)
                t_start = time.perf_counter()
                
                response = chain.invoke({
                    "input": prompt,
                    "chat_history": history,
                    "doc_names": st.session_state.doc_names,
                })
                
                full_answer = response.get("answer", "")
                raw_sources = response.get("context", [])
                
                status_container.empty()
                
                # Parse <think> blocks exactly like the backend
                think_content = None
                think_match = re.search(r"<think>([\s\S]*?)</think>", full_answer, re.IGNORECASE)
                if think_match:
                    think_content = think_match.group(1).strip()
                    full_answer = re.sub(r"<think>[\s\S]*?</think>", "", full_answer, flags=re.IGNORECASE).strip()
                elif full_answer.strip().startswith("<think>"):
                    think_content = re.sub(r"^<think>\s*", "", full_answer.strip(), flags=re.IGNORECASE).strip()
                    full_answer = "I've completed my analysis. Please see the chain of thought above."
                
                # Display Chain of Thought if present
                if think_content:
                    with st.expander("Chain of Thought", expanded=True):
                        st.markdown(think_content)
                
                # Stream the actual answer using typewriter effect
                message_placeholder = st.empty()
                streamed_text = ""
                words = full_answer.split(" ")
                for i, word in enumerate(words):
                    streamed_text += word + " "
                    message_placeholder.markdown(streamed_text + "▌")
                    time.sleep(0.015)
                message_placeholder.markdown(streamed_text)
                
                # Format sources
                formatted_sources = []
                for doc in raw_sources:
                    formatted_sources.append({
                        "file": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", 1),
                        "type": "Section" if doc.metadata.get("is_parent") else "Chunk",
                        "content": doc.page_content
                    })
                    
                if formatted_sources:
                    with st.expander("Sources"):
                        for s in formatted_sources:
                            st.caption(f"**{s['file']}** - Page {s['page']} | {s['type']}")
                            st.text(s['content'][:300] + "...")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": streamed_text,
                    "think": think_content,
                    "sources": formatted_sources
                })
                
            except Exception as e:
                status_container.empty()
                st.error(f"Error generating response: {str(e)}")
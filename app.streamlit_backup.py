"""
Advanced RAG Platform — Streamlit Frontend (Phase 1 Refactored)

This file is now a thin UI orchestrator. All core logic lives in:
  - app/core/config.py          → settings
  - app/ingestion/ocr.py        → OCR engine
  - app/ingestion/pdf_loader.py → PDF processing
  - app/ingestion/chunking.py   → text chunking
  - app/retrieval/vector_store.py → embedding + vector DB
  - app/generation/answer_generator.py → RAG chain

Streamlit-specific concerns (caching, session state, UI) remain here.
"""
import logging
import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# --- Bootstrap: configure environment before any heavy imports ---
from app.core.config import configure_environment, settings
configure_environment()

# --- Module imports ---
from app.ingestion.ocr import init_paddle_ocr
from app.ingestion.pdf_loader import process_pdf_bytes, process_pdf_bytes_structured
from app.ingestion.chunking import chunk_documents, create_parent_child_chunks
from app.retrieval.vector_store import build_vectorstore_retriever, build_hybrid_retriever
from app.generation.answer_generator import create_rag_chain

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="PDF/Paper Analyser RAG App",
    layout="wide",
    page_icon="⚡",
)


# ============================================================================
# CYBERPUNK THEME (preserved from original)
# ============================================================================
def apply_tech_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    .stApp { background-color: #080c16; color: #a9bad4; font-family: 'Share Tech Mono', monospace; }

    h1 {
        text-align: center; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 30px;
        background: linear-gradient(90deg, #00ffcc, #0077ff, #00ffcc); background-size: 200% auto;
        color: transparent !important; -webkit-background-clip: text; animation: shine 3s linear infinite;
        text-shadow: 0 0 20px rgba(0, 255, 204, 0.3);
    }
    @keyframes shine { to { background-position: 200% center; } }

    .metric-box {
        background: rgba(0, 255, 204, 0.05); border: 1px solid #00ffcc; border-radius: 4px;
        padding: 6px 15px; color: #00ffcc; font-size: 0.9rem; display: inline-block;
        margin-top: 15px; animation: pulse-glow 2.5s infinite alternate;
    }
    @keyframes pulse-glow {
        0%   { box-shadow: 0 0 5px rgba(0, 255, 204, 0.2) inset, 0 0 5px rgba(0, 255, 204, 0.2); }
        100% { box-shadow: 0 0 15px rgba(0, 255, 204, 0.6) inset, 0 0 15px rgba(0, 255, 204, 0.6); }
    }

    .stButton > button {
        background-color: #0e1526; border: 1px solid #0077ff; color: #a9bad4;
        border-radius: 5px; transition: all 0.3s ease; width: 100%; height: 100%;
        white-space: normal; min-height: 60px;
    }
    .stButton > button:hover {
        background-color: rgba(0, 255, 204, 0.1); border-color: #00ffcc; color: #00ffcc;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.4); transform: translateY(-2px);
    }

    div.row-widget.stRadio > div { background: #0e1526; padding: 10px; border-radius: 10px; border: 1px solid #1f2d47; }
    .stSelectbox > div > div { background-color: #0e1526; color: #00ffcc; border: 1px solid #1f2d47; }
    .stChatInputContainer, .stChatInput { background-color: #0e1526 !important; border: 1px solid #1a2a44 !important; box-shadow: 0 0 15px rgba(0, 119, 255, 0.1) !important; transition: all 0.3s ease; }
    .stChatInputContainer:focus-within { border-color: #00ffcc !important; box-shadow: 0 0 20px rgba(0, 255, 204, 0.3) !important; }
    .stChatMessage { background-color: #0e1526; border: 1px solid #1f2d47; border-radius: 8px; padding: 15px; margin-bottom: 20px; animation: slideUpFade 0.4s ease-out forwards; }
    @keyframes slideUpFade { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .streamlit-expanderHeader { background-color: rgba(0, 255, 204, 0.05); color: #00ffcc !important; border: 1px solid #1f2d47; border-radius: 5px; }
    [data-testid="stSidebar"] { background-color: #0a0f1c; border-right: 1px solid rgba(0, 255, 204, 0.3); box-shadow: 5px 0 20px rgba(0, 255, 204, 0.05); }
    .stFileUploader { border-radius: 10px; border: 1px dashed #00ffcc !important; background: rgba(0, 255, 204, 0.02); }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    [data-testid="collapsedControl"] {
        color: #00ffcc !important;
        background-color: #0e1526 !important;
        border: 1px solid #1f2d47 !important;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# CACHED RESOURCES (Streamlit-level caching — must live here)
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_ocr_engine():
    """Initialize and cache the PaddleOCR engine (loaded once per Streamlit session)."""
    return init_paddle_ocr()


@st.cache_resource(show_spinner=False)
def get_retriever(file_bytes_tuple: tuple):
    """
    Process uploaded PDFs using structure-aware parsing + parent-child chunking.
    Falls back to basic fixed-size chunking if structure parsing fails.
    Cached per unique set of file contents — only re-runs when files change.
    """
    ocr_engine = get_ocr_engine()
    all_parent_docs = []
    all_child_docs = []
    fallback_chunks = []
    use_parent_child = True

    for i, file_bytes in enumerate(file_bytes_tuple):
        doc_names = st.session_state.get("uploaded_file_names", [])
        doc_name = doc_names[i] if i < len(doc_names) else f"document_{i + 1}.pdf"

        # Phase 2: Try structure-aware parsing first
        doc_id, sections = process_pdf_bytes_structured(
            file_bytes=file_bytes,
            document_name=doc_name,
            ocr_engine=ocr_engine,
        )

        if sections and len(sections) > 0:
            parents, children = create_parent_child_chunks(
                sections=sections,
                document_name=doc_name,
                document_id=doc_id,
            )
            all_parent_docs.extend(parents)
            all_child_docs.extend(children)
        else:
            # Fallback to Phase 1 basic chunking
            use_parent_child = False
            page_chunks = process_pdf_bytes(
                file_bytes=file_bytes,
                document_name=doc_name,
                ocr_engine=ocr_engine,
            )
            fallback_chunks.extend(chunk_documents(page_chunks))

    # Phase 3: Build the Hybrid Retriever (Dense + BM25 + RRF + Reranker)
    if use_parent_child and all_child_docs:
        return build_hybrid_retriever(
            parent_docs=all_parent_docs,
            child_docs=all_child_docs,
            enable_reranking=True,
        )
    else:
        # Fallback: basic vectorstore (for completely unparseable PDFs)
        combined = fallback_chunks + all_child_docs
        return build_vectorstore_retriever(combined if combined else [])


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    defaults = {
        "messages": [],
        "current_files": (),
        "suggestions_generated": False,
        "suggested_questions": [],
        "pending_query": None,
        "uploaded_file_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# UI — SIDEBAR
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ SYSTEM CONFIG")
        app_mode = st.radio(
            "SELECT UPLINK MODE:",
            ["Single PDF Explorer", "Explore Multiple PDFs"],
        )

        if app_mode == "Single PDF Explorer":
            uploaded_file = st.file_uploader(
                "UPLOAD DATABANK (PDF)", type=["pdf"], accept_multiple_files=False
            )
            uploaded_files = [uploaded_file] if uploaded_file else []
        else:
            uploaded_files = st.file_uploader(
                "UPLOAD MULTIPLE DATABANKS", type=["pdf"], accept_multiple_files=True
            )

        st.markdown("---")
        st.markdown("### 🧠 NEURAL CORE OVERRIDE")
        selected_model = st.selectbox(
            "Select AI Engine (Swap if rate-limited):",
            settings.AVAILABLE_LLM_MODELS,
            index=0,
        )

        st.markdown("---")
        st.markdown(f"""
        **System Architecture:**
        - **Pipeline:** LangGraph Agentic RAG (Phase 5)
        - **Brain:** `{selected_model}`
        - **Vision/OCR:** PyMuPDF + PaddleOCR + Tesseract
        - **Memory:** ChromaDB (Ephemeral)
        - **Sensors:** `{settings.EMBEDDING_MODEL}`
        - **Search:** Dense + BM25 → RRF → Reranker
        - **Query:** Multi-Query + Decomposition + (HyDE opt)
        - **Reranker:** `{settings.RERANKER_MODEL.split('/')[-1]}`
        - **Chunking:** Structure-aware + Parent-Child
        """)

        st.markdown("---")
        st.markdown("### 💾 DATA EXPORT")
        if st.session_state.messages:
            chat_log = "PDF/PAPER ANALYSER RAG APP - SESSION LOG\n" + "=" * 40 + "\n\n"
            for msg in st.session_state.messages:
                role = "USER" if msg["role"] == "user" else "AI NODE"
                chat_log += f"[{role}]:\n{msg['content']}\n\n"
                if msg.get("sources"):
                    chat_log += "--- EXTRACTED FRAGMENTS ---\n"
                    for i, doc in enumerate(msg["sources"]):
                        chat_log += (
                            f"Fragment {i+1} "
                            f"[{doc['extraction_method']} | Page {doc['page']}]:\n"
                            f"{doc['content']}\n\n"
                        )
                    chat_log += "---------------------------\n\n"

            st.download_button(
                label="Download Session Log (.txt)",
                data=chat_log,
                file_name=f"Neural_Log_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.caption("No session data to export yet.")

    return uploaded_files, selected_model


# ============================================================================
# UI — CHAT HISTORY
# ============================================================================
def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if msg["role"] == "assistant" and "<think>" in content and "</think>" in content:
                import re
                think_match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                if think_match:
                    thinking_text = think_match.group(1).strip()
                    answer_text = content.split("</think>")[-1].strip()
                    # Render the thinking part in light green, not bold, with a slight left border
                    st.markdown(
                        f"<div style='color: #98fb98; font-weight: 400; margin-bottom: 15px; "
                        f"padding-left: 10px; border-left: 2px solid #98fb98; opacity: 0.9;'>"
                        f"<em>{thinking_text}</em></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(answer_text)
                else:
                    st.write(content)
            else:
                st.write(content)
            if msg.get("sources"):
                with st.expander("🔍 View Extracted Source Fragments"):
                    for i, doc in enumerate(msg["sources"]):
                        section = doc.get('section', '')
                        section_tag = f" · §{section}" if section else ""
                        content_type = doc.get('content_type', 'text')
                        type_icon = {"parent": "📄", "child": "🔍", "table": "📊"}.get(content_type, "📝")
                        st.markdown(
                            f"**{type_icon} Fragment {i+1}** "
                            f"*(via {doc['extraction_method']} — Page {doc['page']}{section_tag})*:"
                        )
                        
                        # Render the scores if available
                        score_parts = []
                        if doc.get("dense_score") is not None: score_parts.append(f"Dense: {doc['dense_score']}")
                        if doc.get("bm25_score") is not None: score_parts.append(f"BM25: {doc['bm25_score']}")
                        if doc.get("rrf_score") is not None: score_parts.append(f"RRF: {doc['rrf_score']}")
                        if doc.get("rerank_score") is not None: score_parts.append(f"Rerank: {doc['rerank_score']}")
                        if score_parts:
                            st.caption("🎯 " + " | ".join(score_parts))
                            
                        st.caption(doc["content"])
                        st.markdown("---")
            if "time" in msg:
                debug = st.session_state.get("last_retrieval_debug", {})
                timings = debug.get("timings", {})
                # Build a compact metric string
                metric_parts = [f"⏱️ {msg['time']:.2f}s"]
                if timings.get("dense_ms"):
                    metric_parts.append(f"Dense {timings['dense_ms']:.0f}ms")
                if timings.get("bm25_ms"):
                    metric_parts.append(f"BM25 {timings['bm25_ms']:.0f}ms")
                if timings.get("rerank_ms"):
                    metric_parts.append(f"Reranker {timings['rerank_ms']:.0f}ms")
                st.markdown(
                    f'<div class="metric-box">{" │ ".join(metric_parts)}</div>',
                    unsafe_allow_html=True,
                )
                # Retrieval debug panel
                if debug:
                    with st.expander("📡 Retrieval Debug"):
                        # Query intelligence row
                        qi_strategy = debug.get("qi_strategy", "")
                        if qi_strategy:
                            st.markdown(f"🧠 **Query Strategy:** `{qi_strategy}`")
                        # Sub-questions (decomposition)
                        if debug.get("sub_questions"):
                            st.markdown("**🔪 Sub-questions (Decomposition):**")
                            for sq in debug["sub_questions"]:
                                st.caption(f"• {sq}")
                        # Multi-query variants
                        if debug.get("multi_queries"):
                            st.markdown("**🔄 Query Variants (Multi-Query):**")
                            for mq in debug["multi_queries"]:
                                st.caption(f"• {mq}")
                        # HyDE
                        if debug.get("hyde_doc"):
                            st.markdown("**👻 HyDE Hypothetical Passage:**")
                            st.caption(debug["hyde_doc"])
                        # LangGraph Agent Trace
                        agent_trace = st.session_state.get("agent_trace", [])
                        if agent_trace:
                            st.markdown("**🤖 Agentic Decision Trace:**")
                            for step in agent_trace:
                                st.caption(f"→ {step}")
                        st.markdown("---")
                        # Retrieval metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("Dense Hits", debug.get("dense_hits", "-"))
                        col_b.metric("BM25 Hits", debug.get("bm25_hits", "-"))
                        col_c.metric("After RRF", debug.get("rrf_candidates", "-"))
                        col_d.metric("Final Docs", debug.get("final_results", "-"))
                        if debug.get("reranker_scores"):
                            scores_str = ", ".join(f"{s:.3f}" for s in debug["reranker_scores"])
                            st.caption(f"🎯 Top reranker scores: {scores_str}")


# ============================================================================
# UI — SUGGESTED QUESTIONS
# ============================================================================
def render_suggested_questions(retriever, selected_model):
    if not st.session_state.suggestions_generated:
        with st.spinner(f"🧠 {selected_model} is analyzing document context..."):
            try:
                # Bypass the strict RAG agent (which would reject the prompt as irrelevant)
                # Just grab a few chunks directly to figure out what the document is about
                from langchain_groq import ChatGroq
                from app.core.config import settings
                
                # Retrieve generic context
                sample_docs = retriever.invoke("What is the main topic of this document?")
                context = "\n\n".join([d.page_content for d in sample_docs[:3]])
                
                llm = ChatGroq(model=selected_model, temperature=0.7)
                prompt = (
                    "Based on the following document excerpts, provide EXACTLY 3 highly "
                    "relevant, insightful questions that a user should ask to understand "
                    "the core content. Format your response strictly as a bulleted list "
                    "with NO intro or outro text. Ensure they end with a question mark.\n\n"
                    f"Excerpts:\n{context}"
                )
                
                response = llm.invoke(prompt)
                raw_text = response.content
                
                questions = [
                    q.strip().lstrip("1234567890.*- ")
                    for q in raw_text.split("\n")
                    if "?" in q
                ]
                st.session_state.suggested_questions = questions[:3]
            except Exception as e:
                logger.error(f"Failed to generate suggested questions: {e}")
                st.session_state.suggested_questions = []
                
            st.session_state.suggestions_generated = True
            st.rerun()

    if st.session_state.suggested_questions:
        st.markdown(
            "<p style='color: #00ffcc; font-size: 1rem; margin-top: 20px; "
            "text-align: center;'>RECOMMENDED NEXT QUERIES:</p>",
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        questions = st.session_state.suggested_questions

        if len(questions) > 0 and col1.button(questions[0], key="btn1"):
            st.session_state.pending_query = questions[0]
        if len(questions) > 1 and col2.button(questions[1], key="btn2"):
            st.session_state.pending_query = questions[1]
        if len(questions) > 2 and col3.button(questions[2], key="btn3"):
            st.session_state.pending_query = questions[2]


# ============================================================================
# MAIN
# ============================================================================
def main():
    apply_tech_theme()
    init_session_state()

    st.title("⚡ PDF/Paper Analyser RAG App ⚡")
    st.markdown(
        "<p style='text-align: center; color: #58a6ff; margin-top: -20px; "
        "font-weight: bold;'>[ Secure Uplink Established — Phase 2: Structure-Aware RAG ]</p>",
        unsafe_allow_html=True,
    )

    uploaded_files, selected_model = render_sidebar()

    # --- Detect file change and reset session ---
    if uploaded_files:
        current_file_names = tuple(sorted([f.name for f in uploaded_files]))
        if st.session_state.current_files != current_file_names:
            st.session_state.messages = []
            st.session_state.current_files = current_file_names
            st.session_state.suggestions_generated = False
            st.session_state.suggested_questions = []
            st.session_state.pending_query = None
            st.session_state.uploaded_file_names = [f.name for f in uploaded_files]

    render_chat_history()

    if uploaded_files:
        with st.spinner("⏳ Compiling neural embeddings & running OCR vision models..."):
            file_bytes_tuple = tuple([f.getvalue() for f in uploaded_files])
            retriever = get_retriever(file_bytes_tuple)

        rag_chain = create_rag_chain(retriever, selected_model)

        render_suggested_questions(retriever, selected_model)

        user_query = st.chat_input("Enter query parameter...")
        active_query = user_query or st.session_state.pending_query

        if active_query:
            st.session_state.pending_query = None
            st.session_state.messages.append({"role": "user", "content": active_query})

            with st.chat_message("assistant"):
                with st.spinner(f"Processing through {selected_model} node..."):
                    start_time = time.time()

                    # Build chat history for the chain
                    chat_history = []
                    for m in st.session_state.messages[:-1]:
                        if m["role"] == "user":
                            chat_history.append(HumanMessage(content=m["content"]))
                        else:
                            chat_history.append(AIMessage(content=m["content"]))

                    response = rag_chain.invoke({
                        "input": active_query,
                        "chat_history": chat_history,
                    })
                    answer = response["answer"]
                    source_docs = [
                        {
                            "content": doc.page_content,
                            "extraction_method": doc.metadata.get("extraction_method", "Unknown"),
                            "page": doc.metadata.get("page", "?"),
                            "section": doc.metadata.get("section", ""),
                            "content_type": doc.metadata.get("content_type", "text"),
                            "document_name": doc.metadata.get("document_name", ""),
                            "dense_score": doc.metadata.get("dense_score"),
                            "bm25_score": doc.metadata.get("bm25_score"),
                            "rrf_score": doc.metadata.get("rrf_score"),
                            "rerank_score": doc.metadata.get("rerank_score"),
                        }
                        for doc in response["context"]
                    ]
                    end_time = time.time()

                # Generate follow-up suggestions using a direct LLM call to save time
                with st.spinner("Calculating next logical query vectors..."):
                    try:
                        from langchain_groq import ChatGroq
                        llm = ChatGroq(model=selected_model, temperature=0.7)
                        
                        followup_prompt = (
                            f"The user just asked: '{active_query}'\n"
                            f"The AI answered: '{answer}'\n\n"
                            "Based on this interaction, suggest EXACTLY 3 short, logical "
                            "follow-up questions the user should ask next to dive deeper. "
                            "Format strictly as a bulleted list ending in question marks. "
                            "Do not include any other text."
                        )
                        
                        followup_response = llm.invoke(followup_prompt)
                        raw_text = followup_response.content
                        
                        new_questions = [
                            q.strip().lstrip("1234567890.*- ")
                            for q in raw_text.split("\n")
                            if "?" in q
                        ]
                        st.session_state.suggested_questions = new_questions[:3]
                    except Exception as e:
                        logger.error(f"Failed to generate follow-ups: {e}")
                        st.session_state.suggested_questions = []

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_docs,
                "time": end_time - start_time,
            })
            st.rerun()

    else:
        st.markdown("""
            <div style='text-align: center; padding: 50px; background: rgba(0, 255, 204, 0.05);
                        border: 1px dashed #00ffcc; border-radius: 10px; margin-top: 50px;'>
                <h2 style='color: #a9bad4;'>SYSTEM STANDBY</h2>
                <p style='color: #58a6ff;'>
                    Awaiting document uplink. Please select a mode and upload
                    databanks in the sidebar to initialize the neural network.
                </p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
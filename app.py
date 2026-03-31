import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Must be the very first Streamlit command
st.set_page_config(page_title="Neural RAG System", layout="wide", page_icon="⚡")

# API Keys are now loaded securely from the .env file

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Fix a common warning with huggingface in Streamlit

# --- CSS STYLING FOR TECH AESTHETIC ---
def apply_tech_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    /* Overall Background */
    .stApp {
        background-color: #0b101e;
        color: #a9bad4;
        font-family: 'Share Tech Mono', monospace;
    }
    
    /* Top Header */
    h1 {
        text-shadow: 0 0 15px rgba(0, 255, 204, 0.8);
        color: #00ffcc !important;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }
    
    /* Neon Boxes */
    .metric-box {
        background: rgba(0, 255, 204, 0.05);
        border: 1px solid #00ffcc;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.2) inset, 0 0 10px rgba(0, 255, 204, 0.2);
        border-radius: 4px;
        padding: 6px 15px;
        color: #00ffcc;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 15px;
    }
    
    /* Chat inputs */
    .stChatInputContainer, .stChatInput {
        background-color: #121929 !important;
        border: 1px solid #1a2a44 !important;
        box-shadow: 0 0 10px rgba(88, 166, 255, 0.1) !important;
    }
    
    /* Chat bubbles */
    .stChatMessage {
        background-color: #111827;
        border: 1px solid #1f2d47;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1221;
        border-right: 1px solid #00ffcc;
        box-shadow: 5px 0 15px rgba(0, 255, 204, 0.1);
    }
    
    /* File Uploader override */
    .stFileUploader {
        border-radius: 10px;
    }
    
    /* Hide default Streamlit marks */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# --- CORE LOGIC ---
@st.cache_resource(show_spinner=False)
def init_rag_system(file_bytes):
    """Initializes the RAG chain. Cached so we don't recalculate embeddings every chat message!"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    # 1. Load Document
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_retries=2)

    # 5. Prompt Setup
    system_prompt = (
        "You are a highly advanced AI Assistant integrated into a neural network interface. "
        "Analyze the retrieved document data and answer the user clearly and concisely.\n\n"
        "Retrieved Data Fragments:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain


# --- UI LAYOUT ---
apply_tech_theme()

st.title("⚡ Neural RAG Node ⚡")
st.markdown("<p style='text-align: center; color: #58a6ff;'>Secure uplink established. Upload your dataset to begin.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONFIG")
    st.markdown("---")
    uploaded_file = st.file_uploader("UPLOAD DATABANK (PDF)", type=["pdf"])
    st.markdown("---")
    st.markdown("""
        **System Specs:**
        - **Core:** LangChain
        - **LLM:** LLaMA 3.3 (70B) via Groq
        - **Vector DB:** Chroma
        - **Embeddings:** all-MiniLM-L6-v2
    """)

# Keep track of history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "time" in msg:
            st.markdown(f'<div class="metric-box">⏱️ Response Metric: {msg["time"]:.2f}s</div>', unsafe_allow_html=True)

# Main interaction loop
if uploaded_file:
    # We pass .getvalue() so Streamlit hashing for @st.cache_resource works reliably on the raw bytes
    with st.spinner("⏳ Compiling neural embeddings... Please wait."):
        rag_chain = init_rag_system(uploaded_file.getvalue())
    
    if user_query := st.chat_input("Enter query parameter..."):
        # Add User message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Add Assistant message
        with st.chat_message("assistant"):
            with st.spinner("Processing through LLaMA-3.3 node..."):
                start_time = time.time()
                
                # Fetch Answer from RAG
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Display Answer
                st.write(answer)
                st.markdown(f'<div class="metric-box">⏱️ Response Metric: {response_time:.2f}s</div>', unsafe_allow_html=True)
                
        # Save to state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "time": response_time
        })
else:
    st.info("Awaiting uplink connection... Please upload a PDF in the sidebar panel.")

# import streamlit as st
# import os
# import time
# import tempfile
# import uuid
# from dotenv import load_dotenv

# load_dotenv()
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_groq import ChatGroq
# from langchain_classic.chains import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

# # Must be the very first Streamlit command
# st.set_page_config(page_title="PDF/Paper analyser RAG System", layout="wide", page_icon="⚡")

# # API Keys are now loaded securely from the .env file

# os.environ["TOKENIZERS_PARALLELISM"] = "false" # Fix a common warning with huggingface in Streamlit

# # --- CSS STYLING FOR TECH AESTHETIC ---
# def apply_tech_theme():
#     st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
#     /* Overall Background */
#     .stApp {
#         background-color: #0b101e;
#         color: #a9bad4;
#         font-family: 'Share Tech Mono', monospace;
#     }
    
#     /* Top Header */
#     h1 {
#         text-shadow: 0 0 15px rgba(0, 255, 204, 0.8);
#         color: #00ffcc !important;
#         text-align: center;
#         text-transform: uppercase;
#         letter-spacing: 2px;
#         margin-bottom: 30px;
#     }
    
#     /* Neon Boxes */
#     .metric-box {
#         background: rgba(0, 255, 204, 0.05);
#         border: 1px solid #00ffcc;
#         box-shadow: 0 0 10px rgba(0, 255, 204, 0.2) inset, 0 0 10px rgba(0, 255, 204, 0.2);
#         border-radius: 4px;
#         padding: 6px 15px;
#         color: #00ffcc;
#         font-size: 0.9rem;
#         display: inline-block;
#         margin-top: 15px;
#     }
    
#     /* Chat inputs */
#     .stChatInputContainer, .stChatInput {
#         background-color: #121929 !important;
#         border: 1px solid #1a2a44 !important;
#         box-shadow: 0 0 10px rgba(88, 166, 255, 0.1) !important;
#     }
    
#     /* Chat bubbles */
#     .stChatMessage {
#         background-color: #111827;
#         border: 1px solid #1f2d47;
#         border-radius: 8px;
#         padding: 15px;
#         margin-bottom: 20px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.3);
#     }
    
#     /* Sidebar */
#     [data-testid="stSidebar"] {
#         background-color: #0d1221;
#         border-right: 1px solid #00ffcc;
#         box-shadow: 5px 0 15px rgba(0, 255, 204, 0.1);
#     }
    
#     /* File Uploader override */
#     .stFileUploader {
#         border-radius: 10px;
#     }
    
#     /* Hide default Streamlit marks */
#     #MainMenu {visibility: hidden;}
#     header {visibility: hidden;}
#     footer {visibility: hidden;}
#     </style>
#     """, unsafe_allow_html=True)


# # --- CORE LOGIC ---
# @st.cache_resource(show_spinner=False)
# def init_rag_system(file_bytes):
#     """Initializes the RAG chain. Cached so we don't recalculate embeddings every chat message!"""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(file_bytes)
#         tmp_file_path = tmp_file.name

#     # 1. Load Document
#     loader = PyPDFLoader(tmp_file_path)
#     docs = loader.load()

#     # 2. Chunking
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(docs)

#     # 3. Embedding
#     embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = Chroma.from_documents(
#         documents=splits, 
#         embedding=embedding_model,
#         collection_name=f"pdf_{uuid.uuid4().hex}"
#     )
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

#     # 4. LLM
#     llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_retries=2)

#     # 5. Prompt Setup
#     system_prompt = (
#         "You are a highly advanced AI Assistant integrated into a neural network interface. "
#         "Analyze the retrieved document data and answer the user clearly and concisely.\n\n"
#         "Retrieved Data Fragments:\n{context}"
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ])

#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
#     return rag_chain


# # --- UI LAYOUT ---
# apply_tech_theme()

# st.title("⚡ Neural RAG Node ⚡")
# st.markdown("<p style='text-align: center; color: #58a6ff;'>Secure uplink established. Upload your dataset to begin.</p>", unsafe_allow_html=True)

# with st.sidebar:
#     st.markdown("### ⚙️ SYSTEM CONFIG")
#     st.markdown("---")
#     uploaded_file = st.file_uploader("UPLOAD DATABANK (PDF)", type=["pdf"])
#     st.markdown("---")
#     st.markdown("""
#         **System Specs:**
#         - **Core:** LangChain
#         - **LLM:** LLaMA 3.3 (70B) via Groq
#         - **Vector DB:** Chroma
#         - **Embeddings:** all-MiniLM-L6-v2
#     """)

# # Keep track of history
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "current_file" not in st.session_state:
#     st.session_state.current_file = None

# if uploaded_file:
#     if st.session_state.current_file != uploaded_file.name:
#         st.session_state.messages = [] # Wipe old chat when new file is uploaded
#         st.session_state.current_file = uploaded_file.name

# # Display chat history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])
#         if "time" in msg:
#             st.markdown(f'<div class="metric-box">⏱️ Response Metric: {msg["time"]:.2f}s</div>', unsafe_allow_html=True)

# # Main interaction loop
# if uploaded_file:
#     # We pass .getvalue() so Streamlit hashing for @st.cache_resource works reliably on the raw bytes
#     with st.spinner("⏳ Compiling neural embeddings... Please wait."):
#         rag_chain = init_rag_system(uploaded_file.getvalue())
    
#     if user_query := st.chat_input("Enter query parameter..."):
#         # Add User message
#         st.session_state.messages.append({"role": "user", "content": user_query})
#         with st.chat_message("user"):
#             st.write(user_query)

#         # Add Assistant message
#         with st.chat_message("assistant"):
#             with st.spinner("Processing through LLaMA-3.3 node..."):
#                 start_time = time.time()
                
#                 # Fetch Answer from RAG
#                 response = rag_chain.invoke({"input": user_query})
#                 answer = response["answer"]
                
#                 end_time = time.time()
#                 response_time = end_time - start_time
                
#                 # Display Answer
#                 st.write(answer)
#                 st.markdown(f'<div class="metric-box">⏱️ Response Metric: {response_time:.2f}s</div>', unsafe_allow_html=True)
                
#         # Save to state
#         st.session_state.messages.append({
#             "role": "assistant", 
#             "content": answer,
#             "time": response_time
#         })
# else:
#     st.info("Awaiting uplink connection... Please upload a PDF in the sidebar panel.")



# import streamlit as st
# import os
# import time
# import tempfile
# import uuid
# import chromadb

# # Your hardcoded API key
# os.environ["GROQ_API_KEY"] = "gsk_******************************"
# os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_groq import ChatGroq
# from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage, AIMessage

# # Must be the very first Streamlit command
# st.set_page_config(page_title="PDF/Paper Analyser RAG App", layout="wide", page_icon="⚡")

# # --- ADVANCED CSS STYLING & ANIMATIONS ---
# def apply_tech_theme():
#     st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
#     .stApp { background-color: #080c16; color: #a9bad4; font-family: 'Share Tech Mono', monospace; }
    
#     h1 { 
#         text-align: center; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 30px;
#         background: linear-gradient(90deg, #00ffcc, #0077ff, #00ffcc); background-size: 200% auto;
#         color: transparent !important; -webkit-background-clip: text; animation: shine 3s linear infinite;
#         text-shadow: 0 0 20px rgba(0, 255, 204, 0.3);
#     }
#     @keyframes shine { to { background-position: 200% center; } }

#     .metric-box { 
#         background: rgba(0, 255, 204, 0.05); border: 1px solid #00ffcc; border-radius: 4px; padding: 6px 15px; 
#         color: #00ffcc; font-size: 0.9rem; display: inline-block; margin-top: 15px; 
#         animation: pulse-glow 2.5s infinite alternate;
#     }
#     @keyframes pulse-glow {
#         0% { box-shadow: 0 0 5px rgba(0, 255, 204, 0.2) inset, 0 0 5px rgba(0, 255, 204, 0.2); }
#         100% { box-shadow: 0 0 15px rgba(0, 255, 204, 0.6) inset, 0 0 15px rgba(0, 255, 204, 0.6); }
#     }

#     .stButton > button {
#         background-color: #0e1526; border: 1px solid #0077ff; color: #a9bad4; border-radius: 5px;
#         transition: all 0.3s ease; width: 100%; height: 100%; white-space: normal; min-height: 60px;
#     }
#     .stButton > button:hover {
#         background-color: rgba(0, 255, 204, 0.1); border-color: #00ffcc; color: #00ffcc;
#         box-shadow: 0 0 15px rgba(0, 255, 204, 0.4); transform: translateY(-2px);
#     }

#     div.row-widget.stRadio > div { background: #0e1526; padding: 10px; border-radius: 10px; border: 1px solid #1f2d47; }
#     .stChatInputContainer, .stChatInput { background-color: #0e1526 !important; border: 1px solid #1a2a44 !important; box-shadow: 0 0 15px rgba(0, 119, 255, 0.1) !important; transition: all 0.3s ease; }
#     .stChatInputContainer:focus-within { border-color: #00ffcc !important; box-shadow: 0 0 20px rgba(0, 255, 204, 0.3) !important; }
    
#     .stChatMessage { background-color: #0e1526; border: 1px solid #1f2d47; border-radius: 8px; padding: 15px; margin-bottom: 20px; animation: slideUpFade 0.4s ease-out forwards; }
#     @keyframes slideUpFade { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

#     .streamlit-expanderHeader { background-color: rgba(0, 255, 204, 0.05); color: #00ffcc !important; border: 1px solid #1f2d47; border-radius: 5px; }
#     [data-testid="stSidebar"] { background-color: #0a0f1c; border-right: 1px solid rgba(0, 255, 204, 0.3); box-shadow: 5px 0 20px rgba(0, 255, 204, 0.05); }
#     .stFileUploader { border-radius: 10px; border: 1px dashed #00ffcc !important; background: rgba(0, 255, 204, 0.02); }
    
#     #MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
#     </style>
#     """, unsafe_allow_html=True)


# # --- CORE LOGIC ---
# @st.cache_resource(show_spinner=False)
# def init_rag_system(file_contents_tuple):
#     all_splits = []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#     for file_bytes in file_contents_tuple:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(file_bytes)
#             tmp_file_path = tmp_file.name

#         loader = PyPDFLoader(tmp_file_path)
#         docs = loader.load()
#         splits = text_splitter.split_documents(docs)
#         all_splits.extend(splits)

#     embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     ephemeral_client = chromadb.EphemeralClient()
#     unique_collection_name = f"pdf_batch_{uuid.uuid4().hex}"
    
#     vectorstore = Chroma.from_documents(
#         documents=all_splits, embedding=embedding_model,
#         client=ephemeral_client, collection_name=unique_collection_name
#     )
    
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_retries=2)

#     # NEW: 1. Contextualize Question Prompt (The Memory Engine)
#     contextualize_q_system_prompt = (
#         "Given a chat history and the latest user question "
#         "which might reference context in the chat history, "
#         "formulate a standalone question which can be understood "
#         "without the chat history. Do NOT answer the question, "
#         "just reformulate it if needed and otherwise return it as is."
#     )
#     contextualize_q_prompt = ChatPromptTemplate.from_messages([
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ])
    
#     # Create the history-aware retriever
#     history_aware_retriever = create_history_aware_retriever(
#         llm, retriever, contextualize_q_prompt
#     )

#     # NEW: 2. Answer Question Prompt (Now supports chat_history)
#     system_prompt = (
#         "You are a highly advanced AI Assistant integrated into a neural network interface. "
#         "Analyze the retrieved document fragments and answer the user clearly, concisely, and accurately based ONLY on the provided context.\n\n"
#         "Retrieved Data Fragments:\n{context}"
#     )
#     qa_prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ])

#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
#     # NEW: 3. Final Memory-Enabled RAG Chain
#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
#     return rag_chain


# # --- UI LAYOUT ---
# apply_tech_theme()

# st.title("⚡ PDF/Paper Analyser RAG App ⚡")
# st.markdown("<p style='text-align: center; color: #58a6ff; margin-top: -20px; font-weight: bold;'>[ Secure Uplink Established ]</p>", unsafe_allow_html=True)

# # Application State Management
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "current_files" not in st.session_state:
#     st.session_state.current_files = ()
# if "suggestions_generated" not in st.session_state:
#     st.session_state.suggestions_generated = False
# if "suggested_questions" not in st.session_state:
#     st.session_state.suggested_questions = []
# if "pending_query" not in st.session_state:
#     st.session_state.pending_query = None

# with st.sidebar:
#     st.markdown("### ⚙️ SYSTEM CONFIG")
#     app_mode = st.radio("SELECT UPLINK MODE:", ["Single PDF Explorer", "Explore Multiple PDFs"])
#     st.markdown("---")
    
#     if app_mode == "Single PDF Explorer":
#         uploaded_file = st.file_uploader("UPLOAD DATABANK (PDF)", type=["pdf"], accept_multiple_files=False)
#         uploaded_files = [uploaded_file] if uploaded_file else []
#     else:
#         uploaded_files = st.file_uploader("UPLOAD MULTIPLE DATABANKS", type=["pdf"], accept_multiple_files=True)
    
#     st.markdown("---")
#     st.markdown("""
#         **System Architecture:**
#         - **Pipeline:** LangChain Core (w/ Memory Cortex)
#         - **Brain:** LLaMA 3.3 (70B) via Groq
#         - **Memory:** ChromaDB (Isolated)
#         - **Sensors:** all-MiniLM-L6-v2
#     """)
    
#     st.markdown("---")
#     st.markdown("### 💾 DATA EXPORT")
#     if st.session_state.messages:
#         chat_log = "PDF/PAPER ANALYSER RAG APP - SESSION LOG\n" + "="*40 + "\n\n"
#         for msg in st.session_state.messages:
#             role = "USER" if msg["role"] == "user" else "AI NODE"
#             chat_log += f"[{role}]:\n{msg['content']}\n\n"
#             if msg.get("sources"):
#                 chat_log += "--- EXTRACTED FRAGMENTS ---\n"
#                 for i, doc in enumerate(msg["sources"]):
#                     chat_log += f"Fragment {i+1}: {doc}\n"
#                 chat_log += "---------------------------\n\n"
                
#         st.download_button(
#             label="Download Session Log (.txt)",
#             data=chat_log, file_name=f"Neural_Log_{int(time.time())}.txt", mime="text/plain", use_container_width=True
#         )
#     else:
#         st.caption("No session data to export yet.")

# if uploaded_files:
#     current_file_names = tuple(sorted([f.name for f in uploaded_files]))
#     if st.session_state.current_files != current_file_names:
#         st.session_state.messages = [] 
#         st.session_state.current_files = current_file_names
#         st.session_state.suggestions_generated = False
#         st.session_state.suggested_questions = []
#         st.session_state.pending_query = None

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])
#         if msg.get("sources"):
#             with st.expander("🔍 View Extracted Source Fragments"):
#                 for i, doc_text in enumerate(msg["sources"]):
#                     st.markdown(f"**Fragment {i+1}:**")
#                     st.caption(doc_text)
#                     st.markdown("---")
#         if "time" in msg:
#             st.markdown(f'<div class="metric-box">⏱️ Response Metric: {msg["time"]:.2f}s</div>', unsafe_allow_html=True)

# if uploaded_files:
#     with st.spinner("⏳ Compiling neural embeddings & isolating vector space..."):
#         file_bytes_tuple = tuple([f.getvalue() for f in uploaded_files])
#         rag_chain = init_rag_system(file_bytes_tuple)
    
#     # 1. INITIAL COLD-START SUGGESTIONS
#     if not st.session_state.suggestions_generated:
#         with st.spinner("🧠 Analyzing document context to generate optimal query vectors..."):
            
#             # Format history for the first invoke (empty)
#             chat_history = []
            
#             init_prompt = "Analyze the provided document context and provide EXACTLY 3 highly relevant, insightful questions that a user should ask to understand the core content. Format your response strictly as a bulleted list. Ensure they end with a question mark."
#             response = rag_chain.invoke({
#                 "input": init_prompt,
#                 "chat_history": chat_history
#             })
            
#             raw_text = response["answer"]
#             questions = [q.strip().lstrip('1234567890.*- ') for q in raw_text.split('\n') if '?' in q]
#             st.session_state.suggested_questions = questions[:3]
#             st.session_state.suggestions_generated = True
#             st.rerun() 
            
#     if st.session_state.suggested_questions:
#         st.markdown("<p style='color: #00ffcc; font-size: 1rem; margin-top: 20px; text-align: center;'>RECOMMENDED NEXT QUERIES:</p>", unsafe_allow_html=True)
#         col1, col2, col3 = st.columns(3)
        
#         if len(st.session_state.suggested_questions) > 0:
#             if col1.button(st.session_state.suggested_questions[0], key="btn1"):
#                 st.session_state.pending_query = st.session_state.suggested_questions[0]
#         if len(st.session_state.suggested_questions) > 1:
#             if col2.button(st.session_state.suggested_questions[1], key="btn2"):
#                 st.session_state.pending_query = st.session_state.suggested_questions[1]
#         if len(st.session_state.suggested_questions) > 2:
#             if col3.button(st.session_state.suggested_questions[2], key="btn3"):
#                 st.session_state.pending_query = st.session_state.suggested_questions[2]

#     user_query = st.chat_input("Enter query parameter...")
#     active_query = user_query or st.session_state.pending_query

#     if active_query:
#         st.session_state.pending_query = None
#         st.session_state.messages.append({"role": "user", "content": active_query})
        
#         with st.chat_message("assistant"):
#             with st.spinner("Processing through LLaMA-3.3 node..."):
#                 start_time = time.time()
                
#                 # NEW: Convert session messages into LangChain Message Objects
#                 chat_history = []
#                 for m in st.session_state.messages[:-1]: # exclude the current user message just added
#                     if m["role"] == "user":
#                         chat_history.append(HumanMessage(content=m["content"]))
#                     else:
#                         chat_history.append(AIMessage(content=m["content"]))
                
#                 # Fetch Answer & Context (Now passing chat history!)
#                 response = rag_chain.invoke({
#                     "input": active_query,
#                     "chat_history": chat_history
#                 })
#                 answer = response["answer"]
#                 source_docs = [doc.page_content for doc in response["context"]] 
                
#                 end_time = time.time()
#                 response_time = end_time - start_time
                
#             with st.spinner("Calculating next logical query vectors..."):
#                 # Append the latest AI answer to history for the follow-up prompt
#                 chat_history.append(HumanMessage(content=active_query))
#                 chat_history.append(AIMessage(content=answer))
                
#                 followup_prompt = f"The user just asked: '{active_query}'. Based on the document context and your answer, suggest EXACTLY 3 short, logical follow-up questions the user should ask next to dive deeper. Format strictly as a bulleted list ending in question marks."
#                 followup_response = rag_chain.invoke({
#                     "input": followup_prompt,
#                     "chat_history": chat_history
#                 })
                
#                 raw_text = followup_response["answer"]
#                 new_questions = [q.strip().lstrip('1234567890.*- ') for q in raw_text.split('\n') if '?' in q]
#                 st.session_state.suggested_questions = new_questions[:3]
                
#         st.session_state.messages.append({
#             "role": "assistant", 
#             "content": answer,
#             "sources": source_docs,
#             "time": response_time
#         })
        
#         st.rerun() 
# else:
#     st.markdown("""
#         <div style='text-align: center; padding: 50px; background: rgba(0, 255, 204, 0.05); border: 1px dashed #00ffcc; border-radius: 10px; margin-top: 50px;'>
#             <h2 style='color: #a9bad4;'>SYSTEM STANDBY</h2>
#             <p style='color: #58a6ff;'>Awaiting document uplink. Please select a mode and upload databanks in the sidebar to initialize the neural network.</p>
#         </div>
#     """, unsafe_allow_html=True)




import streamlit as st
import os
import time
import tempfile
import uuid
import chromadb
import platform
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- TESSERACT PATH CONFIGURATION ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="PDF/Paper Analyser RAG App", layout="wide", page_icon="⚡")

# --- ADVANCED CSS STYLING & ANIMATIONS ---
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
        background: rgba(0, 255, 204, 0.05); border: 1px solid #00ffcc; border-radius: 4px; padding: 6px 15px; 
        color: #00ffcc; font-size: 0.9rem; display: inline-block; margin-top: 15px; 
        animation: pulse-glow 2.5s infinite alternate;
    }
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 5px rgba(0, 255, 204, 0.2) inset, 0 0 5px rgba(0, 255, 204, 0.2); }
        100% { box-shadow: 0 0 15px rgba(0, 255, 204, 0.6) inset, 0 0 15px rgba(0, 255, 204, 0.6); }
    }

    .stButton > button {
        background-color: #0e1526; border: 1px solid #0077ff; color: #a9bad4; border-radius: 5px;
        transition: all 0.3s ease; width: 100%; height: 100%; white-space: normal; min-height: 60px;
    }
    .stButton > button:hover {
        background-color: rgba(0, 255, 204, 0.1); border-color: #00ffcc; color: #00ffcc;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.4); transform: translateY(-2px);
    }

    div.row-widget.stRadio > div { background: #0e1526; padding: 10px; border-radius: 10px; border: 1px solid #1f2d47; }
    
    /* Styled Selectbox for Models */
    .stSelectbox > div > div { background-color: #0e1526; color: #00ffcc; border: 1px solid #1f2d47; }
    
    .stChatInputContainer, .stChatInput { background-color: #0e1526 !important; border: 1px solid #1a2a44 !important; box-shadow: 0 0 15px rgba(0, 119, 255, 0.1) !important; transition: all 0.3s ease; }
    .stChatInputContainer:focus-within { border-color: #00ffcc !important; box-shadow: 0 0 20px rgba(0, 255, 204, 0.3) !important; }
    
    .stChatMessage { background-color: #0e1526; border: 1px solid #1f2d47; border-radius: 8px; padding: 15px; margin-bottom: 20px; animation: slideUpFade 0.4s ease-out forwards; }
    @keyframes slideUpFade { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

    .streamlit-expanderHeader { background-color: rgba(0, 255, 204, 0.05); color: #00ffcc !important; border: 1px solid #1f2d47; border-radius: 5px; }
    [data-testid="stSidebar"] { background-color: #0a0f1c; border-right: 1px solid rgba(0, 255, 204, 0.3); box-shadow: 5px 0 20px rgba(0, 255, 204, 0.05); }
    .stFileUploader { border-radius: 10px; border: 1px dashed #00ffcc !important; background: rgba(0, 255, 204, 0.02); }
    
    /* Hide default Streamlit marks but KEEP the sidebar toggle visible */
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;}
    
    /* Make the top header transparent instead of hidden */
    header {background-color: transparent !important;}
    
    /* Style the sidebar toggle arrow to match the cyberpunk theme */
    [data-testid="collapsedControl"] {
        color: #00ffcc !important; 
        background-color: #0e1526 !important; 
        border: 1px solid #1f2d47 !important;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- ARCHITECTURE SPLIT: Phase 1 (Data Processing - Cached) ---
@st.cache_resource(show_spinner=False)
def init_vectorstore(file_contents_tuple):
    """Processes PDFs, runs OCR, and builds the Vector Database ONCE."""
    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)

    for file_bytes in file_contents_tuple:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        doc = fitz.open(tmp_file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            extraction_source = "PyMuPDF"
            
            if len(text.strip()) < 50:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                extraction_source = "Tesseract OCR"
            
            # Create a document for each page to retain its specific extraction origin
            docs = [Document(page_content=text, metadata={"source": tmp_file_path, "extraction_method": extraction_source, "page": page_num + 1})]
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)

    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    ephemeral_client = chromadb.EphemeralClient()
    unique_collection_name = f"pdf_batch_{uuid.uuid4().hex}"
    
    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=embedding_model,
        client=ephemeral_client, collection_name=unique_collection_name
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 12})


# --- ARCHITECTURE SPLIT: Phase 2 (Brain Assembly - Dynamic) ---
def create_rag_chain(retriever, selected_model):
    """Dynamically builds the chain using whatever model the user selected."""
    llm = ChatGroq(model=selected_model, temperature=0.3, max_retries=2)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are a highly advanced AI Assistant integrated into a neural network interface. "
        "Analyze the retrieved document fragments and answer the user clearly, concisely, and accurately based ONLY on the provided context.\n\n"
        "Retrieved Data Fragments:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- UI LAYOUT ---
apply_tech_theme()

st.title("⚡ PDF/Paper Analyser RAG App ⚡")
st.markdown("<p style='text-align: center; color: #58a6ff; margin-top: -20px; font-weight: bold;'>[ Secure Uplink Established ]</p>", unsafe_allow_html=True)

# Application State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_files" not in st.session_state:
    st.session_state.current_files = ()
if "suggestions_generated" not in st.session_state:
    st.session_state.suggestions_generated = False
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONFIG")
    app_mode = st.radio("SELECT UPLINK MODE:", ["Single PDF Explorer", "Explore Multiple PDFs"])
    
    if app_mode == "Single PDF Explorer":
        uploaded_file = st.file_uploader("UPLOAD DATABANK (PDF)", type=["pdf"], accept_multiple_files=False)
        uploaded_files = [uploaded_file] if uploaded_file else []
    else:
        uploaded_files = st.file_uploader("UPLOAD MULTIPLE DATABANKS", type=["pdf"], accept_multiple_files=True)
    
    st.markdown("---")
    
    # FIX: Updated to currently active, running Groq Models + newly requested high-speed options
    st.markdown("### 🧠 NEURAL CORE OVERRIDE")
    selected_model = st.selectbox(
        "Select AI Engine (Swap if rate-limited):",
        [
            "llama-3.3-70b-versatile",                   # The powerful 70B model
            "openai/gpt-oss-120b",                       # Massive 120B parameter model
            "llama-3.1-8b-instant",                      # The ultra-fast Llama model (500+ t/s)
            "openai/gpt-oss-20b",                        # Ultra-fast 20B Model (1,000+ t/s)
            "meta-llama/llama-4-scout-17b-16e-instruct", # Llama 4 Scout 17B (750+ t/s)
            "qwen/qwen3-32b"                             # Efficient Multilingual 32B model
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown(f"""
        **System Architecture:**
        - **Pipeline:** LangChain Core 
        - **Brain:** `{selected_model}`
        - **Vision/OCR:** PyMuPDF + Tesseract
        - **Memory:** ChromaDB (Isolated)
        - **Sensors:** BAAI/bge-small-en-v1.5
    """)
    
    st.markdown("---")
    st.markdown("### 💾 DATA EXPORT")
    if st.session_state.messages:
        chat_log = "PDF/PAPER ANALYSER RAG APP - SESSION LOG\n" + "="*40 + "\n\n"
        for msg in st.session_state.messages:
            role = "USER" if msg["role"] == "user" else "AI NODE"
            chat_log += f"[{role}]:\n{msg['content']}\n\n"
            if msg.get("sources"):
                chat_log += "--- EXTRACTED FRAGMENTS ---\n"
                for i, doc in enumerate(msg["sources"]):
                    chat_log += f"Fragment {i+1} [Extracted via {doc['extraction_method']} | Page {doc['page']}]:\n"
                    chat_log += f"{doc['content']}\n\n"
                chat_log += "---------------------------\n\n"
                
        st.download_button(
            label="Download Session Log (.txt)",
            data=chat_log, file_name=f"Neural_Log_{int(time.time())}.txt", mime="text/plain", use_container_width=True
        )
    else:
        st.caption("No session data to export yet.")

if uploaded_files:
    current_file_names = tuple(sorted([f.name for f in uploaded_files]))
    if st.session_state.current_files != current_file_names:
        st.session_state.messages = [] 
        st.session_state.current_files = current_file_names
        st.session_state.suggestions_generated = False
        st.session_state.suggested_questions = []
        st.session_state.pending_query = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 View Extracted Source Fragments"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Fragment {i+1}** *(Extracted via {doc['extraction_method']} on Page {doc['page']})*:")
                    st.caption(doc["content"])
                    st.markdown("---")
        if "time" in msg:
            st.markdown(f'<div class="metric-box">⏱️ Response Metric: {msg["time"]:.2f}s</div>', unsafe_allow_html=True)

if uploaded_files:
    # 1. Fetch the cached retriever (Only runs once per PDF upload)
    with st.spinner("⏳ Compiling neural embeddings & running OCR vision models..."):
        file_bytes_tuple = tuple([f.getvalue() for f in uploaded_files])
        retriever = init_vectorstore(file_bytes_tuple)
    
    # 2. Instantly build the chain with the user's currently selected model
    rag_chain = create_rag_chain(retriever, selected_model)
    
    if not st.session_state.suggestions_generated:
        with st.spinner(f"🧠 {selected_model} is analyzing document context..."):
            chat_history = []
            init_prompt = "Analyze the provided document context and provide EXACTLY 3 highly relevant, insightful questions that a user should ask to understand the core content. Format your response strictly as a bulleted list. Ensure they end with a question mark."
            response = rag_chain.invoke({
                "input": init_prompt,
                "chat_history": chat_history
            })
            
            raw_text = response["answer"]
            questions = [q.strip().lstrip('1234567890.*- ') for q in raw_text.split('\n') if '?' in q]
            st.session_state.suggested_questions = questions[:3]
            st.session_state.suggestions_generated = True
            st.rerun() 
            
    if st.session_state.suggested_questions:
        st.markdown("<p style='color: #00ffcc; font-size: 1rem; margin-top: 20px; text-align: center;'>RECOMMENDED NEXT QUERIES:</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        if len(st.session_state.suggested_questions) > 0:
            if col1.button(st.session_state.suggested_questions[0], key="btn1"):
                st.session_state.pending_query = st.session_state.suggested_questions[0]
        if len(st.session_state.suggested_questions) > 1:
            if col2.button(st.session_state.suggested_questions[1], key="btn2"):
                st.session_state.pending_query = st.session_state.suggested_questions[1]
        if len(st.session_state.suggested_questions) > 2:
            if col3.button(st.session_state.suggested_questions[2], key="btn3"):
                st.session_state.pending_query = st.session_state.suggested_questions[2]

    user_query = st.chat_input("Enter query parameter...")
    active_query = user_query or st.session_state.pending_query

    if active_query:
        st.session_state.pending_query = None
        st.session_state.messages.append({"role": "user", "content": active_query})
        
        with st.chat_message("assistant"):
            with st.spinner(f"Processing through {selected_model} node..."):
                start_time = time.time()
                
                chat_history = []
                for m in st.session_state.messages[:-1]: 
                    if m["role"] == "user":
                        chat_history.append(HumanMessage(content=m["content"]))
                    else:
                        chat_history.append(AIMessage(content=m["content"]))
                
                response = rag_chain.invoke({
                    "input": active_query,
                    "chat_history": chat_history
                })
                answer = response["answer"]
                source_docs = [{
                    "content": doc.page_content, 
                    "extraction_method": doc.metadata.get("extraction_method", "Unknown"),
                    "page": doc.metadata.get("page", "?")
                } for doc in response["context"]]
                
                end_time = time.time()
                response_time = end_time - start_time
                
            with st.spinner("Calculating next logical query vectors..."):
                chat_history.append(HumanMessage(content=active_query))
                chat_history.append(AIMessage(content=answer))
                
                followup_prompt = f"The user just asked: '{active_query}'. Based on the document context and your answer, suggest EXACTLY 3 short, logical follow-up questions the user should ask next to dive deeper. Format strictly as a bulleted list ending in question marks."
                followup_response = rag_chain.invoke({
                    "input": followup_prompt,
                    "chat_history": chat_history
                })
                
                raw_text = followup_response["answer"]
                new_questions = [q.strip().lstrip('1234567890.*- ') for q in raw_text.split('\n') if '?' in q]
                st.session_state.suggested_questions = new_questions[:3]
                
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": source_docs,
            "time": response_time
        })
        
        st.rerun() 
else:
    st.markdown("""
        <div style='text-align: center; padding: 50px; background: rgba(0, 255, 204, 0.05); border: 1px dashed #00ffcc; border-radius: 10px; margin-top: 50px;'>
            <h2 style='color: #a9bad4;'>SYSTEM STANDBY</h2>
            <p style='color: #58a6ff;'>Awaiting document uplink. Please select a mode and upload databanks in the sidebar to initialize the neural network.</p>
        </div>
    """, unsafe_allow_html=True)
# 📄 PDF & Research Paper Analyzer 🤖

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Framework-00A0DF.svg?style=flat-square)

**An intelligent document analysis system powered by Retrieval-Augmented Generation (RAG) and advanced LLMs**

[Features](#features) • [Tech Stack](#tech-stack) • [Installation](#installation) • [Usage](#usage) • [Setup](#setup) • [Contributing](#contributing)

</div>

---

## 🎯 Overview

**PDF & Research Paper Analyzer** is a cutting-edge Generative AI application that enables intelligent question-answering over PDF documents and research papers. Using **Retrieval-Augmented Generation (RAG)** with state-of-the-art embeddings and large language models, it understands complex document content and provides accurate, context-aware answers.

### Key Capabilities:
✨ **Smart Document Understanding** - Extract and comprehend content from PDFs  
🔍 **Advanced RAG** - Retrieve relevant context and generate accurate answers  
👁️ **OCR Support** - Process scanned documents and image-based PDFs  
💬 **Conversational AI** - Multi-turn chat with conversation history awareness  
🎨 **Modern UI** - Cyberpunk-themed interface with real-time streaming responses  
⚡ **Fast Processing** - Powered by Groq's ultra-fast LLM inference  

---

## ✨ Features

### 📚 Document Processing
- **Multi-format PDF Support**: Handle text-based and scanned PDFs seamlessly
- **Intelligent Chunking**: Recursive text splitting with overlap to preserve context
- **OCR Capability**: Extract text from scanned documents using Tesseract and PaddleOCR
- **Image Extraction**: Convert PDF pages to PNG images for analysis

### 🧠 RAG Technology
- **Vector Embeddings**: HuggingFace embeddings (all-MiniLM-L6-v2) for semantic understanding
- **ChromaDB Vector Store**: Persistent, efficient vector database for fast retrieval
- **Smart Retrieval**: Context-aware retrieval with configurable k-nearest neighbors
- **Session Management**: Maintain conversation history across multiple queries

### 💡 Intelligent Responses
- **Groq LLM Integration**: Ultra-fast inference using Llama 3.3 (70B) model
- **Context-Aware Answers**: Answers grounded in document content
- **Streaming Responses**: Real-time text streaming for better UX
- **Configurable Behavior**: Adjustable temperature, model selection, and system prompts

### 🎨 User Interface
- **Streamlit-Powered**: Clean, responsive web interface
- **Cyberpunk Theme**: Modern dark theme with animated elements
- **Interactive Elements**: File upload, text input, chat history display
- **Responsive Design**: Works seamlessly on desktop and mobile

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Web UI framework |
| **LLM** | Groq (Llama 3.3 70B) | Large language model inference |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Semantic text embeddings |
| **Vector DB** | ChromaDB | Vector storage & retrieval |
| **Framework** | LangChain | LLM orchestration & chaining |
| **PDF Processing** | PyPDF, PyMuPDF (fitz) | PDF parsing |
| **OCR** | Tesseract, PaddleOCR | Scanned document text extraction |
| **Image Processing** | Pillow | Image manipulation |
| **Environment** | python-dotenv | Environment variable management |

---

## 📋 Prerequisites

- **Python 3.9+**
- **Groq API Key** (free from [groq.com](https://groq.com))
- **Tesseract-OCR** (for OCR support)
- **Virtual Environment** (recommended)

---

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Abhinav-1612/PDF-Researchpaper_analyser_GENAI_APP.git
cd pdf-analyzer-rag
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install System Dependencies

#### Windows
1. **Tesseract-OCR**: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)
   - Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - Update path in `app.py` if installed elsewhere

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install tesseract-ocr libtesseract-dev
```

### Step 5: Configure Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key from [Groq Console](https://console.groq.com/keys)

---

## 💻 Usage

### Option 1: Streamlit Web App (Recommended)
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

**Features:**
- Upload PDF files (drag & drop or click to select)
- Ask questions about the document in the chat interface
- View conversation history
- Select different LLM models
- Real-time streaming responses

### Option 2: CLI Application
```bash
python rag_app.py
```
Then follow the interactive prompts:
1. Place `sample_paper.pdf` in the project folder
2. Type your questions when prompted
3. Type 'quit' to exit

---

## 📁 Project Structure

```
pdf-analyzer-rag/
│
├── app.py                      # Main Streamlit application
├── rag_app.py                  # CLI RAG implementation
├── generate_project_pdf.py     # PDF generation utilities
│
├── requirements.txt            # Python dependencies
├── packages.txt                # System packages (Tesseract)
├── .env.example                # Example environment variables
├── README.md                   # This file
│
└── data/                       # (Optional) Sample PDFs
    └── sample_paper.pdf
```

---

## 🎮 How It Works

### RAG Pipeline Architecture

```
┌─────────────────┐
│  PDF Upload     │
└────────┬────────┘
         ▼
┌─────────────────┐      ┌──────────────────┐
│  PDF Loading    │─────→│  OCR (if needed) │
└────────┬────────┘      └──────────────────┘
         ▼
┌─────────────────┐
│ Text Chunking   │ (Recursive splitting with overlap)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Embeddings     │ (HuggingFace)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Vector Store   │ (ChromaDB)
└────────┬────────┘
         ▼
    ┌────────────────────────────┐
    │  Query Processing          │
    │  1. Get embeddings         │
    │  2. Semantic search        │
    │  3. Retrieve top-3 chunks  │
    └────────┬───────────────────┘
             ▼
    ┌────────────────────────────┐
    │  LLM Processing (Groq)     │
    │  1. Context assembly       │
    │  2. Prompt generation      │
    │  3. Response streaming     │
    └────────┬───────────────────┘
             ▼
    ┌────────────────────────────┐
    │  Answer to User            │
    └────────────────────────────┘
```

### Key Components

1. **Document Loader**: Extracts text from PDFs using PyPDF
2. **Text Splitter**: Breaks content into 1000-char chunks with 200-char overlap
3. **Embedding Model**: Converts text to semantic vectors (384-dimensional)
4. **Vector Database**: Stores embeddings for fast retrieval
5. **Retriever**: Fetches top-3 most relevant chunks for each query
6. **LLM Chain**: Combines context with query for answer generation
7. **UI Layer**: Streamlit interface for seamless interaction

---

## ⚙️ Configuration

### Modify Model Selection
Edit the LLM model in `app.py`:
```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Try: "llama-3-8b-8192", "mixtral-8x7b-32768"
    temperature=0.3,                   # Lower = more deterministic
    max_retries=0
)
```

### Adjust Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Larger = more context per chunk
    chunk_overlap=200     # More overlap = better coherence
)
```

### Change Retrieval Parameters
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Number of chunks to retrieve
)
```

---

## 🔑 Groq API Setup

1. **Sign Up** at [groq.com](https://groq.com)
2. **Create API Key** in [Groq Console](https://console.groq.com/keys)
3. **Free Tier Benefits**:
   - 30 free API calls per minute
   - No credit card required for free tier
   - Access to Llama 3.3 70B model

---

## 📊 Performance Metrics

- **Embedding Generation**: ~100-500ms per document
- **Vector Search**: <50ms for similarity search
- **LLM Response**: 1-5 seconds (Groq inference)
- **Total End-to-End**: 2-10 seconds per query

---

## 🐛 Troubleshooting

### Issue: Tesseract Not Found
**Solution:**
- Windows: Ensure install path matches `app.py` configuration
- macOS/Linux: Run `brew install tesseract` or `apt-get install tesseract-ocr`

### Issue: Groq API Key Invalid
**Solution:**
- Verify API key in `.env` file
- Check key hasn't expired at [console.groq.com](https://console.groq.com/keys)
- Ensure no extra spaces in `.env` file

### Issue: Out of Memory with Large PDFs
**Solution:**
- Reduce `chunk_size` to smaller values
- Process PDFs in separate sessions
- Increase available system RAM

### Issue: OCR Not Working
**Solution:**
- Install Tesseract system package (not just Python package)
- Update Tesseract path in `app.py` if using custom installation
- Ensure PaddleOCR dependencies are installed

---

## 📈 Future Enhancements

- [ ] Multi-document analysis and comparison
- [ ] Advanced document segmentation
- [ ] Custom model fine-tuning
- [ ] Database persistence for conversation history
- [ ] Web scraping capabilities
- [ ] Support for other document formats (DOCX, TXT, etc.)
- [ ] Real-time collaboration features
- [ ] Advanced analytics and document insights

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Lint
flake8 .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Abhinav Singh**
- GitHub: [@Abhinav-1612](https://github.com/Abhinav-1612)
- Email: abhinavishu0311@gmail.com

---

## 🙏 Acknowledgments

- **LangChain**: For excellent LLM orchestration framework
- **Groq**: For providing blazing-fast LLM inference
- **HuggingFace**: For open-source embeddings and transformers
- **Streamlit**: For intuitive web app development
- **ChromaDB**: For efficient vector database

---

## 📞 Support

For issues, questions, or suggestions:
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Abhinav-1612/pdf-analyzer-rag/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Abhinav-1612/pdf-analyzer-rag/discussions)
- 📧 **Email**: abhinavishu0311@gmail.com

---

<div align="center">

Made with ❤️ by [Abhinav Singh]

⭐ If you find this project useful, please consider giving it a star!

</div>

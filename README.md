# 🧠 PaperForge-AI  
### AI-Powered Research Paper Generator from Uploaded Documents

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.10-00A67E?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.0-FF6F00?style=flat-square)](https://www.trychroma.com/)
[![OpenAI API](https://img.shields.io/badge/OpenAI%20API-2.20.0-00D084?style=flat-square&logo=openai&logoColor=white)](https://openai.com/)
[![PDFProcessing](https://img.shields.io/badge/PyPDF-6.7.0-0066CC?style=flat-square)](https://pypi.org/project/pypdf/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Meet--Amin-black?style=flat-square&logo=github)](https://github.com/Meet-Amin/PaperForge-AI)

PaperForge-AI is an end-to-end AI research assistant that transforms uploaded documents (PDF, DOCX, TXT, MD) into a structured academic-style research paper using Retrieval-Augmented Generation (RAG). It combines semantic search and LLM reasoning to synthesize knowledge into a coherent research document.

🔗 **Live Demo:** https://paperforge-ai.streamlit.app/

---

## 🚀 Features

- Upload multiple documents (PDF / DOCX / TXT / MD)
- Semantic vector search using embeddings
- AI-generated research paper synthesis
- Persistent Chroma vector database
- Optimized chunking + retrieval pipeline
- Streamlit web interface
- Secure API key loading via `.env`
- Structured academic-style output

---

## 🧩 How It Works

PaperForge-AI follows a Retrieval-Augmented Generation pipeline:

1. User uploads documents
2. Documents are chunked into smaller segments
3. Text chunks are embedded using OpenAI embeddings
4. Stored inside a Chroma vector database
5. Relevant chunks are retrieved
6. LLM generates a research paper
7. Final paper is displayed to the user

---

## 🏗 Architecture Diagram

![Architecture](architecture.svg)

---

## 🧪 Tech Stack

- **Python** - Core language
- **Streamlit** - Web interface
- **LangChain** - LLM orchestration & RAG framework
- **ChromaDB** - Vector database for embeddings
- **OpenAI API** - LLM & embeddings
- **Python-dotenv** - Environment configuration
- **PyPDF** - PDF document processing
- **python-docx** - DOCX document processing
- **Unstructured** - Document parsing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Pydantic** - Data validation

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Meet-Amin/PaperForge-AI.git
cd PaperForge-AI

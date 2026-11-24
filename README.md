# Modular RAG from Scratch with LangChain & Ollama

A lightweight, modular Retrieval-Augmented Generation (RAG) system built completely from scratch in Python, without relying on heavy abstractions for the indexing pipeline. It uses **ChromaDB** for vector storage, **Sentence Transformers** for embeddings, and integrates with **LangChain & Ollama** for autonomous agentic tool-calling.

---

## Project Structure

```text
RAG Testing/
│
├── RAG/
│   ├── __init__.py       # Public package exports
│   ├── chunking.py       # Text splitting with overlap logic
│   ├── embedding.py      # Vector embedding generation (all-MiniLM-L6-v2)
│   ├── loader.py         # Multi-format document loader (PDF, TXT, MD, JSON, DOCX, Images with OCR) & hash registry
│   ├── rag.py            # Preloading pipeline (reads folder, chunks, embeds, and stores)
│   └── search.py         # Vector similarity search querying ChromaDB
│
├── documents/            # Folder where you drop your local knowledge files
├── venv/                 # Python virtual environment
├── main.py               # Main CLI execution script with the LangChain agent
├── requirements.txt      # Project dependencies
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
```

---

## Features

- **Multi-Format Document Support:** Extract text natively from PDFs, Word documents (`.docx`), Markdown, Text files, JSON structures, and images via Tesseract OCR.
- **Incremental Indexing:** Uses file content hashing (`loaded_files.json`) to prevent reprocessing or duplicating files that are already indexed.
- **Local Embeddings:** Powered by `sentence-transformers/all-MiniLM-L6-v2` running locally on PyTorch/NumPy.
- **Persistent Vector Database:** Uses **ChromaDB** in persistent mode (`./rag_chroma`) so your data survives script restarts.
- **Agentic RAG Integration:** Powered by **LangChain** and **Ollama** (`qwen3.5:9b` or similar), enabling the LLM to decide dynamically when to invoke the RAG search tool based on user queries.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with your model of choice (e.g., `qwen3.5:9b`).]

---

## Installation & Setup

1. **Clone the repository or open the project directory:**
   ```bash
   git clone https://github.com/SebasmarR/RAG-Testing
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your documents:**
   Place your files (`.pdf`, `.docx`, `.txt`, etc.) inside a folder named `documents/` at the root of the project.

---

## Usage

Run the main application script:

```bash
python main.py
```

The script will automatically:
1. Preload and index any new or updated documents from the `documents/` folder.
2. Spin up the conversational agent loop.
3. Listen for your queries in the terminal and fetch relevant context using vector similarity search when needed.

Type `exit` or `quit` to close the session.

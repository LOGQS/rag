# 🔍 RAG Index Builder & Retrieval System

A lightning-fast, intelligent document indexing and retrieval system that turns your codebase and documents into a searchable knowledge base.

## ✨ Features

- **🚀 Blazing Fast**: Parallel processing for indexing and retrieval
- **🧠 Smart Chunking**: Code-aware parsing that understands 40+ programming languages
- **⚡ Incremental Updates**: Only processes changed files using SHA-256 hashing
- **🎯 Precise Retrieval**: Context-aware search with line number precision
- **🖥️ Dual Interface**: Both GUI and CLI for different workflows
- **💾 Persistent Storage**: ChromaDB vector store with automatic persistence

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Scanner  │───▶│  Smart Chunker  │───▶│  Vector Store   │
│   (Parallel)    │    │  (Code/Text)    │    │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Search Results │◀───│   Retriever     │◀───│  Query Engine   │
│  (w/ Context)   │    │  (Similarity)   │    │  (Embeddings)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Your Index
**GUI Mode:**
```bash
python index.py
```
Click "Select Files" → Choose your documents → "Build Index"

**CLI Mode:**
```python
from rag import build_index
build_index(["path/to/your/code", "path/to/docs"])
```

### 3. Search Your Knowledge Base
```bash
python retrieve.py
```
Enter your query and get instant, context-aware results!

## 🎯 What Makes It Special

- **Language-Aware**: Recognizes Python, JavaScript, Java, C++, and 35+ other languages
- **Context-Rich**: Shows surrounding code/text for better understanding and correct indentation
- **Incremental**: Skip re-indexing unchanged files for speed
- **GPU Accelerated**: Automatic CUDA detection for faster embeddings
- **Thread-Safe**: Parallel processing without race conditions

## ⚠️ Project Status

A enhanced, improved version of this RAG will run under a bigger ecosystem called **ATLAS2**. It is currently unfinished.

Stay tuned for the evolution! 🚀

## 📊 Performance

- **Indexing Speed**: ~1000 files/minute (typical codebase)
- **Search Latency**: <200ms for most queries
- **Memory Usage**: Efficient chunking keeps RAM usage low
- **Storage**: Compressed vector embeddings

## 🛠️ Technical Stack

- **Embeddings**: HuggingFace Transformers (sentence-transformers)
- **Vector Store**: ChromaDB with HNSW indexing
- **Chunking**: LlamaIndex with custom code splitters
- **UI**: Tkinter (GUI) + Rich CLI experience
- **Parallelization**: ThreadPoolExecutor + multiprocessing

---
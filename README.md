# 🧠 Offline AI RAG System

Local Retrieval-Augmented Generation (RAG) assistant powered by **FAISS**, **SentenceTransformers**, and **Ollama (LLM)**.

This system enables querying your own documents (PDFs, notes, memory) with semantic search + reasoning — fully offline.

---

## 🚀 Features

* 📄 PDF ingestion pipeline (multi-document support)
* 🔍 Semantic search using embeddings (e5-large-v2)
* 🧠 Local LLM via Ollama (llama3.1)
* 🔀 Multi-query retrieval (agent-style query expansion)
* 🎯 Re-ranking for better context selection
* 💬 Conversational memory (persistent + session)
* 📚 Source attribution (trace answers to documents)
* ⚡ Fully offline (no API required)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Query Expansion (LLM)
    │
    ▼
Embedding (SentenceTransformer)
    │
    ▼
FAISS Vector Search
    │
    ▼
Top-K Chunks
    │
    ▼
Re-ranking (LLM)
    │
    ▼
Context Assembly
    │
    ▼
LLM (Ollama)
    │
    ▼
Final Answer + Sources
```

---

## 📂 Project Structure

```
.
├── rag_ollama.py      # Main RAG pipeline
├── ingest_pdf.py      # PDF ingestion + indexing
├── data/              # Input PDFs
├── memory.txt         # Persistent memory
├── index.faiss        # Vector index (ignored in git)
├── chunks.npy         # Text chunks (ignored)
├── metadata.npy       # Source mapping (ignored)
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install sentence-transformers faiss-cpu pypdf pdfminer.six requests
```

---

### 2. Install Ollama

👉 https://ollama.com

Run:

```bash
ollama pull llama3.1
```

---

### 3. Start Ollama server

```bash
ollama serve
```

---

### 4. Add your PDFs

Place files into:

```
data/
```

---

### 5. Build index

```bash
python ingest_pdf.py
```

---

### 6. Run assistant

```bash
python rag_ollama.py
```

---

## 💡 Example

```
Ask: What is transformer?

→ Retrieves relevant chunks
→ Generates grounded answer

Ask: Compare transformers and CNNs

→ Multi-query retrieval
→ Structured comparison answer
```

---

## 🧠 Memory System

You can store facts:

```
remember The sun is a star
```

Saved into:

```
memory.txt
```

Included in retrieval pipeline.

---

## ⚠️ Limitations

* Depends on chunk quality (semantic chunking improves results)
* No fine-tuning (RAG ≠ training)
* Hallucination possible if context is weak
* Retrieval threshold tuning required

---

## 🔬 Future Improvements

* Hybrid search (BM25 + vector)
* Cross-encoder re-ranking
* Streaming responses
* UI (web interface)
* Knowledge graph integration
* Agent-based reasoning

---

## 🧠 Tech Stack

* SentenceTransformers (embeddings)
* FAISS (vector search)
* Ollama (local LLM inference)
* Python

---

## 🎯 Goal

Build a **local, controllable AI system** capable of:

* understanding private data
* reasoning over documents
* operating fully offline

---

## 📌 Author

Yevhen Biedniakov
AI Engineer (in progress)

---

## ⭐ Notes

This project demonstrates:

* real-world RAG pipeline
* LLM + retrieval integration
* system design thinking

Ideal for AI/ML portfolio.

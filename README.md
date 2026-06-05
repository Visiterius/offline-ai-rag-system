# Offline AI RAG System

> **Private document question-answering that runs locally.**  
> A compact Retrieval-Augmented Generation (RAG) project using SentenceTransformers, FAISS, and Ollama.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#tech-stack)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green)](#architecture)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black)](#setup)
[![Privacy](https://img.shields.io/badge/Privacy-Offline%20First-success)](#why-this-project)

## Why this project

Most RAG demos depend on hosted APIs. This project shows the opposite: a local-first assistant that can index private notes or PDFs, retrieve relevant chunks with embeddings, and answer through a local Ollama model.

It is built as an interview-ready AI engineering project because it demonstrates:

- document ingestion and chunking
- embedding generation with `intfloat/e5-large-v2`
- vector search with FAISS
- local LLM inference through Ollama
- source attribution for retrieved PDF pages
- persistent lightweight memory
- clear boundaries between retrieval and generation

## Demo in 60 seconds

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build a small demo index from data.txt
python build_index.py

# 4. Search the local index
python query.py
```

Example query:

```text
Ask: What is a transformer?
```

Expected behavior:

```text
Top results:
[1] A transformer is a neural network architecture that uses attention mechanisms instead of recurrence...
```

For a full LLM answer with Ollama:

```bash
ollama pull llama3.1
ollama serve
python rag_ollama.py
```

Then ask:

```text
Ask: Explain transformers and cite the source.
```

See [`DEMO.md`](DEMO.md) for the interview walkthrough script.

## Architecture

```text
Documents / notes / memory
        |
        v
Text extraction + chunking
        |
        v
SentenceTransformer embeddings
        |
        v
FAISS vector index
        |
        v
Top-k retrieval + source metadata
        |
        v
Ollama prompt assembly
        |
        v
Grounded answer + sources
```

### Retrieval flow

1. Load text or PDF pages.
2. Split content into overlapping chunks.
3. Embed chunks as `passage: ...` using E5 format.
4. Store vectors in FAISS and chunk/source metadata in NumPy files.
5. Embed user query as `query: ...`.
6. Retrieve nearest chunks.
7. Ask Ollama to answer using retrieved context.
8. Print answer plus source pages when available.

## Features

- **Fully local:** no hosted API required for the core flow.
- **PDF ingestion:** extract pages and keep source/page metadata.
- **Semantic search:** FAISS over SentenceTransformer embeddings.
- **Local generation:** Ollama endpoint for `llama3.1` or another local model.
- **Multi-query retrieval:** uses the LLM to expand a question into related search queries.
- **Re-ranking:** lets the LLM select the strongest retrieved chunks.
- **Memory file:** `remember ...` writes persistent notes into `memory.txt`.
- **Source attribution:** PDF answers include source filename and page.

## Project structure

```text
.
├── build_index.py        # Build a FAISS index from data.txt
├── query.py              # Search the built index without an LLM
├── ingest_pdf.py         # Build a FAISS index from PDFs in data/
├── rag_ollama.py         # Full retrieval + Ollama answer loop
├── ollama.py             # Minimal Ollama API experiment
├── data.txt              # Tiny demo corpus for quick local testing
├── requirements.txt      # Python dependencies
├── DEMO.md               # Interview/demo walkthrough
└── README.md
```

Generated local artifacts are intentionally ignored by git:

```text
index.faiss
chunks.npy
metadata.npy
memory.txt
data/
```

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash / Linux / macOS
pip install -r requirements.txt
```

### 2. Optional: Ollama local LLM

Install Ollama from <https://ollama.com>, then:

```bash
ollama pull llama3.1
ollama serve
```

By default the app calls:

```text
http://localhost:11434/api/generate
```

Override if needed:

```bash
export OLLAMA_URL="http://127.0.0.1:11434/api/generate"
export OLLAMA_MODEL="llama3.1"
```

## Usage

### Option A: quick vector-search demo

Use the included tiny sample corpus:

```bash
python build_index.py
python query.py
```

This mode proves the embedding + FAISS retrieval path without requiring Ollama.

### Option B: PDF RAG

```bash
mkdir -p data
# Put PDFs into data/
python ingest_pdf.py
python rag_ollama.py
```

Inside the assistant:

```text
Ask: Summarize the main topic of the document.
Ask: Compare concept A and concept B.
remember The user prefers short answers with bullet points.
Ask: Use my preference and explain the document.
exit
```

## Tech stack

- Python
- SentenceTransformers
- `intfloat/e5-large-v2`
- FAISS CPU
- NumPy
- pdfminer.six
- Ollama


## Author

**Yevhen Biedniakov**  
Dental CAD/CAM engineer transitioning into AI engineering, focused on local/private AI, RAG systems, and domain-specific AI for MedTech and dental CAD/CAM workflows.

# 🧠 RAG AI Assistant (Local LLM + FAISS)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using:

- SentenceTransformers (embeddings)
- FAISS (vector search)
- Ollama (local LLM)
- PDF ingestion pipeline

## Features
- Multi-document PDF ingestion
- Semantic chunking
- Vector search (FAISS)
- LLM reasoning with context
- Source attribution
- Query expansion (agent-style)
- Reranking

## Architecture
PDF → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer

## Stack
- Python
- FAISS
- SentenceTransformers
- Ollama (llama3.1)

## Example
Ask questions about your PDFs and get answers with sources.

## Future Work
- Cross-encoder reranking
- Memory as vector DB
- Structured document parsing

# Demo walkthrough

Use this as a short interview script. It keeps the explanation technical but easy to follow.

## 1. One-line pitch

> This is an offline-first RAG assistant. It embeds private documents locally, retrieves relevant chunks with FAISS, and uses Ollama to generate grounded answers without sending data to a hosted API.

## 2. Quick terminal demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_index.py
python query.py
```

Ask:

```text
What is a transformer?
```

Expected result:

```text
Top results:
[1] A transformer is a neural network architecture that uses attention mechanisms instead of recurrence...
```

What to say:

> First I prove retrieval works without involving an LLM. The index is built from local text, the query is embedded with the same E5 embedding model, and FAISS returns the nearest chunks.

## 3. Full RAG demo with Ollama

Terminal 1:

```bash
ollama serve
```

Terminal 2:

```bash
ollama pull llama3.1
python rag_ollama.py
```

Ask:

```text
Explain what a transformer is. Use only the retrieved context if possible.
```

What to say:

> This second path adds generation. The model gets only the retrieved context plus the question. For PDFs, the assistant also prints the source file and page number.

## 4. PDF demo path

```bash
mkdir -p data
# put one or more PDFs into data/
python ingest_pdf.py
python rag_ollama.py
```

Ask:

```text
Summarize the document in five bullet points.
Which page supports the answer?
```

What to say:

> The PDF ingestion keeps page metadata, so answers can point back to source pages. That is important for trust in medical, dental, or regulatory workflows.

## 5. Best technical explanation

- `ingest_pdf.py` extracts PDF pages and stores source/page metadata.
- Text is split into chunks to keep retrieval precise.
- Chunks are embedded as `passage: ...` because E5 models use query/passages prefixes.
- User questions are embedded as `query: ...`.
- FAISS performs vector similarity search.
- `rag_ollama.py` builds the prompt from retrieved chunks.
- Ollama runs the local LLM.

## 6. Honest limitations

Say this directly if asked:

- The current UI is terminal-based.
- Retrieval quality depends on chunking quality.
- It needs local model RAM/CPU/GPU depending on the Ollama model.
- It is a working prototype, not a full production product yet.

## 7. Strong next steps

If I had another week, I would add:

1. Streamlit/FastAPI web UI.
2. Hybrid BM25 + vector search.
3. A small evaluation set to measure retrieval accuracy.
4. Docker Compose for repeatable setup.
5. Better document metadata filters.

## 8. Why this fits my profile

> My background is dental CAD/CAM and private technical data workflows. This kind of local RAG architecture is directly relevant to labs or clinics that want AI over internal manuals, CAD/CAM workflows, order notes, or SOPs without exposing sensitive data to external APIs.

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re
from pdfminer.high_level import extract_text

model = SentenceTransformer('intfloat/e5-large-v2')


def chunk_text(text, max_sentences=5, overlap=2):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    i = 0

    while i < len(sentences):
        chunk = " ".join(sentences[i:i + max_sentences])

        if len(chunk) > 50:
            chunks.append(chunk)

        i += max_sentences - overlap

    return chunks


def is_bad_chunk(text):
    return (
        len(text) < 50 or
        "references" in text.lower() or
        ("[" in text and "]" in text and len(text) < 300)
    )



def load_pdfs(folder="data"):
    documents = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)

            print("Processing:", file)

            text = extract_text(path)

            if text:
                pages = text.split("\f")

                for i, page in enumerate(pages):
                    page = page.strip()

                    if len(page) > 50:
                        documents.append({
                            "text": page,
                            "source": file,
                            "page": i + 1
                        })

    if os.path.exists("memory.txt"):
        with open("memory.txt", "r") as f:
            memory_text = f.read()

            if memory_text.strip():
                documents.append({
                    "text": memory_text,
                    "source": "memory",
                    "page": 0
                })

    return documents



documents = load_pdfs("data")

chunks = []
metadata = []

for doc in documents:
    pieces = chunk_text(doc["text"])

    for p in pieces:
        if not is_bad_chunk(p):
            chunks.append(p)
            metadata.append({
                "source": doc["source"],
                "page": doc["page"]
            })



np.save("metadata.npy", metadata)

embeddings = model.encode(
    [f"passage: {c}" for c in chunks],
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "index.faiss")
np.save("chunks.npy", chunks)

print("PDF index built")

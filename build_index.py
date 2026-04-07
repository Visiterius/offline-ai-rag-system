from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('intfloat/e5-large-v2')


def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


with open("data.txt", "r") as f:
    text = f.read()


chunks = chunk_text(text)


embeddings = model.encode(
    [f"passage: {c}" for c in chunks],
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "index.faiss")
np.save("chunks.npy", chunks)

print("Index built")

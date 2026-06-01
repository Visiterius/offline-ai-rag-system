import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-large-v2')

# Load index + chunks
index = faiss.read_index("index.faiss")
chunks = np.load("chunks.npy", allow_pickle=True)

def search(query, top_k=5):
    query_embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
    )
    query_embedding = np.array(query_embedding).astype('float32')

    top_k = min(top_k, index.ntotal)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    seen = set()
    for i in indices[0]:
        if i == -1 or i in seen:
            continue
        seen.add(i)
        results.append(chunks[i])

    return results

if __name__ == "__main__":
    while True:
        try:
            q = input("\nAsk: ").strip()
        except EOFError:
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        results = search(q)

        print("\nTop results:\n")
        for i, r in enumerate(results):
            print(f"[{i+1}] {r[:500]}\n")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-large-v2')

# Load index + chunks
index = faiss.read_index("index.faiss")
chunks = np.load("chunks.npy", allow_pickle=True)

def search(query, top_k=5):
    query_embedding = model.encode(["query: " + query])
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append(chunks[i])

    return results

if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        results = search(q)

        print("\nTop results:\n")
        for i, r in enumerate(results):
            print(f"[{i+1}] {r[:500]}\n")

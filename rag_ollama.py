import faiss
import numpy as np
import requests
import re
from sentence_transformers import SentenceTransformer

chat_history = []

model = SentenceTransformer('intfloat/e5-large-v2')

index = faiss.read_index("index.faiss")
chunks = np.load("chunks.npy", allow_pickle=True)
metadata = np.load("metadata.npy", allow_pickle=True)

OLLAMA_URL = "http://172.23.208.1:11434/api/generate"
MODEL_NAME = "llama3.1"

def rerank(query, chunks):
    joined = "\n\n".join([f"[{i}] {c}" for i, c in enumerate(chunks)])

    prompt = f"""
Select most relevant chunks.

Question: {query}

Chunks:
{joined}

Return best indices:
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
    )

    text = response.json()["response"]

    try:
        indices = [int(x) for x in re.findall(r"\d+", text)]
        return [chunks[i] for i in indices if i < len(chunks)]
    except:
        return chunks[:5]

def generate_subqueries(query):
    prompt = f"""
Break the question into 2-3 short search queries.
Return each on a new line.

Question: {query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    text = response.json()["response"]

    lines = text.split("\n")
    return [l.strip("-•123. ").strip() for l in lines if len(l.strip()) > 3]

def split_query(query):
    return query.split(" and ")

def retrieve(query, top_k=20):
    all_queries = [query] + generate_subqueries(query)

    results = []
    sources = []

    for sub in all_queries:
        emb = model.encode([f"query: {sub}"])
        emb = np.array(emb).astype('float32')

        distances, indices = index.search(emb, top_k)

        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and dist < 1.2:
                results.append(chunks[idx])
                sources.append(metadata[idx])

    if not results:
        return "", []

    seen = set()
    unique_results = []
    unique_sources = []

    for r, s in zip(results, sources):
        if r not in seen:
            seen.add(r)
            unique_results.append(r)
            unique_sources.append(s)


    unique_results = rerank(query, unique_results)


    filtered_sources = []
    for r in unique_results:
        for i, orig in enumerate(results):
            if r == orig:
                filtered_sources.append(sources[i])
                break

    unique_results = unique_results[:10]
    filtered_sources = filtered_sources[:10]

    context = "\n\n---\n\n".join(unique_results)

    print("\n--- Retrieved Context ---\n")
    print(context[:1000])

    return context, filtered_sources

def ask_llm(query, context, sources):
    history_text = "\n".join(chat_history[-5:])

    prompt = f"""
You are an AI assistant.

Use only provided context as primary source.
If needed, supplement with general knowledge.
Clearly distinguish both.

If multiple topics are present:
- explain each topic
- compare them
- find relationships if possible

If unrelated:
- explain both separately

If the answer is not in the context, say:
"I don't know based on provided data."

If unsure → explicitly say uncertainty clearly
Do NOT infer missing facts

If any part is uncertain:
explicitly state uncertainty

TASK:
1. Identify main concepts in the question
2. For each concept:
   - find relevant parts of context
   - explain clearly
3. If multiple concepts:
   - compare them
   - explain relationships if any
4. If unrelated:
   - clearly separate answers

OUTPUT FORMAT:
- Structured answer
- Clear sections
- Concise explanations

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )

    try:
        answer = response.json()["response"]
    except Exception:
        return "Error: Failed to get response from LLM."

    if sources:
        seen = set()
        source_text = "\n\nSources:\n"

        for s in sources:
            key = (s['source'], s['page'])
            if key not in seen:
                seen.add(key)
                source_text += f"- {s['source']} (page {s['page']})\n"
    else:
        source_text = ""
    final_answer = answer + source_text

    chat_history.append(f"User: {query}")
    chat_history.append(f"Assistant: {answer}")

    return final_answer

if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")

        if q.lower() in ["exit", "quit"]:
            break

        if q.lower().startswith("remember"):
            cleaned = q.replace("remember", "").strip()

            with open("memory.txt", "a") as f:
                f.write(cleaned + "\n")

            print("Saved to memory.")
            continue

        context,sources = retrieve(q)
        answer = ask_llm(q, context, sources)

        print("\n=== AI Answer ===\n")
        print(answer)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# --- Load FAISS index ---
index = faiss.read_index("dnd_index.faiss")

# --- Load documents (to map back from vectors to text) ---
with open("dnd_data.json") as f:
    raw_data = json.load(f)

# Collect all chunks in the same way we did before
from langchain.text_splitter import RecursiveCharacterTextSplitter
def clean_text(text):
    import re
    return re.sub(r"\s+", " ", text).strip()

documents = []
for d in raw_data:
    title = d.get("title", "untitled")
    text = clean_text(d.get("text", ""))
    url = d.get("url", "")
    if text:
        documents.append({"title": title, "text": text, "url": url})

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = []
for d in documents:
    for chunk in splitter.split_text(d["text"]):
        docs.append({"title": d["title"], "text": chunk, "url": d["url"]})

# --- Load the same embedding model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, k=3):
    """Search the FAISS index with a query string"""
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)  # distances & indices
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({"text": docs[idx]["text"], "title": docs[idx]["title"], "url": docs[idx]["url"], "score": float(score)})
    return results

# --- Example interactive query ---
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        hits = search(q, k=3)
        for h in hits:
            print(f"\n[Title] {h['title']} (Score: {h['score']:.2f})\n{h['text']}\nURL: {h['url']}")
        print("-" * 80)
# query_index.py
# This script allows querying the FAISS index created from D&D data
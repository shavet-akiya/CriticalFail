import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Load the scraped data ---
with open("dnd_data.json") as f:
    data = json.load(f)

# --- Clean function ---
def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()

# --- Normalize into documents ---
documents = []
for d in data:
    # Defensive checks in case some keys are missing
    title = d.get("title", "untitled")
    text = clean_text(d.get("text", ""))
    url = d.get("url", "")

    if text:  # only keep non-empty docs
        documents.append({"title": title, "text": text, "url": url})

# --- Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = []
for d in documents:
    for chunk in splitter.split_text(d["text"]):
        docs.append({"title": d["title"], "text": chunk, "url": d["url"]})

print(f"Prepared {len(docs)} chunks")

# --- Embedding ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode([d["text"] for d in docs], convert_to_numpy=True)

# --- Store in FAISS ---
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print(f"Stored {len(embeddings)} vectors in FAISS index")

# --- (Optional) Save the index ---
faiss.write_index(index, "dnd_index.faiss")

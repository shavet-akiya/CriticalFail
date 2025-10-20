import os
import chromadb
from chromadb.config import Settings

# Resolve absolute path to CriticalFail/chroma/chroma_data
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "CriticalFail", "chroma")
)
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_data")

# Ensure the directory exists
os.makedirs(PERSIST_DIR, exist_ok=True)

# Initialize Chroma client with persistent directory
chroma_client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR)
)

# Create or get the main collection
session_collection = chroma_client.get_or_create_collection(name="dnd_sessions")

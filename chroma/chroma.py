import os
import chromadb
from chromadb.config import Settings

# Resolve path to persist inside the server folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_data")

# Initialize Chroma client
chroma_client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR)
)

# Create or get the main collection
session_collection = chroma_client.get_or_create_collection(name="dnd_sessions")
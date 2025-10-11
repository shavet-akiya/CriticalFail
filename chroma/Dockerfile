# Use the official ChromaDB image
FROM chromadb/chroma:latest

# Set working directory inside the container
WORKDIR /app

# Optional: create a volume mount point for persistence
VOLUME ["/app/chroma_data"]

# Expose ChromaDB port
EXPOSE 8000

# Start ChromaDB server
CMD ["chroma", "run", "--host", "0.0.0.0", "--port", "8000", "--path", "/app/chroma_data"]

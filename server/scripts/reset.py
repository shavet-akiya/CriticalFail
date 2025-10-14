

from ./main import chroma_client

collection = chroma_client.get_or_create_collection("dnd_sessions")
collection.delete()  # wipes everything

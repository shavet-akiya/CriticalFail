import os
import uuid
import datetime
from chromadb import HttpClient
from fastapi import APIRouter

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
session_collection = chroma_client.get_or_create_collection(name="dnd_sessions")

router = APIRouter()


def save_campaign_to_chroma(campaign_name: str, session_ids: list[str]) -> str:
    """
    Save a campaign as a separate type in ChromaDB.
    """
    campaign_id = str(uuid.uuid4())

    session_collection.add(
        documents=[campaign_name],
        ids=[campaign_id],
        metadatas=[
            {
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "session_ids": session_ids,
                "created_at": str(datetime.datetime.utcnow()),
                "type": "campaign",
            }
        ],
    )

    return campaign_id


def save_session_to_chroma(session_data: dict) -> str:
    chroma_id = str(uuid.uuid4())
    summary = session_data.get("summary", {})
    summary_text = summary.get("session_summary", "No summary")

    # Save main session
    session_collection.add(
        documents=[summary_text],
        ids=[chroma_id],
        metadatas=[
            {
                "session_id": session_data.get("session_id", str(uuid.uuid4())),
                "campaign_id": session_data.get("campaign_id", "Unassigned"),
                "processed_at": session_data.get(
                    "processed_at", str(datetime.datetime.utcnow())
                ),
                "type": "session",
            }
        ],
    )

    # Save characters, locations, events...
    from ._characters import save_characters
    from ._locations import save_locations
    from ._events import save_events

    save_characters(session_collection, summary, session_data)
    save_locations(session_collection, summary, session_data)
    save_events(session_collection, summary, session_data)

    return chroma_id




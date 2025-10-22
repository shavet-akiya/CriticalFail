import os
import uuid
import datetime
from chromadb import HttpClient
from chromadb.config import Settings
from fastapi import APIRouter
import json

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
session_collection = chroma_client.get_or_create_collection(name="dnd_sessions")

router = APIRouter()


def save_campaign_to_chroma(campaign_name: str, session_ids: list[str]) -> str:
    """
    Save a campaign as a separate type in ChromaDB.
    """
    campaign_id = str(uuid.uuid4())[:6]

    session_collection.add(
        documents=[campaign_name],
        ids=[campaign_id],
        metadatas=[
            {
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "session_ids": session_ids,
                "characters": [],
                "locations": [],
                "created_at": str(datetime.datetime.utcnow()),
                "type": "campaign",
            }
        ],
    )

    return campaign_id


def save_session_to_chroma(session_data: dict) -> str:
    chroma_id = str(uuid.uuid4())[:6]
    summary = session_data.get("summary", {})
    summary_text = summary.get("session_summary", "No summary")
    session_id = session_data.get("session_id", str(uuid.uuid4())[:6])
    campaign_id = session_data.get("campaign_id", "Unassigned")

    # --- Prepare metadata ---
    metadata = {
        "session_id": session_id,
        "campaign_id": campaign_id,
        "processed_at": session_data.get(
            "processed_at", str(datetime.datetime.utcnow())
        ),
        "type": "session",
        "characters": json.dumps(session_data.get("characters", [])),
        "locations": json.dumps(session_data.get("locations", [])),
        "events": json.dumps(session_data.get("events", [])),
    }

    # Save main session
    session_collection.add(
        documents=[summary_text],
        ids=[chroma_id],
        metadatas=[metadata],
    )

    # --- Save characters, ensuring consistent character_id ---
    from ._characters import save_characters
    from ._locations import save_locations
    from ._events import save_events

    save_characters(session_collection, summary, session_data, campaign_id)
    save_locations(session_collection, summary, session_data, campaign_id)
    save_events(session_collection, summary, session_data)

    return chroma_id


# --- When fetching sessions, deserialize lists ---
def deserialize_session_metadata(meta: dict) -> dict:
    """
    Convert JSON strings back to Python objects for API responses
    """
    for key in ["characters", "locations", "events"]:
        if key in meta and isinstance(meta[key], str):
            try:
                meta[key] = json.loads(meta[key])
            except Exception:
                meta[key] = []
    return meta

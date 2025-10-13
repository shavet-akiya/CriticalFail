import sys
import os
import json
import uuid
import traceback
import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Path, Body
from pydantic import BaseModel
from typing import Optional, List

from chromadb import HttpClient
from llm import dnd_ai

# --- ChromaDB Setup (via HTTP client) ---
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")  # container name in docker-compose
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
session_collection = chroma_client.get_or_create_collection(name="dnd_sessions")


def save_session_to_chroma(session_data: dict) -> str:
    chroma_id = str(uuid.uuid4())
    summary_text = session_data["summary"].get("session_summary", "")

    # Save the session itself
    session_collection.add(
        documents=[summary_text],
        ids=[chroma_id],
        metadatas=[
            {
                "session_id": session_data["session_id"],
                "campaign_id": session_data.get("campaign_id", 0),
                "processed_at": session_data["processed_at"],
                "type": "session",
            }
        ],
    )

    # Save each character
    for character in session_data["summary"].get("characters", []):
        character_id = character["character_id"]
        session_collection.add(
            documents=[character.get("name", "")],
            ids=[character_id],
            metadatas=[
                {
                    "character_id": character_id,
                    "session_id": session_data["session_id"],
                    "type": "character",
                    **character,
                }
            ],
        )

    # Save each location
    for loc in session_data["summary"].get("locations", []):
        loc_id = loc.get("location_id", str(uuid.uuid4()))
        session_collection.add(
            documents=[loc.get("location_name", loc.get("name", ""))],
            ids=[loc_id],
            metadatas=[
                {
                    "location_id": loc_id,
                    "session_id": session_data["session_id"],
                    "type": "location",
                    **loc,
                }
            ],
        )

    # Save each event
    for ev in session_data["summary"].get("events", []):
        ev_id = ev.get("event_id", str(uuid.uuid4()))
        session_collection.add(
            documents=[ev.get("event", "")],
            ids=[ev_id],
            metadatas=[
                {
                    "event_id": ev_id,
                    "session_id": session_data["session_id"],
                    "type": "event",
                    **ev,
                }
            ],
        )

    return chroma_id


# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SESSIONS = 10
recent_sessions: List[dict] = []


class TranscriptInput(BaseModel):
    transcript: str


class DeleteRequest(BaseModel):
    session_id: str


# --- Routes ---
@app.post("/sessions")
async def process_session(input_data: TranscriptInput):
    try:
        structured_json = dnd_ai.extract_session_data(input_data.transcript)

        recent_sessions.insert(0, structured_json)
        if len(recent_sessions) > MAX_SESSIONS:
            recent_sessions.pop()

        chroma_id = save_session_to_chroma(structured_json)

        return {
            "status": "success",
            "session_data": structured_json,
            "chroma_id": chroma_id,
        }

    except Exception as e:
        traceback.print_exc()
        if hasattr(e, "request") or hasattr(e, "response"):
            return JSONResponse(
                status_code=502,
                content={"error": "Ollama API error", "details": str(e)},
            )
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process session", "details": str(e)},
        )


@app.get("/sessions/recent")
async def list_recent_sessions():
    hydrated = []
    for s in recent_sessions:
        session_id = s["session_id"]

        # Fetch characters
        chars = session_collection.get(
            where={"$and": [{"type": "character"}, {"session_id": session_id}]}
        )

        # Fetch locations
        locs = session_collection.get(
            where={"$and": [{"type": "location"}, {"session_id": session_id}]}
        )

        # Fetch events
        evs = session_collection.get(
            where={"$and": [{"type": "event"}, {"session_id": session_id}]}
        )

        s_copy = s.copy()
        s_copy["summary"]["characters"] = chars["metadatas"]
        s_copy["summary"]["locations"] = locs["metadatas"]
        s_copy["summary"]["events"] = evs["metadatas"]

        hydrated.append(s_copy)

    return hydrated


@app.get("/sessions")
async def list_chroma_sessions():
    try:
        sessions = session_collection.get(where={"type": "session"})
        decoded = {
            "ids": sessions["ids"],
            "documents": sessions["documents"],
            "metadatas": [],
        }

        for md in sessions["metadatas"]:
            session_id = md["session_id"]

            # Fetch characters
            chars = session_collection.get(
                where={"$and": [{"type": "character"}, {"session_id": session_id}]}
            )

            # Fetch locations
            locs = session_collection.get(
                where={"$and": [{"type": "location"}, {"session_id": session_id}]}
            )

            # Fetch events
            evs = session_collection.get(
                where={"$and": [{"type": "event"}, {"session_id": session_id}]}
            )

            md_with_data = md.copy()
            md_with_data["characters"] = chars["metadatas"]
            md_with_data["locations"] = locs["metadatas"]
            md_with_data["events"] = evs["metadatas"]

            decoded["metadatas"].append(md_with_data)

        return decoded

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch from Chroma", "details": str(e)},
        )


@app.delete("/sessions")
async def delete_session(req: DeleteRequest):
    try:
        session_collection.delete(where={"session_id": req.session_id})
        return {"status": "deleted", "session_id": req.session_id}
    except Exception as e:
        return {"status": "error", "details": str(e)}


@app.get("/")
async def root():
    return {"message": "FastAPI D&D server is running!"}


class UpdateCampaignRequest(BaseModel):
    session_id: str
    campaign_id: Optional[str]


@app.put("/sessions/campaign")
async def update_campaign_id(req: UpdateCampaignRequest):
    try:
        # Fetch existing session metadata
        results = session_collection.get(where={"session_id": req.session_id})
        if not results["ids"]:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Session not found",
                    "session_id": req.session_id,
                },
            )

        # Extract existing metadata and document
        old_metadata = results["metadatas"][0]
        old_document = results["documents"][0]
        old_id = results["ids"][0]

        # Update campaign_id
        old_metadata["campaign_id"] = req.campaign_id

        # Delete old record and reinsert with updated metadata
        session_collection.delete(ids=[old_id])
        session_collection.add(
            documents=[old_document], ids=[old_id], metadatas=[old_metadata]
        )

        return {
            "status": "updated",
            "session_id": req.session_id,
            "campaign_id": req.campaign_id,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to update campaign ID", "details": str(e)},
        )


class CharacterUpdate(BaseModel):
    id: str
    name: str
    race: Optional[str]
    class_: Optional[str]
    npc: Optional[bool] = False

    AC: Optional[int]
    HP: Optional[int]
    STR: Optional[int]
    DEX: Optional[int]
    CON: Optional[int]
    INT: Optional[int]
    WIS: Optional[int]
    CHA: Optional[int]


@app.get("/characters")
async def list_characters():
    results = session_collection.get(where={"type": "character"})
    # results["metadatas"] is a list of dicts, each containing character_id, name, stats, etc.
    return {"characters": results["metadatas"]}


@app.get("/characters/{character_id}")
async def get_character(character_id: str):
    # Direct lookup by Chroma ID (since we stored each character with character_id as its id)
    results = session_collection.get(ids=[character_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Character not found"})
    return {"character": results["metadatas"][0]}


@app.patch("/characters/{character_id}")
async def patch_character(character_id: str, update: dict = Body(...)):
    results = session_collection.get(ids=[character_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Character not found"})

    old_metadata = results["metadatas"][0]
    old_document = results["documents"][0]

    # Only merge keys that are explicitly provided
    merged = old_metadata.copy()
    for key, value in update.items():
        if value is not None:
            merged[key] = value

    # Use updated name for the document if provided
    new_document = merged.get("name", old_document)

    # Delete old record and re-add updated one
    session_collection.delete(ids=[character_id])
    session_collection.add(
        documents=[new_document],
        ids=[character_id],
        metadatas=[merged],
    )

    return {"status": "updated", "character": merged}


@app.delete("/reset")
async def reset_database():
    try:
        all_ids = session_collection.get()["ids"]
        if all_ids:
            session_collection.delete(ids=all_ids)
        return {"status": "all data deleted"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


@app.get("/locations")
async def list_locations():
    try:
        results = session_collection.get(where={"type": "location"})
        # results["metadatas"] contains all the location data
        return {"locations": results["metadatas"]}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch locations", "details": str(e)},
        )


@app.get("/locations/{location_id}")
async def get_location(location_id: str):
    results = session_collection.get(ids=[location_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Location not found"})
    # return the first metadata, which is the location
    return {"location": results["metadatas"][0]}


# --- Event Endpoints ---


class EventUpdate(BaseModel):
    event: Optional[str]
    event_summary: Optional[str]
    participants: Optional[List[str]]
    location: Optional[str]
    timeline_order: Optional[int]
    event_tags: Optional[List[str]]


@app.get("/events")
async def list_events():
    try:
        results = session_collection.get(where={"type": "event"})
        return {"events": results["metadatas"]}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch events", "details": str(e)},
        )


@app.get("/events/{event_id}")
async def get_event(event_id: str):
    results = session_collection.get(ids=[event_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Event not found"})
    return {"event": results["metadatas"][0]}


@app.put("/events/{event_id}")
async def update_event(event_id: str, update: EventUpdate):
    results = session_collection.get(ids=[event_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Event not found"})

    old_metadata = results["metadatas"][0]
    old_document = results["documents"][0]

    new_data = update.dict(exclude_unset=True)
    merged = {**old_metadata, **new_data}

    # Replace old record
    session_collection.delete(ids=[event_id])
    session_collection.add(
        documents=[merged.get("event", old_document)],
        ids=[event_id],
        metadatas=[merged],
    )

    return {"status": "updated", "event": merged}


@app.delete("/events/{event_id}")
async def delete_event(event_id: str):
    try:
        session_collection.delete(ids=[event_id])
        return {"status": "deleted", "event_id": event_id}
    except Exception as e:
        return {"status": "error", "details": str(e)}

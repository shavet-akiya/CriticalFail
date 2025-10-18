import os
import uuid
import traceback
import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Path, Body, UploadFile, Form, File
from pydantic import BaseModel
from typing import Optional, List
import asyncio
from fastapi import BackgroundTasks

from chromadb import HttpClient
from llm import dnd_ai

llm_jobs: dict[str, dict] = (
    {}
)  # job_id -> {"status": "processing"|"completed"|"error", "result": ..., "error": ...}


try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - dev env without httpx
    httpx = None  # type: ignore

# Speech service configuration
SPEECH_SERVICE_URL = os.getenv("SPEECH_SERVICE_URL", "http://speech:8001")


# --- ChromaDB Setup (via HTTP client) ---
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")  # container name in docker-compose
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
session_collection = chroma_client.get_or_create_collection(name="dnd_sessions")


def save_session_to_chroma(session_data: dict) -> str:
    chroma_id = str(uuid.uuid4())
    summary = session_data.get("summary", {})
    summary_text = summary.get("session_summary", "")

    # --- Save main session summary ---
    session_collection.add(
        documents=[summary_text or "No summary"],
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

    # --- Save characters safely ---
    for character in summary.get("characters", []):
        if not isinstance(character, dict):
            print(f"⚠️ Skipping invalid character entry: {character}")
            continue

        character_id = character.get("character_id") or str(uuid.uuid4())
        name = character.get("name", "Unknown Character")

        try:
            session_collection.add(
                documents=[name],
                ids=[character_id],
                metadatas=[
                    {
                        "character_id": character_id,
                        "session_id": session_data.get("session_id"),
                        "type": "character",
                        **character,
                    }
                ],
            )
        except Exception as e:
            print(f"❌ Failed to save character {name}: {e}")

    # --- Save locations safely ---
    for loc in summary.get("locations", []):
        if not isinstance(loc, dict):
            print(f"⚠️ Skipping invalid location entry: {loc}")
            continue

        loc_id = loc.get("location_id") or str(uuid.uuid4())
        loc_name = loc.get("location_name") or loc.get("name", "Unknown Location")

        try:
            session_collection.add(
                documents=[loc_name],
                ids=[loc_id],
                metadatas=[
                    {
                        "location_id": loc_id,
                        "session_id": session_data.get("session_id"),
                        "type": "location",
                        **loc,
                    }
                ],
            )
        except Exception as e:
            print(f"❌ Failed to save location {loc_name}: {e}")

    # --- Save events safely ---
    for ev in summary.get("events", []):
        if not isinstance(ev, dict):
            print(f"⚠️ Skipping invalid event entry: {ev}")
            continue

        ev_id = ev.get("event_id") or str(uuid.uuid4())
        event_text = ev.get("event", "Unnamed Event")

        # Make a shallow copy so we can safely modify it
        ev_metadata = ev.copy()

        # Flatten lists that Chroma won't accept
        for key in ["participants", "event_tags"]:
            if isinstance(ev_metadata.get(key), list):
                ev_metadata[key] = ", ".join(map(str, ev_metadata[key]))

        # Add Chroma-required fields
        ev_metadata.update(
            {
                "event_id": ev_id,
                "session_id": session_data.get("session_id"),
                "type": "event",
            }
        )

        try:
            session_collection.add(
                documents=[event_text],
                ids=[ev_id],
                metadatas=[ev_metadata],
            )
        except Exception as e:
            print(f"❌ Failed to save event {event_text}: {e}")

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
    session_code: str


# --- Speech Proxy Routes ---
@app.post("/speech/upload")
async def proxy_speech_upload(
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(2),
    max_speakers: Optional[int] = Form(8),
):
    """Proxy audio uploads to the speech processing service"""
    print("\n" + "=" * 80)
    print("[SERVER PROXY] RECEIVED UPLOAD REQUEST")
    print("=" * 80)
    print(f"[SERVER PROXY] File: {file.filename}")
    print(
        f"[SERVER PROXY] Size: {file.size / 1024 / 1024:.2f} MB"
        if file.size
        else "[SERVER PROXY] Size: unknown"
    )

    if not httpx:
        return JSONResponse(
            status_code=503,
            content={"error": "httpx not available"},
        )

    try:
        print("[SERVER PROXY] Forwarding to speech service...")

        # Forward the file to the speech service (returns job_id immediately)
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            data = {
                "min_speakers": str(min_speakers),
                "max_speakers": str(max_speakers),
            }

            print(
                f"[SERVER PROXY] Calling speech service at {SPEECH_SERVICE_URL}/process"
            )

            response = await client.post(
                f"{SPEECH_SERVICE_URL}/process",
                files=files,
                data=data,
            )

            print(f"[SERVER PROXY] ✓ Got response: {response.status_code}")

            return JSONResponse(
                status_code=response.status_code,
                content=response.json(),
            )

    except Exception as e:
        print(f"[SERVER PROXY] ❌ ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process audio: {str(e)}"},
        )


@app.get("/speech/status/{job_id}")
async def proxy_speech_job_status(job_id: str):
    """Proxy job status check to speech service and auto-process completed jobs"""
    if not httpx:
        return JSONResponse(status_code=503, content={"error": "httpx not available"})

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{SPEECH_SERVICE_URL}/status/{job_id}")
            data = response.json()

            # If the speech job is finished, auto-process the transcript
            if data.get("status") == "completed" and "transcript" in data:
                transcript_text = data["transcript"]

                # Run through LLM (existing logic)
                structured_json = await dnd_ai.extract_session_data(transcript_text)

                # Save to Chroma
                chroma_id = save_session_to_chroma(structured_json)

                # Store in recent memory
                recent_sessions.insert(0, structured_json)
                if len(recent_sessions) > MAX_SESSIONS:
                    recent_sessions.pop()

                return {
                    "status": "completed",
                    "job_id": job_id,
                    "transcript": transcript_text,
                    "session_data": structured_json,
                    "chroma_id": chroma_id,
                }

            return data  # not yet complete or error

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get speech job status: {str(e)}"},
        )


@app.get("/speech/jobs")
async def proxy_speech_jobs():
    """Proxy jobs list to speech service"""
    if not httpx:
        return JSONResponse(
            status_code=503,
            content={"error": "httpx not available"},
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{SPEECH_SERVICE_URL}/jobs")
            return response.json()
    except Exception as e:
        return {"error": str(e), "jobs": []}


@app.get("/speech/status")
async def proxy_speech_status():
    """Check if speech service is ready"""
    if not httpx:
        return {"initialized": False, "error": "httpx not available"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SPEECH_SERVICE_URL}/")
            return response.json()
    except Exception as e:
        return {"initialized": False, "error": str(e)}


# # --- Routes ---
# @app.post("/sessions")
# async def process_session(input_data: TranscriptInput):
#     try:
#         structured_json = await dnd_ai.extract_session_data(input_data.transcript)

#         recent_sessions.insert(0, structured_json)
#         if len(recent_sessions) > MAX_SESSIONS:
#             recent_sessions.pop()

#         chroma_id = save_session_to_chroma(structured_json)

#         return {
#             "status": "success",
#             "session_data": structured_json,
#             "chroma_id": chroma_id,
#         }

#     except Exception as e:
#         traceback.print_exc()
#         if hasattr(e, "request") or hasattr(e, "response"):
#             return JSONResponse(
#                 status_code=502,
#                 content={"error": "Ollama API error", "details": str(e)},
#             )
#         return JSONResponse(
#             status_code=500,
#             content={"error": "Failed to process session", "details": str(e)},
#         )


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


# --- Event Endpoints ---
@app.get("/events")
async def list_events():
    try:
        results = session_collection.get(where={"type": "event"})
        # results["metadatas"] contains all the event data
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
    # return the first metadata, which is the event
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


# Background task
async def process_and_save_session(job_id: str, transcript: str):
    try:
        llm_jobs[job_id] = {"status": "processing"}
        # Run LLM
        structured_json = await dnd_ai.extract_session_data(transcript)
        # Save to Chroma in a thread (blocking code)
        chroma_id = await asyncio.to_thread(save_session_to_chroma, structured_json)
        structured_json["chroma_id"] = chroma_id

        # Store result
        llm_jobs[job_id] = {"status": "completed", "result": structured_json}
        # Also update recent_sessions
        recent_sessions.insert(0, structured_json)
        if len(recent_sessions) > MAX_SESSIONS:
            recent_sessions.pop()
    except Exception as e:
        llm_jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/sessions")
async def create_session(
    input_data: TranscriptInput, background_tasks: BackgroundTasks
):
    job_id = str(uuid.uuid4())
    # Start background processing
    background_tasks.add_task(process_and_save_session, job_id, input_data.transcript)
    # Return immediately
    return {"status": "processing", "job_id": job_id}


@app.get("/sessions/status/{job_id}")
async def get_session_status(job_id: str):
    job = llm_jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job

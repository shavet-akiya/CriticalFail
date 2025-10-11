import sys
import os
import json
import uuid
import traceback
import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

    session_collection.add(
        documents=[summary_text],
        ids=[chroma_id],
        metadatas=[
            {
                "session_code": session_data["session_code"],
                "campaign_id": session_data.get("campaign_id", 0),
                # JSON-encode lists/dicts so Chroma accepts them
                "characters": json.dumps(session_data["summary"].get("characters", [])),
                "locations": json.dumps(session_data["summary"].get("locations", [])),
                "events": json.dumps(session_data["summary"].get("events", [])),
                "tags": json.dumps(session_data["summary"].get("tags", [])),
                "processed_at": session_data["processed_at"],
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
    session_code: str


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
    return recent_sessions


@app.get("/sessions")
async def list_chroma_sessions():
    try:
        results = session_collection.get()
        decoded = {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": [],
        }
        for md in results["metadatas"]:
            decoded_md = md.copy()
            # Safely decode JSON fields
            for field in ["characters", "locations", "events"]:
                try:
                    decoded_md[field] = json.loads(md.get(field, "[]"))
                except Exception:
                    decoded_md[field] = []
            decoded["metadatas"].append(decoded_md)
        return decoded
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch from Chroma", "details": str(e)},
        )


@app.delete("/sessions")
async def delete_session(req: DeleteRequest):
    try:
        session_collection.delete(where={"session_code": req.session_code})
        return {"status": "deleted", "session_code": req.session_code}
    except Exception as e:
        return {"status": "error", "details": str(e)}


@app.get("/")
async def root():
    return {"message": "FastAPI D&D server is running!"}


class UpdateCampaignRequest(BaseModel):
    session_code: str
    campaign_id: Optional[str]


@app.put("/sessions/campaign")
async def update_campaign_id(req: UpdateCampaignRequest):
    try:
        # Fetch existing session metadata
        results = session_collection.get(where={"session_code": req.session_code})
        if not results["ids"]:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Session not found",
                    "session_code": req.session_code,
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
            "session_code": req.session_code,
            "campaign_id": req.campaign_id,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to update campaign ID", "details": str(e)},
        )


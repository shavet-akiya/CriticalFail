import sys
import os
import json
import uuid
import traceback
import datetime

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

from chromadb import HttpClient
from llm import dnd_ai

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
    print(f"[SERVER PROXY] Size: {file.size / 1024 / 1024:.2f} MB" if file.size else "[SERVER PROXY] Size: unknown")
    
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
            
            print(f"[SERVER PROXY] Calling speech service at {SPEECH_SERVICE_URL}/process")
            
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
    """Proxy job status check to speech service"""
    if not httpx:
        return JSONResponse(
            status_code=503,
            content={"error": "httpx not available"},
        )
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{SPEECH_SERVICE_URL}/status/{job_id}")
            return JSONResponse(
                status_code=response.status_code,
                content=response.json(),
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
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


# --- Main Routes ---
@app.get("/")
async def root():
    return {"message": "FastAPI D&D server is running!"}


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
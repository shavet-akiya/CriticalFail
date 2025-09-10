import sys
import os
import json
from fastapi import FastAPI  # may be cooked
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import traceback


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from llm import dnd_ai


try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - dev env without httpx
    httpx = None  # type: ignore

app = FastAPI()

# Allow CORS for local UI/dev servers
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

# Keep last N sessions in memory
MAX_SESSIONS = 10
recent_sessions: List[dict] = []


class TranscriptInput(BaseModel):
    transcript: str


@app.post("/sessions")
async def process_session(input_data: TranscriptInput):
    try:
        structured_json = dnd_ai.extract_session_data(input_data.transcript)
        # Save in memory (prepend so newest is first)
        recent_sessions.insert(0, structured_json)
        if len(recent_sessions) > MAX_SESSIONS:
            recent_sessions.pop()
        return structured_json
    except Exception as e:
        # Print full traceback for debugging inside Docker
        traceback.print_exc()

        # If the error is from Ollama connection or response
        if hasattr(e, "request") or hasattr(e, "response"):
            return JSONResponse(
                status_code=502,
                content={
                    "error": "Ollama API error",
                    "details": str(e),
                },
            )

        # Otherwise return a generic error
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to process session",
                "details": str(e),
            },
        )


@app.get("/sessions")
async def list_sessions():
    print("Current sessions:", recent_sessions)
    return recent_sessions


@app.get("/")
async def root():
    return {"message": "FastAPI D&D server is running!"}

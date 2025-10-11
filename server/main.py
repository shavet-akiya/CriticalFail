import sys
import os
import json
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import traceback
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from llm import dnd_ai
from .src.speech_client import speech_service

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - dev env without httpx
    httpx = None  # type: ignore

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


@app.on_event("startup")
async def startup_event():
    await speech_service.initialize()


@app.get("/api/speech/status")
async def get_speech_status():
    return {
        "initialized": speech_service.initialization_complete,
        "error": speech_service.initialization_error,
        "is_recording": speech_service.is_recording,
    }


@app.post("/api/speech/start")
async def start_recording():
    return speech_service.start_recording()


@app.post("/api/speech/pause")
async def pause_recording():
    return speech_service.pause_recording()


@app.post("/api/speech/resume")
async def resume_recording():
    return speech_service.resume_recording()


@app.post("/api/speech/stop")
async def stop_recording():
    result = speech_service.stop_recording()

    if result.get("success") and result.get("transcript"):
        try:
            structured_json = dnd_ai.extract_session_data(result["transcript"])
            result["structured_data"] = structured_json

            recent_sessions.insert(0, structured_json)
            if len(recent_sessions) > MAX_SESSIONS:
                recent_sessions.pop()
        except Exception as e:
            print(f"Error processing transcript: {e}")
            result["ai_error"] = str(e)

    return result


@app.post("/api/speech/upload")
async def upload_audio(
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(2),
    max_speakers: Optional[int] = Form(8),
):
    try:
        recordings_dir = Path("src/recordings")
        recordings_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix.lower() or ".webm"
        saved_filename = f"recording_{timestamp}{file_extension}"
        saved_path = recordings_dir / saved_filename

        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = await speech_service.process_audio_file(
            str(saved_path), min_speakers=min_speakers, max_speakers=max_speakers
        )

        result["file_info"] = {
            "original_name": file.filename,
            "saved_name": saved_filename,
            "size_mb": saved_path.stat().st_size / (1024 * 1024),
        }

        if result.get("success") and result.get("transcript"):
            try:
                structured_json = dnd_ai.extract_session_data(result["transcript"])
                result["structured_data"] = structured_json

                recent_sessions.insert(0, structured_json)
                if len(recent_sessions) > MAX_SESSIONS:
                    recent_sessions.pop()
            except Exception as e:
                print(f"Error processing transcript with AI: {e}")
                result["ai_error"] = str(e)

        return result
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to process audio: {str(e)}"}
        )


@app.post("/sessions")
async def process_session(input_data: TranscriptInput):
    try:
        structured_json = dnd_ai.extract_session_data(input_data.transcript)
        recent_sessions.insert(0, structured_json)
        if len(recent_sessions) > MAX_SESSIONS:
            recent_sessions.pop()
        return structured_json
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


@app.get("/sessions")
async def list_sessions():
    print("Current sessions:", recent_sessions)
    return recent_sessions


@app.get("/")
async def root():
    return {"message": "FastAPI D&D server with Speech-to-Text is running!"}

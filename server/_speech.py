from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import httpx
import os
import traceback
from ._database import save_session_to_chroma
from llm import dnd_ai

router = APIRouter()
SPEECH_SERVICE_URL = os.getenv("SPEECH_SERVICE_URL", "http://speech:8001")
llm_jobs: dict[str, dict] = {}


@router.post("/upload")
async def upload_audio(
    file: UploadFile, min_speakers: int = Form(2), max_speakers: int = Form(8)
):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            data = {
                "min_speakers": str(min_speakers),
                "max_speakers": str(max_speakers),
            }
            response = await client.post(
                f"{SPEECH_SERVICE_URL}/process", files=files, data=data
            )
            return JSONResponse(
                status_code=response.status_code, content=response.json()
            )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/status/{job_id}")
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


@router.get("/jobs")
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


@router.get("/status")
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

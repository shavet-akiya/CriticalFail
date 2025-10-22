# server/transcripts_api.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os

router = APIRouter()
TRANSCRIPTS_DIR = "./server/src/transcripts"  # adjust path


# List all transcripts
@router.get("/transcripts")
async def list_transcripts():
    try:
        files = os.listdir(TRANSCRIPTS_DIR)
        return {"transcripts": files}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Get single transcript
@router.get("/transcripts/{filename}")
async def get_transcript(filename: str):
    file_path = os.path.join(TRANSCRIPTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Transcript not found")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {"filename": filename, "content": content}

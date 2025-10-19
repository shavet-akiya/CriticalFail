from fastapi import APIRouter, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid, asyncio
from ._database import save_session_to_chroma
from ._speech import llm_jobs
from llm import dnd_ai
from ._database import session_collection

router = APIRouter()
MAX_SESSIONS = 1000
recent_sessions = []


class TranscriptInput(BaseModel):
    transcript: str
    campaign_id: str


async def process_and_save_session(job_id: str, transcript: str, campaign_id: str):
    try:
        llm_jobs[job_id] = {"status": "processing"}

        structured_json = await dnd_ai.extract_session_data(transcript)

        # Assign campaign ID to the session
        structured_json["campaign_id"] = campaign_id
        # Save to Chroma
        chroma_id = await asyncio.to_thread(save_session_to_chroma, structured_json)
        structured_json["chroma_id"] = chroma_id

        llm_jobs[job_id] = {"status": "completed", "result": structured_json}

    except Exception as e:
        llm_jobs[job_id] = {"status": "error", "error": str(e)}


@router.post("/")
async def create_session(
    input_data: TranscriptInput, background_tasks: BackgroundTasks
):
    job_id = str(uuid.uuid4())
    # Start background processing with campaign_id
    background_tasks.add_task(
        process_and_save_session,
        job_id,
        input_data.transcript,
        input_data.campaign_id,  # <-- add this
    )
    # Return immediately
    return {"status": "processing", "job_id": job_id}


@router.delete("/")
async def reset_database():
    try:
        all_ids = session_collection.get()["ids"]
        if all_ids:
            session_collection.delete(ids=all_ids)
        return {"status": "all data deleted"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


@router.get("/status/{job_id}")
async def get_session_status(job_id: str):
    job = llm_jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job


@router.get("/")
async def list_chroma_sessions(
    campaign_id: str | None = Query(None, description="Filter by campaign ID")
):
    """
    List sessions from Chroma, optionally filtered by campaign_id.
    """
    try:
        # Fetch all sessions
        sessions = session_collection.get(where={"type": "session"})
        decoded = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        for i, md in enumerate(sessions["metadatas"]):
            # If campaign_id is specified, skip sessions that don't match
            if campaign_id is not None and str(md.get("campaign_id")) != str(
                campaign_id
            ):
                continue
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

            decoded["ids"].append(sessions["ids"][i])
            decoded["documents"].append(sessions["documents"][i])
            decoded["metadatas"].append(md_with_data)

        return decoded

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch from Chroma", "details": str(e)},
        )

from fastapi import APIRouter, BackgroundTasks, Body, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid, asyncio
from ._database import save_session_to_chroma
from ._speech import llm_jobs
from llm import dnd_ai
from ._database import session_collection
import json
from datetime import datetime
import json
import uuid
import asyncio


router = APIRouter()
MAX_SESSIONS = 1000
recent_sessions = []


class TranscriptInput(BaseModel):
    transcript: str
    campaign_id: str


async def process_and_save_session(job_id: str, transcript: str, campaign_id: str):
    try:
        llm_jobs[job_id] = {"status": "processing"}

        # --- Fetch existing campaign metadata first ---
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if campaign and campaign.get("ids"):
            campaign_meta = dict(campaign["metadatas"][0])

            try:
                existing_chars = json.loads(campaign_meta.get("characters", "[]"))
            except:
                existing_chars = []

            try:
                existing_locs = json.loads(campaign_meta.get("locations", "[]"))
            except:
                existing_locs = []
        else:
            campaign_meta = {"campaign_id": campaign_id}
            existing_chars = []
            existing_locs = []

        # --- Extract structured session data from transcript ---
        structured_json, new_chars, new_locs = await dnd_ai.extract_session_data(
            transcript,
            existing_chars=existing_chars,
            existing_locs=existing_locs,
            campaign_id=campaign_id,
        )
        structured_json["campaign_id"] = campaign_id
        session_id = structured_json.get("session_id")
        structured_json["session_id"] = session_id

        # --- Save session to Chroma ---
        chroma_id = await asyncio.to_thread(save_session_to_chroma, structured_json)
        structured_json["chroma_id"] = chroma_id

        # --- Merge characters back into campaign ---
        char_lookup = {
            c["name"].strip().lower(): c for c in existing_chars if "name" in c
        }
        for char in new_chars:
            name = char.get("name", "Unknown").strip()
            key = name.lower()
            if key in char_lookup:
                existing_char = char_lookup[key]
                merged_char = {**existing_char, **char}
                merged_char["character_id"] = existing_char["character_id"]
                merged_char["session_ids"] = list(
                    set(existing_char.get("session_ids", []) + [session_id])
                )
                char_lookup[key] = merged_char
            else:
                char["character_id"] = str(uuid.uuid4())[:6]
                char["session_ids"] = [session_id]
                char_lookup[key] = char
        campaign_meta["characters"] = json.dumps(list(char_lookup.values()))

        # --- Merge locations back into campaign ---
        loc_lookup = {
            l["location_name"].strip().lower(): l
            for l in existing_locs
            if "location_name" in l
        }
        for loc in new_locs:
            loc_name = loc.get("location_name", "Unknown").strip()
            key = loc_name.lower()
            if key in loc_lookup:
                existing_loc = loc_lookup[key]
                merged_loc = {**existing_loc, **loc}
                merged_loc["location_id"] = existing_loc["location_id"]
                merged_loc["session_ids"] = list(
                    set(existing_loc.get("session_ids", []) + [session_id])
                )
                loc_lookup[key] = merged_loc
            else:
                loc["location_id"] = str(uuid.uuid4())[:6]
                loc["location_name"] = loc_name
                loc.setdefault("location_description", "No description provided")
                loc["session_ids"] = [session_id]
                loc_lookup[key] = loc
        campaign_meta["locations"] = json.dumps(list(loc_lookup.values()))

        # --- Merge session IDs ---
        try:
            existing_session_ids = json.loads(campaign_meta.get("session_ids", "[]"))
        except:
            existing_session_ids = []
        if session_id not in existing_session_ids:
            existing_session_ids.append(session_id)
        campaign_meta["session_ids"] = json.dumps(existing_session_ids)

        # --- Save updated campaign metadata ---
        if campaign and campaign.get("ids"):
            session_collection.update(
                ids=[campaign["ids"][0]], metadatas=[campaign_meta]
            )
        else:
            # If campaign doesn't exist yet, create it
            session_collection.add(
                documents=[""], ids=[campaign_id], metadatas=[campaign_meta]
            )

        # --- Done ---
        llm_jobs[job_id] = {"status": "completed", "result": structured_json}

    except Exception as e:
        llm_jobs[job_id] = {"status": "error", "error": str(e)}


@router.post("/")
async def create_session(
    input_data: TranscriptInput, background_tasks: BackgroundTasks
):
    job_id = str(uuid.uuid4())[:6]
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


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    try:
        # Delete the session document directly by session_id
        session_collection.delete(ids=[session_id])

        # Remove from any campaign's session_ids
        campaigns = session_collection.get(where={"type": "campaign"})
        for i, cmeta in enumerate(campaigns.get("metadatas", [])):
            try:
                session_ids = json.loads(cmeta.get("session_ids", "[]"))
            except:
                session_ids = []
            if session_id in session_ids:
                session_ids.remove(session_id)
                cmeta["session_ids"] = json.dumps(session_ids)
                session_collection.update(ids=[campaigns["ids"][i]], metadatas=[cmeta])

        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/{session_id}")
async def get_session(session_id: str):
    """
    Fetch a single session and include its characters, locations, and events.
    """
    try:
        # Fetch the session
        results = session_collection.get(
            where={"$and": [{"type": "session"}, {"session_id": session_id}]}
        )
        if not results or not results.get("ids"):
            return JSONResponse(status_code=404, content={"error": "Session not found"})

        session_meta = results["metadatas"][0]

        # --- Fetch related records ---
        chars = session_collection.get(
            where={"$and": [{"type": "character"}, {"session_id": session_id}]}
        )
        locs = session_collection.get(
            where={"$and": [{"type": "location"}, {"session_id": session_id}]}
        )
        evs = session_collection.get(
            where={"$and": [{"type": "event"}, {"session_id": session_id}]}
        )

        # --- Attach related data ---
        session_meta["characters"] = chars.get("metadatas", [])
        session_meta["locations"] = locs.get("metadatas", [])
        session_meta["events"] = evs.get("metadatas", [])

        return session_meta

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.patch("/{session_id}")
async def patch_session(session_id: str, update: dict = Body(...)):
    """
    Partially update a session by session_id.
    Only fields provided in `update` will be merged into existing metadata.
    """
    try:
        # Fetch existing session
        results = session_collection.get(where={"session_id": session_id})
        if not results.get("ids"):
            return JSONResponse(status_code=404, content={"error": "Session not found"})

        old_metadata = results["metadatas"][0]
        old_document = results["documents"][0]

        # Merge updated fields (ignore None)
        merged = {**old_metadata, **{k: v for k, v in update.items() if v is not None}}

        # Optional: if name/title field exists, update the document text
        new_document = merged.get("name", old_document)

        # Remove old entry and save the updated one
        session_collection.delete(ids=[results["ids"][0]])
        session_collection.add(
            documents=[new_document],
            ids=[results["ids"][0]],
            metadatas=[merged],
        )

        return {"status": "updated", "session": merged}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Helper to get session IDs for a campaign
def get_session_ids_for_campaign(campaign_id: str):
    sessions = session_collection.get(
        where={"$and": [{"type": "session"}, {"campaign_id": campaign_id}]}
    ).get("metadatas", [])
    return [s["session_id"] for s in sessions]

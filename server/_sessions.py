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

router = APIRouter()
MAX_SESSIONS = 1000
recent_sessions = []


class TranscriptInput(BaseModel):
    transcript: str
    campaign_id: str


import httpx
import json
import uuid
import asyncio
from fastapi.responses import JSONResponse


async def process_and_save_session(job_id: str, transcript: str, campaign_id: str):
    try:
        llm_jobs[job_id] = {"status": "processing"}

        # --- Extract structured session data from transcript ---
        structured_json, chars, locs = await dnd_ai.extract_session_data(transcript)
        structured_json["campaign_id"] = campaign_id
        session_id = structured_json.get("session_id") or str(uuid.uuid4())[:12]
        structured_json["session_id"] = session_id

        # --- Save session to Chroma ---
        chroma_id = await asyncio.to_thread(save_session_to_chroma, structured_json)
        structured_json["chroma_id"] = chroma_id

        # --- Fetch campaign metadata ---
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if campaign and campaign.get("ids"):
            campaign_meta = dict(campaign["metadatas"][0])

            # -------------------
            # --- CHARACTERS ----
            # -------------------
            try:
                existing_chars = json.loads(campaign_meta.get("characters", "[]"))
            except:
                existing_chars = []
            char_lookup = {c["name"]: c for c in existing_chars if "name" in c}

            for char in chars:
                name = char.get("name", "Unknown")
                if name in char_lookup:
                    # Merge existing character
                    existing_char = char_lookup[name]
                    merged_char = {**existing_char, **char}
                    merged_char["character_id"] = existing_char["character_id"]
                    merged_char["session_ids"] = list(
                        set(existing_char.get("session_ids", []) + [session_id])
                    )
                    char_lookup[name] = merged_char
                else:
                    # New character
                    char["character_id"] = str(uuid.uuid4())[:6]
                    char["session_ids"] = [session_id]
                    char_lookup[name] = char

            campaign_meta["characters"] = json.dumps(list(char_lookup.values()))

            # -------------------
            # --- LOCATIONS -----
            # -------------------
            try:
                existing_locs = json.loads(campaign_meta.get("locations", "[]"))
            except:
                existing_locs = []
            loc_lookup = {
                l["location_name"]: l for l in existing_locs if "location_name" in l
            }

            for i, loc in enumerate(locs):
                loc_name = loc.get("location_name", f"Location {i+1}")
                if loc_name in loc_lookup:
                    existing_loc = loc_lookup[loc_name]
                    merged_loc = {**existing_loc, **loc}
                    merged_loc["location_id"] = existing_loc["location_id"]
                    merged_loc["session_ids"] = list(
                        set(existing_loc.get("session_ids", []) + [session_id])
                    )
                    loc_lookup[loc_name] = merged_loc
                else:
                    # New location
                    loc["location_id"] = str(uuid.uuid4())[:6]
                    loc["location_name"] = loc_name
                    loc.setdefault("location_description", "No description provided")
                    loc["session_ids"] = [session_id]
                    loc_lookup[loc_name] = loc

            campaign_meta["locations"] = json.dumps(list(loc_lookup.values()))

            # -------------------
            # --- SESSIONS -----
            # -------------------
            try:
                existing_session_ids = json.loads(
                    campaign_meta.get("session_ids", "[]")
                )
            except:
                existing_session_ids = []
            if session_id not in existing_session_ids:
                existing_session_ids.append(session_id)
            campaign_meta["session_ids"] = json.dumps(existing_session_ids)

            # --- Save updated campaign metadata ---
            session_collection.update(
                ids=[campaign["ids"][0]], metadatas=[campaign_meta]
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
        # Remove session document
        results = session_collection.get(where={"session_id": session_id})
        if results and results.get("ids"):
            session_collection.delete(ids=[results["ids"][0]])

        # Remove from any campaign's session_ids
        campaigns = session_collection.get(where={"type": "campaign"})
        for i, cmeta in enumerate(campaigns["metadatas"]):
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


# EVENTS
@router.get("/{campaign_id}/events")
async def get_campaign_events(campaign_id: str):
    """
    Return all events for a campaign, flattened across sessions.
    """
    try:
        session_ids = get_session_ids_for_campaign(campaign_id)
        events = []

        for sid in session_ids:
            ev_results = session_collection.get(
                where={"$and": [{"type": "event"}, {"session_id": sid}]}
            )
            events.extend(ev_results.get("metadatas", []))

        # Optional: sort events by timeline_order per session
        events.sort(key=lambda e: (e.get("session_id", ""), e.get("timeline_order", 0)))

        return {"events": events}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.patch("/{campaign_id}/events/{event_id}")
async def patch_event(event_id: str, update: dict = Body(...)):
    """
    Partially update an event by ID within a campaign.
    Only fields provided in `update` will be merged into existing metadata.
    """
    try:
        # Fetch the event by ID
        results = session_collection.get(ids=[event_id])
        if not results["ids"]:
            return JSONResponse(status_code=404, content={"error": "Event not found"})

        old_metadata = results["metadatas"][0]
        old_document = results["documents"][0]

        # Merge only the fields provided (non-None)
        merged = {**old_metadata, **{k: v for k, v in update.items() if v is not None}}

        # If "name" exists in the merged metadata, use it as the new document; otherwise, preserve old
        new_document = merged.get("name", old_document)

        # Remove old entry and save the updated one
        session_collection.delete(ids=[event_id])
        session_collection.add(
            documents=[new_document],
            ids=[event_id],
            metadatas=[merged],
        )

        return {"status": "updated", "event": merged}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Fetch a single event by ID within a campaign
@router.get("/{campaign_id}/events/{event_id}")
async def get_campaign_event(campaign_id: str, event_id: str):
    """
    Fetch a single event by event_id within a given campaign.
    """
    try:
        # Get all session IDs for the campaign
        session_ids = get_session_ids_for_campaign(campaign_id)
        if not session_ids:
            raise HTTPException(
                status_code=404, detail="Campaign not found or has no sessions"
            )

        # Search for the event across all sessions
        for sid in session_ids:
            event_results = session_collection.get(
                where={
                    "$and": [
                        {"type": "event"},
                        {"session_id": sid},
                        {"event_id": event_id},
                    ]
                }
            )
            events = event_results.get("metadatas", [])
            if events:
                return {"event": events[0]}

        # If we reach here, event was not found
        raise HTTPException(status_code=404, detail="Event not found in this campaign")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.delete("/{campaign_id}/events/{event_id}")
async def delete_event(campaign_id: str, event_id: str):
    """
    Delete a specific event by ID from a given campaign.
    """
    try:
        # Get all session IDs for the campaign
        session_ids = get_session_ids_for_campaign(campaign_id)
        if not session_ids:
            return JSONResponse(
                status_code=404,
                content={"error": "Campaign not found or has no sessions"},
            )

        # Search for the event across all sessions
        found = False
        for sid in session_ids:
            event_results = session_collection.get(
                where={
                    "$and": [
                        {"type": "event"},
                        {"session_id": sid},
                        {"event_id": event_id},
                    ]
                }
            )
            if event_results.get("ids"):
                # Delete the event
                session_collection.delete(ids=event_results["ids"])
                found = True
                break

        if not found:
            return JSONResponse(
                status_code=404,
                content={"error": "Event not found in this campaign"},
            )

        return {"status": "deleted", "event_id": event_id}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Locations


@router.get("/{campaign_id}/locations")
async def get_campaign_locations(campaign_id: str):
    """
    Return all locations for a campaign, flattened across sessions.
    """
    try:
        session_ids = get_session_ids_for_campaign(campaign_id)
        locations = []

        for sid in session_ids:
            loc_results = session_collection.get(
                where={"$and": [{"type": "location"}, {"session_id": sid}]}
            )
            locations.extend(loc_results.get("metadatas", []))

        return {"locations": locations}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/{campaign_id}/locations/{location_id}")
async def get_campaign_location(campaign_id: str, location_id: str):
    """
    Fetch a single location by location_id within a given campaign.
    """
    try:
        # Get all session IDs for the campaign
        session_ids = get_session_ids_for_campaign(campaign_id)
        if not session_ids:
            raise HTTPException(
                status_code=404, detail="Campaign not found or has no sessions"
            )

        # Search for the location across all sessions in the campaign
        for sid in session_ids:
            loc_results = session_collection.get(
                where={
                    "$and": [
                        {"type": "location"},
                        {"session_id": sid},
                        {"location_id": location_id},
                    ]
                }
            )
            locs = loc_results.get("metadatas", [])
            if locs:
                return {"location": locs[0]}

        # If we reach here, location was not found
        raise HTTPException(
            status_code=404, detail="Location not found in this campaign"
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.patch("/{campaign_id}/locations/{location_id}")
async def patch_location(location_id: str, update: dict = Body(...)):
    """
    Partially update a location by ID.
    Only fields provided in `update` will be merged into existing metadata.
    """
    try:
        results = session_collection.get(ids=[location_id])
        if not results["ids"]:
            return JSONResponse(
                status_code=404, content={"error": "Location not found"}
            )

        old_metadata = results["metadatas"][0]
        old_document = results["documents"][0]

        # Merge only provided fields (non-null)
        merged = {**old_metadata, **{k: v for k, v in update.items() if v is not None}}

        # Use name as the document title if provided
        new_document = merged.get("name", old_document)

        # Replace old entry with updated one
        session_collection.delete(ids=[location_id])
        session_collection.add(
            documents=[new_document],
            ids=[location_id],
            metadatas=[merged],
        )

        return {"status": "updated", "location": merged}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.delete("/{campaign_id}/locations/{location_id}")
async def delete_location(campaign_id: str, location_id: str):
    """
    Delete a specific location by ID from a given campaign.
    """
    try:
        # Get all session IDs for this campaign
        session_ids = get_session_ids_for_campaign(campaign_id)
        if not session_ids:
            return JSONResponse(
                status_code=404,
                content={"error": "Campaign not found or has no sessions"},
            )

        # Search and delete location
        found = False
        for sid in session_ids:
            loc_results = session_collection.get(
                where={
                    "$and": [
                        {"type": "location"},
                        {"session_id": sid},
                        {"location_id": location_id},
                    ]
                }
            )
            if loc_results.get("ids"):
                session_collection.delete(ids=loc_results["ids"])
                found = True
                break

        if not found:
            return JSONResponse(
                status_code=404,
                content={"error": "Location not found in this campaign"},
            )

        return {"status": "deleted", "location_id": location_id}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

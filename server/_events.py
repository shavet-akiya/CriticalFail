import uuid
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ._database import session_collection
from ._sessions import get_session_ids_for_campaign

router = APIRouter()


def save_events(collection, summary, session_data):
    for ev in summary.get("events", []):
        if not isinstance(ev, dict):
            continue
        ev_id = ev.get("event_id") or str(uuid.uuid4())[:6]
        ev_text = ev.get("event", "Unnamed Event")
        ev_metadata = ev.copy()
        for key in ["participants", "event_tags"]:
            if isinstance(ev_metadata.get(key), list):
                ev_metadata[key] = ", ".join(map(str, ev_metadata[key]))
        ev_metadata.update(
            {
                "event_id": ev_id,
                "session_id": session_data.get("session_id"),
                "type": "event",
            }
        )
        collection.add(documents=[ev_text], ids=[ev_id], metadatas=[ev_metadata])


# list events, optionally filtered by campaign_id
@router.get("/")
async def list_campaign_events():
    try:
        # fetch all events
        results = session_collection.get(where={"type": "event"})
        events = results.get("metadatas", [])
        # ensure participants and tags are arrays
        for ev in events:
            for key in ["participants", "event_tags"]:
                val = ev.get(key)
                if isinstance(val, str):
                    # convert comma-separated string back to list
                    ev[key] = [v.strip() for v in val.split(",") if v.strip()]
                elif not isinstance(val, list):
                    ev[key] = []

        return {"events": events}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# EVENTS
@router.get("/{campaign_id}")
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


@router.patch("/{campaign_id}/{event_id}")
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


# events/campaign_id/event_id
@router.get("/{campaign_id}/{event_id}")
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


# events/campaign_id/event_id
@router.delete("/{campaign_id}/{event_id}")
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

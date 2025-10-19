import uuid
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ._database import session_collection

router = APIRouter()


def save_events(collection, summary, session_data):
    for ev in summary.get("events", []):
        if not isinstance(ev, dict):
            continue
        ev_id = ev.get("event_id") or str(uuid.uuid4())
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


# list all events
@router.get("/")
async def list_events():
    try:
        results = session_collection.get(where={"type": "event"})
        return {"events": results["metadatas"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# get event by id
@router.get("/{event_id}")
async def get_event(event_id: str):
    results = session_collection.get(ids=[event_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Event not found"})
    return {"event": results["metadatas"][0]}


# Update specific event
@router.patch("/{event_id}")
async def patch_event(event_id: str, update: dict = Body(...)):
    results = session_collection.get(ids=[event_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Event not found"})

    old_metadata = results["metadatas"][0]
    old_document = results["documents"][0]

    # Merge only fields provided
    merged = {**old_metadata, **{k: v for k, v in update.items() if v is not None}}
    new_document = merged.get("event", old_document)

    session_collection.delete(ids=[event_id])
    session_collection.add(
        documents=[new_document],
        ids=[event_id],
        metadatas=[merged],
    )

    return {"status": "updated", "event": merged}


# Delete specific event
@router.delete("/{event_id}")
async def delete_event(event_id: str):
    results = session_collection.get(ids=[event_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Event not found"})

    try:
        session_collection.delete(ids=[event_id])
        return {"status": "deleted", "event_id": event_id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to delete event", "details": str(e)},
        )

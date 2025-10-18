import uuid
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from ._database import session_collection

router = APIRouter()


def save_locations(collection, summary, session_data):
    for loc in summary.get("locations", []):
        if not isinstance(loc, dict):
            continue
        loc_id = loc.get("location_id") or str(uuid.uuid4())
        loc_name = loc.get("location_name") or loc.get("name", "Unknown Location")
        collection.add(
            documents=[loc_name],
            ids=[loc_id],
            metadatas={
                "location_id": loc_id,
                "session_id": session_data.get("session_id"),
                "type": "location",
                **loc,
            },
        )


# list all locations
@router.get("/")
async def list_locations():
    try:
        results = session_collection.get(where={"type": "location"})
        return {"locations": results["metadatas"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# get location by id
@router.get("/{location_id}")
async def get_location(location_id: str):
    results = session_collection.get(ids=[location_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Location not found"})
    return {"location": results["metadatas"][0]}


# update specific location
@router.patch("/{location_id}")
async def patch_location(location_id: str, update: dict = Body(...)):
    results = session_collection.get(ids=[location_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Location not found"})

    old_metadata = results["metadatas"][0]
    old_document = results["documents"][0]

    # Only merge keys explicitly provided
    merged = {**old_metadata, **{k: v for k, v in update.items() if v is not None}}
    new_document = merged.get("location_name", old_document)

    session_collection.delete(ids=[location_id])
    session_collection.add(
        documents=[new_document], ids=[location_id], metadatas=[merged]
    )

    return {"status": "updated", "location": merged}


# delete specific location
@router.delete("/{location_id}")
async def delete_location(location_id: str):
    results = session_collection.get(ids=[location_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Location not found"})

    try:
        session_collection.delete(ids=[location_id])
        return {"status": "deleted", "location_id": location_id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to delete location", "details": str(e)},
        )

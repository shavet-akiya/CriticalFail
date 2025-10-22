import uuid
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from ._database import session_collection
from pydantic import BaseModel
import json

router = APIRouter()


def save_locations(collection, summary, session_data):
    for loc in summary.get("locations", []):
        if not isinstance(loc, dict):
            continue
        loc_id = loc.get("location_id") or str(uuid.uuid4())[:6]
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


#############################
#############################
#############################
class CreateLocationRequest(BaseModel):
    location_name: str
    location_description: str = "No description provided"
    campaign_id: str
    session_ids: list[str] = []  # optional, can pre-assign session IDs


@router.post("/{campaign_id}/locations", status_code=201)
async def add_location_to_campaign(campaign_id: str, req: CreateLocationRequest):
    """
    Add a location to a campaign.
    If the location name already exists, merge session_ids and update fields.
    """
    try:
        # Fetch the campaign
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign_meta = campaign["metadatas"][0]

        # Load existing locations
        existing_locs = campaign_meta.get("locations", "[]")
        try:
            existing_locs = json.loads(existing_locs)
        except Exception:
            existing_locs = []

        # Lookup by name
        existing_lookup = {
            l["location_name"]: l for l in existing_locs if "location_name" in l
        }

        if req.location_name in existing_lookup:
            # Merge existing location
            existing_loc = existing_lookup[req.location_name]
            merged_sids = list(
                set(existing_loc.get("session_ids", []) + req.session_ids)
            )
            existing_loc.update(req.dict(exclude={"campaign_id", "session_ids"}))
            existing_loc["session_ids"] = merged_sids
            existing_lookup[req.location_name] = existing_loc
        else:
            # New location
            new_loc = req.dict()
            new_loc["location_id"] = str(uuid.uuid4())[:6]
            if not new_loc.get("session_ids"):
                new_loc["session_ids"] = []
            existing_lookup[req.location_name] = new_loc

        # Save back to campaign
        campaign_meta["locations"] = json.dumps(list(existing_lookup.values()))
        session_collection.update(ids=[campaign["ids"][0]], metadatas=[campaign_meta])

        return {
            "status": "created",
            "campaign_id": campaign_id,
            "location": existing_lookup[req.location_name],
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

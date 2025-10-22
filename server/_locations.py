import uuid
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from ._database import session_collection
from pydantic import BaseModel
import json
from ._sessions import get_session_ids_for_campaign

router = APIRouter()


# --- Save location(s) for a session and link to campaign ---
def save_locations(collection, summary, session_data, campaign_id):
    session_id = session_data.get("session_id")
    existing_locs = {}

    # --- Fetch existing campaign locations ---
    campaign = collection.get(
        where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
    )
    if campaign and campaign.get("ids"):
        campaign_meta = campaign["metadatas"][0]
        try:
            locs = json.loads(campaign_meta.get("locations", "[]"))
            existing_locs = {
                l["location_name"]: l["location_id"]
                for l in locs
                if "location_name" in l and "location_id" in l
            }
        except Exception:
            existing_locs = {}

    # --- Save each location ---
    for loc in summary.get("locations", []):
        if not isinstance(loc, dict):
            continue

        loc_name = loc.get("location_name", loc.get("name", "Unknown Location"))
        loc_id = existing_locs.get(loc_name, str(uuid.uuid4())[:6])

        # Save session-level location
        collection.add(
            documents=[loc_name],
            ids=[loc_id],
            metadatas={
                "location_id": loc_id,
                "session_id": session_id,
                "campaign_id": campaign_id,
                "type": "location",
                **loc,
            },
        )

    # --- After adding, update campaign-level metadata ---
    campaign = collection.get(
        where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
    )
    if not campaign or not campaign.get("ids"):
        return

    campaign_meta = campaign["metadatas"][0]
    campaign_locs = json.loads(campaign_meta.get("locations", "[]"))

    # Merge new locations
    for loc in summary.get("locations", []):
        loc_name = loc.get("location_name", loc.get("name", "Unknown Location"))
        loc_id = existing_locs.get(loc_name, str(uuid.uuid4())[:6])

        # Find if exists
        existing = next(
            (l for l in campaign_locs if l["location_name"] == loc_name), None
        )
        if existing:
            sids = set(existing.get("session_ids", []))
            sids.add(session_id)
            existing["session_ids"] = list(sids)
        else:
            new_entry = {
                "location_id": loc_id,
                "location_name": loc_name,
                "location_description": loc.get(
                    "location_description", "No description"
                ),
                "session_ids": [session_id],
            }
            campaign_locs.append(new_entry)

    # Save back to campaign
    campaign_meta["locations"] = json.dumps(campaign_locs)
    session_collection.update(ids=[campaign["ids"][0]], metadatas=[campaign_meta])


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


@router.post("/{campaign_id}", status_code=201)
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


# Locations


@router.get("/{campaign_id}")
async def get_campaign_locations(campaign_id: str):
    """
    Return only the locations for a given campaign.
    """
    try:
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            return JSONResponse(
                status_code=404, content={"error": "Campaign not found"}
            )

        campaign_meta = campaign["metadatas"][0]
        locations_str = campaign_meta.get("locations", "[]")

        try:
            locations = json.loads(locations_str)
        except Exception:
            locations = []

        return {"location": {"campaign_id": campaign_id, "locations": locations}}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


########################################
# Update (patch) location
########################################
@router.patch("/{campaign_id}/{location_id}")
async def patch_campaign_location(
    campaign_id: str, location_id: str, update: dict = Body(...)
):
    """
    Update a location across a campaign by location_id.
    """
    try:
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign_meta = campaign["metadatas"][0]
        locations = json.loads(campaign_meta.get("locations", "[]"))

        location = next(
            (l for l in locations if l.get("location_id") == location_id), None
        )
        if not location:
            raise HTTPException(
                status_code=404, detail="Location not found in this campaign"
            )

        for k, v in update.items():
            if v is not None:
                location[k] = v

        session_ids = location.get("session_ids", [])
        if not isinstance(session_ids, list):
            session_ids = []

        # Update session-level copies
        for sid in session_ids:
            session_locs = session_collection.get(
                where={
                    "$and": [
                        {"type": "location"},
                        {"session_id": sid},
                        {"location_id": location_id},
                    ]
                }
            )
            for i, sl in enumerate(session_locs.get("metadatas", [])):
                merged = {**sl, **{k: v for k, v in update.items() if v is not None}}
                session_collection.update(
                    ids=[session_locs["ids"][i]], metadatas=[merged]
                )

        campaign_meta["locations"] = json.dumps(locations)
        session_collection.update(ids=[campaign["ids"][0]], metadatas=[campaign_meta])

        location.pop("session_ids", None)
        return {"status": "updated", "location": location}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


########################################
# Delete a location from campaign
########################################
@router.delete("/{campaign_id}/{location_id}")
async def delete_campaign_location(campaign_id: str, location_id: str):
    """
    Delete a specific location by ID from a given campaign.
    """
    try:
        session_ids = get_session_ids_for_campaign(campaign_id)
        if not session_ids:
            return JSONResponse(
                status_code=404,
                content={"error": "Campaign not found or has no sessions"},
            )

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

        # Remove from campaign metadata
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if campaign and campaign.get("ids"):
            campaign_meta = campaign["metadatas"][0]
            locations = json.loads(campaign_meta.get("locations", "[]"))
            locations = [l for l in locations if l.get("location_id") != location_id]
            campaign_meta["locations"] = json.dumps(locations)
            session_collection.update(
                ids=[campaign["ids"][0]], metadatas=[campaign_meta]
            )

        return {"status": "deleted", "location_id": location_id}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

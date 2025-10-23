from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import uuid
import datetime
from ._database import session_collection


router = APIRouter()


class AssignSessionRequest(BaseModel):
    session_id: str


class CreateCampaignRequest(BaseModel):
    campaign_name: str
    session_ids: list[str] = []  # Optional – can be empty


class CreateCampaignRequest(BaseModel):
    campaign_name: str
    session_ids: list[str] = []  # Optional – can be empty


class UpdateCharactersRequest(BaseModel):
    campaign_id: str
    characters: list[dict]  # list of character objects to patch


@router.patch("/{campaign_id}/characters")
async def patch_campaign_characters(campaign_id: str, req: UpdateCharactersRequest):
    """
    Patch character info into a campaign.
    Existing characters with the same name will be updated.
    New characters will be appended.
    """
    try:
        # Fetch campaign
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign_meta = campaign["metadatas"][0]

        # Load existing characters
        existing_chars = campaign_meta.get("characters", "[]")
        if isinstance(existing_chars, str):
            try:
                existing_chars = json.loads(existing_chars)
            except:
                existing_chars = []

        # Create a lookup by name for easy patching
        existing_lookup = {c["name"]: c for c in existing_chars}

        # Merge/update incoming characters
        for char in req.characters:
            name = char.get("name")
            if name in existing_lookup:
                # Update existing character fields
                existing_lookup[name].update(char)
            else:
                # Add new character
                existing_lookup[name] = char

        # Save back as JSON string
        campaign_meta["characters"] = json.dumps(list(existing_lookup.values()))

        # Update in ChromaDB
        session_collection.update(ids=[campaign["ids"][0]], metadatas=[campaign_meta])

        return {
            "status": "updated",
            "campaign_id": campaign_id,
            "characters": list(existing_lookup.values()),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# New folder for campaign images
UPLOAD_DIR = "server/images/campaign_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/")
async def create_campaign(
    campaign_name: str = Form(...),
    campaign_description: str = Form(""),
    campaign_image: UploadFile | None = File(None),
):
    campaign_id = str(uuid.uuid4())[:6]
    filename = None
    image_url = None

    # Save uploaded image
    if campaign_image:
        file_ext = os.path.splitext(campaign_image.filename)[1]
        filename = f"{campaign_id}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(await campaign_image.read())

        # Public URL for frontend
        image_url = f"/campaign_images/{filename}"

    # Store campaign metadata
    metadata = {
        "type": "campaign",
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "campaign_description": campaign_description or "",
        "characters": json.dumps([]),
        "locations": json.dumps([]),
        "session_ids": json.dumps([]),
        "campaign_image_url": image_url or "",  # always string
        "created_at": str(datetime.datetime.utcnow()),
    }

    try:
        session_collection.add(
            documents=[campaign_name],
            ids=[campaign_id],
            metadatas=[metadata],
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "status": "created",
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "characters": json.dumps([]),
        "locations": json.dumps([]),
        "image_url": image_url,
        "session_count": 0,
    }


@router.get("/{campaign_id}/sessions")
async def get_campaign_sessions(campaign_id: str):
    """
    Get all sessions associated with a specific campaign.
    Includes their characters, locations, and events.
    """
    try:
        # Fetch all sessions tied to this campaign
        results = session_collection.get(
            where={"$and": [{"type": "session"}, {"campaign_id": campaign_id}]}
        )

        if not results or not results.get("metadatas"):
            return JSONResponse(
                status_code=404,
                content={"error": f"No sessions found for campaign {campaign_id}"},
            )

        sessions = []
        for i, session_meta in enumerate(results["metadatas"]):
            session_id = session_meta.get("session_id")

            # Fetch associated records
            chars = session_collection.get(
                where={"$and": [{"type": "character"}, {"session_id": session_id}]}
            )
            locs = session_collection.get(
                where={"$and": [{"type": "location"}, {"session_id": session_id}]}
            )
            evs = session_collection.get(
                where={"$and": [{"type": "event"}, {"session_id": session_id}]}
            )

            session_meta["characters"] = chars.get("metadatas", [])
            session_meta["locations"] = locs.get("metadatas", [])
            session_meta["events"] = evs.get("metadatas", [])
            session_meta["document"] = results["documents"][i]

            sessions.append(session_meta)

        return {"campaign_id": campaign_id, "sessions": sessions}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/{campaign_id}/sessions", status_code=201)
async def create_session_in_campaign(campaign_id: str, req: AssignSessionRequest):
    """
    Assign an existing session to this campaign or create a session entry.
    Expects body: {"session_id": "..."}.
    """
    try:
        # Find session by session_id
        results = session_collection.get(
            where={"type": "session", "session_id": req.session_id}
        )
        if results and results.get("ids"):
            # Update existing session metadata
            idx = 0
            meta = results["metadatas"][idx]
            chroma_id = results["ids"][idx]
            meta["campaign_id"] = campaign_id
            meta["updated_at"] = str(datetime.datetime.utcnow())
            session_collection.update(ids=[chroma_id], metadatas=[meta])
        else:
            # Create a minimal session record if not present
            chroma_doc_id = str(uuid.uuid4())[:6]
            session_collection.add(
                documents=[req.session_id],
                ids=[chroma_doc_id],
                metadatas=[
                    {
                        "type": "session",
                        "session_id": req.session_id,
                        "campaign_id": campaign_id,
                        "created_at": str(datetime.datetime.utcnow()),
                    }
                ],
            )

        # Also ensure campaign's session list includes this session
        campaign = session_collection.get(
            where={"type": "campaign", "campaign_id": campaign_id}
        )
        if campaign and campaign.get("ids"):
            cmeta = campaign["metadatas"][0]
            session_ids = set(cmeta.get("session_ids", []))
            session_ids.add(req.session_id)
            cmeta["session_ids"] = list(session_ids)
            session_collection.update(ids=[campaign["ids"][0]], metadatas=[cmeta])

        return {
            "status": "assigned",
            "session_id": req.session_id,
            "campaign_id": campaign_id,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.delete("/{campaign_id}")
async def delete_campaign(campaign_id: str):
    """
    Delete a campaign and all sessions associated with it.
    """
    try:
        # Find the campaign by ID
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )

        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign_meta = campaign["metadatas"][0]
        campaign_chroma_id = campaign["ids"][0]

        # Delete associated sessions
        sessions = session_collection.get(
            where={"$and": [{"type": "session"}, {"campaign_id": campaign_id}]}
        )

        deleted_sessions = []
        if sessions and sessions.get("ids"):
            session_ids_to_delete = sessions["ids"]
            session_collection.delete(ids=session_ids_to_delete)
            deleted_sessions = [m["session_id"] for m in sessions["metadatas"]]

        # Delete the campaign itself
        session_collection.delete(ids=[campaign_chroma_id])

        return {
            "status": "deleted",
            "campaign_id": campaign_id,
            "deleted_sessions": deleted_sessions,
            "message": f"Campaign {campaign_id} and {len(deleted_sessions)} associated sessions deleted.",
        }

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/")
async def get_campaigns():
    """
    Return all campaigns, including aggregated characters and locations
    across all sessions.
    """
    try:
        # Fetch all campaigns
        campaigns = session_collection.get(where={"type": "campaign"})
        if not campaigns or not campaigns.get("metadatas"):
            return []

        full_list = []

        for i, cmeta in enumerate(campaigns["metadatas"]):
            campaign_id = cmeta.get("campaign_id")
            session_ids = json.loads(cmeta.get("session_ids", "[]"))

            # Aggregate characters and locations from all sessions
            all_chars = []
            all_locs = []

            for sid in session_ids:
                chars = session_collection.get(
                    where={"$and": [{"type": "character"}, {"session_id": sid}]}
                )
                locs = session_collection.get(
                    where={"$and": [{"type": "location"}, {"session_id": sid}]}
                )
                if chars.get("metadatas"):
                    all_chars.extend(chars["metadatas"])
                if locs.get("metadatas"):
                    all_locs.extend(locs["metadatas"])

            # Deduplicate by id or name
            unique_chars = {
                char.get("character_id") or char.get("name"): char for char in all_chars
            }.values()
            unique_locs = {
                loc.get("location_id") or loc.get("location_name"): loc
                for loc in all_locs
            }.values()

            full_list.append(
                {
                    "campaign_id": campaign_id,
                    "campaign_name": cmeta.get("campaign_name"),
                    "campaign_description": cmeta.get("campaign_description", ""),
                    "session_ids": session_ids,
                    "campaign_image_url": cmeta.get("campaign_image_url", ""),
                    "characters": list(unique_chars),
                    "locations": list(unique_locs),
                }
            )

        return full_list

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/{campaign_id}")
async def get_campaign(campaign_id: str):
    """
    Return a single campaign with sessions included.
    Characters and locations are returned as arrays of objects.
    """
    try:
        # Fetch campaign metadata
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        meta = campaign["metadatas"][0]

        # Parse characters, locations, session_ids from strings into proper arrays
        def parse_json_field(field):
            val = meta.get(field, "[]")
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except:
                    return []
            return val

        characters = parse_json_field("characters")
        locations = parse_json_field("locations")
        session_ids = parse_json_field("session_ids")

        return {
            "campaign_id": meta.get("campaign_id"),
            "campaign_name": meta.get("campaign_name"),
            "campaign_description": meta.get("campaign_description", ""),
            "campaign_image_url": meta.get("campaign_image_url", ""),
            "created_at": meta.get("created_at"),
            "characters": characters,
            "locations": locations,
            "session_ids": session_ids,
        }

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class UpdateCampaignRequest(BaseModel):
    campaign_id: str
    campaign_name: str | None = None
    campaign_image_url: str | None = None
    campaign_description: str | None = None


@router.patch("/{campaign_id}")
async def update_campaign(campaign_id: str, req: UpdateCampaignRequest):
    try:
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        meta = campaign["metadatas"][0]

        if req.campaign_name is not None:
            meta["campaign_name"] = req.campaign_name
        if req.campaign_description is not None:
            meta["campaign_description"] = req.campaign_description
        if req.campaign_image_url is not None:
            meta["campaign_image_url"] = req.campaign_image_url

        session_collection.update(ids=[campaign["ids"][0]], metadatas=[meta])

        return {
            "campaign_name": meta.get("campaign_name"),
            "campaign_description": meta.get("campaign_description", ""),
            "campaign_image_url": meta.get("campaign_image_url", ""),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/{campaign_id}/image")
async def update_campaign_image(
    campaign_id: str,
    campaign_image: UploadFile = File(...),
):
    """
    Upload a new image for a campaign.
    Replaces existing image if present.
    """
    try:
        # Fetch campaign
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        meta = campaign["metadatas"][0]
        campaign_chroma_id = campaign["ids"][0]

        # Save new image
        file_ext = os.path.splitext(campaign_image.filename)[1]
        filename = f"{campaign_id}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(await campaign_image.read())

        # Update campaign metadata
        image_url = f"/campaign_images/{filename}"
        meta["campaign_image_url"] = image_url
        session_collection.update(ids=[campaign_chroma_id], metadatas=[meta])

        return {"status": "updated", "campaign_image_url": image_url}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.delete("/{campaign_id}/image")
async def delete_campaign_image(campaign_id: str):
    """
    Remove the campaign image: deletes the file (if exists) and clears the URL in metadata.
    """
    try:
        # Fetch campaign
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        meta = campaign["metadatas"][0]
        chroma_id = campaign["ids"][0]

        # Delete file from disk if exists
        image_url = meta.get("campaign_image_url", "")
        if image_url:
            filename = os.path.basename(image_url)
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Clear image URL in metadata
        meta["campaign_image_url"] = ""
        session_collection.update(ids=[chroma_id], metadatas=[meta])

        return {
            "status": "removed",
            "campaign_id": campaign_id,
            "campaign_image_url": "",
        }

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

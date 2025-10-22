import uuid
import json
from pydantic import BaseModel
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from ._database import session_collection
from ._sessions import get_session_ids_for_campaign

router = APIRouter()


# save character
def save_characters(collection, summary, session_data, campaign_id):
    session_id = session_data.get("session_id")
    existing_chars = {}

    # Fetch existing campaign characters
    campaign = collection.get(
        where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
    )
    if campaign and campaign.get("ids"):
        campaign_meta = campaign["metadatas"][0]
        try:
            chars = json.loads(campaign_meta.get("characters", "[]"))
            existing_chars = {
                c["name"]: c["character_id"]
                for c in chars
                if "name" in c and "character_id" in c
            }
        except Exception:
            existing_chars = {}

    # Save each character
    for character in summary.get("characters", []):
        if not isinstance(character, dict):
            continue
        name = character.get("name", "Unknown Character")
        # Reuse existing character_id if name exists
        character_id = existing_chars.get(name, str(uuid.uuid4())[:6])

        collection.add(
            documents=[name],
            ids=[character_id],
            metadatas={
                "character_id": character_id,
                "session_id": session_id,
                "campaign_id": campaign_id,
                "type": "character",
                **character,
            },
        )


# --- Get all characters for a campaign ---
@router.get("/{campaign_id}")
async def get_campaign_characters(campaign_id: str):
    """
    Return all characters for a campaign from the campaign metadata.
    """
    try:
        # Fetch the campaign metadata
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            return JSONResponse(
                status_code=404, content={"error": "Campaign not found"}
            )

        campaign_meta = campaign["metadatas"][0]

        # Load characters directly from campaign
        characters = json.loads(campaign_meta.get("characters", "[]"))

        # Remove session_ids for cleaner response if desired
        for c in characters:
            c.pop("session_ids", None)

        return {"characters": characters}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Get specific character in a campaign ---
@router.get("/{campaign_id}/{character_id}")
async def get_campaign_character(campaign_id: str, character_id: str):
    """
    Fetch a single character by character_id within a given campaign.
    """
    try:
        # Fetch the campaign metadata
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign_meta = campaign["metadatas"][0]
        characters = json.loads(campaign_meta.get("characters", "[]"))

        # Find the character by ID
        character = next(
            (c for c in characters if c.get("character_id") == character_id), None
        )

        if not character:
            raise HTTPException(
                status_code=404, detail="Character not found in this campaign"
            )

        # Optionally remove session_ids if you don’t want to expose them
        character.pop("session_ids", None)

        return {"character": character}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Edit character in a campaign and update session-level characters via name and session_id ---
@router.patch("/{campaign_id}/{character_id}")
async def edit_character(campaign_id: str, character_id: str, update: dict = Body(...)):
    campaign = session_collection.get(
        where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
    )
    if not campaign or not campaign.get("ids"):
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign_meta = campaign["metadatas"][0]
    characters = json.loads(campaign_meta.get("characters", "[]"))

    character = next(
        (c for c in characters if c.get("character_id") == character_id), None
    )
    if not character:
        raise HTTPException(
            status_code=404, detail="Character not found in this campaign"
        )

    # Merge updates
    for k, v in update.items():
        if v is not None:
            character[k] = v

    session_ids = character.get("session_ids", [])
    if not isinstance(session_ids, list):
        session_ids = []

    # Update session-level characters
    for sid in session_ids:
        session_chars = session_collection.get(
            where={
                "$and": [
                    {"type": "character"},
                    {"session_id": sid},
                    {"character_id": character_id},
                ]
            }
        )
        for i, sc in enumerate(session_chars.get("metadatas", [])):
            merged = {**sc, **{k: v for k, v in update.items() if v is not None}}
            session_collection.update(ids=[session_chars["ids"][i]], metadatas=[merged])

    # Save back to campaign metadata
    campaign_meta["characters"] = json.dumps(characters)
    session_collection.update(ids=[campaign["ids"][0]], metadatas=[campaign_meta])

    character.pop("session_ids", None)
    return {"status": "updated", "character": character}


@router.delete("/{campaign_id}/{character_id}")
async def delete_character(campaign_id: str, character_id: str):
    """
    Delete a specific character by ID from a given campaign.
    Works even if the campaign has no sessions.
    """
    try:
        # --- Try campaign-level delete first ---
        campaign_result = session_collection.get(
            where={
                "$and": [
                    {"type": "character"},
                    {"campaign_id": campaign_id},
                    {"character_id": character_id},
                ]
            }
        )

        if campaign_result.get("ids"):
            session_collection.delete(ids=campaign_result["ids"])
            return {
                "status": "deleted (campaign-level)",
                "character_id": character_id,
            }

        # --- Fallback: try deleting from session-based characters ---
        session_ids = get_session_ids_for_campaign(campaign_id)
        if not session_ids:
            return JSONResponse(
                status_code=404,
                content={"error": "Character not found in campaign or sessions"},
            )

        found = False
        for sid in session_ids:
            char_results = session_collection.get(
                where={
                    "$and": [
                        {"type": "character"},
                        {"session_id": sid},
                        {"character_id": character_id},
                    ]
                }
            )
            if char_results.get("ids"):
                session_collection.delete(ids=char_results["ids"])
                found = True
                break

        if not found:
            return JSONResponse(
                status_code=404,
                content={"error": "Character not found in this campaign"},
            )

        return {"status": "deleted", "character_id": character_id}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


########################################
########################################
########################################

import json


class CreateCharacterRequest(BaseModel):
    name: str
    char_class: str = "Unknown"
    role: str = "Unknown"
    npc: bool = False
    campaign_id: str
    session_ids: list[str] = []  # optional, can pre-assign session IDs
    race: str = "Unknown"
    AC: int = 0
    HP: int = 0
    STR: int = 0
    DEX: int = 0
    CON: int = 0
    INT: int = 0
    WIS: int = 0
    CHA: int = 0


# create character campaign - /characters/campaign_id
@router.post("/{campaign_id}", status_code=201)
async def create_character(campaign_id: str, req: CreateCharacterRequest):
    """
    Add a character to a campaign.
    If the character name already exists, merge session_ids and update fields.
    """
    try:
        # Fetch the campaign
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign_meta = campaign["metadatas"][0]

        # Load existing characters
        existing_chars = campaign_meta.get("characters", "[]")
        try:
            existing_chars = json.loads(existing_chars)
        except Exception:
            existing_chars = []

        # Lookup by name
        existing_lookup = {c["name"]: c for c in existing_chars if "name" in c}

        if req.name in existing_lookup:
            # Merge existing character
            existing_char = existing_lookup[req.name]
            merged_sids = list(
                set(existing_char.get("session_ids", []) + req.session_ids)
            )
            existing_char.update(req.dict(exclude={"campaign_id", "session_ids"}))
            existing_char["session_ids"] = merged_sids
            existing_lookup[req.name] = existing_char
        else:
            # New character
            new_char = req.dict()
            new_char["character_id"] = str(uuid.uuid4())[:6]
            if not new_char.get("session_ids"):
                new_char["session_ids"] = []
            existing_lookup[req.name] = new_char

        # Save back to campaign
        campaign_meta["characters"] = json.dumps(list(existing_lookup.values()))
        session_collection.update(ids=[campaign["ids"][0]], metadatas=[campaign_meta])

        return {
            "status": "created",
            "campaign_id": campaign_id,
            "character": existing_lookup[req.name],
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.patch("/{campaign_id}/{name}")
async def patch_campaign_character_by_name(
    campaign_id: str, name: str, update: dict = Body(...)
):
    """
    Update a character across a campaign by name.
    The updates propagate to all sessions the character appears in.
    """
    try:
        # --- Fetch campaign ---
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign_meta = campaign["metadatas"][0]

        # --- Fetch characters from campaign ---
        characters = json.loads(campaign_meta.get("characters", "[]"))
        char = next((c for c in characters if c.get("name") == name), None)
        if not char:
            raise HTTPException(status_code=404, detail="Character not found")

        # --- Merge updates ---
        for k, v in update.items():
            if v is not None:
                char[k] = v

        session_ids = char.get("session_ids", [])
        if not isinstance(session_ids, list):
            session_ids = []

        # --- Update all session-level characters by name ---
        for sid in session_ids:
            session_chars = session_collection.get(
                where={"$and": [{"type": "character"}, {"session_id": sid}]}
            )
            for i, sc in enumerate(session_chars.get("metadatas", [])):
                if sc.get("name") == name:
                    merged = {
                        **sc,
                        **{k: v for k, v in update.items() if v is not None},
                    }
                    session_collection.update(
                        ids=[session_chars["ids"][i]], metadatas=[merged]
                    )

        # --- Update campaign metadata ---
        campaign_meta["characters"] = json.dumps(characters)
        session_collection.update(ids=[campaign["ids"][0]], metadatas=[campaign_meta])

        # Remove session_ids in response
        char.pop("session_ids", None)
        return {"status": "updated", "character": char}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/{campaign_id}/{name}")
async def get_campaign_character_instances(campaign_id: str, name: str):
    """
    Fetch all instances of a character by name across all sessions in a campaign.
    """
    try:
        # --- Fetch campaign ---
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign_meta = campaign["metadatas"][0]

        # --- Get session IDs for this campaign ---
        session_ids = campaign_meta.get("session_ids", [])
        if not isinstance(session_ids, list):
            session_ids = []

        all_instances = []

        # --- Iterate sessions and fetch character instances by name ---
        for sid in session_ids:
            session_chars = session_collection.get(
                where={"$and": [{"type": "character"}, {"session_id": sid}]}
            )
            for i, sc in enumerate(session_chars.get("metadatas", [])):
                if sc.get("name") == name:
                    # Optionally exclude session_id or keep it
                    all_instances.append({k: v for k, v in sc.items()})

        if not all_instances:
            raise HTTPException(
                status_code=404, detail="Character not found in any session"
            )

        return {"instances": all_instances}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

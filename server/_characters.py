import uuid
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from ._database import session_collection

router = APIRouter()


def save_characters(collection, summary, session_data):
    session_id = session_data.get("session_id")
    campaign_id = session_data.get("campaign_id")  # ← grab it from session

    for character in summary.get("characters", []):
        if not isinstance(character, dict):
            continue
        character_id = character.get("character_id") or str(uuid.uuid4())
        name = character.get("name", "Unknown Character")

        collection.add(
            documents=[name],
            ids=[character_id],
            metadatas={
                "character_id": character_id,
                "session_id": session_id,
                "campaign_id": campaign_id,  # ← include campaign
                "type": "character",
                **character,
            },
        )


@router.get("/")
async def get_campaign_characters(campaign_id: str):
    """
    Fetch all characters belonging to a specific campaign.
    """
    try:
        # Get all characters for this campaign
        results = session_collection.get(
            where={"$and": [{"type": "character"}, {"campaign_id": campaign_id}]}
        )

        if not results or not results.get("metadatas"):
            return {"characters": []}

        characters = results["metadatas"]

        # Ensure proper ordering or filtering if needed
        return {"characters": characters}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch characters", "details": str(e)},
        )


# get specific character
@router.get("/{character_id}")
async def get_character(character_id: str):
    results = session_collection.get(ids=[character_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Character not found"})
    return {"character": results["metadatas"][0]}


# delete specific character
@router.delete("/{character_id}")
async def delete_character(character_id: str):
    results = session_collection.get(ids=[character_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Character not found"})

    try:
        session_collection.delete(ids=[character_id])
        return {"status": "deleted", "character_id": character_id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to delete character", "details": str(e)},
        )

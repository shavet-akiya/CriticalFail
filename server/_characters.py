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


@router.get("/campaign/{campaign_id}")
async def list_campaign_characters(campaign_id: str):
    """List all characters linked to a campaign (through its sessions)."""
    results = session_collection.get(
        where={
            "type": "character",
            "campaign_id": campaign_id,
        }
    )
    return {"characters": results["metadatas"]}


# get specific character
@router.get("/{character_id}")
async def get_character(character_id: str):
    results = session_collection.get(ids=[character_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Character not found"})
    return {"character": results["metadatas"][0]}


# update specific character
@router.patch("/{character_id}")
async def patch_character(character_id: str, update: dict = Body(...)):
    results = session_collection.get(ids=[character_id])
    if not results["ids"]:
        return JSONResponse(status_code=404, content={"error": "Character not found"})

    old_metadata = results["metadatas"][0]
    old_document = results["documents"][0]

    merged = {**old_metadata, **{k: v for k, v in update.items() if v is not None}}
    new_document = merged.get("name", old_document)

    session_collection.delete(ids=[character_id])
    session_collection.add(
        documents=[new_document], ids=[character_id], metadatas=[merged]
    )

    return {"status": "updated", "character": merged}


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

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


class UpdateCampaignRequest(BaseModel):
    session_id: str
    campaign_id: str = None


class CreateCampaignRequest(BaseModel):
    campaign_name: str
    session_ids: list[str] = []  # Optional – can be empty


# ✅ New folder for campaign images
UPLOAD_DIR = "server/images/campaign_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/")
async def create_campaign(
    campaign_name: str = Form(...),
    campaign_image: UploadFile | None = File(None),
):
    campaign_id = str(uuid.uuid4())[:6]
    filename = None
    image_url = None

    # ✅ Save uploaded image
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
        "image_url": image_url,
        "session_count": 0,
    }


@router.put("/")
async def update_campaign(req: UpdateCampaignRequest):
    try:
        results = session_collection.get(where={"session_id": req.session_id})
        if not results["ids"]:
            return JSONResponse(status_code=404, content={"error": "Session not found"})

        old_metadata = results["metadatas"][0]
        old_document = results["documents"][0]
        old_id = results["ids"][0]

        old_metadata["campaign_id"] = req.campaign_id
        session_collection.delete(ids=[old_id])
        session_collection.add(
            documents=[old_document], ids=[old_id], metadatas=[old_metadata]
        )

        return {
            "status": "updated",
            "session_id": req.session_id,
            "campaign_id": req.campaign_id,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/")
async def list_campaigns():
    try:
        results = session_collection.get(where={"type": "campaign"})
        if not results or "metadatas" not in results:
            return []

        campaigns = results.get("metadatas", [])
        # Ensure campaigns is always a list
        if not isinstance(campaigns, list):
            campaigns = [campaigns]

        # Optional: normalize fields
        normalized = [
            {
                "campaign_id": c.get("campaign_id"),
                "campaign_name": c.get("campaign_name"),
                "session_ids": json.loads(c.get("session_ids", "[]")),  # ✅ decode JSON
                "campaign_image_url": c.get("campaign_image_url", ""),
            }
            for c in campaigns
        ]

        return normalized
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.patch("/sessions/{session_id}/campaign")
def update_session_campaign(
    session_id: str, new_campaign_id: str = Body(..., embed=True)
):
    """
    Update a session's campaign assignment in ChromaDB.
    """

    # Fetch all session metadata to find the matching session
    results = session_collection.get(
        where={"type": "session", "session_id": session_id}
    )

    if not results or not results.get("ids"):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    session_idx = 0  # Assuming unique session_id
    current_metadata = results["metadatas"][session_idx]
    chroma_id = results["ids"][session_idx]

    # Update the metadata with the new campaign
    current_metadata["campaign_id"] = new_campaign_id
    current_metadata["updated_at"] = str(datetime.datetime.utcnow())

    # Re-add or update the session document
    session_collection.update(
        ids=[chroma_id],
        metadatas=[current_metadata],
    )

    # Optionally update the campaign’s session list
    campaign = session_collection.get(
        where={"type": "campaign", "campaign_id": new_campaign_id}
    )
    if campaign and campaign.get("ids"):
        campaign_metadata = campaign["metadatas"][0]
        session_ids = set(campaign_metadata.get("session_ids", []))
        session_ids.add(session_id)
        campaign_metadata["session_ids"] = list(session_ids)

        session_collection.update(
            ids=[campaign["ids"][0]],
            metadatas=[campaign_metadata],
        )

    return {"message": f"Session {session_id} updated to campaign {new_campaign_id}."}


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


@router.get("/{campaign_id}")
async def get_campaign(campaign_id: str):
    try:
        campaign = session_collection.get(
            where={"$and": [{"type": "campaign"}, {"campaign_id": campaign_id}]}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")
        return campaign["metadatas"][0]
    except HTTPException:
        raise
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

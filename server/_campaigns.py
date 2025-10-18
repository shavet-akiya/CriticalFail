from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import datetime
from ._database import session_collection

router = APIRouter(tags=["campaigns"])


class AssignSessionRequest(BaseModel):
    session_id: str


class UpdateCampaignRequest(BaseModel):
    session_id: str
    campaign_id: str = None


class CreateCampaignRequest(BaseModel):
    campaign_name: str
    session_ids: list[str] = []  # Optional – can be empty


@router.post("/campaigns")
async def create_campaign(req: CreateCampaignRequest):
    """
    Create a new campaign in ChromaDB, optionally with session IDs.
    """
    try:
        campaign_id = str(uuid.uuid4())

        session_collection.add(
            documents=[req.campaign_name],
            ids=[campaign_id],
            metadatas=[
                {
                    "type": "campaign",
                    "campaign_id": campaign_id,
                    "campaign_name": req.campaign_name,
                    "session_ids": req.session_ids,
                    "created_at": str(datetime.datetime.utcnow()),
                }
            ],
        )

        return JSONResponse(
            status_code=201,
            content={
                "status": "created",
                "campaign_id": campaign_id,
                "campaign_name": req.campaign_name,
                "session_count": len(req.session_ids),
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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


@router.post("/campaigns/{campaign_id}/sessions", status_code=201)
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
            chroma_doc_id = str(uuid.uuid4())
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


@router.get("/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    try:
        campaign = session_collection.get(
            where={"type": "campaign", "campaign_id": campaign_id}
        )
        if not campaign or not campaign.get("ids"):
            raise HTTPException(status_code=404, detail="Campaign not found")
        return campaign["metadatas"][0]
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/campaigns/{campaign_id}/sessions")
async def get_campaign_sessions(campaign_id: str):
    """
    Return list of session metadata that belong to campaign_id.
    """
    try:
        # Query all sessions with matching campaign_id
        results = session_collection.get(
            where={"type": "session", "campaign_id": campaign_id}
        )
        sessions = results.get("metadatas", [])
        return {"sessions": sessions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

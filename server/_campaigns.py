from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ._database import session_collection

router = APIRouter()


class UpdateCampaignRequest(BaseModel):
    session_id: str
    campaign_id: str = None


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

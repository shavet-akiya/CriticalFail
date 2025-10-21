from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ._speech import router as speech_router
from ._sessions import router as sessions_router
from ._characters import router as characters_router
from ._locations import router as locations_router
from ._events import router as events_router
from ._campaigns import router as campaigns_router
from ._database import router as database_router

import os

app = FastAPI()

# origins = [
#     "http://localhost:3000",  # for local dev
#     "http://ui:3000",  # inside Docker
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(speech_router, prefix="/speech")
app.include_router(sessions_router, prefix="/sessions")
app.include_router(characters_router, prefix="/characters")
app.include_router(locations_router, prefix="/locations")
app.include_router(events_router, prefix="/events")
app.include_router(campaigns_router, prefix="/campaign")
app.include_router(database_router, prefix="/database")

os.makedirs("uploads/campaign_images", exist_ok=True)
app.mount(
    "/campaign_images",
    StaticFiles(directory="server/images/campaign_images"),
    name="campaign_images",
)

# Ensure folder exists


@app.get("/")
async def root():
    return {"message": "FastAPI D&D server running!"}

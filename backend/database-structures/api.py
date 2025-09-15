from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Event import Event

database: dict[int: Event] = {}

database[0] = (
    Event(
        summary="he died", 
        characters=["Jim"], 
        places=["USA", "Jim's house"], 
        themes=["Loss", "A Deep sense of mourning"], 
        tags=[]
    )
)

database[1] = (
    Event(
        summary="he got revived by a wizard", 
        characters=["Jim", "an awesome wizard"],
        places=["USA", "Jim's house"], 
        themes=["jubilee", "mirth", "rebirth"], 
        tags=["idk"])
)

app = FastAPI()

origins = [
    "http://localhost:5173",
    "localhost:5173"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
async def read_root() -> dict:
    return {"message": "Welcome to the thing!"}


def validate(event: Event, characters, places, themes, tags):
    if characters:
        for character in characters:
            if character not in event.characters:
                return False
    
    if places:
        for place in places:
            if place not in event.places:
                return False

    if themes:  
        for theme in themes:
            if theme not in event.themes:
                return False
    
    if tags:
        for tag in tags:
            if tag not in event.tags:
                return False
    
    return True

@app.get("/event")
async def search(
    characters: list[str] | None=None, 
    places: list[str] | None=None, 
    themes: list[str] | None=None,
    tags: list[str] | None=None,
    ) -> dict:

    returning = []
    for event in database.values():
        if validate(event, characters, places, themes, tags):
            returning.append(event)
    return {"data": returning}


@app.post("/event")
async def add_event(event: Event, at_id: int, infront: bool):
    database[at_id] = event
    return {"data": "Event added."}


@app.delete("/event")
async def remove_event(event_id: int):
    removing = database[event_id]
    database.pop(event_id)
    return {"data": removing}


@app.put("/event")
async def edit_event(new_event: Event, event_id: int):
    database[event_id] = new_event

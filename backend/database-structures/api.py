from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Event import Event

# Before doing the database more thoroughly,
# I will test FastAPI using a python dict of events

database: list[Event] = []

database.append(
    Event(
        id= 0, 
        summary="he dieded", 
        characters=["Joe Biden"], 
        places=["USA", "Whitehouse"], 
        themes=["Loss", "A Deep sense of mourning"], 
        tags=[]
    )
)

database.append(
    Event(
        id= 1, 
        summary="he got revived by a wizard", 
        characters=["Joe Biden", "an awesome wizard"],
        places=["USA", "Whitehouse"], 
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


@app.get("/event")
async def get_database() -> dict:
    return {"data": database}


@app.put("/event")
async def add_event(event: Event):
    database.append(event)
    return event

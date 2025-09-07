from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Event import Event

# Before doing the database more thoroughly,
# I will test FastAPI using a python dict of events

database: dict[int:Event] = {}

database[0] = Event(id= 0, summary="he dieded", characters=["Joe Biden"], places=["USA", "Whitehouse"], themes=["Loss", "A Deep sense of mourning"], tags=[])

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
    return "Welcome to the thing!"


@app.get("/event")
async def get_database() -> dict:
    return database


@app.put("/event")
async def add_event(event: Event):
    database[event.id] = event
    return event


@app.delete("/event")
async def pop_event(id: int):
    deleting = database[id]
    database.pop(id)
    return deleting



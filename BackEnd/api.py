from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Event import Event

# Before doing the database more thoroughly,
# I will test FastAPI using a regular python list of events

database: dict[Event] = {}

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
async def read_root() -> str:
    return "Welcome to the thing!!"


@app.get("/database")
async def get_database() -> dict[Event]:
    return database


@app.post("/todo", tags=["todos"])
async def add_event(event: Event) -> str:
    database.append(event)
    return "event added."

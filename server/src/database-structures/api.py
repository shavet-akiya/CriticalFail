from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Structures import Event, validate, TableEntry
from Database import Database

database = Database()

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
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

''' TABLE METHODS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
@app.get("/table/get_keys/{table_name}")
async def get_ids(table_name):
    return {"data": database.get_keys(table_name)}

@app.get("/table/{table_name}")
async def get_element(table_name, name):
    return {"data": database.get_element(table_name, name)}

@app.post("/table/{table_name}")
async def add_or_edit_element(table_name, item: TableEntry):
    return {"data": database.add_or_edit_element(table_name, item)}

@app.delete("/table/{table_name}")
async def remove_element(table_name, name):
    return {"data": database.remove_element(table_name, name)}
''' EVENT METHODS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

@app.get("/event")
async def search(
    characters: list[str] | None=None, 
    locations: list[str] | None=None, 
    themes: list[str] | None=None,
    tags: list[str] | None=None,
    ) -> dict:

    returning = []
    for event in database.timeline:
        if validate(event, characters, locations, themes, tags):
            returning.append(event)
    return {"data": returning}


@app.post("/event")
async def add_event(event: Event, id: int, infront: bool):
    database.timeline.write_future(event=event, id=id)
    return {"data": "Event added."}


@app.delete("/event")
async def remove_event(id: int):
    database.timeline.burn_record(id)
    return {"data": "Event removed"}


@app.put("/event")
async def edit_event(new_event: Event, id: int):
    database.timeline.change_history(event=new_event, id=id)
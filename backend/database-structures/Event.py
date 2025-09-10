<<<<<<< HEAD:backend/database-structures/Event.py
from pydantic import BaseModel

class Event(BaseModel):
    summary: str
    characters: list[str]
    places: list[str]
    themes: list[str]
=======
from pydantic import BaseModel

class Event(BaseModel):
    id: int
    summary: str
    characters: list[str]
    places: list[str]
    themes: list[str]
>>>>>>> main:BackEnd/Event.py
    tags: list[str]
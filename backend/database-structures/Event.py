from pydantic import BaseModel

class Event(BaseModel):
    id: int
    summary: str
    characters: list[str]
    places: list[str]
    themes: list[str]
    tags: list[str]

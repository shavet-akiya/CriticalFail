from pydantic import BaseModel, Field
from typing import TypeVar, Generic

MetadataModelT = TypeVar("MetadataModelT", bound=BaseModel)

class Metadata(Generic[MetadataModelT]):
    def __init__(self, model: type[MetadataModelT]):
        self.model = model

    def _get_metadata(self) -> MetadataModelT:
        pass
        
    def _set_metadata(self, metadata: MetadataModelT):
        pass

    def get_next_id(self) -> int:
        pass


class Event(BaseModel):
    id: int
    title: str
    summary: str
    characters: list[str]
    locations: list[str]
    themes: list[str]
    tags: list[str]
    session: int

class TableEntry(BaseModel):
    name: str

class Character(TableEntry):
    class_: str | None = Field(alias='class', default=None)
    race: str | None = None
    armour_class: int | None = None
    npc: bool = False
    enemy: bool | None = None
    hp: int | None = None
    str: int | None = None
    dex: int | None = None
    con: int | None = None
    wis: int | None = None
    cha: int | None = None
    img: int | None = None
    int_: int | None = Field(alias='int', default=None)

class Location(TableEntry):
    description: str

class Session(TableEntry):
    summary: str

class Tag(TableEntry):
    pass


# Idk if this function is useful, probably don't worry about it for now
def validate(event: Event, characters, locations, themes, tags):
    if characters:
        for character in characters:
            if character not in event.characters:
                return False
    
    if locations:
        for location in locations:
            if location not in event.locations:
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
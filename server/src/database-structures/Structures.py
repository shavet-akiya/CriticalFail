from pydantic import BaseModel, Field

class TableEntry(BaseModel):
    id: int | None = None

class Event(TableEntry):
    title: str
    summary: str
    characters: list[str]
    locations: list[str]
    themes: list[str]
    tags: list[str]
    session: int

class Character(TableEntry):
    name: str
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
    name: str
    description: str

class Session(TableEntry):
    summary: int

class Tag(TableEntry):
    tag: str

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
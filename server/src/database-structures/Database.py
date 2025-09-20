from Event_Timeline_Graph import *
from Table import Table
from Structures import Character, Location, Session, Tag
import io

class Database:
    def __init__(self, address):
        self.address = address + "database/"
        os.makedirs(self.address, exist_ok=True)
        self.timeline: Timeline = Timeline(self.address + "timeline/")
        self.characters: Table[Character] = Table[Character](self.address + "characters/", Character)
        self.locations: Table[Location] = Table[Location](self.address + "locations/", Location)
        self.sessions: Table[Session] = Table[Session](self.address + "sessions/", Session)
        self.tags: Table[Tag] = Table[Tag](self.address + "tags/", Tag)
    
    def get_table(self, table_name: str):
        if table_name == "characters":
            return self.characters
        if table_name == "locations":
            return self.locations
        if table_name == "sessions":
            return self.sessions
        if table_name == "tags":
            return self.tags

from Event_Timeline_Graph import *
from Table import Table
from Structures import Character, Location, Session, Tag
import io
import chromadb

class Database:
    def __init__(self, address):
        self.address = address + "DND_database/"

        chroma_client = chromadb.HttpClient(host='localhost', port=8011)

        self.collections = {}
        tables = ["events", "characters", "locations", "sessions", "tags"]

        for table in tables:
                self.collections[table] = self.chroma_client.get_or_create_collection(
                name=table,
            )
                
        self.events = self.collections["events"]
        self.characters = self.collections["characters"]
        self.locations = self.collections["locations"]
        self.sessions = self.collections["sessions"]
        self.tags = self.collections["tags"]

        # i dont know what to do with this at all
        self.timeline = None


        # os.makedirs(self.address, exist_ok=True)
        # self.timeline: Timeline = Timeline(self.address + "timeline/")
        # self.characters: Table[Character] = Table[Character](self.address + "characters/", Character)
        # self.locations: Table[Location] = Table[Location](self.address + "locations/", Location)
        # self.sessions: Table[Session] = Table[Session](self.address + "sessions/", Session)
        # self.tags: Table[Tag] = Table[Tag](self.address + "tags/", Tag)
    
    def get_table(self, table_name: str):
        if table_name == "characters":
            return self.characters
        if table_name == "locations":
            return self.locations
        if table_name == "sessions":
            return self.sessions
        if table_name == "tags":
            return self.tags

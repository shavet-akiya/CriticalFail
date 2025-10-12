from Event_Timeline_Graph import *
from Table import Table
from Structures import Character, Location, Session, Tag
import io
import chromadb

class Database:
    def __init__(self):
        self.dnd_database = chromadb.HttpClient(host='localhost', port=8011)
        self.timeline = Timeline(
            self.dnd_database.get_or_create_collection(name="timeline"),
            self.dnd_database.get_or_create_collection(name="timeline_meta")
        )

    
    def reset_timeline(self):
        self.dnd_database.delete_collection(name="timeline")
        self.dnd_database.delete_collection(name="timeline_meta")

        self.timeline = Timeline(
            self.dnd_database.get_or_create_collection(name="timeline"),
            self.dnd_database.get_or_create_collection(name="timeline_meta")
        )
        '''
        self.collections = {}
        tables = ["timeline", "events", "characters", "locations", "sessions", "tags", "timeline_meta"]

        for table in tables:
                self.collections[table] = self.dnd_database.get_or_create_collection(
                name=table,
            )

        self.timeline = self.collections["timeline"]
        self.timeline_meta = self.collections["timeline_meta"]
        self.events = self.collections["events"]
        self.characters = self.collections["characters"]
        self.locations = self.collections["locations"]
        self.sessions = self.collections["sessions"]
        self.tags = self.collections["tags"]
        '''

        # i dont know what to do with this at all
        


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

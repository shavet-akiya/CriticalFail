from Event_Timeline_Graph import *
from Structures import TableEntry
import chromadb

class Database:
    def __init__(self):
        self.dnd_database = chromadb.HttpClient(host='localhost', port=8011)

        self.collections: list[Collection] = {}
        tables = ["timeline", "characters", "locations", "sessions", "tags", "timeline_meta"]

        for table in tables:
                self.collections[table] = self.dnd_database.get_or_create_collection(
                name=table,
            )

        self.timeline = Timeline(
            self.collections["timeline"],
            self.collections["timeline_meta"]
        )

    ''' Should return a list of all of the keys in the table '''
    def get_keys(self, table_name: str) -> list:
        pass

    ''' Return the tabe entry for an element at a given id'''
    def get_element(self, table_name: str, name: str) -> TableEntry:
        pass

    ''' Add the element to the table given its json'''
    def add_element(self, table_name: str, entry: TableEntry):
        id = [str(entry.name)]
        document = [entry.model_dump_json()]
        metadata = []

        self.collections[table_name].upsert(    
            ids=id,
            documents=document,
            metadatas=metadata
        )

    ''' Edit the element at the id: entry.name to be entry '''
    def edit_element(self, table_name: str, entry: str):
        pass

    ''' Remove the element at the id: name '''
    def remove_element(self, table_name: str, name: str):
        pass


    def get_table(self, table_name: str) -> Collection | None:
        if table_name in self.collections:
            return self.collections[table_name]
        else:
            raise KeyError



    # i dont know what to do with this at all
    
    # self.timeline = self.collections["timeline"]
    # self.timeline_meta = self.collections["timeline_meta"]
    # self.events = self.collections["events"]
    # self.characters = self.collections["characters"]
    # self.locations = self.collections["locations"]
    # self.sessions = self.collections["sessions"]
    # self.tags = self.collections["tags"]

    # def reset_timeline(self):
    #     self.dnd_database.delete_collection(name="timeline")
    #     self.dnd_database.delete_collection(name="timeline_meta")

    #     self.collections["timeline"] = self.dnd_database.get_or_create_collection(name="timeline")
    #     self.collections["timeline_meta"] = self.dnd_database.get_or_create_collection(name="timeline_meta")

    #     self.timeline = Timeline(
    #         self.collections["timeline"],
    #         self.collections["timeline_meta"]
    #     )
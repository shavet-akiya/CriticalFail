import os
import json
from typing import TypeVar, Generic
from pydantic import BaseModel
from Structures import TableEntry
    
MetadataModelT = TypeVar("MetadataModelT", bound=BaseModel)

class Metadata(Generic[MetadataModelT]):
    def __init__(self, address, model: type[MetadataModelT]):
        self.address = address + "metadata"
        self.model = model

    def _get_metadata(self) -> MetadataModelT:
        try:
            file = open(self.address, "r")
            jsonfile = json.load(file)
            return self.model(**jsonfile)
        except FileNotFoundError:
            metadata = self.model()
            file = open(self.address, "w")
            file.write(metadata.model_dump_json())
            return metadata
        
    def _set_metadata(self, metadata: MetadataModelT):
        file = open(self.address, "w")
        file.write(metadata.model_dump_json())

    def get_next_id(self) -> int:
        data = self._get_metadata()
        data.next_id += 1
        self._set_metadata(data)
        return data.next_id


class TableMetadataModel(BaseModel):
    ids: list[int] = []

TableEntryT = TypeVar("TableEntryT", bound=TableEntry)

'''
A table of json files.
'''
class Table(Generic[TableEntryT]):
    def __init__(self, address, type: type[TableEntryT]):
        self.address = address
        self._data: Metadata[TableMetadataModel] = Metadata[TableMetadataModel](address, TableMetadataModel)
        self.type = type
        os.makedirs(self.address, exist_ok=True) 

    def get_ids(self):
        data = self._data._get_metadata()
        return data.ids
    
    def edit(self, entry: TableEntryT):
        file = open(self.address + str(entry.id), "w")
        file.write(entry.model_dump_json(by_alias=True))

    def get(self, id: int) -> TableEntryT:
        with open(self.address + str(id), "r") as f:
            return self.type(**(json.load(f)))
        
    ''' Too similar
    '''
    def add(self, entry: TableEntryT):
        file = open(self.address + str(entry.id), "w")
        file.write(entry.model_dump_json(by_alias=True))
        data = self._data._get_metadata()
        if entry.id not in data.ids:
            data.ids.append(entry.id)
            self._data._set_metadata(data)

    def remove(self, id):
        
        data = self._data._get_metadata()
        data.ids.remove(id)
        self._data._set_metadata(data)
        os.remove(self.address + str(id))
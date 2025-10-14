import os
import json
from typing import TypeVar, Generic
from pydantic import BaseModel
from Structures import TableEntry, Metadata

'''

**************************** I THINK THIS CODE IS OBSOLETE, SO DON'T WORRY ABOUT IT ****************************

'''

class TableMetadataModel(BaseModel):
    ids: list[int] = []

TableEntryT = TypeVar("TableEntryT", bound=TableEntry)

'''
A table of json files.
'''
class Table(Generic[TableEntryT]):
    def __init__(self, type: type[TableEntryT]):
        self._data: Metadata[TableMetadataModel] = Metadata[TableMetadataModel](TableMetadataModel)
        self.type = type

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
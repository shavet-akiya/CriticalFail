from __future__ import annotations
from Event import Event
from pydantic import BaseModel
import json
import os


class OrbModel(BaseModel):
    id: int
    event: Event
    past: int | None = None
    future: int | None = None


    # def write_future(self, event : Event):
    #     new_orb = Orb(event, self, self.future)
    #     if self.future:
    #         self.future.past = new_orb
    #     self.future = new_orb
    #     return new_orb

    # def write_past(self, event : Event):
    #     new_orb = Orb(event, self.past, self)
    #     if self.past:
    #         self.past.future = new_orb
    #     self.past = new_orb
    #     return new_orb


class TimelineMetadataModel(BaseModel):
    start: int | None = None
    end: int | None = None
    length: int = 0
    next_id: int = -1


class TimelineMetadata:
    def __init__(self, address):
        self.address = address + "metadata"

    def _get_metadata(self) -> TimelineMetadataModel:
        try:
            file = open(self.address, "r")
            jsonfile = json.load(file)
            return TimelineMetadataModel(**jsonfile)
        except FileNotFoundError:
            metadata = TimelineMetadataModel()
            file = open(self.address, "w")
            file.write(metadata.model_dump_json())
            return metadata
        
    def _set_metadata(self, metadata: TimelineMetadataModel):
        file = open(self.address, "w")
        file.write(metadata.model_dump_json())

    def get_start(self):
        data = self._get_metadata()
        return data.start
    
    def set_start(self, start):
        data = self._get_metadata()
        data.start = start
        self._set_metadata(data)
    
    def get_end(self):
        data = self._get_metadata()
        return data.end
    
    def set_end(self, end):
        data = self._get_metadata()
        data.end = end
        self._set_metadata(data)
    
    def get_length(self):
        data = self._get_metadata()
        return data.length
    
    def add_length(self, i: int):
        data = self._get_metadata()
        data.length += i
        self._set_metadata(data)

    def get_next_id(self) -> int:
        data = self._get_metadata()
        data.next_id += 1
        self._set_metadata(data)
        return data.next_id


class Timeline:
    def __init__(self, address):
        self.address = address + "timeline/"
        self._data = TimelineMetadata(self.address)

    def __iter__(self):
        self.pointer = self._data.get_end()
        return self
    
    def __next__(self):
        if self.pointer is None:
            raise StopIteration
        else:
            orb = self.get_orb(self.pointer)
            self.pointer = orb.past
            event = orb.event
            event.id = orb.id
            return event

    def _event_filename(self, id: int):
        return self.address + "events/" + str(id)

    def _save_orb(self, orb: OrbModel):
        file = open(self._event_filename(orb.id), "w")
        file.write(orb.model_dump_json())

    def get_orb(self, id: int) -> OrbModel:
        with open(self._event_filename(id), "r") as f:
            return OrbModel(**(json.load(f)))
        
    
    '''
    If id is None or end, then it appends it to the end. If end is none but id is not none then error
    Otherwise it inserts it inbetween the current and next node.
    '''
    def write_future(self, event: Event, id: int = None):
        end = self._data.get_end()
        if id is None:
            id = end

        new_orb = OrbModel(id=self._data.get_next_id(), event=event, past=id)

        if id is None:
            self._data.set_start(new_orb.id)
            self._data.set_end(new_orb.id)

        else:
            old_orb = self.get_orb(id)
            new_orb.future = old_orb.future
            
            if old_orb.future is None:
                self._data.set_end(new_orb.id)

            else:
                future_orb = self.get_orb(old_orb.future)
                future_orb.past = new_orb.id
                self._save_orb(future_orb)

            old_orb.future = new_orb.id
            self._save_orb(old_orb)

        self._save_orb(new_orb)
        self._data.add_length(1)

    def change_history(self, event: Event, id: int):
        orb = self.get_orb(id)
        orb.event = event
        self._save_orb(orb)

    def burn_record(self, id: int):
        marked_orb = self.get_orb(id)
        if marked_orb.future is not None:
            future_orb = self.get_orb(marked_orb.future)
            future_orb.past = marked_orb.past
            self._save_orb(future_orb)

        if marked_orb.past is not None:
            past_orb = self.get_orb(marked_orb.past)
            past_orb.future = marked_orb.future
            self._save_orb(past_orb)

        os.remove(self._event_filename(id))
        self._data.add_length(-1)
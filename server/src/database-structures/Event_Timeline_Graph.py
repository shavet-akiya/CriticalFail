from __future__ import annotations
from Structures import Event
from pydantic import BaseModel
import json
import os
from Table import Metadata
import chromadb

class OrbModel(BaseModel):
    id: int
    event: Event
    past: int | None = None
    future: int | None = None


class TimelineMetadataModel(BaseModel):
    start: int | None = None
    end: int | None = None
    length: int = 0
    next_id: int = -1

class TimelineMetadata(Metadata[TimelineMetadataModel]):
    def __init__(self, address):
        super().__init__(address, TimelineMetadataModel)

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


class Timeline:
    def __init__(self):
        self.dnd_database = chromadb.HttpClient(host='localhost', port=8011)

        self.timeline = self.dnd_database.get_or_create_collection(name="timeline")
        self.timeline_meta = self.dnd_database.get_or_create_collection(name="timeline_meta")

        self.initialise_timeline()
        # self.address = address
        # self._data = TimelineMetadata(self.address)
        # os.makedirs(self.address, exist_ok=True) 
        # os.makedirs(self.address + "events/", exist_ok=True) 


    def initialise_timeline(self):
        result = self.timeline_meta.get(ids=["META"], include=["metadatas"])

        if not result or not result.get('metadatas') or not result['metadatas'][0]:
            initial_metadata = {
                "start": 0,
                "end": 0,
                "length": 0,
                "next_id": 0
            }

            self.timeline_meta.add(ids=["META"], documents=["Timeline Metadata"], metadatas=[initial_metadata])

    def get_timeline_state(self):
        result_meta = self.timeline_meta.get(ids=["META"], include=["metadatas"])

        if result_meta and result_meta.get("metadatas") and result_meta["metadatas"][0]:
            metadata = result_meta["metadatas"][0]
            return TimelineMetadataModel(**metadata)

        raise Exception("Timeline Metadata missing")

    def _save_timeline_state(self, state: TimelineMetadataModel):
        metadata_dict = state.model_dump()
        self.timeline_meta.upsert(
            ids = ["META"],
            documents=["Timeline Metadata"],
            metadatas=[metadata_dict]
        )

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
        id = [str(orb.id)]
        document = [orb.event.model_dump_json()]
        metadata = [{"past": orb.past, "future": orb.future}]

        self.timeline.upsert(    
            ids=id,
            documents=document,
            metadatas=metadata
        )

    def get_orb(self, id: int) -> OrbModel:
        data = self.timeline.get(ids = [str(id)], include=["documents", "metadatas"])

        if not data or not data.get("documents") or not data["documents"][0]:
            raise FileNotFoundError(f"Orb with ID {id} not found in ChromaDB.")
        
        print(data)

        orb_data = data["documents"][0]
        orb_metadata = data["metadatas"][0]

        orb_data_dict = json.loads(orb_data)
        event = Event(**orb_data_dict)

        past = orb_metadata.get("past")
        future = orb_metadata.get("future")

        orb = OrbModel(id=id, event = event, past = past, future = future)

        return orb

    '''
    If id is None or end, then it appends it to the end. If end is none but id is not none then error
    Otherwise it inserts it inbetween the current and next node.
    '''
    def write_future(self, event: Event, id: int = None):

        state = self.get_timeline_state()

        end = state.end

        if id is None:
            id = end

        new_orb = OrbModel(id=state.next_id, event=event, past=id)
    
        if not state.next_id and not state.end and not state.start:
            state.start = new_orb.id
            state.end = new_orb.id

        else:
            old_orb = self.get_orb(id)
            new_orb.future = old_orb.future
        
            if old_orb.future is None:
                state.end = new_orb.id

            else:
                future_orb = self.get_orb(old_orb.future)
                future_orb.past = new_orb.id
                self._save_orb(future_orb)
            
            old_orb.future = new_orb.id
            self._save_orb(old_orb)

        self._save_orb(new_orb)
        state.length += 1
        state.next_id += 1
        self._save_timeline_state(state)

    def change_history(self, event: Event, id: int):
        orb = self.get_orb(id)
        orb.event = event
        self._save_orb(orb)

    ''' TODO this doesn't update the metata to get a new start if you delete the start
    '''
    def burn_record(self, id: int):
        marked_orb = self.get_orb(id)
        if marked_orb.future is not None:
            future_orb = self.get_orb(marked_orb.future)
            future_orb.past = marked_orb.past
            self._save_orb(future_orb)
        else:
            # update end
            pass

        if marked_orb.past is not None:
            past_orb = self.get_orb(marked_orb.past)
            past_orb.future = marked_orb.future
            self._save_orb(past_orb)
        else:
            # update future
            pass

        os.remove(self._event_filename(id))
        self._data.add_length(-1)

    def reset_timeline(self):
        self.dnd_database.delete_collection(name="timeline")
        self.dnd_database.delete_collection(name="timeline_meta")


        self.timeline = self.dnd_database.get_or_create_collection(name="timeline")
        self.timeline_meta = self.dnd_database.get_or_create_collection(name="timeline_meta")
        self.initialise_timeline()


 #hell

    # def _save_orb(self, orb: OrbModel):

    #     file = open(self._event_filename(orb.id), "w")
    #     file.write(orb.model_dump_json())

    # def get_orb(self, id: int) -> OrbModel:
    #     with open(self._event_filename(id), "r") as f:
    #         return OrbModel(**(json.load(f)))
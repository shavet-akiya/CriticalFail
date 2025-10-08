import chromadb

from Structures import Event, validate
from Database import Database
from Event_Timeline_Graph import Timeline, OrbModel

# chroma_client = chromadb.HttpClient(host='localhost', port=8011)

# collection = chroma_client.get_or_create_collection(name="DND_Database")

timeline = Timeline()

test_id = 0

test_event_data = {
    "id":test_id,
    "title":"the tradgedy",
    "summary":"he died",
    "characters":["jim"],
    "locations":["USA","Jim's house"],
    "themes":["Loss","A Deep sense of mourning"],
    "tags":[],
    "session":1
}
test_event = Event(**test_event_data)

# test_orb = OrbModel(
#     id=test_id, 
#     event=test_event, 
#     past=41, 
#     future=43 # Pointers are saved in metadata
# )

# timeline._save_orb(test_orb)

timeline.reset_timeline()

timeline.write_future(test_event)

timeline.write_future(test_event)
timeline.write_future(test_event)

query_orb = timeline.get_orb(2)

print()
print()
print()
print(query_orb)

print(timeline.get_timeline_state())
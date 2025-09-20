from Database import Database
from Structures import Event, Character, Location
from Event_Timeline_Graph import Timeline


event = Event(id= 0, title= "the tradgedy", summary="he died", characters=["jim"], locations=["USA", "Jim's house"], themes=["Loss", "A Deep sense of mourning"], tags=[], session= 1)
# "{"summary":"he died","characters":["Jim"],"locations":["USA","Jim's house"],"themes":["Loss","A Deep sense of mourning"],"tags":[],"id":null}"
database = Database("server/src/database-structures/")
# database.timeline.write_future(event)

chars = database.get_table("characters")
jims_data = {
    "id": 5,
    "name": "jim", 
    "race": "humanoid",
    "armour_class": 3,
    "wis": 5,
    "int": 400,
    "class": "wizard"

}
jim = Character(**jims_data)

usa = Location(id= 3, name= "USA", description="str")

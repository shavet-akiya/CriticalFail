from Event_Timeline_Graph import *

class Database:
    def __init__(self):
        self.timeline: Timeline = Timeline()
        self.characters: {str: [str]} = {}
        self.places: {str: [str]} = {}
        self.themes: {str: [str]} = {}
        self.tags: [str] = []
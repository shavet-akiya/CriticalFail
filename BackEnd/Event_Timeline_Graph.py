from __future__ import annotations

class Event:
    def __init__(self, summary : str, characters: [str], places : [str], themes : [str], tags : [str]):
        self.summary = summary
        self.characters = characters
        self.places = places
        self.themes = themes
        self.tags = tags

class Orb:
    def __init__(self, event : Event, past : Orb = None, future : Orb = None):
        self.event: Event = event
        self.past: Orb = past
        self.future: Orb = future

    def write_future(self, event : Event):
        new_orb = Orb(event, self, self.future)
        if self.future:
            self.future.past = new_orb
        self.future = new_orb
        return new_orb

    def write_past(self, event : Event):
        new_orb = Orb(event, self.past, self)
        if self.past:
            self.past.future = new_orb
        self.past = new_orb
        return new_orb

class Timeline:
    def __init__(self, start: Orb = None):
        self.start: Orb = start
        self.current: Orb = self.start
        self.length: int = self.start is None

    def continue_journey(self, event : Event):
        self.length += 1

        if not self.start:
            self.start = Orb(event)
            self.current = self.start
            return

        self.current = self.current.write_future(event)
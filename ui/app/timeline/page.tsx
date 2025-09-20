"use client";

import React, { useEffect, useState, createContext, useContext } from "react";

// Define the Event interface
interface Event {
  id: number;
  summary: string;
  characters: [string];
  places: [string];
  themes: [string];
  tags: [string];
}

// Context setup
const EventsContext = createContext({
  events: [] as Event[],
  fetchEvents: () => {},
});

export default function Timeline() {
  const [events, setEvents] = useState<Event[]>([]);

  const fetchEvents = async () => {
    const response = await fetch("http://localhost:8000/event");
    const events = await response.json();
    setEvents(events.data);
  };

  useEffect(() => {
    fetchEvents();
  }, []);

  return (
    <EventsContext.Provider value={{ events, fetchEvents }}>
      <div className="container">
        <h2>Here begins the events</h2>
        <div className="events-list">
          {events.map((event: Event) => (
            <div className="event-card" key={event.id}>
              <p>Look, it's event {event.id}!!!</p>
              <dl>
                <dt><strong>Summary</strong></dt>
                <dd>- {event.summary}</dd>

                <dt><strong>Characters</strong></dt>
                <dd>- {event.characters.join(" and ")}</dd>

                <dt><strong>Places</strong></dt>
                <dd>- {event.places.join(", ")}</dd>

                <dt><strong>Themes</strong></dt>
                <dd>- {event.themes.join(", ")}</dd>
              </dl>
            </div>
          ))}
        </div>
      </div>
    </EventsContext.Provider>
  );
}
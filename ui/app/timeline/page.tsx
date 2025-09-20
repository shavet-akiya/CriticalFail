<<<<<<< HEAD
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
=======
// /app/timeline/page.tsx
'use client';

import React from 'react';
import { EventsProvider, useEvents } from './events';

function Timeline() {
  const { events } = useEvents();

  return (
    <div className="max-w-screen-xl pt-24">
      <h1 className="text-2xl font-bold mb-4">Timeline</h1>
      <ul className="space-y-4">
        {events.map((event) => (
          <li key={event.id} className="border-l-4 border-blue-500 pl-4">
            <p className="text-lg font-semibold">{event.summary}</p>
            <p className="text-sm text-gray-600">
              Characters: {event.characters.join(", ")}
            </p>
            <p className="text-sm text-gray-600">
              Places: {event.places.join(", ")}
            </p>
            <p className="text-sm text-gray-600">
              Themes: {event.themes.join(", ")}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Page() {
  return (
    <EventsProvider>
      <Timeline />
    </EventsProvider>
  );
>>>>>>> f0d92fd05c4a232b432e5e360ddad04ac4259b5b
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
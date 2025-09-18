'use client';

import React, { useEffect, useState, createContext, useContext, ReactNode } from "react";

interface Event {
  id: number;
  summary: string;
  characters: [string];
  places: [string];
  themes: [string];
  tags: [string];
}

interface EventsContextType {
  events: Event[];
  fetchEvents: () => void;
}

export const EventsContext = createContext<EventsContextType>({
  events: [],
  fetchEvents: () => {}
});


export function EventsProvider({ children }: { children: ReactNode }) {
  const [events, setEvents] = useState<Event[]>([]);

  const fetchEvents = async () => {
    const response = await fetch("http://localhost:3000/event");
    const json = await response.json();
    setEvents(json.data);
  };

  useEffect(() => {
    fetchEvents();
  }, []);

  return (
    <EventsContext.Provider value={{ events, fetchEvents }}>
      {children}
    </EventsContext.Provider>
  );
}


export const useEvents = () => useContext(EventsContext);

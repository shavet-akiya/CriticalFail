"use client";

import { createContext, useContext, ReactNode, useState } from "react";
interface Event {
    event_id: string;
    session_id: string;
    campaign_id?: string;
    timeline_order?: number;
    event?: string;
    event_summary?: string;
    participants?: string[];
    location?: string;
    event_tags?: string[];
    type?: string;
}

interface EventContextType {
    currentEvent: Event | null;
    setCurrentEvent: (e: Event) => void;
}

const EventContext = createContext<EventContextType | undefined>(undefined);

export function EventProvider({ children }: { children: ReactNode }) {
    const [currentEvent, setCurrentEvent] = useState<Event | null>(null);

    return (
        <EventContext.Provider value={{ currentEvent, setCurrentEvent }}>
            {children}
        </EventContext.Provider>
    );
}

export function useEvent() {
    const context = useContext(EventContext);
    if (!context) throw new Error("useEvent must be used within EventProvider");
    return context;
}

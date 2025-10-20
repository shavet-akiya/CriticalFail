"use client";

import React from "react";

interface Character {
    character_id: string;
    name: string;
    class: string;
    race?: string;
    HP: number;
    AC: number;
    STR: number;
    DEX: number;
    CON: number;
    INT: number;
    WIS: number;
    CHA: number;
}

interface Location {
    location_id: string;
    location_name: string;
    description: string;
}

interface Event {
    event_id: string;
    event: string;
    event_summary: string;
    participants?: string;
    location?: string;
    event_tags?: string;
}

interface Session {
    session_id: string;
    processed_at?: string;
    characters?: Character[];
    locations?: Location[];
    events?: Event[];
}

interface SessionCardProps {
    session: Session;
    formatSessionDate: (id: string) => string;
}

export default function SessionCard({ session, formatSessionDate }: SessionCardProps) {
    return (
        <div
            key={session.session_id}
            className="bg-[#e0d6cb] p-6 rounded-lg shadow-md hover:shadow-lg transition-all duration-200"
        >
            <p className="font-bold mb-2 text-lg">
                {formatSessionDate(session.session_id)} Session
            </p>
            <p className="mb-4 text-sm text-gray-500 italic">
                Processed at: {session.processed_at || "N/A"}
            </p>

            <div className="mb-4">
                <h3 className="font-semibold text-lg">Characters</h3>
                {session.characters?.length ? (
                    <ul className="list-disc list-inside text-sm mt-1 space-y-1">
                        {session.characters.map((c) => (
                            <li key={c.character_id}>
                                {c.name} ({c.class}, {c.race || "unknown"}) – HP:{c.HP}, AC:{c.AC},{" "}
                                STR:{c.STR}, DEX:{c.DEX}, CON:{c.CON}, INT:{c.INT}, WIS:{c.WIS}, CHA:{c.CHA}
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="text-gray-500 text-sm">No characters.</p>
                )}
            </div>

            <div className="mb-4">
                <h3 className="font-semibold text-lg">Locations</h3>
                {session.locations?.length ? (
                    <ul className="list-disc list-inside text-sm mt-1 space-y-1">
                        {session.locations.map((l) => (
                            <li key={l.location_id}>
                                {l.location_name}: {l.description}
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="text-gray-500 text-sm">No locations.</p>
                )}
            </div>

            <div>
                <h3 className="font-semibold text-lg">Events</h3>
                {session.events?.length ? (
                    <ul className="list-disc list-inside text-sm mt-1 space-y-1">
                        {session.events.map((e) => (
                            <li key={e.event_id}>
                                <strong>{e.event}</strong> – {e.event_summary}
                                {e.participants && <> (Participants: {e.participants})</>}
                                {e.location && <> @ {e.location}</>}
                                {e.event_tags && <> [{e.event_tags}]</>}
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="text-gray-500 text-sm">No events.</p>
                )}
            </div>
        </div>
    );
}

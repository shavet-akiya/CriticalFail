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
    npc?: boolean;
    session_id: string;
    campaign_id: string;
    imageURL?: string;
}

interface Session {
    session_id: string;
    processed_at?: string;
    characters?: Character[];
    locations?: Location[];
    events?: Event[];
    campaign_id: string;
}

interface Location {
    location_id: string;
    location_name: string;
    location_description: string;
}

interface Event {
    event_id: string;
    event: string;
    event_summary: string;
    participants?: string;
    location?: string;
    event_tags?: string;
}

interface SessionCardProps {
    session: Session;
    formatSessionDate: (id: string) => string;
}

function formatProcessedAt(isoString?: string) {
    if (!isoString) return "N/A";
    const [datePart, timePart] = isoString.split("T");
    const time = timePart.replace("Z", "");
    const [year, month, day] = datePart.split("-");
    return `${day}/${month}/${year} ${time}`;
}

export default function SessionCard({
    session,
    formatSessionDate,
}: SessionCardProps) {
    return (
        <div className="p-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 bg-white-colour">
            <p className="font-bold mb-2 text-lg obsidian-colour">
                {formatSessionDate(session.session_id)} Session
            </p>
            <p className="mb-4 text-sm text-gray-500 italic">
                Created at: {formatProcessedAt(session.processed_at) || "N/A"}
            </p>

            <div className="mb-6">
                <h3 className="font-semibold text-lg obsidian-colour mb-2">
                    Characters
                </h3>
                {session.characters?.length ? (
                    <div className="flex flex-wrap gap-2">
                        {session.characters.map((c) => (
                            <div
                                key={c.character_id}
                                className="bg-purple-50 border border-purple-200 rounded-lg p-2 text-sm flex flex-col items-center w-42"
                            >
                                <p className="font-semibold text-sm text-gray-800">
                                    {c.name}
                                </p>
                                <p className="text-xs text-gray-500">
                                    {c.class}, {c.race || "unknown"}
                                </p>
                                <div className="flex gap-1 text-xs mt-1">
                                    <span className="bg-gray-200 rounded px-1 text-gray-500">
                                        HP: {c.HP}
                                    </span>
                                    <span className="bg-gray-200 rounded px-1 text-gray-500">
                                        AC: {c.AC}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-gray-500 text-sm">No characters.</p>
                )}
            </div>

            <div className="mb-6">
                <h3 className="font-semibold text-lg obsidian-colour mb-2">
                    Locations
                </h3>
                {session.locations?.length ? (
                    <div className="flex flex-wrap gap-2">
                        {session.locations.map((l) => (
                            <div
                                key={l.location_id}
                                className="bg-green-50 border-l-4 border-green-400 p-2 rounded shadow-sm text-sm"
                            >
                                <p className="font-semibold text-gray-800">
                                    {l.location_name}
                                </p>
                                <p className="text-gray-500">
                                    {l.location_description}
                                </p>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-gray-500 text-sm">No locations.</p>
                )}
            </div>

            <div>
                <h3 className="font-semibold text-lg obsidian-colour mb-2">
                    Events
                </h3>
                {session.events?.length ? (
                    <div className="flex flex-col gap-4">
                        {session.events.map((e) => (
                            <div
                                key={e.event_id}
                                className="border-l-4 border-blue-400 bg-blue-50 p-3 rounded shadow-sm"
                            >
                                <h4 className="font-semibold text-sm obsidian-colour mb-1">
                                    {e.event}
                                </h4>
                                <p className="text-sm obsidian-colour mb-1">
                                    {e.event_summary}
                                </p>
                                <div className="flex flex-wrap gap-1 text-xs">
                                    {e.participants && (
                                        <span className="bg-blue-100 rounded px-1 text-gray-500">
                                            {e.participants}
                                        </span>
                                    )}
                                    {e.location && (
                                        <span className="bg-blue-100 rounded px-1 text-gray-500">
                                            {e.location}
                                        </span>
                                    )}
                                    {e.event_tags && (
                                        <span className="bg-blue-100 rounded px-1 text-gray-500">
                                            {e.event_tags}
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-gray-500 text-sm">No events.</p>
                )}
            </div>
        </div>
    );
}

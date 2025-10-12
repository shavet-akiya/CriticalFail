"use client";

import { useEffect, useState } from "react";
import type { Event, Character } from "@/types/types";

type EventCardProps = {
    event: Event;
};

export default function EventCard({ event }: EventCardProps) {
    const [characters, setCharacters] = useState<Character[]>([]);
    const [error, setError] = useState<string | null>(null);

    // fetch characters from your DB API
    useEffect(() => {
        fetch("/api/characters", { cache: "no-store" })
            .then((res) => {
                if (!res.ok)
                    throw new Error(
                        `Failed to fetch characters: ${res.status}`
                    );
                return res.json();
            })
            .then((data) => setCharacters(data.characters ?? []))
            .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    }, []);

    // resolve character IDs to names
    const eventCharacters: Character[] = event.characterIds
        .map((id) => characters.find((c) => c.characterId === id))
        .filter((c): c is Character => c !== undefined);
    return (
        <div className="card bg-base-100 border shadow-sm mb-4">
            <div className="card-body">
                <h2 className="card-title">{event.eventSummary}</h2>

                {/* Characters */}
                <div className="mt-2">
                    <strong>Characters:</strong>{" "}
                    {eventCharacters.length > 0
                        ? eventCharacters.map((c) => c.name).join(", ")
                        : "None"}
                </div>

                {/* Locations */}
                <div className="mt-2">
                    <strong>Locations:</strong> {event.locationIds.join(", ")}
                </div>

                {/* Tags */}
                <div className="flex gap-2 mt-3 flex-wrap">
                    {event.eventTags.map((tag, i) => (
                        <span key={i} className="badge badge-outline">
                            {tag}
                        </span>
                    ))}
                </div>

                {error && <div className="text-error mt-2">{error}</div>}
            </div>
        </div>
    );
}

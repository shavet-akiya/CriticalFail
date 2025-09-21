// ui/components/eventCard.tsx
import { Event, Character } from "@/types/types";
import { characters } from "@/types/mockData"; // adjust path if needed

type EventCardProps = {
    event: Event;
};

export default function EventCard({ event }: EventCardProps) {
    // resolve character IDs to names
    const eventCharacters: Character[] = event.characterIDs
        .map((id) => characters.find((c) => c.id === id))
        .filter((c): c is Character => c !== undefined);

    return (
        <div className="card bg-base-100 border shadow-sm mb-4">
            <div className="card-body">
                <h2 className="card-title">{event.summary}</h2>
                { /*<p className="text-sm text-gray-500">Session {event.sessionId}</p> */}

                {/* Characters */}
                <div className="mt-2">
                    <strong>Characters:</strong>{" "}
                    {eventCharacters.length > 0
                        ? eventCharacters.map((c) => c.name).join(", ")
                        : "None"}
                </div>

                {/* Locations (still showing IDs until we have mockData for locations) */}
                <div className="mt-2">
                    <strong>Locations:</strong>{" "}
                    {event.locationIDs.join(", ")}
                </div>

                {/* Tags */}
                <div className="flex gap-2 mt-3 flex-wrap">
                    {event.tags.map((tag, i) => (
                        <span key={i} className="badge badge-outline">
                            {tag}
                        </span>
                    ))}
                </div>

                {/* Themes */}
                <div className="flex gap-2 mt-2 flex-wrap">
                    {event.themes.map((theme, i) => (
                        <span key={i} className="badge badge-secondary">
                            {theme}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
}

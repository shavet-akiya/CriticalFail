import React from "react";
import { SessionCharacter, SessionLocation } from "@/helpers/types";

interface Event {
    event: string;
    event_summary?: string;
}

interface SessionMetadata {
    session_id?: string;
    campaign_id?: string;
    characters?: SessionCharacter[];
    locations?: SessionLocation[];
    events?: Event[];
}

interface Session {
    id: string;
    metadata?: SessionMetadata;
    document?: string;
}

interface SessionCardProps {
    session: Session;
    baseUrl?: string;
    fetchSessions: () => Promise<void>;
    setError: (msg: string) => void;
}

const AltSessionCard: React.FC<SessionCardProps> = ({
    session,
    baseUrl,
    fetchSessions,
    setError,
}) => {
    const handleDelete = async () => {
        if (!confirm(`Delete session ${session.metadata?.session_id}?`)) return;

        try {
            const res = await fetch(
                `${baseUrl}/sessions/${encodeURIComponent(
                    session.metadata?.session_id ?? session.id
                )}`,
                { method: "DELETE" }
            );
            if (!res.ok) {
                const msg = await res.text();
                throw new Error(`Delete failed: ${res.status} ${msg}`);
            }
            await fetchSessions();
        } catch (err: any) {
            setError(err.message);
        }
    };

    return (
        <li className="p-4 bg-gray-50 rounded shadow text-black flex flex-col gap-2">
            <div className="flex justify-between items-center">
                <div>
                    <p>
                        <strong>Session ID:</strong> {session.metadata?.session_id ?? "N/A"}
                    </p>
                    <p>
                        <strong>Campaign:</strong> {session.metadata?.campaign_id ?? "N/A"}
                    </p>
                </div>
                <button
                    onClick={handleDelete}
                    className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
                >
                    Delete
                </button>
            </div>

            <p>
                <strong>Characters:</strong>{" "}
                {session.metadata?.characters?.length
                    ? session.metadata.characters.map((c) => c.name).join(", ")
                    : "None"}
            </p>
            <p>
                <strong>Locations:</strong>{" "}
                {session.metadata?.locations?.length
                    ? session.metadata.locations
                        .map((l) => l.location_name || l.name)
                        .join(", ")
                    : "None"}
            </p>
            <p>
                <strong>Events:</strong>{" "}
                {session.metadata?.events?.length
                    ? session.metadata.events
                        .map((e) => `${e.event} — ${e.event_summary ?? ""}`)
                        .join("; ")
                    : "None"}
            </p>
            <p className="mt-2">
                <strong>Summary:</strong>
            </p>
            <p className="text-sm text-black">{session.document}</p>
        </li>
    );
};

export default AltSessionCard;

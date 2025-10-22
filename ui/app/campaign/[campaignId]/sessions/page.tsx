"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation"; // for app directory
import Loading from "@/components/Loading";

interface Character {
    name: string;
}

interface Location {
    name?: string;
    location_name?: string;
}

interface Event {
    event: string;
    event_summary?: string;
}

interface SessionMetadata {
    session_id: string;
    campaign_id?: string;
    characters?: Character[];
    locations?: Location[];
    events?: Event[];
}

interface Session {
    id: string;
    document: string;
    metadata?: SessionMetadata;
}

export default function SessionList() {
    const [sessions, setSessions] = useState<Session[]>([]);
    const [fetching, setFetching] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [latestSession] = useState<any>(null);

    const params = useParams();
    const campaignId = params?.campaignId;

    // Use environment variable with fallback
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    async function resetDatabase() {
        try {
            const res = await fetch(`${baseUrl}/sessions`, {
                method: "DELETE",
            });
            if (!res.ok) throw new Error(`RESET failed: ${res.status}`);
            await fetchSessions();
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }

    async function fetchSessions() {
        setFetching(true);
        setError(null);
        try {
            // Add campaignId as query param
            const url = campaignId
                ? `${baseUrl}/sessions?campaign_id=${campaignId}`
                : `${baseUrl}/sessions`;

            const res = await fetch(url, { cache: "no-store" });
            if (!res.ok) throw new Error(`GET failed: ${res.status}`);
            const data = await res.json();

            const mapped = data.documents.map((doc: string, i: number) => ({
                id: data.ids[i],
                document: doc,
                metadata: data.metadatas[i],
            }));

            setSessions(mapped);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setFetching(false);
        }
    }

    useEffect(() => {
        fetchSessions();
    }, [campaignId]);

    return (
        <div className="min-h-screen p-6 bg-gray-100 text-black">
            <h1 className="text-4xl font-bold text-center mb-8">
                Dungeons & Dragons AI Sessions
            </h1>
            <button
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
                onClick={resetDatabase}
            >
                Delete All Sessions
            </button>

            {/* Past Sessions */}
            <div className="bg-white shadow rounded p-6">
                <h2 className="text-xl font-semibold mb-4">Past Sessions</h2>
                {fetching ? (
                    <Loading />
                ) : sessions.length === 0 ? (
                    <p className="text-black">No sessions available.</p>
                ) : (
                    <ul className="space-y-4">
                        {sessions.map((s) => (
                            <li
                                key={s.id}
                                className="p-4 bg-gray-50 rounded shadow text-black flex flex-col gap-2"
                            >
                                <div className="flex justify-between items-center">
                                    <div>
                                        <p>
                                            <strong>Session ID:</strong>{" "}
                                            {s.metadata?.session_id ?? "N/A"}
                                        </p>
                                        <p>
                                            <strong>Campaign:</strong>{" "}
                                            {s.metadata?.campaign_id ?? "N/A"}
                                        </p>
                                    </div>
                                    <button
                                        onClick={async () => {
                                            if (
                                                !confirm(
                                                    `Delete session ${s.metadata?.session_id}?`
                                                )
                                            )
                                                return;
                                            try {
                                                const res = await fetch(
                                                    `${baseUrl}/sessions/${encodeURIComponent(
                                                        s.metadata
                                                            ?.session_id ?? s.id
                                                    )}`,
                                                    { method: "DELETE" }
                                                );
                                                if (!res.ok) {
                                                    const msg =
                                                        await res.text();
                                                    throw new Error(
                                                        `Delete failed: ${res.status} ${msg}`
                                                    );
                                                }
                                                // Refresh sessions after successful delete
                                                await fetchSessions();
                                            } catch (err: any) {
                                                setError(err.message);
                                            }
                                        }}
                                        className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
                                    >
                                        Delete
                                    </button>
                                </div>

                                <p>
                                    <strong>Characters:</strong>{" "}
                                    {s.metadata?.characters?.length
                                        ? s.metadata.characters
                                            .map((c) => c.name)
                                            .join(", ")
                                        : "None"}
                                </p>
                                <p>
                                    <strong>Locations:</strong>{" "}
                                    {s.metadata?.locations?.length
                                        ? s.metadata.locations
                                            .map(
                                                (l) =>
                                                    l.location_name || l.name
                                            )
                                            .join(", ")
                                        : "None"}
                                </p>
                                <p>
                                    <strong>Events:</strong>{" "}
                                    {s.metadata?.events?.length
                                        ? s.metadata.events
                                            .map(
                                                (e) =>
                                                    `${e.event} — ${e.event_summary ?? ""
                                                    }`
                                            )
                                            .join("; ")
                                        : "None"}
                                </p>
                                <p className="mt-2">
                                    <strong>Summary:</strong>
                                </p>
                                <p className="text-sm text-black">
                                    {s.document}
                                </p>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

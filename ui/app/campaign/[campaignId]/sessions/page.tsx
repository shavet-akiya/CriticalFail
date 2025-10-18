"use client";

import { useEffect, useState } from "react";

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
    const [transcript, setTranscript] = useState("");
    const [sessions, setSessions] = useState<Session[]>([]);
    const [posting, setPosting] = useState(false);
    const [fetching, setFetching] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [latestSession, setLatestSession] = useState<any>(null);

    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    async function submitTranscript() {
        if (!transcript.trim()) return;
        setPosting(true);
        setError(null);
        try {
            const res = await fetch(`${baseUrl}/sessions`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ transcript }),
            });
            if (!res.ok) throw new Error(`POST failed: ${res.status}`);
            const data = await res.json();
            setLatestSession(data);
            setTranscript("");
            await fetchSessions();
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setPosting(false);
        }
    }

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
            const res = await fetch(`${baseUrl}/sessions`, {
                cache: "no-store",
            });
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
    }, []);

    return (
        <div className="min-h-screen p-6 bg-gray-100">
            <h1 className="text-4xl font-bold text-center mb-8">
                Dungeons & Dragons AI Sessions
            </h1>

            <button
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
                onClick={resetDatabase}
            >
                Delete All Sessions
            </button>
            {/* Latest Session */}
            {latestSession && (
                <div className="bg-white shadow rounded p-6 mb-8">
                    <h2 className="text-xl font-semibold mb-2">
                        Latest Session
                    </h2>
                    <pre className="bg-gray-100 p-2 rounded overflow-x-auto">
                        {JSON.stringify(latestSession, null, 2)}
                    </pre>
                </div>
            )}

            {/* Past Sessions */}
            <div className="bg-white shadow rounded p-6">
                <h2 className="text-xl font-semibold mb-4">Past Sessions</h2>
                {fetching ? (
                    <p className="text-gray-600">Loading sessions…</p>
                ) : sessions.length === 0 ? (
                    <p className="text-gray-600">No sessions available.</p>
                ) : (
                    <ul className="space-y-4">
                        {sessions.map((s) => (
                            <li
                                key={s.id}
                                className="p-4 bg-gray-50 rounded shadow"
                            >
                                <p>
                                    <strong>Session ID:</strong>{" "}
                                    {s.metadata?.session_id ?? "N/A"}
                                </p>
                                <p>
                                    <strong>Campaign:</strong>{" "}
                                    {s.metadata?.campaign_id ?? "N/A"}
                                </p>
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
                                                      `${e.event} — ${
                                                          e.event_summary ?? ""
                                                      }`
                                              )
                                              .join("; ")
                                        : "None"}
                                </p>
                                <p className="mt-2">
                                    <strong>Summary:</strong>
                                </p>
                                <p className="text-sm text-gray-700">
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

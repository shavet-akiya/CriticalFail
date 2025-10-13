"use client";

import { useEffect, useState } from "react";

export default function Home() {
    const [transcript, setTranscript] = useState("");
    const [session, setSession] = useState<any | null>(null);
    const [sessions, setSessions] = useState<any[]>([]);
    const [posting, setPosting] = useState(false);
    const [fetching, setFetching] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const baseUrl = "/api";

    async function submitTranscript() {
        if (!transcript.trim() || posting) return;
        setPosting(true);
        setError(null);

        try {
            // Optional: show temporary "processing" session
            setSession({ status: "processing", transcript });

            const res = await fetch(`${baseUrl}/sessions`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ transcript }),
            });

            if (!res.ok) throw new Error(`POST failed: ${res.status}`);

            const newSession = await res.json();

            // Replace placeholder with final session
            setSession(newSession);

            // Append to past sessions
            setSessions((prev) => [
                {
                    id: newSession.id || crypto.randomUUID(),
                    document: newSession.document,
                    metadata: newSession.metadata,
                },
                ...prev,
            ]);

            setTranscript("");
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setPosting(false);
        }
    }

    async function resetDatabase() {
        try {
            const res = await fetch(`${baseUrl}/reset`, { method: "DELETE" });
            if (!res.ok) throw new Error(`RESET failed: ${res.status}`);
            await fetchSessions();
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }

    async function fetchSessions() {
        setFetching(true);
        try {
            const res = await fetch(`${baseUrl}/sessions`, {
                cache: "no-store",
            });
            if (!res.ok) throw new Error(`GET failed: ${res.status}`);
            const data = await res.json();
            setSessions(
                data.documents.map((doc: string, i: number) => ({
                    id: data.ids[i],
                    document: doc,
                    metadata: data.metadatas[i],
                }))
            );
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
        <div className="flex flex-col p-6 space-y-6 bg-white min-h-screen">
            <h1 className="text-4xl font-bold text-center text-black mb-6">
                Dungeons & Dragons AI Processor
            </h1>

            {/* Submit Transcript */}
            <div className="card w-full bg-gray-900 shadow-xl p-6">
                <h2 className="card-title mb-4 text-white">
                    Submit Transcript
                </h2>
                <textarea
                    className="textarea textarea-bordered w-full h-40 bg-gray-800 text-white"
                    placeholder="Paste your D&D transcript here..."
                    value={transcript}
                    onChange={(e) => setTranscript(e.target.value)}
                    disabled={posting}
                />
                <div className="card-actions justify-end mt-4">
                    <button
                        className="btn btn-primary rounded-xl"
                        onClick={submitTranscript}
                        disabled={posting || !transcript.trim()}
                    >
                        {posting ? "Processing…" : "Submit"}
                    </button>
                </div>
                {error && <p className="text-sm text-error mt-2">{error}</p>}
            </div>

            {/* Latest Session */}
            {session && (
                <div className="card bg-gray-900 shadow-xl p-6">
                    <h2 className="card-title text-white">Latest Session</h2>
                    <pre className="mt-2 p-2 rounded text-xs text-white bg-gray-800 whitespace-pre-wrap break-words">
                        {JSON.stringify(session, null, 2)}
                    </pre>
                </div>
            )}

            {/* Past Sessions */}
            <div className="card bg-gray-900 shadow-xl p-6">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="card-title text-white">Past Sessions</h2>
                    <button
                        className="btn btn-error rounded-xl"
                        onClick={resetDatabase}
                    >
                        Delete All Sessions
                    </button>
                </div>

                {fetching ? (
                    <p className="text-gray-400">Loading sessions…</p>
                ) : (
                    <ul className="list-none mt-2 space-y-4">
                        {sessions.map((s) => (
                            <li
                                key={s.id}
                                className="p-4 bg-gray-800 rounded shadow-sm text-white"
                            >
                                <strong>Session Code:</strong>{" "}
                                {s.metadata.session_id}
                                <br />
                                <strong>Campaign:</strong>{" "}
                                {s.metadata.campaign_id || "Unassigned"}
                                <br />
                                <strong>Characters:</strong>{" "}
                                {s.metadata.characters
                                    ?.map((c: any) => c.name)
                                    .join(", ") || "None"}
                                <br />
                                <strong>Locations:</strong>{" "}
                                {s.metadata.locations
                                    ?.map((l: any) => l.location_name || l.name)
                                    .join(", ") || "None"}
                                <br />
                                <strong>Events:</strong>{" "}
                                {s.metadata.events
                                    ?.map(
                                        (e: any) =>
                                            `${e.event} — ${e.event_summary}`
                                    )
                                    .join("; ") || "None"}
                                <br />
                                <strong>Summary:</strong>
                                <p className="mt-2 text-sm text-gray-300">
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

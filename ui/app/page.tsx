"use client";

import { useEffect, useState } from "react";

export default function Home() {
    const [transcript, setTranscript] = useState("");
    const [session, setSession] = useState<any | null>(null);
    const [sessions, setSessions] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const baseUrl = "/api";

    async function submitTranscript() {
        if (!transcript.trim()) return;
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(`${baseUrl}/sessions`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ transcript }),
            });
            if (!res.ok) throw new Error(`POST failed: ${res.status}`);
            const data = await res.json();
            setSession(data);
            setTranscript("");
            await fetchSessions();
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(false);
        }
    }

    async function fetchSessions() {
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
        }
    }

    // 🔥 New: delete session
    async function deleteSession(sessionCode: string) {
        try {
            const res = await fetch(`${baseUrl}/sessions`, {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_code: sessionCode }),
            });
            if (!res.ok) throw new Error(`DELETE failed: ${res.status}`);
            await fetchSessions(); // refresh list
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
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
            <h2 className="text-xl text-center text-gray-700 mb-6">
                For testing transcripts and AI processing.
            </h2>

            <div className="card w-full bg-gray-900 shadow-xl p-6">
                <h2 className="card-title mb-4 text-white">
                    Submit Transcript
                </h2>
                <textarea
                    className="textarea textarea-bordered w-full h-40 bg-gray-800 text-white"
                    placeholder="Paste your D&D transcript here..."
                    value={transcript}
                    onChange={(e) => setTranscript(e.target.value)}
                    disabled={loading}
                />
                <div className="card-actions justify-end mt-4">
                    <button
                        className="btn btn-primary rounded-xl"
                        onClick={submitTranscript}
                        disabled={loading || !transcript.trim()}
                    >
                        {loading ? "Processing…" : "Submit"}
                    </button>
                </div>
                {error && <p className="text-sm text-error mt-2">{error}</p>}
            </div>

            {session && (
                <div className="card bg-gray-900 shadow-xl p-6">
                    <h2 className="card-title text-white">Latest Session</h2>
                    <pre className="mt-2 p-2 rounded text-xs text-white bg-gray-800 whitespace-pre-wrap break-words">
                        {JSON.stringify(session, null, 2)}
                    </pre>
                </div>
            )}

            <div className="card bg-gray-900 shadow-xl p-6">
                <h2 className="card-title text-white">Past Sessions</h2>
                <ul className="list-none mt-2 space-y-4">
                    {sessions.map((s) => (
                        <li
                            key={s.id}
                            className="p-4 bg-gray-800 rounded shadow-sm text-white"
                        >
                            <div className="flex justify-between items-start">
                                <div>
                                    <strong>Session Code:</strong>{" "}
                                    {s.metadata.session_code}
                                    <br />
                                    <strong>Campaign:</strong>{" "}
                                    {s.metadata.campaign_id || "Unassigned"}
                                    <br />
                                    <strong>Characters:</strong>{" "}
                                    {s.metadata.characters?.join(", ") ||
                                        "None"}
                                    <br />
                                    <strong>Locations:</strong>{" "}
                                    {s.metadata.locations?.join(", ") || "None"}
                                    <br />
                                    <strong>Events:</strong>{" "}
                                    {s.metadata.events?.join(", ") || "None"}
                                    <br />
                                    <strong>Summary:</strong>
                                    <p className="mt-2 text-sm text-gray-300">
                                        {s.document}
                                    </p>
                                </div>
                                {/* 🔥 Delete button */}
                                <button
                                    className="btn btn-error btn-sm ml-4"
                                    onClick={() =>
                                        deleteSession(s.metadata.session_code)
                                    }
                                >
                                    Delete
                                </button>
                            </div>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}

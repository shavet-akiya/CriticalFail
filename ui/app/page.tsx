"use client";

import { useEffect, useState } from "react";

export default function Home() {
    const [transcript, setTranscript] = useState("");
    const [session, setSession] = useState<any | null>(null);
    const [sessions, setSessions] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Same-origin proxy: Next.js rewrites /api/* → backend
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
            setSessions(data);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }

    useEffect(() => {
        fetchSessions();
    }, []);

    return (
        <div className="flex flex-col p-6 space-y-6 bg-white min-h-screen">
            {/* Main Page Title */}
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
                            key={s.session_code}
                            className="p-4 bg-gray-800 rounded shadow-sm text-white"
                        >
                            <strong>Session Code:</strong> {s.session_code}
                            <pre className="mt-2 p-2 rounded text-xs text-white bg-gray-700 whitespace-pre-wrap break-words">
                                {JSON.stringify(s, null, 2)}
                            </pre>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}

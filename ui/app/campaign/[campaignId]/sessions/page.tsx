"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

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
    id: string; // maps to the API 'id'
    document: string;
    metadata?: SessionMetadata;
}

export default function SessionList() {
    const [sessions, setSessions] = useState<Session[]>([]);
    const [fetching, setFetching] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [editingSession, setEditingSession] = useState<Session | null>(null);
    const [editedDocument, setEditedDocument] = useState<string>("");

    const params = useParams();
    const campaignId = params?.campaignId;
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    async function fetchSessions() {
        setFetching(true);
        setError(null);
        try {
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

    async function saveDocument(session: Session) {
        try {
            const res = await fetch(
                `${baseUrl}/sessions/${encodeURIComponent(session.id)}`,
                {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ document: editedDocument }),
                }
            );
            if (!res.ok) throw new Error(`Update failed: ${res.status}`);

            setSessions((prev) =>
                prev.map((s) =>
                    s.id === session.id ? { ...s, document: editedDocument } : s
                )
            );
            setEditingSession(null);
        } catch (err: any) {
            setError(err.message);
        }
    }

    async function deleteSession(session: Session) {
        if (
            !confirm(
                `Delete session ${session.metadata?.session_id ?? session.id}?`
            )
        )
            return;
        try {
            const res = await fetch(
                `${baseUrl}/sessions/${encodeURIComponent(session.id)}`,
                {
                    method: "DELETE",
                }
            );
            if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
            setSessions((prev) => prev.filter((s) => s.id !== session.id));
        } catch (err: any) {
            setError(err.message);
        }
    }

    return (
        <div className="min-h-screen p-6 bg-gray-100 text-black">
            <h1 className="text-4xl font-bold text-center mb-8">
                Dungeons & Dragons AI Sessions
            </h1>

            <div className="bg-white shadow rounded p-6">
                <h2 className="text-xl font-semibold mb-4">Past Sessions</h2>
                {fetching ? (
                    <p>Loading sessions…</p>
                ) : sessions.length === 0 ? (
                    <p>No sessions available.</p>
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
                                            {s.metadata?.session_id ?? s.id}
                                        </p>
                                        <p>
                                            <strong>Campaign:</strong>{" "}
                                            {s.metadata?.campaign_id ?? "N/A"}
                                        </p>
                                    </div>
                                    <div className="flex gap-2">
                                        <button
                                            onClick={() => {
                                                setEditingSession(s);
                                                setEditedDocument(s.document);
                                            }}
                                            className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                                        >
                                            Edit Document
                                        </button>
                                        <button
                                            onClick={() => deleteSession(s)}
                                            className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
                                        >
                                            Delete
                                        </button>
                                    </div>
                                </div>

                                <p>
                                    <strong>Summary:</strong>
                                </p>
                                <p className="text-sm">{s.document}</p>
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            {/* Modal for editing document */}
            {editingSession && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
                    <div className="bg-white rounded shadow-lg p-6 w-full max-w-lg">
                        <h3 className="text-xl font-semibold mb-4">
                            Edit Document for Session{" "}
                            {editingSession.metadata?.session_id ??
                                editingSession.id}
                        </h3>
                        <textarea
                            className="w-full border p-2 rounded text-black"
                            rows={8}
                            value={editedDocument}
                            onChange={(e) => setEditedDocument(e.target.value)}
                        />
                        <div className="flex justify-end gap-2 mt-4">
                            <button
                                onClick={() => setEditingSession(null)}
                                className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => saveDocument(editingSession)}
                                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                            >
                                Save
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {error && <p className="text-red-500 mt-4">{error}</p>}
        </div>
    );
}

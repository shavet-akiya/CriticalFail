"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { formatSessionDate } from "@/helpers/helper_functions";
import Loading from "@/components/Loading";
import { useCampaign } from "@/contexts/CampaignContext";


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
    const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
    const [editedDocument, setEditedDocument] = useState<string>("");

    const params = useParams();
    const { selectedCampaign } = useCampaign();

    const campaignId = selectedCampaign?.campaign_id;
    const campaignName = selectedCampaign?.campaign_name;


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

            // Update the session locally
            setSessions((prev) =>
                prev.map((s) =>
                    s.id === session.id ? { ...s, document: editedDocument } : s
                )
            );

            // Exit edit mode after saving
            setEditingSessionId(null);
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

    if (fetching) return <Loading />

    return (
        <div className="min-h-screen p-6 obsidian-colour">
            <h1 className="text-4xl text-center mb-4 ">
                {campaignName} sessions
            </h1>

            <div className="p-6">
                {fetching ? (
                    <p>Loading sessions…</p>
                ) : sessions.length === 0 ? (
                    <p>No sessions available.</p>
                ) : (
                    <ul className="space-y-4">
                        {sessions.map((s) => {
                            const isEditing = editingSessionId === s.id;
                            return (
                                <li
                                    key={s.id}
                                    className="p-8 bg-gray-50 rounded shadow text-black flex flex-col gap-4"
                                >
                                    <div className="flex justify-between items-center ">
                                        <div>
                                            <p className="text-xl">
                                                <strong>Session:</strong>{" "}
                                                {formatSessionDate(s.metadata?.session_id ?? s.id)}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex flex-row justify-between items-center">
                                        <p className="text-lg font-semibold">Summary Notes</p>

                                    </div>

                                    {isEditing ? (
                                        <textarea
                                            className="w-full border p-2 rounded text-black"
                                            rows={6}
                                            value={editedDocument}
                                            onChange={(e) =>
                                                setEditedDocument(e.target.value)
                                            }
                                        />
                                    ) : (
                                        <p className="text-md whitespace-pre-wrap">
                                            {s.document}
                                        </p>
                                    )}

                                    {isEditing ? (
                                        <div className="flex gap-2 justify-end">
                                            <button
                                                onClick={() => saveDocument(s)}
                                                className="btn px-3 py-1 bg-green-600 white-colour text-sm rounded hover:bg-green-700"
                                            >
                                                Save
                                            </button>
                                            <button
                                                onClick={() => setEditingSessionId(null)}
                                                className="btn px-3 py-1 bg-gray-400 white-colour text-sm rounded hover:bg-gray-500"
                                            >
                                                Cancel
                                            </button>
                                        </div>
                                    ) : (
                                        <div className="flex flex-row gap-2 justify-end">
                                            <button
                                                onClick={() => {
                                                    setEditingSessionId(s.id);
                                                    setEditedDocument(s.document);
                                                }}
                                                className="btn px-3 py-1 bg-blue-600 white-colour text-sm rounded hover:bg-blue-700"
                                            >
                                                Edit
                                            </button>
                                            <button
                                                onClick={() => deleteSession(s)}
                                                className="btn px-3 py-1 bg-red-600 text-sm rounded white-colour hover:bg-red-700"
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    )}
                                </li>
                            );
                        })}
                    </ul>
                )}
            </div>

            {error && <p className="text-red-500 mt-4">{error}</p>}
        </div>
    );
}

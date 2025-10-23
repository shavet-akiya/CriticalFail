"use client";

import { useEffect, useState } from "react";
import { formatSessionDate } from "@/helpers/helper_functions";
import Loading from "@/components/Loading";
import { useCampaign } from "@/contexts/CampaignContext";
import TextareaAutosize from "react-textarea-autosize";

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

const prompts = [
    "Scribe your schemes, tally your treasures, and prepare for chaos.",
    "The party awaits your wisdom. What quests shall unfold next?",
    "Pen your plans, oh Dungeon Master — your players’ fate lies within these notes.",
    "Record your secrets here before your players inevitably ruin them.",
    "The ink is still wet, and destiny is unwritten. Chronicle your next campaign chapter.",
    "Let your parchment bear the whispers of dragons, taverns, and tragic backstories.",
    "From crypt to castle, weave the tale that shall echo through taverns for ages.",
    "Inscribe your prophecy, adventurer — for tomorrow’s session awaits.",
    "What mischief brews next?",
    "The adventure starts with a note.",
    "Your next session begins with a keystroke.",
];

export default function SessionList() {
    const [sessions, setSessions] = useState<Session[]>([]);
    const [fetching, setFetching] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
    const [editedDocument, setEditedDocument] = useState<string>("");
    const [prompt, setPrompt] = useState("");
    const [savingSessionId, setSavingSessionId] = useState<string | null>(null);

    const { selectedCampaign } = useCampaign();
    const campaignId = selectedCampaign?.campaign_id;
    const campaignName = selectedCampaign?.campaign_name;
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    // Picks a random prompt line to display as the instruction
    useEffect(() => {
        const randomIndex = Math.floor(Math.random() * prompts.length);
        setPrompt(prompts[randomIndex]);
    }, []);

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
        setFetching(true);
    }, [campaignId]);


    async function saveDocument(session: Session) {
        setSavingSessionId(session.id);
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

            setEditingSessionId(null);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setSavingSessionId(null);
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
            <div className="p-6">
                {fetching ? (
                    <Loading />
                ) : sessions.length === 0 ? (
                    <div>
                        <div>
                            <h1 className="text-4xl text-center mb-4 ">
                                {campaignName} sessions
                            </h1>
                            <h2>{prompt}</h2>
                        </div>
                        <p>No sessions available.</p>
                    </div>
                ) : (
                    <>
                        <div className="flex flex-col items-center mb-8 obsidian-colour">
                            <h1 className="text-4xl text-center mb-4 select-none">
                                {campaignName} sessions
                            </h1>
                            <h2 className="text-lg italic red-colour select-none">{prompt}</h2>
                        </div>
                        <ul className="space-y-4">
                            {sessions.map((s) => {
                                const isEditing = editingSessionId === s.id;
                                return (
                                    <li
                                        key={s.id}
                                        className="p-8 bg-gray-50 rounded shadow obsidian-colour flex flex-col gap-4"
                                    >
                                        <div className="flex justify-between items-center ">
                                            <div>
                                                <p className="text-xl font-semibold select-none">
                                                    <p>Session: {formatSessionDate(s.metadata?.session_id ?? s.id)}</p>{" "}
                                                </p>
                                            </div>
                                        </div>

                                        <div className="flex flex-row justify-between items-center">
                                            <p className="text-lg font-semibold select-none">Session Notes</p>

                                        </div>

                                        {isEditing ? (
                                            <TextareaAutosize
                                                className="max-w-6xl border border-red-600 p-4 rounded obsidian-colour resize-none"
                                                minRows={2}
                                                value={editedDocument}
                                                disabled={savingSessionId === s.id}
                                                onChange={(e) => setEditedDocument(e.target.value)}
                                            />
                                        ) : (
                                            <p className="max-w-6xl w-full p-4 border border-gray-300 rounded obsidian-colour text-md whitespace-pre-wrap">
                                                {s.document}
                                            </p>
                                        )}


                                        {isEditing ? (
                                            <div className="flex gap-2 justify-end">
                                                <button
                                                    onClick={() => setEditingSessionId(null)}
                                                    disabled={savingSessionId === s.id}
                                                    className={`btn px-3 py-1 text-sm rounded ${savingSessionId === s.id
                                                        ? "bg-gray-300 cursor-not-allowed"
                                                        : "bg-gray-400 hover:bg-gray-500 white-colour"
                                                        }`}
                                                >
                                                    Cancel
                                                </button>
                                                <button
                                                    onClick={() => saveDocument(s)}
                                                    disabled={savingSessionId === s.id}
                                                    className={`btn px-3 py-1 text-sm rounded ${savingSessionId === s.id
                                                        ? "bg-green-400 cursor-not-allowed obsidian-colour"
                                                        : "bg-green-600 hover:bg-green-700"
                                                        }`}
                                                >
                                                    {savingSessionId === s.id ? "Saving..." : "Save"}
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
                    </>
                )}
            </div>

            {error && <p className="text-red-500 mt-4">{error}</p>}
        </div>
    );
}

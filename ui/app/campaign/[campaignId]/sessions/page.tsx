"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation"; // for app directory
import Loading from "@/components/Loading";
import AltSessionCard from "@/components/AltSessionCard";

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
                            <AltSessionCard
                                key={s.id}
                                session={s}
                                baseUrl={baseUrl}
                                fetchSessions={fetchSessions}
                                setError={setError}
                            />
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

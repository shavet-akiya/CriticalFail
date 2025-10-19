"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

interface Campaign {
    campaign_id: string;
    campaign_name: string;
    session_ids: string[];
    campaign_image_url?: string;
}

interface Character {
    character_id: string;
    name: string;
    class: string;
    race?: string;
    HP: number;
    AC: number;
    STR: number;
    DEX: number;
    CON: number;
    INT: number;
    WIS: number;
    CHA: number;
    npc?: boolean;
    session_id: string;
    campaign_id: string;
}

interface Location {
    location_id: string;
    location_name: string;
    description: string;
    session_id: string;
}

interface Event {
    event_id: string;
    event: string;
    event_summary: string;
    participants?: string;
    location?: string;
    event_tags?: string;
    session_id: string;
}

interface Session {
    session_id: string;
    processed_at?: string;
    characters?: Character[];
    locations?: Location[];
    events?: Event[];
    campaign_id: string;
}

export default function CampaignSummaryPage() {
    const { campaignId } = useParams();
    const [campaign, setCampaign] = useState<Campaign | null>(null);
    const [sessions, setSessions] = useState<Session[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    useEffect(() => {
        async function fetchCampaignAndSessions() {
            try {
                // 1️⃣ Fetch the campaign
                const resCampaign = await fetch(
                    `${baseUrl}/campaign/${campaignId}`
                );
                if (!resCampaign.ok)
                    throw new Error(
                        `Campaign not found: ${resCampaign.status}`
                    );

                const raw = await resCampaign.json();
                const data: Campaign = Array.isArray(raw) ? raw[0] : raw;

                // Normalize session_ids
                let sessionIds: string[] = [];
                if (Array.isArray(data.session_ids))
                    sessionIds = data.session_ids;
                else if (typeof data.session_ids === "string") {
                    try {
                        sessionIds = JSON.parse(data.session_ids);
                    } catch {
                        sessionIds = [];
                    }
                }
                setCampaign({ ...data, session_ids: sessionIds });

                // 2️⃣ Fetch all session details in parallel
                const sessionPromises = sessionIds.map(async (id) => {
                    const res = await fetch(`${baseUrl}/sessions/${id}`);
                    if (!res.ok) throw new Error(`Session ${id} not found`);
                    return res.json() as Promise<Session>;
                });

                const sessionData = await Promise.all(sessionPromises);
                setSessions(sessionData);
            } catch (e: any) {
                setError(e.message);
            } finally {
                setLoading(false);
            }
        }

        if (campaignId) fetchCampaignAndSessions();
    }, [campaignId, baseUrl]);

    if (loading)
        return <div className="p-6">Loading campaign and sessions...</div>;
    if (error) return <div className="p-6 text-red-500">{error}</div>;
    if (!campaign) return <div className="p-6">Campaign not found.</div>;

    return (
        <div className="p-6 flex flex-col items-center bg-gray-900 text-white min-h-screen w-full">
            <h1 className="text-4xl font-bold mb-4">
                {campaign.campaign_name}
            </h1>
            <p className="mb-2">Campaign ID: {campaign.campaign_id}</p>
            <p className="mb-4">Sessions: {sessions.length}</p>

            {sessions.length === 0 ? (
                <p className="text-gray-400">No sessions yet.</p>
            ) : (
                <ul className="space-y-6 w-full max-w-4xl">
                    {sessions.map((session) => (
                        <li
                            key={session.session_id}
                            className="bg-gray-800 p-4 rounded shadow-sm"
                        >
                            <p className="font-bold mb-1">
                                Session ID: {session.session_id}
                            </p>
                            <p className="text-gray-300 mb-2">
                                Processed at: {session.processed_at || "N/A"}
                            </p>

                            <div className="mb-2">
                                <h3 className="font-semibold">Characters</h3>
                                {session.characters?.length ? (
                                    <ul className="list-disc list-inside">
                                        {session.characters.map((c) => (
                                            <li key={c.character_id}>
                                                {c.name} ({c.class},{" "}
                                                {c.race || "unknown"}) – HP:
                                                {c.HP}, AC:{c.AC}, STR:{c.STR},
                                                DEX:{c.DEX}, CON:{c.CON}, INT:
                                                {c.INT}, WIS:{c.WIS}, CHA:
                                                {c.CHA}
                                            </li>
                                        ))}
                                    </ul>
                                ) : (
                                    <p className="text-gray-400">
                                        No characters.
                                    </p>
                                )}
                            </div>

                            <div className="mb-2">
                                <h3 className="font-semibold">Locations</h3>
                                {session.locations?.length ? (
                                    <ul className="list-disc list-inside">
                                        {session.locations.map((l) => (
                                            <li key={l.location_id}>
                                                {l.location_name}:{" "}
                                                {l.description}
                                            </li>
                                        ))}
                                    </ul>
                                ) : (
                                    <p className="text-gray-400">
                                        No locations.
                                    </p>
                                )}
                            </div>

                            <div className="mb-2">
                                <h3 className="font-semibold">Events</h3>
                                {session.events?.length ? (
                                    <ul className="list-disc list-inside">
                                        {session.events.map((e) => (
                                            <li key={e.event_id}>
                                                <strong>{e.event}</strong> –{" "}
                                                {e.event_summary}
                                                {e.participants && (
                                                    <>
                                                        {" "}
                                                        (Participants:{" "}
                                                        {e.participants})
                                                    </>
                                                )}
                                                {e.location && (
                                                    <> @ {e.location}</>
                                                )}
                                                {e.event_tags && (
                                                    <> [{e.event_tags}]</>
                                                )}
                                            </li>
                                        ))}
                                    </ul>
                                ) : (
                                    <p className="text-gray-400">No events.</p>
                                )}
                            </div>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

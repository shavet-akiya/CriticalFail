"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

type Event = {
    event_id: string;
    session_id: string;
    timeline_order: number;
    event: string;
    event_summary: string;
    participants: string;
    location: string;
    event_tags: string;
    type: string;
};

export default function Timeline() {
    const { campaignId } = useParams();
    const [events, setEvents] = useState<Event[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!campaignId) return;

        const fetchEvents = async () => {
            try {
                const res = await fetch(
                    `${baseUrl}/campaigns/${campaignId}/events`
                );
                if (!res.ok)
                    throw new Error(`Failed to fetch events: ${res.status}`);
                const data = await res.json();
                setEvents(data.events || []);
            } catch (err: any) {
                console.error("Failed to fetch events:", err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchEvents();
    }, [campaignId]);

    if (loading) return <div>Loading timeline…</div>;
    if (error) return <div className="text-red-500">Error: {error}</div>;
    if (events.length === 0)
        return <div>No events found for this campaign.</div>;

    // Group events by session_id
    const sessionsMap: Record<string, Event[]> = {};
    events.forEach((ev) => {
        if (!sessionsMap[ev.session_id]) sessionsMap[ev.session_id] = [];
        sessionsMap[ev.session_id].push(ev);
    });

    // Sort events in each session by timeline_order
    Object.keys(sessionsMap).forEach((sessionId) => {
        sessionsMap[sessionId].sort(
            (a, b) => a.timeline_order - b.timeline_order
        );
    });

    return (
        <div className="max-w-4xl mx-auto p-4 space-y-8">
            {Object.entries(sessionsMap).map(([sessionId, sessionEvents]) => (
                <div key={sessionId} className="border p-4 rounded shadow-sm">
                    <h2 className="text-xl font-bold mb-4">
                        Session {sessionId}
                    </h2>
                    <ul className="space-y-4">
                        {sessionEvents.map((ev) => (
                            <li
                                key={ev.event_id}
                                className="p-4 border rounded hover:bg-gray-50"
                            >
                                <h3 className="font-semibold text-lg">
                                    {ev.event}
                                </h3>
                                <p className="text-gray-700">
                                    {ev.event_summary}
                                </p>
                                <p className="text-sm text-gray-500">
                                    <strong>Participants:</strong>{" "}
                                    {ev.participants}
                                </p>
                                <p className="text-sm text-gray-500">
                                    <strong>Location:</strong> {ev.location}
                                </p>
                                <p className="text-sm text-gray-500">
                                    <strong>Tags:</strong> {ev.event_tags}
                                </p>
                            </li>
                        ))}
                    </ul>
                </div>
            ))}
        </div>
    );
}

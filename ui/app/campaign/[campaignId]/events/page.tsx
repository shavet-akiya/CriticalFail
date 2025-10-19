"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import EventCard from "@/components/eventCard"; // import the EventCard component

const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

type Event = {
    event_id: string;
    session_id: string;
    campaign_id: string;
    timeline_order: number;
    event: string;
    event_summary: string;
    participants?: string[];
    location?: string;
    event_tags?: string[];
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
                    `${baseUrl}/sessions/${campaignId}/events`,
                    { cache: "no-store" }
                );
                if (!res.ok)
                    throw new Error(`Failed to fetch events: ${res.status}`);
                const data = await res.json();

                const allEvents: Event[] = data.events || [];

                // Sort events globally
                allEvents.sort((a, b) => a.timeline_order - b.timeline_order);

                setEvents(allEvents);
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

    // Sort events within each session
    Object.values(sessionsMap).forEach((evList) =>
        evList.sort((a, b) => a.timeline_order - b.timeline_order)
    );

    return (
        <div className="max-w-4xl mx-auto p-4 space-y-8">
            {Object.entries(sessionsMap).map(([sessionId, sessionEvents]) => (
                <div key={sessionId} className="border p-4 rounded shadow-sm">
                    <h2 className="text-xl font-bold mb-4">
                        Session {sessionId}
                    </h2>
                    <div className="space-y-4">
                        {sessionEvents.map((ev) => (
                            <EventCard key={ev.event_id} event={ev} />
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
}

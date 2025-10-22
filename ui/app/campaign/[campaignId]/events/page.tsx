"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import EventCard, { Event } from "@/components/eventCard";

const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

export default function CampaignEventsPage() {
    const { campaignId } = useParams<{ campaignId: string }>();
    const [events, setEvents] = useState<Event[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!campaignId) return;

        const fetchEvents = async () => {
            try {
                const res = await fetch(`${baseUrl}/events/${campaignId}/`, {
                    cache: "no-store",
                });
                if (!res.ok)
                    throw new Error(`Failed to fetch events: ${res.status}`);
                const data = await res.json();

                // Transform participants and tags into arrays
                const allEvents: Event[] = (data.events || []).map(
                    (ev: any) => ({
                        ...ev,
                        participants: ev.participants
                            ? ev.participants
                                  .split(",")
                                  .map((p: string) => p.trim())
                            : [],
                        event_tags: ev.event_tags
                            ? ev.event_tags
                                  .split(",")
                                  .map((t: string) => t.trim())
                            : [],
                    })
                );

                // Sort by session_id, then timeline_order
                allEvents.sort((a, b) => {
                    if (a.session_id === b.session_id) {
                        return (
                            (a.timeline_order || 0) - (b.timeline_order || 0)
                        );
                    }
                    return a.session_id.localeCompare(b.session_id);
                });

                setEvents(allEvents);
            } catch (err: any) {
                console.error(err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchEvents();
    }, [campaignId]);

    if (loading) return <div>Loading events…</div>;
    if (error) return <div className="text-red-500">Error: {error}</div>;
    if (events.length === 0)
        return <div>No events found for this campaign.</div>;

    // Group by session_id
    const sessionsMap: Record<string, Event[]> = {};
    events.forEach((ev) => {
        if (!sessionsMap[ev.session_id]) sessionsMap[ev.session_id] = [];
        sessionsMap[ev.session_id].push(ev);
    });

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

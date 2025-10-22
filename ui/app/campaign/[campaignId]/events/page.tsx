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
    const [searchTerm, setSearchTerm] = useState("");

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

    // Filter events by search term safely
    const filteredEvents = events.filter((ev) => {
        const keyword = searchTerm.toLowerCase();

        const title = ev.event?.toLowerCase() || "";
        const summary = ev.event_summary?.toLowerCase() || "";
        const tags = (ev.event_tags || []).map((t) => t.toLowerCase());
        const participants = (ev.participants || []).map((p) =>
            p.toLowerCase()
        );

        return (
            title.includes(keyword) ||
            summary.includes(keyword) ||
            tags.some((t) => t.includes(keyword)) ||
            participants.some((p) => p.includes(keyword))
        );
    });

    // Group events by session_id
    const sessionsMap: Record<string, Event[]> = {};
    filteredEvents.forEach((ev) => {
        if (!sessionsMap[ev.session_id]) sessionsMap[ev.session_id] = [];
        sessionsMap[ev.session_id].push(ev);
    });

    return (
        <div className="max-w-4xl mx-auto p-4 space-y-4 text-black">
            <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search events..."
                className="border p-2 rounded w-full mb-4"
            />

            {Object.entries(sessionsMap).length === 0 ? (
                <div>No events match your search.</div>
            ) : (
                Object.entries(sessionsMap).map(
                    ([sessionId, sessionEvents]) => (
                        <div
                            key={sessionId}
                            className="border p-4 rounded shadow-sm text-black"
                        >
                            <h2 className="text-xl font-bold mb-4 text-black">
                                Session {sessionId}
                            </h2>
                            <div className="space-y-4">
                                {sessionEvents.map((ev) => (
                                    <EventCard key={ev.event_id} event={ev} />
                                ))}
                            </div>
                        </div>
                    )
                )
            )}
        </div>
    );
}

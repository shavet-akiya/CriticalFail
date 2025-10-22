"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import EventCard, { Event } from "@/components/eventCard";
import FilterSidebar from "@/components/FilterSidebar";
import Loading from "@/components/Loading"

const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

export default function CampaignEventsPage() {
    const { campaignId } = useParams<{ campaignId: string }>();
    const [events, setEvents] = useState<Event[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState("");
    const [filters, setFilters] = useState<string[]>([]);

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

                const allEvents: Event[] = (data.events || []).map((ev: any) => ({
                    ...ev,
                    participants: ev.participants
                        ? ev.participants.split(",").map((p: string) => p.trim())
                        : [],
                    event_tags: ev.event_tags
                        ? ev.event_tags.split(",").map((t: string) => t.trim())
                        : [],
                }));

                allEvents.sort((a, b) => {
                    if (a.session_id === b.session_id) {
                        return (a.timeline_order || 0) - (b.timeline_order || 0);
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

    const filteredEvents = useMemo(() => {
        if (!events || events.length === 0) return [];
        const keyword = searchTerm.toLowerCase();

        return events.filter((ev) => {
            const title = ev.event?.toLowerCase() || "";
            const summary = ev.event_summary?.toLowerCase() || "";
            const tags = (ev.event_tags || []).map((t) => t.toLowerCase());
            const participants = (ev.participants || []).map((p) => p.toLowerCase());

            const matchesSearch =
                title.includes(keyword) ||
                summary.includes(keyword) ||
                tags.some((t) => t.includes(keyword)) ||
                participants.some((p) => p.includes(keyword));

            const matchesFilters =
                filters.length === 0 ||
                filters.every((f) =>
                    (ev.event_tags || []).map((t) => t.toLowerCase()).includes(f.toLowerCase())
                );

            return matchesSearch && matchesFilters;
        });
    }, [events, searchTerm, filters]);

    const sessionsMap: Record<string, Event[]> = useMemo(() => {
        const map: Record<string, Event[]> = {};
        filteredEvents.forEach((ev) => {
            if (!map[ev.session_id]) map[ev.session_id] = [];
            map[ev.session_id].push(ev);
        });
        return map;
    }, [filteredEvents]);

    if (loading) return <Loading />;
    if (error) return <div className="text-red-500">Error: {error}</div>;

    return (
        <div className="flex flex-col sm:flex-row max-w-6xl mx-auto gap-6 p-4">
            <div className="sm:w-64 flex-shrink-0">
                <FilterSidebar
                    events={events}
                    filters={filters}
                    setFilters={setFilters}
                />
            </div>

            <div className="flex-1 space-y-4 obsidian-colour">
                <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search events..."
                    className="border p-2 rounded w-full mb-4"
                />

                {Object.entries(sessionsMap).length === 0 ? (
                    <div>No events match your search or filters.</div>
                ) : (
                    Object.entries(sessionsMap).map(([sessionId, sessionEvents]) => (
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
                    ))
                )}
            </div>
        </div>
    );
}

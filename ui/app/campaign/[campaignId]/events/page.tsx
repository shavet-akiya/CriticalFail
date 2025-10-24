"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import EventCard, { Event } from "@/components/eventCard";
import FilterSidebar from "@/components/FilterSidebar";
import Loading from "@/components/Loading";
import { formatSessionDate } from "@/helpers/helper_functions";

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

    const filteredEvents = useMemo(() => {
        if (!events || events.length === 0) return [];
        const keyword = searchTerm.toLowerCase();

        return events.filter((ev) => {
            const title = ev.event?.toLowerCase() || "";
            const summary = ev.event_summary?.toLowerCase() || "";
            const tags = (ev.event_tags || []).map((t) => t.toLowerCase());
            const participants = (ev.participants || []).map((p) =>
                p.toLowerCase()
            );

            const matchesSearch =
                title.includes(keyword) ||
                summary.includes(keyword) ||
                tags.some((t) => t.includes(keyword)) ||
                participants.some((p) => p.includes(keyword));

            const matchesFilters =
                filters.length === 0 ||
                filters.every((f) =>
                    (ev.event_tags || [])
                        .map((t) => t.toLowerCase())
                        .includes(f.toLowerCase())
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
        <div className="pl-16 pr-16 text-black h-screen flex flex-col pb-16">
            <div className="w-full bg-purple-colour rounded-b-lg shadow-md py-6 px-4 mb-6 text-center sticky top-0 z-20">
                <h1 className="text-4xl font-bold text-white tracking-wide">
                    Campaign Event History
                </h1>
            </div>

            <div className="flex flex-col md:flex-row gap-6">
                {/* Sidebar */}
                <div className="md:w-1/3 w-full sticky top-16 self-start">
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        placeholder="Search events..."
                        className="border p-2 rounded w-full mb-4 obsidian-colour"
                    />
                    <div className="border border-obsidian rounded-md shadow-md bg-white-colour p-4 flex flex-col">
                        <FilterSidebar
                            events={events}
                            filters={filters}
                            setFilters={setFilters}
                        />
                    </div>
                </div>

                {/* Main content */}
                <div className="md:w-2/3 w-full min-h-screen overflow-y-auto md:p-6 text-black flex flex-col justify-start">
                    {Object.entries(sessionsMap).length === 0 ? (
                        <div className="mt-0">
                            No events match your search or filters.
                        </div>
                    ) : (
                        Object.entries(sessionsMap).map(
                            ([sessionId, sessionEvents]) => (
                                <div
                                    key={sessionId}
                                    className="w-full border p-4 rounded shadow-sm bg-white mb-6"
                                >
                                    <h2 className="text-xl font-bold mb-4">
                                        Session: {formatSessionDate(sessionId)}
                                    </h2>
                                    <div className="space-y-4 w-full">
                                        {sessionEvents.map((ev) => (
                                            <EventCard
                                                key={ev.event_id}
                                                event={ev}
                                                showEdit={true}
                                            />
                                        ))}
                                    </div>
                                </div>
                            )
                        )
                    )}
                </div>
            </div>
        </div>
    );
}

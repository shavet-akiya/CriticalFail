// "use client";
// import { useState } from "react";
// import EventCard from "@/components/eventCard";
// import FilterDrawer from "@/components/filterDrawer";

// export default function Timeline() {
//     const [characterFilter, setCharacterFilter] = useState<
//         "all" | "players" | "npc"
//     >("all");
//     const [tagFilter, setTagFilter] = useState<string[]>([]);
//     const [themeFilter, setThemeFilter] = useState<string[]>([]);

//     return (
//         <div className="w-full">
//             <div className="grid grid-cols-[250px_1fr] min-h-screen"></div>
//         </div>
//     );
// }

"use client";
import { useEffect, useState } from "react";
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
    const [events, setEvents] = useState<Event[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchEvents = async () => {
            try {
                const res = await fetch(`/${baseUrl}/events`);
                const data = await res.json();
                setEvents(data.events || []);
            } catch (err) {
                console.error("Failed to fetch events:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchEvents();
    }, []);

    if (loading) return <div>Loading timeline…</div>;

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

"use client";

import EventCard from "@/components/eventCard";
import eventsData from "@/types/events.json";
import type { Event, CampaignTags } from "@/types/types";
import SearchBar from "@/components/SearchBar"

export default function Timeline() {
    const events: Event[] = eventsData.events
        .map((e) => ({
            ...e,
            event_tags: e.event_tags as CampaignTags[],
        }))
        .sort((a, b) => b.timeline_order - a.timeline_order);

    return (
        <div className="w-full flex flex-col items-center p-4">
            <h1 className="text-2xl font-bold mb-6 w-full max-w-3xl">
                Timeline
            </h1>
            <SearchBar />
            <div className="flex flex-col gap-4 w-full max-w-3xl">
                {events.map((event) => (
                    <EventCard key={event.event_id} event={event} />
                ))}
            </div>
        </div>
    );
}



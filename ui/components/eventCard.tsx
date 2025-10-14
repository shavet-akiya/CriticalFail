"use client";

import { useEffect, useState } from "react";
import type { Event, Character } from "@/types/types";

type EventCardProps = {
    event: Event;
};

export default function EventCard({ event }: EventCardProps) {
    return (
        <div className="card bg-base-100 border shadow-sm mb-4 p-4">
            <h2 className="text-lg font-extrabold">
                {event.timeline_order}. {event.event}
            </h2>
            <p className="mt-2">{event.event_summary}</p>
            <p className="mt-2">
                <strong>Participants:</strong> {event.participants.join(", ")}
            </p>
            <p className="mt-2">
                <strong>Location:</strong> {event.location}
            </p>
            <div className="flex gap-2 mt-2 flex-wrap">
                {event.event_tags.map((tag, i) => (
                    <span key={i} className="badge badge-outline">
                        {tag}
                    </span>
                ))}
            </div>
        </div>
    );
}

"use client";

import { useParams } from "next/navigation";
import Link from "next/link";

export interface Event {
    event_id: string;
    session_id: string;
    campaign_id?: string;
    timeline_order?: number;
    event?: string;
    event_summary?: string;
    participants?: string[];
    location?: string;
    event_tags?: string[];
    type?: string;
}

interface EventCardProps {
    event: Event;
    showEdit?: boolean;
}

export default function EventCard({ event, showEdit = false }: EventCardProps) {
    const { campaignId } = useParams<{ campaignId: string }>();

    const participants = Array.isArray(event.participants)
        ? event.participants
        : [];
    const tags = Array.isArray(event.event_tags) ? event.event_tags : [];

    return (
        <div className="w-full max-w-sm white-colour">
            <div className="card bg-white-colour shadow-md obsidian-colour rounded-lg select-none group p-4">
                <h2 className="text-xl font-bold">
                    {event.timeline_order || "?"}.{" "}
                    {event.event || "Unnamed Event"}
                </h2>
                <p className="text-sm">{event.event_summary || ""}</p>

                {participants.length > 0 && (
                    <p className="mt-2">
                        <strong>Participants:</strong>{" "}
                        {participants.join(", ")}
                    </p>
                )}
                {tags.length > 0 && (
                    <p className="mt-1">
                        <strong>Tags:</strong> {tags.join(", ")}
                    </p>
                )}
                {event.location && (
                    <p className="mt-1">
                        <strong>Location:</strong> {event.location}
                    </p>
                )}

                {showEdit && (
                    <Link
                        href={`/campaign/${campaignId}/events/${event.event_id}`}
                        className="btn btn-primary z-10 w-auto ml-auto block"
                        onClick={(e) => e.stopPropagation()}
                    >
                        Edit
                    </Link>
                )}


            </div>
        </div>
    );
}

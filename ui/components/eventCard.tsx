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
        <div className="w-full white-colour">
            <div className="card bg-white-colour shadow-md obsidian-colour rounded-lg select-none group p-6">
                <h2 className="text-xl font-bold pb-2">
                    {event.timeline_order || "?"}.{" "}
                    {event.event || "Unnamed Event"}
                </h2>

                <p className="pb-2">{event.event_summary || ""}</p>

                <hr className="h-0.5 bg-red-100 " />

                {participants.length > 0 && (
                    <p className="mt-2">
                        <strong>Participants:</strong>{" "}
                        {participants.join(", ")}
                    </p>
                )}

                {event.location && (
                    <p className="mt-1">
                        <strong>Location:</strong> {event.location}
                    </p>
                )}

                {tags.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-2 items-center pb-4">
                        <strong className="mr-2">Tags:</strong>
                        {tags.map((tag) => (
                            <span
                                key={tag}
                                className="badge badge-sm badge-outline"
                            >
                                {tag.charAt(0).toUpperCase() + tag.slice(1)}
                            </span>
                        ))}
                    </div>
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

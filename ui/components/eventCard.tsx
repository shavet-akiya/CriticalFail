"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
interface Event {
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

export default function EventCard({ event }: { event: Event }) {
    const { campaignId } = useParams<{ campaignId: string }>();

    return (
        <Link
            href={`/campaign/${campaignId}/events/${event.event_id}`}
            className="block"
        >
            <div className="card bg-base-100 w-full max-w-sm shadow-sm hover:bg-gray-700 rounded-lg cursor-pointer group relative">
                {/* Edit button */}
                <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg">
                    <button className="btn btn-primary rounded-full w-auto flex items-center gap-1">
                        <img
                            src="/svg/edit.svg"
                            alt="Edit"
                            className="w-4 h-4"
                        />
                        Edit
                    </button>
                </div>

                {/* Event header */}
                <div className="card-body text-white">
                    <div className="flex items-center justify-between">
                        <div>
                            <h2 className="text-xl font-bold">
                                {event.timeline_order}.{" "}
                                {event.event || "Unnamed Event"}
                            </h2>
                            <p className="text-sm">
                                {event.event_summary || ""}
                            </p>
                        </div>
                    </div>

                    {/* Event details */}
                    <div className="mt-2 text-sm">
                        {event.participants &&
                            event.participants.length > 0 && (
                                <p className="mt-2">
                                    <strong>Participants:</strong>{" "}
                                    {(Array.isArray(event.participants)
                                        ? event.participants
                                        : []
                                    ).join(", ")}
                                </p>
                            )}
                        {event.location && (
                            <p>
                                <strong>Location:</strong> {event.location}
                            </p>
                        )}
                        {event.event_tags && event.event_tags.length > 0 && (
                            <div className="flex gap-2 mt-2 flex-wrap">
                                {(Array.isArray(event.event_tags)
                                    ? event.event_tags
                                    : []
                                ).map((tag, i) => (
                                    <span
                                        key={i}
                                        className="badge badge-outline"
                                    >
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </Link>
    );
}

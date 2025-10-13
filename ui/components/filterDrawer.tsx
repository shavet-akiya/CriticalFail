"use client";

import { useEffect, useMemo, useState } from "react";
import type { Event } from "@/types/types";

interface FilterDrawerProps {
    filters: string[];
    setFilters: (filters: string[]) => void;
}

export default function FilterDrawer({
    filters,
    setFilters,
}: FilterDrawerProps) {
    const [events, setEvents] = useState<Event[]>([]);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetch("/api/events", { cache: "no-store" })
            .then((res) => {
                if (!res.ok)
                    throw new Error(`Failed to fetch events: ${res.status}`);
                return res.json();
            })
            .then((data) => setEvents(data.events ?? []))
            .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    }, []);

    const allTags = useMemo(() => {
        const tagSet = new Set<string>();
        events.forEach((e) => e.event_tags.forEach((t) => tagSet.add(t)));
        return Array.from(tagSet);
    }, [events]);

    const toggleFilter = (tag: string) => {
        setFilters(
            filters.includes(tag)
                ? filters.filter((f) => f !== tag)
                : [...filters, tag]
        );
    };

    const openModal = () => {
        const modal = document.getElementById(
            "my_modal_1"
        ) as HTMLDialogElement;
        if (modal) modal.showModal();
    };

    return (
        <div className="p-4">
            {error && <div className="text-error mb-2">{error}</div>}

            <div>
                {filters.map((tag) => (
                    <span
                        key={tag}
                        className="btn btn-outline rounded-lg"
                        onClick={() => toggleFilter(tag)}
                    >
                        {tag} ✕
                    </span>
                ))}

                <button
                    className="btn btn-outline rounded-lg"
                    onClick={openModal}
                >
                    + filter tag
                </button>
            </div>

            <dialog id="my_modal_1" className="modal">
                <div className="modal-box">
                    <h3 className="font-bold text-lg mb-2">Select Tags</h3>

                    <div className="flex flex-wrap gap-2">
                        {allTags.map((tag) => (
                            <span
                                key={tag}
                                className={`badge cursor-pointer ${
                                    filters.includes(tag)
                                        ? "badge-secondary"
                                        : "badge-outline"
                                }`}
                                onClick={() => toggleFilter(tag)}
                            >
                                {tag}
                            </span>
                        ))}
                    </div>

                    <div className="modal-action">
                        <form method="dialog">
                            <button className="btn">Close</button>
                        </form>
                    </div>
                </div>
            </dialog>
        </div>
    );
}

"use client";

import { useMemo } from "react";

export type CampaignTags =
    | "combat"
    | "exploration"
    | "player-to-player interaction"
    | "npc interaction"
    | "resting"
    | "investigation"
    | "world expansion"
    | "character expansion"
    | "lore expansion"
    | "misc";

interface Event {
    event_tags?: string[];
}

interface FilterSidebarProps {
    events: Event[];
    filters: string[];
    setFilters: (filters: string[]) => void;
}

const ALL_POSSIBLE_TAGS: CampaignTags[] = [
    "combat",
    "exploration",
    "player-to-player interaction",
    "npc interaction",
    "resting",
    "investigation",
    "world expansion",
    "character expansion",
    "lore expansion",
    "misc",
];

export default function FilterSidebar({
    events,
    filters,
    setFilters,
}: FilterSidebarProps) {
    // ✅ Collect which tags exist in the current events
    const existingTags = useMemo(() => {
        const tagSet = new Set<string>();
        events.forEach((e) => (e.event_tags ?? []).forEach((t) => tagSet.add(t)));
        return tagSet;
    }, [events]);

    const toggleFilter = (tag: string) => {
        if (!existingTags.has(tag)) return; // prevent clicking unavailable tags
        setFilters(
            filters.includes(tag)
                ? filters.filter((f) => f !== tag)
                : [...filters, tag]
        );
    };

    return (
        <div className="bg-white border rounded-xl p-4 shadow-sm text-black w-full sm:w-64 sticky top-4 self-start">
            <h3 className="text-lg font-semibold mb-3">Filters</h3>

            <div className="flex flex-wrap gap-2">
                {ALL_POSSIBLE_TAGS.map((tag) => {
                    const isAvailable = existingTags.has(tag);
                    const isActive = filters.includes(tag);

                    return (
                        <button
                            key={tag}
                            onClick={() => toggleFilter(tag)}
                            disabled={!isAvailable}
                            className={`px-3 py-1 rounded-full border text-sm transition 
                ${isActive
                                    ? "bg-blue-500 text-white border-blue-500"
                                    : isAvailable
                                        ? "bg-gray-100 hover:bg-gray-200 text-gray-800 border-gray-300"
                                        : "bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed opacity-60"
                                }`}
                        >
                            {tag}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}

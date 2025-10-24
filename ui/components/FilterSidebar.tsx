"use client";

import { useMemo, useState } from "react";

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
    const [collapsed, setCollapsed] = useState(true);

    const existingTags = useMemo(() => {
        const tagSet = new Set<string>();
        events.forEach((e) => (e.event_tags ?? []).forEach((t) => tagSet.add(t)));
        return tagSet;
    }, [events]);

    const toggleFilter = (tag: string) => {
        if (!existingTags.has(tag)) return;
        setFilters(
            filters.includes(tag)
                ? filters.filter((f) => f !== tag)
                : [...filters, tag]
        );
    };

    return (
        <div className="select-none flex flex-col">
            {/* Header with toggle only visible on md and smaller */}
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-semibold obsidian-colour">Filters</h3>
                <button
                    className="md:hidden p-2"
                    onClick={() => setCollapsed(!collapsed)}
                >
                    <img
                        src="/svg/arrow-down.svg"
                        alt="Toggle filters"
                        className={`w-4 h-4 transition-transform duration-200 ${collapsed ? "" : "rotate-180"}`}
                    />
                </button>
            </div>

            {/* Filters container */}
            <div
                className={`
            ${collapsed ? "hidden md:flex" : "flex"} 
            flex-wrap gap-2
        `}
            >
                {ALL_POSSIBLE_TAGS.map((tag) => {
                    const isAvailable = existingTags.has(tag);
                    const isActive = filters.includes(tag);
                    const formattedTag = tag.charAt(0).toUpperCase() + tag.slice(1);

                    return (
                        <button
                            key={tag}
                            onClick={() => toggleFilter(tag)}
                            disabled={!isAvailable}
                            className={`px-3 py-1 rounded-full border text-sm transition 
                        ${isActive
                                    ? "bg-blue-500 white-colour border-blue-500"
                                    : isAvailable
                                        ? "bg-gray-100 hover:bg-gray-200 text-gray-800 border-gray-300"
                                        : "bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed opacity-60"
                                }`}
                        >
                            {formattedTag}
                        </button>
                    );
                })}
            </div>
        </div>

    );
}

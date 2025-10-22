"use client";

import { useMemo } from "react";

interface Event {
    event_tags?: string[];
}

interface FilterSidebarProps {
    events: Event[];
    filters: string[];
    setFilters: (filters: string[]) => void;
}

export default function FilterSidebar({ events, filters, setFilters }: FilterSidebarProps) {
    // ✅ Derive unique tags from events
    const allTags = useMemo(() => {
        const tagSet = new Set<string>();
        events.forEach((e) => (e.event_tags ?? []).forEach((t) => tagSet.add(t)));
        return Array.from(tagSet).sort();
    }, [events]);

    const toggleFilter = (tag: string) => {
        setFilters(
            filters.includes(tag)
                ? filters.filter((f) => f !== tag)
                : [...filters, tag]
        );
    };

    return (
        <div className="bg-white border rounded-xl p-4 shadow-sm text-black w-full sm:w-64">
            <h3 className="text-lg font-semibold mb-3">Filters</h3>

            {/* Selected tags */}
            {filters.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-4">
                    {filters.map((tag) => (
                        <span
                            key={tag}
                            onClick={() => toggleFilter(tag)}
                            className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm cursor-pointer"
                        >
                            {tag} ✕
                        </span>
                    ))}
                </div>
            )}

            {/* Available tags */}
            <div className="flex flex-wrap gap-2">
                {allTags.length === 0 ? (
                    <p className="text-gray-500 text-sm">No tags available.</p>
                ) : (
                    allTags.map((tag) => (
                        <button
                            key={tag}
                            onClick={() => toggleFilter(tag)}
                            className={`px-3 py-1 rounded-full border text-sm transition ${filters.includes(tag)
                                    ? "bg-blue-500 text-white border-blue-500"
                                    : "bg-gray-100 hover:bg-gray-200 text-gray-800 border-gray-300"
                                }`}
                        >
                            {tag}
                        </button>
                    ))
                )}
            </div>
        </div>
    );
}

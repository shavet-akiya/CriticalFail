"use client";

import { useMemo } from "react";
import { events } from "@/types/mockData";

interface FilterDrawerProps {
    filters: string[];
    setFilters: (filters: string[]) => void;
}

export default function FilterDrawer({ filters, setFilters }: FilterDrawerProps) {
    const allTags = useMemo(() => {
        const tagSet = new Set<string>();
        events.forEach((e) => e.tags.forEach((t) => tagSet.add(t)));
        return Array.from(tagSet);
    }, []);

    const toggleFilter = (tag: string) => {
        setFilters(
            filters.includes(tag)
                ? filters.filter((f) => f !== tag)
                : [...filters, tag]
        );
    };

    const openModal = () => {
        const modal = document.getElementById("my_modal_1") as HTMLDialogElement;
        if (modal) modal.showModal();
    };

    return (
        <div className="p-4">

            {/* filtering tags + joins on the last */}
            <div>
                {filters.map((tag) => (
                    <span
                        key={tag}
                        className="btn btn-outline rounded-lg"
                        onClick={() => toggleFilter(tag)}> {tag}  ✕ </span>
                ))
                }

                <button className="btn btn-outline rounded-lg" onClick={openModal}>
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
                                className={`badge cursor-pointer ${filters.includes(tag)
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

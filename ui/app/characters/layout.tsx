"use client";

import { useState } from "react";
import { FilterProvider, useFilter } from "@/contexts/FilterContext";
import { CharacterProvider } from "@/contexts/CharacterContext";



export default function CharactersLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const [open, setOpen] = useState(false);

    return (
        <CharacterProvider>
            <FilterProvider>
                <div className="relative min-h-screen mt-8">
                    {children}
                    <FAB open={open} setOpen={setOpen} />
                </div>
            </FilterProvider>
        </CharacterProvider>
    );
}

function FAB({ open, setOpen }: { open: boolean; setOpen: (o: boolean) => void }) {
    const { setFilter } = useFilter();

    return (
        <div className="fixed bottom-8 right-8 flex flex-col items-end space-y-2">
            <div
                className={`flex flex-col items-end space-y-2 transition-all duration-300 ${open
                    ? "opacity-100 translate-y-0"
                    : "opacity-0 translate-y-4 pointer-events-none"
                    }`}
            >
                <button className="btn btn-lg shadow" onClick={() => setFilter("all")}>
                    All characters
                </button>
                <button className="btn btn-lg shadow" onClick={() => setFilter("players")}>
                    Players
                </button>
                <button className="btn btn-lg shadow" onClick={() => setFilter("npc")}>
                    NPC
                </button>
            </div>

            <button
                tabIndex={0}
                role="button"
                className="btn btn-lg rounded-md btn-success shadow-lg"
                onClick={() => setOpen(!open)}
            >
                Filter
            </button>
        </div>
    );
}
